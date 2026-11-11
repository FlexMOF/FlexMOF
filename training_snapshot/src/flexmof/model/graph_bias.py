"""MOF bond-path attention bias computed with the CUDA graph preprocessor."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

try:
    import flexmof_preprocess_cuda
except ImportError as error:
    raise ImportError(
        "flexmof_preprocess_cuda is required; run "
        "`python setup_cuda.py build_ext --inplace`"
    ) from error

FLOYD_WARSHALL_MAX_DISTANCE = 1022
DISCONNECTED_SPATIAL_POSITION = 1023


def build_mof_dense_adj_from_all_adj(
    all_adj: torch.Tensor, n_node: int
) -> torch.Tensor:
    """Recreate the dense MOF topology adjacency currently used before CUDA FW."""
    n_graph, _, path_n_node = all_adj.size()
    adj = torch.zeros(
        [n_graph, path_n_node, path_n_node],
        device=all_adj.device,
        dtype=torch.bool,
    )
    adj[:, :n_node] = all_adj
    adj[:, :, :n_node] = all_adj.transpose(1, 2)
    return adj


def _run_cuda_floyd_warshall(
    adj: torch.Tensor,
    num_atoms_to_consider: torch.Tensor,
    max_distance: int = FLOYD_WARSHALL_MAX_DISTANCE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_graph, path_n_node, _ = adj.size()
    spatial_pos = torch.zeros(
        [n_graph, path_n_node, path_n_node],
        device=adj.device,
        dtype=torch.int16,
    )
    floyd_pred = torch.zeros_like(spatial_pos)

    flexmof_preprocess_cuda.floyd_warshall_batch(
        adj.to(dtype=torch.int16),
        num_atoms_to_consider.to(device=adj.device, dtype=torch.int16),
        path_n_node,
        max_distance,
        spatial_pos,
        floyd_pred,
    )
    return spatial_pos, floyd_pred


def find_connected_components_from_adj(
    adj: torch.Tensor,
    valid_n: int,
    include_singletons: bool = True,
) -> List[torch.Tensor]:
    """Find undirected connected components with union-find, excluding padding nodes."""
    valid_n = int(valid_n)
    if valid_n <= 0:
        return []

    parent = list(range(valid_n))
    rank = [0] * valid_n

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        root_x = _find(x)
        root_y = _find(y)
        if root_x == root_y:
            return
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

    valid_adj = adj[:valid_n, :valid_n].detach().to(dtype=torch.bool)
    valid_adj = valid_adj | valid_adj.transpose(0, 1)
    edge_index = valid_adj.nonzero(as_tuple=False).cpu()
    for source, target in edge_index.tolist():
        if source != target:
            _union(source, target)

    grouped: Dict[int, List[int]] = {}
    for node_idx in range(valid_n):
        grouped.setdefault(_find(node_idx), []).append(node_idx)

    return [
        torch.tensor(nodes, dtype=torch.long, device=adj.device)
        for nodes in grouped.values()
        if include_singletons or len(nodes) > 1
    ]


def _remap_local_floyd_pred_to_global(
    floyd_pred: torch.Tensor,
    component_nodes: torch.Tensor,
    spatial_pos: Optional[torch.Tensor] = None,
    disconnected_spatial_pos: Optional[int] = None,
) -> torch.Tensor:
    mapped = floyd_pred.clone()
    valid_pred = (mapped >= 0) & (mapped < component_nodes.numel())
    if spatial_pos is not None:
        valid_pred = valid_pred & (spatial_pos > 2)
        if disconnected_spatial_pos is not None:
            valid_pred = valid_pred & (spatial_pos != disconnected_spatial_pos)
    # Boolean advanced indexing is a no-op for an empty mask.  Avoid testing
    # ``valid_pred.any()`` in Python because that forces a CUDA synchronization
    # once per connected component.
    mapped[valid_pred] = component_nodes[mapped[valid_pred].long()].to(mapped.dtype)
    return mapped


def component_wise_floyd_warshall_for_mof(
    adj: torch.Tensor,
    num_atoms_to_consider: torch.Tensor,
    disconnected_spatial_pos: int = DISCONNECTED_SPATIAL_POSITION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Production component-wise Floyd-Warshall without profiling barriers.

    The debug implementation below intentionally synchronizes CUDA around
    packing, kernel execution, and scattering and computes extensive Python
    statistics.  Calling it from every training forward serialized the whole
    GPU pipeline.  This routine performs the same tensor operations but leaves
    CUDA asynchronous and omits statistics that are unused in production.
    """
    if adj.dim() != 3 or adj.size(1) != adj.size(2):
        raise ValueError("adj must have shape [B, P, P].")
    n_graph, path_n_node, _ = adj.size()
    spatial_pos = torch.zeros(
        [n_graph, path_n_node, path_n_node], device=adj.device, dtype=torch.int16
    )
    floyd_pred = torch.zeros_like(spatial_pos)
    packed_components = []

    for sample_idx in range(n_graph):
        valid_n = int(num_atoms_to_consider[sample_idx].item())
        valid_n = max(0, min(valid_n, path_n_node))
        if valid_n <= 0:
            continue
        components = find_connected_components_from_adj(
            adj[sample_idx], valid_n, include_singletons=False
        )
        spatial_pos[sample_idx, :valid_n, :valid_n] = disconnected_spatial_pos
        diag_idx = torch.arange(valid_n, device=adj.device)
        spatial_pos[sample_idx, diag_idx, diag_idx] = 1
        packed_components.extend(
            (sample_idx, nodes) for nodes in components if nodes.numel() > 1
        )

    if not packed_components:
        return spatial_pos, floyd_pred

    component_sizes = [int(nodes.numel()) for _, nodes in packed_components]
    max_comp_size = max(component_sizes)
    packed_sizes = torch.tensor(component_sizes, device=adj.device, dtype=torch.int16)
    padded_adj = torch.zeros(
        [len(packed_components), max_comp_size, max_comp_size],
        device=adj.device,
        dtype=adj.dtype,
    )
    for packed_idx, (sample_idx, nodes) in enumerate(packed_components):
        size = int(nodes.numel())
        padded_adj[packed_idx, :size, :size] = (
            adj[sample_idx].index_select(0, nodes).index_select(1, nodes)
        )

    packed_spatial_pos, packed_floyd_pred = _run_cuda_floyd_warshall(
        padded_adj, packed_sizes
    )
    for packed_idx, (sample_idx, nodes) in enumerate(packed_components):
        size = int(nodes.numel())
        rows = nodes[:, None].expand(size, size)
        cols = nodes[None, :].expand(size, size)
        local_spatial = packed_spatial_pos[packed_idx, :size, :size]
        spatial_pos[sample_idx, rows, cols] = local_spatial
        floyd_pred[sample_idx, rows, cols] = _remap_local_floyd_pred_to_global(
            packed_floyd_pred[packed_idx, :size, :size],
            nodes,
            local_spatial,
            disconnected_spatial_pos,
        )
    return spatial_pos, floyd_pred


def component_wise_gen_edge_input_for_mof(
    adj: torch.Tensor,
    attn_edge_type: torch.Tensor,
    num_atoms_to_consider: torch.Tensor,
    multi_hop_max_dist: int,
    max_dist_token: int = FLOYD_WARSHALL_MAX_DISTANCE,
) -> torch.Tensor:
    """Component-wise gen_edge_input for MOF.

    The CUDA ``gen_edge_input_batch`` kernel templates its per-thread path buffer on
    the number of nodes and only supports up to 15360 nodes per graph. Under large
    radius (e.g. 40A) the per-atom expanded graph P can reach 2-3 x 10^4 nodes, which
    overflows that limit. Cross-component node pairs are disconnected and contribute
    only zeros to ``edge_input``, so running the kernel within each connected
    component and scattering the results back into the dense ``edge_input`` yields an
    element-wise identical result while keeping the per-call node count down to the
    largest component size.
    """
    if adj.dim() != 3 or adj.size(1) != adj.size(2):
        raise ValueError("adj must have shape [B, P, P].")
    n_graph, path_n_node, _ = adj.size()
    num_edge_features = int(attn_edge_type.size(-1))
    multi_hop_max_dist = int(multi_hop_max_dist)

    edge_input = torch.zeros(
        [n_graph, path_n_node, path_n_node, multi_hop_max_dist, num_edge_features],
        device=attn_edge_type.device,
        dtype=attn_edge_type.dtype,
    )
    if multi_hop_max_dist <= 0 or path_n_node == 0:
        return edge_input

    # Collect all non-singleton connected components across the batch. Singletons have
    # no edges and therefore contribute nothing to edge_input.
    packed_components = []
    for sample_idx in range(n_graph):
        valid_n = int(num_atoms_to_consider[sample_idx].item())
        valid_n = max(0, min(valid_n, path_n_node))
        if valid_n <= 0:
            continue
        components = find_connected_components_from_adj(
            adj[sample_idx], valid_n, include_singletons=False
        )
        for component_nodes in components:
            if int(component_nodes.numel()) > 1:
                packed_components.append((sample_idx, component_nodes))

    n_packed = len(packed_components)
    if n_packed == 0:
        return edge_input

    component_sizes = [int(nodes.numel()) for _, nodes in packed_components]
    max_comp_size = max(component_sizes)
    packed_sizes = torch.tensor(component_sizes, device=adj.device, dtype=torch.int16)

    padded_adj = torch.zeros(
        [n_packed, max_comp_size, max_comp_size],
        device=adj.device,
        dtype=adj.dtype,
    )
    padded_edge_type = torch.zeros(
        [n_packed, max_comp_size, max_comp_size, num_edge_features],
        device=attn_edge_type.device,
        dtype=attn_edge_type.dtype,
    )
    for packed_idx, (sample_idx, component_nodes) in enumerate(packed_components):
        component_size = int(component_nodes.numel())
        padded_adj[packed_idx, :component_size, :component_size] = (
            adj[sample_idx]
            .index_select(0, component_nodes)
            .index_select(1, component_nodes)
        )
        padded_edge_type[packed_idx, :component_size, :component_size] = (
            attn_edge_type[sample_idx]
            .index_select(0, component_nodes)
            .index_select(1, component_nodes)
        )

    # Local-index Floyd-Warshall on the packed components so that the predecessor
    # matrix fed to gen_edge_input uses component-local indices (required by the path
    # reconstruction inside the kernel).
    batched_spatial_pos, batched_floyd_pred = _run_cuda_floyd_warshall(
        padded_adj, packed_sizes
    )

    packed_edge_input = torch.zeros(
        [n_packed, max_comp_size, max_comp_size, multi_hop_max_dist, num_edge_features],
        device=attn_edge_type.device,
        dtype=attn_edge_type.dtype,
    )
    flexmof_preprocess_cuda.gen_edge_input_batch(
        max_dist_token,
        max_comp_size,
        num_edge_features,
        multi_hop_max_dist,
        packed_sizes,
        batched_floyd_pred,
        batched_spatial_pos,
        padded_edge_type,
        packed_edge_input,
    )

    for packed_idx, (sample_idx, component_nodes) in enumerate(packed_components):
        component_size = int(component_nodes.numel())
        rows = component_nodes[:, None].expand(component_size, component_size)
        cols = component_nodes[None, :].expand(component_size, component_size)
        edge_input[sample_idx, rows, cols] = packed_edge_input[
            packed_idx, :component_size, :component_size
        ]

    return edge_input


class MOFGraphAttentionBias(nn.Module):
    """Shortest-path bond bias for an expanded periodic MOF graph."""

    PERIODIC_GRAPH_PREFIX = "full_central_tensor_v1"

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist
        self.edge_hidden_dim = config.encoder_embed_dim // config.num_attention_heads
        self.edge_encoder = nn.Embedding(
            config.num_edges + 1, self.edge_hidden_dim, padding_idx=0
        )
        self.edge_dis_encoder = nn.Embedding(
            config.num_edge_dis * self.edge_hidden_dim * self.num_heads, 1
        )
        self.mof_spatial_pos_encoder = nn.Embedding(
            config.mof_max_shortest_path_distance + 10,
            self.num_heads,
            padding_idx=0,
        )

    def forward(self, batch, periodic):
        node_count = batch["node_attr"].size(1)
        all_adj = periodic["all_adj"]
        all_edge_type = periodic["all_attn_edge_type"]
        graph_count, _, path_node_count = all_adj.shape

        adjacency = build_mof_dense_adj_from_all_adj(all_adj, node_count)
        edge_type = torch.zeros(
            [graph_count, path_node_count, path_node_count, all_edge_type.size(-1)],
            device=all_adj.device,
            dtype=all_edge_type.dtype,
        )
        edge_type[:, :node_count] = all_edge_type
        edge_type[:, :, :node_count] = all_edge_type.transpose(1, 2)
        if "all_expand_len" in periodic:
            # Union images are stored after the batch-padded central-cell width,
            # so include that entire prefix. Padding nodes are isolated and are
            # skipped by the component-wise preprocessing.
            considered = periodic["all_expand_len"] + node_count
        else:
            # Expanded nodes start after the full central-cell tensor, including
            # its lattice corners and padding, not after the real atom count.
            # Truncating this prefix treats the last expanded nodes differently
            # depending on which atoms happen to occupy those array positions.
            considered = node_count + periodic["expand_len"]

        spatial_pos, _ = component_wise_floyd_warshall_for_mof(adjacency, considered)
        edge_input = component_wise_gen_edge_input_for_mof(
            adjacency,
            edge_type,
            considered,
            self.multi_hop_max_dist,
        )

        spatial_pos = spatial_pos[:, :node_count].int()
        edge_input = edge_input[:, :node_count].int()
        if "expand_len" in periodic and periodic["expand_len"].dim() == 2:
            gather_index = periodic["softmax_gather_index"]
            expanded_spatial_pos = torch.gather(spatial_pos, dim=2, index=gather_index)
            spatial_pos = torch.cat(
                [spatial_pos[:, :, :node_count], expanded_spatial_pos], dim=2
            )
            expanded_edge_input = torch.gather(
                edge_input,
                dim=2,
                index=gather_index[..., None, None].expand(
                    -1,
                    -1,
                    -1,
                    edge_input.size(3),
                    edge_input.size(4),
                ),
            )
            edge_input = torch.cat(
                [edge_input[:, :, :node_count], expanded_edge_input], dim=2
            )
            path_node_count = node_count + gather_index.size(-1)
        graph_bias = torch.zeros(
            [graph_count, node_count, path_node_count],
            device=all_adj.device,
        )
        graph_bias = graph_bias[:, None].repeat(1, self.num_heads, 1, 1)
        graph_bias += self.mof_spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)

        distance = spatial_pos.clone()
        distance[distance == 0] = 1
        distance = torch.where(distance > 1, distance - 1, distance)
        distance = distance.clamp(0, self.multi_hop_max_dist)
        edge_input = edge_input[:, :, :, : self.multi_hop_max_dist]
        edge_input = self.edge_encoder(edge_input).mean(-2)
        max_distance = edge_input.size(-2)
        flattened = edge_input.permute(3, 0, 1, 2, 4).reshape(
            max_distance,
            node_count * path_node_count * graph_count,
            self.edge_hidden_dim,
        )
        flattened = torch.bmm(
            flattened,
            self.edge_dis_encoder.weight.reshape(
                -1, self.edge_hidden_dim, self.num_heads
            )[:max_distance],
        )
        edge_input = flattened.reshape(
            max_distance,
            graph_count,
            node_count,
            path_node_count,
            self.num_heads,
        ).permute(1, 2, 3, 0, 4)
        edge_input = (edge_input.sum(-2) / distance.float().unsqueeze(-1)).permute(
            0, 3, 1, 2
        )
        output = (graph_bias + edge_input).contiguous()
        if periodic.get("streaming_per_atom", False):
            disconnected = self.mof_spatial_pos_encoder.weight[
                DISCONNECTED_SPATIAL_POSITION
            ]
            return {
                "central": output[:, :, :, :node_count],
                "expanded": output[:, :, :, node_count:],
                "disconnected": disconnected,
            }
        return output
