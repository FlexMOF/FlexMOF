"""MOF-only token, pair, and periodic-position embeddings."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .graph_bias import MOFGraphAttentionBias
from .noise import TimeStepEncoder
from .periodic import iter_periodic_edge_slices, polynomial_cutoff
from .position import SpatialBias


class SequenceEmbedding(nn.Module):
    """Embedding used by the invariant sequence transformer."""

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(160, config.encoder_embed_dim)
        self.chain_id_proj = nn.Embedding(1000, config.encoder_embed_dim)
        self.time_step_encoder = TimeStepEncoder(
            config.num_timesteps,
            config.embedding_dim,
        )
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(config.encoder_embed_dim),
            nn.Linear(
                config.encoder_embed_dim,
                config.encoder_pair_embed_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                config.encoder_pair_embed_dim,
                config.encoder_pair_embed_dim,
                bias=False,
            ),
        )

    def forward(self, batch, time_step, clean_mask):
        padding_mask = batch["token_id"].eq(0)
        token_embedding = self.embed(batch["masked_token_type"])
        token_embedding += self.chain_id_proj(batch["chain_ids"])
        time_embedding = self.time_step_encoder(time_step, clean_mask).to(
            token_embedding.dtype
        )
        pair = self.pair_proj(token_embedding)
        pair = torch.einsum("blh,bkh->blkh", pair, pair).permute(1, 2, 0, 3)
        return token_embedding, padding_mask, time_embedding, pair


class SpatialEmbedding(nn.Module):
    """MOF spatial and bond bias used by the structure transformer."""

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(160, config.encoder_embed_dim)
        self.pos_embedding_bias = SpatialBias(config, key_prefix="")
        self.init_pos_embedding_bias = SpatialBias(config, key_prefix="init_")
        self.graph_2d_attention_bias = MOFGraphAttentionBias(config)
        # Preserve checkpoint keys, excluding this unused path from optimizer/DDP.
        if config.mof_disable_graphormer_2d:
            self.graph_2d_attention_bias.requires_grad_(False)
        self.config = config

    def forward(self, batch, padding_mask, clean_mask, periodic):
        if periodic.get("streaming_per_atom", False):
            raise ValueError("Use forward_streaming for a streaming periodic graph")
        embedding = self.embed(batch["masked_token_type"])

        # MOF degree is the only non-token atom feature used by this checkpoint.
        if not self.config.mof_disable_graphormer_2d:
            self._set_degrees(batch, periodic)
        position_bias, position_embedding = self.pos_embedding_bias(
            batch, padding_mask, periodic
        )
        embedding += position_embedding

        batch_size, _, node_count, expanded_count = position_bias.shape
        position_bias = position_bias.reshape(
            batch_size,
            self.config.num_attention_heads,
            node_count,
            expanded_count,
        ).contiguous()

        initial_bias, initial_embedding = self.init_pos_embedding_bias(
            batch, padding_mask, periodic
        )
        initial_mask = (batch["init_pos"] != 0.0).any(dim=-1).any(dim=-1)
        embedding[initial_mask] += initial_embedding[initial_mask]
        initial_bias = (
            initial_bias.masked_fill(~initial_mask[:, None, None, None], 0.0)
            .reshape(
                batch_size,
                self.config.num_attention_heads,
                node_count,
                expanded_count,
            )
            .contiguous()
        )
        position_bias += initial_bias

        if not self.config.mof_disable_graphormer_2d:
            bond_bias = self.graph_2d_attention_bias(batch, periodic)
            bond_bias = bond_bias.masked_fill(clean_mask[:, None, :, None], 0.0)
            position_bias += bond_bias
        return embedding, position_bias

    @staticmethod
    def _set_degrees(batch, periodic):
        batch["node_attr"] = torch.cat(
            [batch["node_attr"][:, :, :1],
             periodic["all_adj"].long().sum(dim=-1, keepdim=True),
             batch["node_attr"][:, :, 2:]],
            dim=-1,
        )

    def _streaming_edge_terms(
        self,
        batch,
        periodic,
        initial_present,
        edge_offset,
        edges,
    ):
        (
            edge_batch,
            query,
            cell_index,
            source,
            current_edge_bias,
            current_edge_features,
            current_edge_distances,
        ) = self.pos_embedding_bias.edge_terms(batch, periodic, edge_offset, edges)
        (
            _,
            _,
            _,
            _,
            initial_edge_bias,
            initial_edge_features,
            _,
        ) = self.init_pos_embedding_bias.edge_terms(batch, periodic, edge_offset, edges)
        initial_edge_bias = initial_edge_bias * initial_present[edge_batch, None]
        geometric_logits = current_edge_bias + initial_edge_bias
        edge_local_weight = polynomial_cutoff(
            current_edge_distances, periodic["cutoff"]
        ).to(current_edge_features.dtype)
        return (
            edge_batch,
            query,
            cell_index,
            source,
            geometric_logits,
            current_edge_features,
            initial_edge_features,
            edge_local_weight,
        )

    def _streaming_edge_reduction(
        self,
        batch,
        clean_mask,
        periodic,
        pair_bias,
        bond,
        initial_present,
        running_max,
        edge_offset,
        edges,
    ):
        batch_size, node_count = batch["token_id"].shape
        num_heads = self.config.num_attention_heads
        (
            edge_batch,
            query,
            cell_index,
            source,
            geometric_logits,
            current_edge_features,
            initial_edge_features,
            edge_local_weight,
        ) = self._streaming_edge_terms(
            batch,
            periodic,
            initial_present,
            edge_offset,
            edges,
        )
        group_index = (
            edge_batch[:, None] * num_heads
            + torch.arange(num_heads, device=edges.device)[None]
        ) * node_count + query[:, None]
        geometric_weight = torch.exp(geometric_logits) * edge_local_weight[:, None]
        central_only = periodic.get("central_cell_graph", False)
        if bond is None or central_only:
            # Every periodic image uses only geometric and learned pair bias;
            # even connected/disconnected classification must ignore topology.
            disconnected_weight = geometric_weight
        else:
            graph_position = periodic["graph_lookup"][edge_batch, cell_index, source].long()
            connected = graph_position.ge(0)
            disconnected_weight = geometric_weight * ~connected[:, None]
        flat_target = (group_index * node_count + source[:, None]).reshape(-1)
        disconnected_addition = (
            torch.zeros(
                batch_size * num_heads * node_count * node_count,
                dtype=geometric_weight.dtype,
                device=geometric_weight.device,
            )
            .scatter_add(
                0,
                flat_target,
                disconnected_weight.reshape(-1),
            )
            .view(batch_size, num_heads, node_count, node_count)
        )
        connected_addition = torch.zeros_like(disconnected_addition)
        if bond is not None and not central_only and bond["expanded"].size(-1):
            selected = connected.nonzero(as_tuple=True)[0]
            connected_logits = (
                geometric_logits[selected]
                + pair_bias[edge_batch[selected], :, query[selected], source[selected]]
                + bond["expanded"][
                    edge_batch[selected],
                    :,
                    query[selected],
                    graph_position[selected],
                ].masked_fill(
                    clean_mask[edge_batch[selected], query[selected]][:, None], 0.0
                )
            )
            connected_weight = (
                torch.exp(
                    connected_logits
                    - running_max[edge_batch[selected], :, query[selected]]
                )
                * edge_local_weight[selected, None]
            )
            selected_target = (
                group_index[selected] * node_count + source[selected, None]
            ).reshape(-1)
            connected_addition = (
                connected_addition.view(-1)
                .scatter_add(
                    0,
                    selected_target,
                    connected_weight.to(connected_addition.dtype).reshape(-1),
                )
                .view_as(connected_addition)
            )
        feature_target = edge_batch * node_count + query
        feature_index = feature_target[:, None].expand(
            -1, current_edge_features.size(-1)
        )
        current_addition = (
            torch.zeros(
                batch_size * node_count,
                current_edge_features.size(-1),
                dtype=current_edge_features.dtype,
                device=current_edge_features.device,
            )
            .scatter_add(
                0,
                feature_index,
                (current_edge_features * edge_local_weight[:, None]).to(
                    current_edge_features.dtype
                ),
            )
            .view(batch_size, node_count, -1)
        )
        initial_addition = (
            torch.zeros_like(current_addition)
            .view(batch_size * node_count, -1)
            .scatter_add(
                0,
                feature_index,
                (
                    initial_edge_features
                    * edge_local_weight[:, None]
                    * initial_present[edge_batch, None]
                ).to(current_addition.dtype),
            )
            .view_as(current_addition)
        )
        return (
            disconnected_addition,
            connected_addition,
            current_addition,
            initial_addition,
        )

    def forward_streaming(
        self,
        batch,
        padding_mask,
        clean_mask,
        periodic,
        pair_bias,
    ):
        """Collapse a sparse periodic multigraph without neighbor padding."""
        embedding = self.embed(batch["masked_token_type"])
        batch_size, node_count = padding_mask.shape

        bond = None
        if not self.config.mof_disable_graphormer_2d:
            bond = self.graph_2d_attention_bias(batch, periodic)
            self._set_degrees(batch, periodic)
        current_bias, current_features, current_distances = (
            self.pos_embedding_bias.central_terms(batch, padding_mask)
        )
        initial_bias, initial_features, _ = self.init_pos_embedding_bias.central_terms(
            batch, padding_mask
        )
        initial_present = (batch["init_pos"] != 0.0).any(dim=-1).any(dim=-1)
        initial_bias = initial_bias.masked_fill(
            ~initial_present[:, None, None, None], 0.0
        )
        local_weight = polynomial_cutoff(
            current_distances, periodic["cutoff"]
        ).to(current_features.dtype)
        pair_mask = padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(1)
        local_weight = local_weight.masked_fill(pair_mask, 0.0)
        current_sum = (current_features * local_weight[..., None]).sum(dim=2)
        initial_sum = (initial_features * local_weight[..., None]).sum(dim=2)

        central_logits = current_bias + initial_bias + pair_bias
        if bond is not None:
            central_logits = central_logits + bond["central"]
        central_logits = central_logits.masked_fill(
            padding_mask[:, None, None, :], float("-inf")
        )
        central_logits = central_logits.masked_fill(padding_mask[:, None, :, None], 0.0)
        running_max = central_logits.detach().amax(dim=-1)
        collapsed = (
            torch.exp(central_logits - running_max[..., None]) * local_weight[:, None]
        )
        disconnected_mass = torch.zeros_like(collapsed)

        for edge_offset, edges in iter_periodic_edge_slices(
            batch["pos"], batch["masked_token_type"], periodic
        ):
            reduction = partial(
                self._streaming_edge_reduction,
                batch,
                clean_mask,
                periodic,
                pair_bias,
                bond,
                initial_present,
                running_max,
                edge_offset,
            )
            if (
                self.training
                and torch.is_grad_enabled()
                and periodic["checkpoint_edge_activations"]
            ):
                (
                    disconnected_addition,
                    connected_addition,
                    current_addition,
                    initial_addition,
                ) = checkpoint(reduction, edges, use_reentrant=False)
            else:
                (
                    disconnected_addition,
                    connected_addition,
                    current_addition,
                    initial_addition,
                ) = reduction(edges)
            disconnected_mass = disconnected_mass + disconnected_addition
            collapsed = collapsed + connected_addition
            current_sum = current_sum + current_addition
            initial_sum = initial_sum + initial_addition

        disconnected_bias = (
            torch.zeros_like(collapsed) if bond is None else
            bond["disconnected"][None, :, None, None].expand_as(collapsed)
        )
        disconnected_bias = disconnected_bias.masked_fill(
            clean_mask[:, None, :, None], 0.0
        )
        disconnected_factor = torch.exp(
            pair_bias + disconnected_bias - running_max[..., None]
        )
        collapsed = collapsed + disconnected_mass * disconnected_factor

        collapsed = collapsed / collapsed.sum(dim=-1, keepdim=True).clamp_min(
            self.config.normalize_softmax_probs_after_local_attn_weights_eps
        )
        position_embedding = self.pos_embedding_bias.pos_embedding_proj(current_sum)
        initial_embedding = self.init_pos_embedding_bias.pos_embedding_proj(initial_sum)
        position_embedding += initial_embedding * initial_present[:, None, None]
        position_embedding = position_embedding.masked_fill(
            padding_mask[..., None], 0.0
        )
        embedding += position_embedding
        return embedding, central_logits, collapsed
