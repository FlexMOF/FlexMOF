import torch
import warnings


def polynomial_cutoff(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """C2 quintic envelope, exactly zero at and beyond the cutoff radius."""
    ratio = (dist / cutoff).clamp(min=0.0, max=1.0)
    # Factored form of 1 - 10*x**3 + 15*x**4 - 6*x**5 avoids cancellation
    # near the boundary and cannot regain positive weight outside the radius.
    return (1 - ratio).pow(3) * (1 + 3 * ratio + 6 * ratio.square())


def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


def create_2d_expand_mask(
    batch_size: int,
    max_num_node: int,
    max_expand_num_node: int,
    num_atoms: torch.Tensor,
    num_expand_atoms: torch.Tensor,
):
    assert num_atoms.size(0) == batch_size
    assert num_expand_atoms.size(0) == batch_size
    assert num_atoms.max() <= max_num_node
    assert num_expand_atoms.max() <= max_expand_num_node
    device = num_atoms.device
    mask = torch.zeros(
        [batch_size, max_num_node, max_expand_num_node + 1], device=device
    )
    ones = torch.ones([batch_size, max_num_node, 1], dtype=mask.dtype, device=device)
    mask = mask.scatter(src=ones, dim=-1, index=num_expand_atoms[:, :, None])
    mask = mask.cumsum(dim=-1)[:, :, :-1]
    return mask.type(torch.bool)


def unpack_periodic_edges(periodic, packed_edges):
    """Decode packed ``(batch, query, cell, source)`` multigraph edges."""
    max_num_atoms = periodic["max_num_atoms"]
    candidate_count = periodic["candidate_count"]
    query_record = torch.div(packed_edges, candidate_count, rounding_mode="floor")
    candidate = packed_edges.remainder(candidate_count)
    batch_index = torch.div(query_record, max_num_atoms, rounding_mode="floor")
    query_index = query_record.remainder(max_num_atoms)
    cell_index = torch.div(candidate, max_num_atoms, rounding_mode="floor")
    source_index = candidate.remainder(max_num_atoms)
    return batch_index, query_index, cell_index, source_index


def iter_periodic_edge_slices(positions, atoms, periodic):
    """Yield ``(offset, packed edges)`` for access to aligned geometry caches."""
    offset = 0
    for edges in iter_periodic_edge_chunks(positions, atoms, periodic):
        yield offset, edges
        offset += edges.numel()


def iter_periodic_edge_chunks(positions, atoms, periodic):
    """Yield valid multigraph edges while bounding the candidate distance tensor."""
    if "packed_edges" in periodic:
        packed = periodic["packed_edges"]
        for start in range(0, packed.numel(), periodic["edge_chunk_size"]):
            yield packed[start : start + periodic["edge_chunk_size"]]
        return
    _, max_num_atoms = atoms.shape
    candidate_count = periodic["candidate_count"]
    for start in range(0, candidate_count, periodic["candidate_chunk_size"]):
        end = min(start + periodic["candidate_chunk_size"], candidate_count)
        flat_index = torch.arange(start, end, device=positions.device)
        source = flat_index.remainder(max_num_atoms)
        cell_index = torch.div(flat_index, max_num_atoms, rounding_mode="floor")
        candidate = positions[:, source] + periodic["cell_offsets"][:, cell_index]
        distances = torch.norm(
            positions.unsqueeze(2) - candidate.unsqueeze(1), p=2, dim=-1
        )
        valid = distances < periodic["cutoff"]
        valid &= atoms.ne(0).unsqueeze(-1)
        valid &= atoms[:, source].ne(0).unsqueeze(1)
        valid &= periodic["cell_allowed"][:, cell_index].unsqueeze(1)
        edge_batch, query, chunk_index = valid.nonzero(as_tuple=True)
        packed = (edge_batch * max_num_atoms + query) * candidate_count + flat_index[
            chunk_index
        ]
        for edge_start in range(0, packed.numel(), periodic["edge_chunk_size"]):
            yield packed[edge_start : edge_start + periodic["edge_chunk_size"]]


class CellExpander:

    def __init__(
        self,
        cutoff=40.0,
        expanded_token_cutoff=256,
        pbc_expanded_num_cell_per_direction=10,
        pbc_collision_threshold=1e-05,
        distance_chunk_size=4096,
        sparse_edge_chunk_size=262144,
        activation_checkpoint_edges=8_000_000,
    ):
        self.cells = []
        for i in range(
            -pbc_expanded_num_cell_per_direction,
            pbc_expanded_num_cell_per_direction + 1,
        ):
            for j in range(
                -pbc_expanded_num_cell_per_direction,
                pbc_expanded_num_cell_per_direction + 1,
            ):
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                ):
                    if i == 0 and j == 0 and (k == 0):
                        continue
                    self.cells.append([i, j, k])
        self.cells = torch.tensor(self.cells)
        self.cell_mask_for_pbc = self.cells != 0
        self.cutoff = cutoff
        self.expanded_token_cutoff = expanded_token_cutoff
        self.pbc_expanded_num_cell_per_direction = pbc_expanded_num_cell_per_direction
        self.conflict_cell_offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i != 0 or j != 0 or k != 0:
                        self.conflict_cell_offsets.append([i, j, k])
        self.conflict_cell_offsets = torch.tensor(self.conflict_cell_offsets)
        conflict_to_consider = self.cells.unsqueeze(
            1
        ) - self.conflict_cell_offsets.unsqueeze(0)
        conflict_to_consider_mask = (
            (conflict_to_consider * self.cells.unsqueeze(1) >= 0)
            & (torch.abs(conflict_to_consider) <= self.cells.unsqueeze(1).abs())
        ).all(dim=-1)
        conflict_to_consider_mask &= (
            (conflict_to_consider <= pbc_expanded_num_cell_per_direction)
            & (conflict_to_consider >= -pbc_expanded_num_cell_per_direction)
        ).all(dim=-1)
        self.conflict_to_consider_mask = conflict_to_consider_mask
        self.pbc_collision_threshold = pbc_collision_threshold
        self.distance_chunk_size = int(distance_chunk_size)
        self.sparse_edge_chunk_size = int(sparse_edge_chunk_size)
        self.activation_checkpoint_edges = int(activation_checkpoint_edges)

    def _chunked_expand_distances(self, pos, atoms, expand_pos, cell_mask):
        """Compute expansion membership with bounded distance temporaries.

        The legacy implementation materialized B x T x (C*T) distances and
        atom-type masks.  Only a per-candidate membership bit and minimum
        distance survive that computation, so reduce each candidate chunk
        immediately instead.
        """
        batch_size, max_num_atoms = pos.shape[:2]
        num_candidates = expand_pos.size(1)
        chunk_size = max(1, self.distance_chunk_size)
        expand_mask = torch.zeros(
            batch_size, num_candidates, dtype=torch.bool, device=pos.device
        )
        min_expand_dist = torch.full(
            (batch_size, num_candidates), torch.inf, dtype=pos.dtype, device=pos.device
        )
        original_padding = atoms.eq(0)
        num_expanded_cell = num_candidates // max_num_atoms
        candidate_atoms = atoms.repeat(1, num_expanded_cell)
        for start in range(0, num_candidates, chunk_size):
            end = min(start + chunk_size, num_candidates)
            candidate_pos = expand_pos[:, start:end]
            distances = torch.norm(
                pos.unsqueeze(2) - candidate_pos.unsqueeze(1), p=2, dim=-1
            )
            candidate_types = candidate_atoms[:, start:end]
            identical = atoms.unsqueeze(-1).eq(candidate_types.unsqueeze(1))
            valid_pair = (distances < self.cutoff) & ~identical
            valid_pair = valid_pair.masked_fill(original_padding.unsqueeze(-1), False)
            valid_candidate = candidate_types.ne(0) & cell_mask[:, start:end]
            expand_mask[:, start:end] = valid_pair.any(dim=1) & valid_candidate
            distances = distances.masked_fill(original_padding.unsqueeze(-1), torch.inf)
            distances = distances.masked_fill(~valid_candidate.unsqueeze(1), torch.inf)
            min_expand_dist[:, start:end] = distances.amin(dim=1)
        return (expand_mask, min_expand_dist)

    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
        Args:
            dist (tf.Tensor): distance tensor
            cutoff (float): cutoff distance
        Returns: polynomial cutoff functions
        """
        return polynomial_cutoff(dist, cutoff)

    def hard_threshold(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
        Args:
            dist (tf.Tensor): distance tensor
            cutoff (float): cutoff distance
        Returns: polynomial cutoff functions
        """
        result = (dist <= cutoff).float()
        return result

    def _get_cell_tensors(self, cell, use_local_attention):

        def _get_max_offset_for_dim(cell, dim):
            lattice_vec_0 = cell[:, dim, :]
            lattice_vec_1_2 = cell[
                :, torch.arange(3, dtype=torch.long, device=cell.device) != dim, :
            ]
            normal_vec = torch.cross(
                lattice_vec_1_2[:, 0, :], lattice_vec_1_2[:, 1, :], dim=-1
            )
            normal_vec = normal_vec / normal_vec.norm(dim=-1, keepdim=True)
            cutoff = self.cutoff
            max_offset = int(
                torch.max(
                    torch.ceil(
                        cutoff
                        / torch.abs(torch.sum(normal_vec * lattice_vec_0, dim=-1))
                    )
                )
            )
            return max_offset

        max_offsets = []
        for i in range(3):
            try:
                max_offset = _get_max_offset_for_dim(cell, i)
            except Exception as e:
                warnings.warn(f"{e} with cell {cell}")
                max_offset = self.pbc_expanded_num_cell_per_direction
            max_offsets.append(max_offset)
        max_offsets = torch.tensor(max_offsets, device=cell.device)
        self.cells = self.cells.to(device=cell.device)
        self.cell_mask_for_pbc = self.cell_mask_for_pbc.to(device=cell.device)
        mask = (self.cells.abs() <= max_offsets).all(dim=-1)
        selected_cell = self.cells[mask, :]
        return (selected_cell, self.cell_mask_for_pbc[mask, :], mask)

    def _get_conflict_mask(self, cell, pos, atoms):
        batch_size, max_num_atoms = pos.size()[:2]
        self.conflict_cell_offsets = self.conflict_cell_offsets.to(device=pos.device)
        self.conflict_to_consider_mask = self.conflict_to_consider_mask.to(
            device=pos.device
        )
        offset = torch.bmm(
            self.conflict_cell_offsets.unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(dtype=cell.dtype),
            cell,
        )
        expand_pos = (pos.unsqueeze(1) + offset.unsqueeze(2)).reshape(batch_size, -1, 3)
        expand_dist = (pos.unsqueeze(2) - expand_pos.unsqueeze(1)).norm(p=2, dim=-1)
        expand_atoms = atoms.repeat(1, self.conflict_cell_offsets.size()[0])
        atoms_identical_mask = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)
        conflict_mask = (
            ((expand_dist < self.pbc_collision_threshold) & atoms_identical_mask)
            .any(dim=1)
            .reshape(batch_size, -1, max_num_atoms)
        )
        all_conflict_mask = (
            torch.bmm(
                self.conflict_to_consider_mask.unsqueeze(0)
                .to(dtype=cell.dtype)
                .repeat(batch_size, 1, 1),
                conflict_mask.to(dtype=cell.dtype),
            )
            .long()
            .bool()
        )
        return all_conflict_mask

    def _get_per_atom_cell_tensors(self, pos, atoms, cell, pbc):
        """Enumerate every translation that can contain a cutoff neighbor.

        For a periodic basis A and its dual D (A @ D = I), a neighbor
        displacement d = x_source + n @ A - x_query with ||d|| < R obeys
        |n_i| <= R * ||D[:, i]|| + span(x @ D)_i. The span accounts for
        unwrapped linkers and noisy positions outside the primary cell.
        Neither the legacy cell-grid limit nor the token cap is used here.
        """
        pbc = pbc.bool()
        if not pbc.any():
            return (
                torch.empty(0, 3, dtype=torch.long, device=pos.device),
                torch.empty(pos.size(0), 0, dtype=torch.bool, device=pos.device),
            )
        basis = cell.double() * pbc.unsqueeze(-1)
        if not torch.isfinite(basis).all():
            raise ValueError("Periodic cell vectors must be finite")
        # Zero nonperiodic rows let the pseudoinverse handle 1D/2D periodicity
        # as well as fully periodic cells, without inventing image directions.
        dual = torch.linalg.pinv(basis, rtol=1e-12)
        if not torch.allclose(
            basis @ dual, torch.diag_embed(pbc.double()), atol=1e-7, rtol=1e-7
        ):
            raise ValueError("Periodic cell vectors must be linearly independent")
        valid = atoms.ne(0)
        fractional = pos.double().masked_fill(~valid[..., None], 0.0) @ dual
        upper = fractional.masked_fill(~valid[..., None], -torch.inf).amax(dim=1)
        lower = fractional.masked_fill(~valid[..., None], torch.inf).amin(dim=1)
        span = (upper - lower).masked_fill(~valid.any(dim=1, keepdim=True), 0.0)
        extent = self.cutoff * dual.norm(dim=-2) + span
        if not torch.isfinite(extent).all():
            raise ValueError("Periodic positions and translation bounds must be finite")
        bounds = torch.ceil(extent).long().masked_fill(~pbc, 0)
        bounds = bounds.masked_fill(~valid.any(dim=1, keepdim=True), 0)
        axes = [
            torch.arange(-limit, limit + 1, device=pos.device)
            for limit in bounds.amax(dim=0).tolist()
        ]
        selected = torch.cartesian_prod(*axes)
        selected = selected[selected.ne(0).any(dim=-1)]
        allowed = (selected.abs()[None] <= bounds[:, None]).all(dim=-1)
        used = allowed.any(dim=0)
        return selected[used], allowed[:, used]

    def _per_atom_candidate_chunk(
        self,
        pos,
        atoms,
        offsets,
        cell_allowed,
        start,
        end,
    ):
        """Return one bounded chunk of periodic candidates and its local mask."""
        max_num_atoms = pos.size(1)
        flat_index = torch.arange(start, end, device=pos.device)
        source_index = flat_index.remainder(max_num_atoms)
        cell_index = torch.div(flat_index, max_num_atoms, rounding_mode="floor")
        candidate_pos = pos[:, source_index] + offsets[:, cell_index]
        distances = torch.norm(
            pos.unsqueeze(2) - candidate_pos.unsqueeze(1), p=2, dim=-1
        )
        valid = distances < self.cutoff
        valid &= atoms.ne(0).unsqueeze(-1)
        valid &= atoms[:, source_index].ne(0).unsqueeze(1)
        valid &= cell_allowed[:, cell_index].unsqueeze(1)
        return flat_index, source_index, candidate_pos, distances, valid

    def _build_per_atom_graph_fields(
        self,
        selected_cells,
        union_flat_index,
        union_len,
        max_num_atoms,
        edge_index,
        target_cell_index,
        edge_attr,
        num_edges,
    ):
        """Build the union topology directly, without a cells-times-atoms graph."""
        batch_size, max_union_len = union_flat_index.shape
        device = edge_index.device
        all_adj = torch.zeros(
            [batch_size, max_num_atoms, max_num_atoms + max_union_len],
            dtype=torch.bool,
            device=device,
        )
        all_attn_edge_type = torch.zeros(
            [
                batch_size,
                max_num_atoms,
                max_num_atoms + max_union_len,
                edge_attr.size(-1),
            ],
            dtype=torch.uint8,
            device=device,
        )
        # Encode the actual dynamic range, including bond offsets outside the
        # selected set, so translations beyond the legacy grid cannot alias.
        radius = max(
            int(selected_cells.abs().max()) if selected_cells.numel() else 0,
            int(target_cell_index.abs().max()) if target_cell_index.numel() else 0,
        )
        factor = radius * 2 + 1

        def encode_cells(cell_offsets):
            shifted = cell_offsets + radius
            return (shifted[:, 0] * factor + shifted[:, 1]) * factor + shifted[:, 2]

        for batch_index in range(batch_size):
            sample_union_len = int(union_len[batch_index].item())
            sample_flat = union_flat_index[batch_index, :sample_union_len]
            union_atom = sample_flat.remainder(max_num_atoms)
            union_cell_position = torch.div(
                sample_flat, max_num_atoms, rounding_mode="floor"
            )
            union_keys = (
                encode_cells(selected_cells[union_cell_position]) * max_num_atoms
                + union_atom
            )

            sample_num_edges = int(num_edges[batch_index].item())
            sample_edges = edge_index[batch_index, :, :sample_num_edges]
            sample_cells = target_cell_index[batch_index, :sample_num_edges]
            sample_attr = edge_attr[batch_index, :sample_num_edges].to(torch.uint8)
            sources = torch.cat([sample_edges[0], sample_edges[1]])
            target_atoms = torch.cat([sample_edges[1], sample_edges[0]])
            target_cells = torch.cat([sample_cells, -sample_cells])
            attrs = torch.cat([sample_attr, sample_attr])

            central = target_cells.eq(0).all(dim=-1)
            columns = target_atoms.clone()
            valid = central.clone()
            periodic_edge = ~central
            if periodic_edge.any() and sample_union_len:
                desired_keys = (
                    encode_cells(target_cells[periodic_edge]) * max_num_atoms
                    + target_atoms[periodic_edge]
                )
                positions = torch.searchsorted(union_keys, desired_keys)
                in_range = positions < sample_union_len
                matches = torch.zeros_like(in_range)
                matches[in_range] = (
                    union_keys[positions[in_range]] == desired_keys[in_range]
                )
                periodic_indices = periodic_edge.nonzero(as_tuple=True)[0]
                matched_indices = periodic_indices[matches]
                columns[matched_indices] = max_num_atoms + positions[matches]
                valid[matched_indices] = True

            all_adj[batch_index, sources[valid], columns[valid]] = True
            all_attn_edge_type[batch_index, sources[valid], columns[valid]] = attrs[
                valid
            ]
        return all_adj, all_attn_edge_type

    @staticmethod
    def _build_central_graph_fields(atoms, edge_index, edge_attr, num_edges):
        """Linker topology in the original cell, independent of image count."""
        batch_size, node_count = atoms.shape
        adjacency = torch.zeros(batch_size, node_count, node_count,
                                dtype=torch.bool, device=atoms.device)
        edge_type = torch.zeros(batch_size, node_count, node_count, edge_attr.size(-1),
                                dtype=torch.uint8, device=atoms.device)
        valid = torch.arange(edge_index.size(-1), device=atoms.device)[None] < num_edges[:, None]
        batch, slot = valid.nonzero(as_tuple=True)
        source, target = edge_index[batch, :, slot].unbind(-1)
        attrs = edge_attr[batch, slot].to(torch.uint8)
        adjacency[batch, source, target] = True
        adjacency[batch, target, source] = True
        edge_type[batch, source, target] = attrs
        edge_type[batch, target, source] = attrs
        return {
            "central_cell_graph": True,
            "all_adj": adjacency,
            "all_attn_edge_type": edge_type,
            "all_expand_len": torch.zeros(batch_size, dtype=torch.long, device=atoms.device),
        }

    def _build_compact_graph_fields(
        self,
        pos,
        atoms,
        selected_cells,
        offsets,
        cell_allowed,
        edge_index,
        target_cell_index,
        edge_attr,
        num_edges,
    ):
        """Retain only periodic image nodes incident to a chemical bond.

        Geometric image nodes without a bond are isolated in the legacy union
        graph. They all receive the disconnected shortest-path token and do not
        need to be materialized for Floyd-Warshall.
        """
        batch_size, max_num_atoms = atoms.shape
        periodic_nodes = []
        directed_edges = []
        for batch_index in range(batch_size):
            count = int(num_edges[batch_index].item())
            sample_edges = edge_index[batch_index, :, :count]
            sample_cells = target_cell_index[batch_index, :count]
            sample_attr = edge_attr[batch_index, :count].to(torch.uint8)
            sources = torch.cat([sample_edges[0], sample_edges[1]])
            targets = torch.cat([sample_edges[1], sample_edges[0]])
            cells = torch.cat([sample_cells, -sample_cells])
            attrs = torch.cat([sample_attr, sample_attr])

            central = cells.eq(0).all(dim=-1)
            records = []
            for edge_position in (~central).nonzero(as_tuple=True)[0].tolist():
                cell_match = (selected_cells == cells[edge_position]).all(dim=-1)
                if not cell_match.any():
                    continue
                selected_index = int(cell_match.nonzero(as_tuple=True)[0][0].item())
                target = int(targets[edge_position].item())
                if not bool(cell_allowed[batch_index, selected_index]) or not bool(
                    atoms[batch_index, target].ne(0)
                ):
                    continue
                candidate = (
                    pos[batch_index, target] + offsets[batch_index, selected_index]
                )
                real_query = atoms[batch_index].ne(0)
                if not bool(
                    ((pos[batch_index] - candidate).norm(dim=-1) < self.cutoff)[
                        real_query
                    ].any()
                ):
                    continue
                records.append(
                    (
                        selected_index,
                        target,
                        int(sources[edge_position].item()),
                        attrs[edge_position],
                    )
                )
            keys = sorted(
                {(cell_index, target) for cell_index, target, _, _ in records}
            )
            key_to_column = {key: index for index, key in enumerate(keys)}
            periodic_nodes.append(keys)
            directed_edges.append(
                (
                    sources[central],
                    targets[central],
                    attrs[central],
                    records,
                    key_to_column,
                )
            )

        max_periodic_nodes = max((len(nodes) for nodes in periodic_nodes), default=0)
        device = pos.device
        all_adj = torch.zeros(
            batch_size,
            max_num_atoms,
            max_num_atoms + max_periodic_nodes,
            dtype=torch.bool,
            device=device,
        )
        all_edge_type = torch.zeros(
            batch_size,
            max_num_atoms,
            max_num_atoms + max_periodic_nodes,
            edge_attr.size(-1),
            dtype=torch.uint8,
            device=device,
        )
        graph_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        graph_lookup = torch.full(
            (batch_size, selected_cells.size(0), max_num_atoms),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for batch_index, nodes in enumerate(periodic_nodes):
            graph_len[batch_index] = len(nodes)
            if nodes:
                node_cells = torch.tensor(
                    [node[0] for node in nodes], dtype=torch.long, device=device
                )
                node_sources = torch.tensor(
                    [node[1] for node in nodes], dtype=torch.long, device=device
                )
                graph_lookup[
                    batch_index,
                    node_cells,
                    node_sources,
                ] = torch.arange(len(nodes), dtype=torch.int32, device=device)
            central_source, central_target, central_attr, records, key_to_column = (
                directed_edges[batch_index]
            )
            all_adj[batch_index, central_source, central_target] = True
            all_edge_type[batch_index, central_source, central_target] = central_attr
            for cell_index, target, source, attr in records:
                column = max_num_atoms + key_to_column[(cell_index, target)]
                all_adj[batch_index, source, column] = True
                all_edge_type[batch_index, source, column] = attr
        return {
            "all_adj": all_adj,
            "all_attn_edge_type": all_edge_type,
            "all_expand_len": graph_len,
            "graph_lookup": graph_lookup,
        }

    def expand_per_atom_streaming(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        pair_token_type,
        use_grad=False,
        edge_index=None,
        target_cell_index=None,
        edge_attr=None,
        num_edges=None,
        central_cell_graph=False,
    ):
        """Return an implicit per-atom multigraph streamed one cell at a time."""
        del num_atoms, pair_token_type
        with torch.set_grad_enabled(use_grad):
            pos = pos.float()
            init_pos = init_pos.float()
            cell = cell.float()
            selected_cells, cell_allowed = self._get_per_atom_cell_tensors(
                pos, atoms, cell, pbc
            )
            offsets = torch.matmul(selected_cells.to(cell.dtype).unsqueeze(0), cell)
            _, max_num_atoms = atoms.shape
            candidate_count = selected_cells.size(0) * max_num_atoms
            chunk_size = max(1, self.distance_chunk_size)
            packed_edges = []
            edge_relative = []
            edge_distance = []
            edge_init_distance = []
            for start in range(0, candidate_count, chunk_size):
                end = min(start + chunk_size, candidate_count)
                flat_index, source_index, candidate_pos, distances, valid = (
                    self._per_atom_candidate_chunk(
                        pos, atoms, offsets, cell_allowed, start, end
                    )
                )
                edge_batch, query, chunk_index = valid.nonzero(as_tuple=True)
                packed_edges.append(
                    (edge_batch * max_num_atoms + query) * candidate_count
                    + flat_index[chunk_index]
                )
                relative = (
                    pos[edge_batch, query] - candidate_pos[edge_batch, chunk_index]
                )
                edge_relative.append(relative)
                edge_distance.append(distances[edge_batch, query, chunk_index])
                cell_index = torch.div(
                    flat_index[chunk_index], max_num_atoms, rounding_mode="floor"
                )
                source = source_index[chunk_index]
                init_expanded = (
                    init_pos[edge_batch, source] + offsets[edge_batch, cell_index]
                )
                edge_init_distance.append(
                    (init_pos[edge_batch, query] - init_expanded).norm(dim=-1)
                )
            packed_edges = (
                torch.cat(packed_edges)
                if packed_edges
                else torch.empty(0, dtype=torch.long, device=pos.device)
            )
            empty_scalar = torch.empty(0, dtype=pos.dtype, device=pos.device)
            empty_vector = torch.empty(0, 3, dtype=pos.dtype, device=pos.device)
            result = {
                "streaming_per_atom": True,
                "selected_cells": selected_cells,
                "cell_offsets": offsets,
                "cell_allowed": cell_allowed,
                "cutoff": self.cutoff,
                "candidate_count": candidate_count,
                "max_num_atoms": max_num_atoms,
                "candidate_chunk_size": chunk_size,
                "edge_chunk_size": self.sparse_edge_chunk_size,
                "packed_edges": packed_edges,
                "edge_relative": (
                    torch.cat(edge_relative) if edge_relative else empty_vector
                ),
                "edge_distance": (
                    torch.cat(edge_distance) if edge_distance else empty_scalar
                ),
                "edge_init_distance": (
                    torch.cat(edge_init_distance)
                    if edge_init_distance
                    else empty_scalar
                ),
                "checkpoint_edge_activations": (
                    packed_edges.numel() > self.activation_checkpoint_edges
                ),
            }
            if edge_index is not None:
                if central_cell_graph:
                    result.update(self._build_central_graph_fields(
                        atoms, edge_index, edge_attr, num_edges
                    ))
                else:
                    result.update(self._build_compact_graph_fields(
                        pos,
                        atoms,
                        selected_cells,
                        offsets,
                        cell_allowed,
                        edge_index,
                        target_cell_index,
                        edge_attr,
                        num_edges,
                    ))
            return result

    def expand_per_atom(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        pair_token_type,
        use_grad=False,
        edge_index=None,
        target_cell_index=None,
        edge_attr=None,
        num_edges=None,
    ):
        """Construct an uncapped local periodic multigraph for every atom.

        Candidate images are visited twice in bounded chunks: once to size the
        packed tensors, and once to fill them. This retains every image inside
        ``cutoff`` without materializing the legacy
        ``B x N x (cells*N)`` distance tensor.
        """
        with torch.set_grad_enabled(use_grad):
            pos = pos.float()
            cell = cell.float()
            batch_size, max_num_atoms = pos.shape[:2]
            selected_cells, cell_allowed = self._get_per_atom_cell_tensors(
                pos, atoms, cell, pbc
            )
            offsets = torch.matmul(selected_cells.to(cell.dtype).unsqueeze(0), cell)
            candidate_count = selected_cells.size(0) * max_num_atoms
            chunk_size = max(1, self.distance_chunk_size)

            expand_len = torch.zeros(
                [batch_size, max_num_atoms], dtype=torch.long, device=pos.device
            )
            union_len = torch.zeros(batch_size, dtype=torch.long, device=pos.device)
            for start in range(0, candidate_count, chunk_size):
                end = min(start + chunk_size, candidate_count)
                _, _, _, _, valid = self._per_atom_candidate_chunk(
                    pos, atoms, offsets, cell_allowed, start, end
                )
                expand_len += valid.sum(dim=-1)
                union_len += valid.any(dim=1).sum(dim=-1)

            max_expand_len = int(expand_len.max().item()) if expand_len.numel() else 0
            max_union_len = int(union_len.max().item()) if union_len.numel() else 0
            outcell_index = torch.zeros(
                [batch_size, max_num_atoms, max_expand_len],
                dtype=torch.long,
                device=pos.device,
            )
            expand_pos = torch.zeros(
                [batch_size, max_num_atoms, max_expand_len, 3],
                dtype=pos.dtype,
                device=pos.device,
            )
            softmax_gather_index = torch.zeros_like(outcell_index)
            union_flat_index = torch.zeros(
                [batch_size, max_union_len], dtype=torch.long, device=pos.device
            )
            local_write = torch.zeros_like(expand_len)
            union_write = torch.zeros_like(union_len)

            for start in range(0, candidate_count, chunk_size):
                end = min(start + chunk_size, candidate_count)
                flat_index, source_index, candidate_pos, _, valid = (
                    self._per_atom_candidate_chunk(
                        pos, atoms, offsets, cell_allowed, start, end
                    )
                )
                union_valid = valid.any(dim=1)
                union_rank = (
                    torch.cumsum(union_valid, dim=-1, dtype=torch.int32).long() - 1
                )
                union_destination = union_write[:, None] + union_rank
                union_batch, union_chunk = union_valid.nonzero(as_tuple=True)
                union_flat_index[
                    union_batch, union_destination[union_batch, union_chunk]
                ] = flat_index[union_chunk]

                local_rank = torch.cumsum(valid, dim=-1, dtype=torch.int32).long() - 1
                local_destination = local_write[:, :, None] + local_rank
                valid_batch, valid_atom, valid_chunk = valid.nonzero(as_tuple=True)
                destination = local_destination[valid_batch, valid_atom, valid_chunk]
                outcell_index[valid_batch, valid_atom, destination] = source_index[
                    valid_chunk
                ]
                expand_pos[valid_batch, valid_atom, destination] = candidate_pos[
                    valid_batch, valid_chunk
                ]
                softmax_gather_index[valid_batch, valid_atom, destination] = (
                    max_num_atoms + union_destination[valid_batch, valid_chunk]
                )
                local_write += valid.sum(dim=-1)
                union_write += union_valid.sum(dim=-1)

            slot = torch.arange(max_expand_len, device=pos.device)
            expand_mask = slot.view(1, 1, -1) >= expand_len.unsqueeze(-1)
            gather_index = outcell_index.unsqueeze(-1).expand(-1, -1, -1, 3)
            source_pos = torch.gather(
                pos[:, None].expand(-1, max_num_atoms, -1, -1),
                dim=2,
                index=gather_index,
            )
            source_init_pos = torch.gather(
                init_pos[:, None].expand(-1, max_num_atoms, -1, -1),
                dim=2,
                index=gather_index,
            )
            init_expand_pos = source_init_pos + expand_pos - source_pos
            expand_pos = expand_pos.masked_fill(expand_mask.unsqueeze(-1), 0.0)
            init_expand_pos = init_expand_pos.masked_fill(
                expand_mask.unsqueeze(-1), 0.0
            )

            expand_pair_token_type = torch.gather(
                pair_token_type,
                dim=2,
                index=outcell_index.unsqueeze(-1).expand(
                    -1, -1, -1, pair_token_type.size(-1)
                ),
            )
            expand_node_type_edge = torch.cat(
                [pair_token_type, expand_pair_token_type], dim=2
            )
            central_dist = torch.norm(pos.unsqueeze(2) - pos.unsqueeze(1), p=2, dim=-1)
            expand_dist = torch.norm(pos.unsqueeze(2) - expand_pos, p=2, dim=-1)
            local_attention_weight = self.polynomial(
                torch.cat([central_dist, expand_dist], dim=-1),
                self.cutoff,
            ).float()
            padding_mask = atoms.eq(0)
            full_mask = torch.cat(
                [
                    padding_mask.unsqueeze(1).expand(-1, max_num_atoms, -1)
                    | padding_mask.unsqueeze(2),
                    expand_mask,
                ],
                dim=-1,
            )
            local_attention_weight = local_attention_weight.masked_fill(full_mask, 0.0)

            result = {
                "expand_pos": expand_pos,
                "outcell_index": outcell_index,
                "expand_mask": expand_mask,
                "local_attention_weight": local_attention_weight,
                "expand_node_type_edge": expand_node_type_edge,
                "init_expand_pos": init_expand_pos,
                "expand_len": expand_len,
                "all_expand_len": union_len,
                "softmax_gather_index": softmax_gather_index,
            }
            if edge_index is not None:
                all_adj, all_attn_edge_type = self._build_per_atom_graph_fields(
                    selected_cells,
                    union_flat_index,
                    union_len,
                    max_num_atoms,
                    edge_index,
                    target_cell_index,
                    edge_attr,
                    num_edges,
                )
                result.update(
                    all_adj=all_adj,
                    all_attn_edge_type=all_attn_edge_type,
                )
            return result

    def check_conflict(self, pos, atoms, pbc_expand_batched):
        expand_pos = pbc_expand_batched["expand_pos"]
        all_pos = torch.cat([pos, expand_pos], dim=1)
        num_expanded_atoms = all_pos.size()[1]
        all_dist = (all_pos.unsqueeze(1) - all_pos.unsqueeze(2)).norm(p=2, dim=-1)
        outcell_index = pbc_expand_batched["outcell_index"]
        all_atoms = torch.cat(
            [atoms, torch.gather(atoms, dim=-1, index=outcell_index)], dim=-1
        )
        atom_identical_mask = all_atoms.unsqueeze(1) == all_atoms.unsqueeze(-1)
        full_mask = torch.cat([atoms.eq(0), pbc_expand_batched["expand_mask"]], dim=-1)
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(-1), False
        )
        atom_identical_mask = atom_identical_mask.masked_fill(
            full_mask.unsqueeze(1), False
        )
        conflict_mask = (all_dist < self.pbc_collision_threshold) & atom_identical_mask
        conflict_mask[
            :,
            torch.arange(num_expanded_atoms, device=all_pos.device),
            torch.arange(num_expanded_atoms, device=all_pos.device),
        ] = False
        assert (
            ~conflict_mask.any()
        ), f"{all_dist[conflict_mask]} {all_atoms[conflict_mask.any(dim=-2)]}"

    def expand(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        pair_token_type,
        use_local_attention=True,
        use_grad=False,
        edge_index=None,
        target_cell_index=None,
        edge_attr=None,
        num_edges=None,
    ):
        with torch.set_grad_enabled(use_grad):
            pos = pos.float()
            cell = cell.float()
            batch_size, max_num_atoms = pos.size()[:2]
            non_batch_cell_tensor, cell_mask, selected_cell_mask = (
                self._get_cell_tensors(cell, use_local_attention)
            )
            cell_tensor = (
                non_batch_cell_tensor.unsqueeze(0)
                .repeat(batch_size, 1, 1)
                .to(dtype=cell.dtype)
            )
            num_expanded_cell = cell_tensor.size()[1]
            offset = torch.bmm(cell_tensor, cell)
            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(2)
            expand_pos = expand_pos.view(batch_size, -1, 3)
            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .expand(-1, -1, max_num_atoms)
                .reshape(batch_size, -1)
            )
            expand_mask, min_expand_dist = self._chunked_expand_distances(
                pos, atoms, expand_pos, cell_mask
            )
            expand_len = torch.sum(expand_mask, dim=-1)
            threshold_num_expanded_token = torch.clamp(
                self.expanded_token_cutoff - num_atoms, min=0
            )
            max_expand_len = torch.max(expand_len)
            need_threshold = expand_len > threshold_num_expanded_token
            if need_threshold.any():
                need_threshold_distances = min_expand_dist[need_threshold]
                threshold_num_expanded_token = threshold_num_expanded_token[
                    need_threshold
                ]
                threshold_dist = torch.sort(
                    need_threshold_distances, dim=-1, descending=False
                )[0]
                threshold_dist = torch.gather(
                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1)
                )
                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist
                expand_mask[need_threshold] &= new_expand_mask
                expand_len = torch.sum(expand_mask, dim=-1)
                max_expand_len = torch.max(expand_len)
            outcell_index = torch.zeros(
                [batch_size, max_expand_len], dtype=torch.long, device=pos.device
            )
            expand_pos_compressed = torch.zeros(
                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
            )
            outcell_all_index = torch.arange(
                max_num_atoms, dtype=torch.long, device=pos.device
            ).repeat(num_expanded_cell)
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]
            expand_pair_token_type = torch.gather(
                pair_token_type,
                dim=2,
                index=outcell_index.unsqueeze(1)
                .unsqueeze(-1)
                .repeat(1, max_num_atoms, 1, pair_token_type.size()[-1]),
            )
            expand_node_type_edge = torch.cat(
                [pair_token_type, expand_pair_token_type], dim=2
            )
            if use_local_attention:
                dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(p=2, dim=-1)
                expand_dist_compress = (
                    pos.unsqueeze(2) - expand_pos_compressed.unsqueeze(1)
                ).norm(p=2, dim=-1)
                local_attention_weight = self.polynomial(
                    torch.cat([dist, expand_dist_compress], dim=2),
                    cutoff=self.cutoff,
                )
                is_periodic = pbc.any(dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    ~is_periodic.unsqueeze(-1).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                expand_mask_compressed = mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                )
                full_mask = torch.cat([atoms.eq(0), expand_mask_compressed], dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    full_mask.unsqueeze(1), 0.0
                )
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": expand_mask_compressed,
                    "local_attention_weight": local_attention_weight,
                    "expand_node_type_edge": expand_node_type_edge,
                    "expand_len": expand_len,
                }
            else:
                expand_mask_compressed = mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                )
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": expand_mask_compressed,
                    "local_attention_weight": None,
                    "expand_node_type_edge": expand_node_type_edge,
                    "expand_len": expand_len,
                }
            expand_pos_no_offset = torch.gather(
                pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            offset = expand_pos_compressed - expand_pos_no_offset
            init_expand_pos_no_offset = torch.gather(
                init_pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            init_expand_pos = init_expand_pos_no_offset + offset
            init_expand_pos = init_expand_pos.masked_fill(
                pbc_expand_batched["expand_mask"].unsqueeze(-1), 0.0
            )
            pbc_expand_batched["init_expand_pos"] = init_expand_pos
            if edge_index is not None:
                device = edge_index.device
                all_adj = torch.zeros(
                    [batch_size, max_num_atoms, max_num_atoms + max_expand_len],
                    dtype=torch.bool,
                    device=device,
                )
                all_attn_edge_type = torch.zeros(
                    [
                        batch_size,
                        max_num_atoms,
                        max_num_atoms + max_expand_len,
                        edge_attr.size(-1),
                    ],
                    dtype=torch.uint8,
                    device=device,
                )
                for i in range(batch_size):
                    sample_num_edges = num_edges[i]
                    sample_all_adj, sample_all_attn_edge_type = self._gen_expand_adj(
                        edge_index=edge_index[i, :, :sample_num_edges].unsqueeze(0),
                        target_cell_index=target_cell_index[
                            i, :sample_num_edges, :
                        ].unsqueeze(0),
                        max_num_atoms=max_num_atoms,
                        cell_tensor=non_batch_cell_tensor,
                        edge_attr=edge_attr[i, :sample_num_edges, :].unsqueeze(0),
                    )
                    all_adj[i, :, :max_num_atoms] = sample_all_adj[:, :max_num_atoms]
                    all_adj[i, :, max_num_atoms : max_num_atoms + expand_len[i]] = (
                        sample_all_adj[:, max_num_atoms:][:, expand_mask[i, :]]
                    )
                    all_attn_edge_type[i, :, :max_num_atoms] = (
                        sample_all_attn_edge_type[:, :max_num_atoms]
                    )
                    all_attn_edge_type[
                        i, :, max_num_atoms : max_num_atoms + expand_len[i]
                    ] = sample_all_attn_edge_type[:, max_num_atoms:][
                        :, expand_mask[i, :]
                    ]
                pbc_expand_batched.update(
                    {"all_adj": all_adj, "all_attn_edge_type": all_attn_edge_type}
                )
            return pbc_expand_batched

    def _gen_expand_adj(
        self, edge_index, target_cell_index, max_num_atoms, cell_tensor, edge_attr
    ):
        assert edge_index.size(0) == 1
        assert target_cell_index.size(0) == 1
        assert edge_attr.size(0) == 1
        edge_index = edge_index[0]
        target_cell_index = target_cell_index[0]
        edge_attr = edge_attr[0]
        device = edge_index.device
        augmented_cell_tensor = torch.cat(
            [torch.zeros([1, 3], dtype=torch.long, device=device), cell_tensor], dim=0
        )
        cell_index_scaler = (
            (augmented_cell_tensor[:, 0] + 1) * 9
            + (augmented_cell_tensor[:, 1] + 1) * 3
            + (augmented_cell_tensor[:, 2] + 1)
        )
        is_valid_target_cell: torch.Tensor = (augmented_cell_tensor.abs() <= 1).all(
            dim=-1
        )
        is_valid_target_cell_position = is_valid_target_cell.nonzero(as_tuple=True)[0]
        valid_target_cell_index_scaler = cell_index_scaler[is_valid_target_cell]
        target_cell_index_to_position_map = torch.full(
            [27], -1, dtype=torch.long, device=device
        )
        target_cell_index_to_position_map[valid_target_cell_index_scaler] = (
            is_valid_target_cell_position
        )
        num_all_expand_atoms = augmented_cell_tensor.size(0) * max_num_atoms
        all_adj = torch.zeros([max_num_atoms, num_all_expand_atoms], device=device)
        target_cell_index_scaler = (
            (target_cell_index[:, 0] + 1) * 9
            + (target_cell_index[:, 1] + 1) * 3
            + (target_cell_index[:, 2] + 1)
        )
        edge_wise_target_cell_position = target_cell_index_to_position_map[
            target_cell_index_scaler
        ]
        is_valid_edge = edge_wise_target_cell_position >= 0
        source_node = edge_index[0, is_valid_edge]
        target_node = (
            edge_index[1, is_valid_edge]
            + edge_wise_target_cell_position[is_valid_edge] * max_num_atoms
        )
        all_adj[source_node, target_node] = True
        mirror_target_cell_index_scaler = (
            (-target_cell_index[:, 0] + 1) * 9
            + (-target_cell_index[:, 1] + 1) * 3
            + (-target_cell_index[:, 2] + 1)
        )
        mirror_edge_wise_target_cell_position = target_cell_index_to_position_map[
            mirror_target_cell_index_scaler
        ]
        is_valid_mirror_edge = mirror_edge_wise_target_cell_position >= 0
        mirror_source_node = edge_index[1, is_valid_mirror_edge]
        mirror_target_node = (
            edge_index[0, is_valid_mirror_edge]
            + mirror_edge_wise_target_cell_position[is_valid_mirror_edge]
            * max_num_atoms
        )
        all_adj[mirror_source_node, mirror_target_node] = True
        all_attn_edge_type = torch.zeros(
            [max_num_atoms, num_all_expand_atoms, edge_attr.size(1)],
            dtype=torch.uint8,
            device=device,
        )
        all_attn_edge_type[source_node, target_node] = edge_attr[is_valid_edge, :].to(
            dtype=all_attn_edge_type.dtype
        )
        all_attn_edge_type[mirror_source_node, mirror_target_node] = edge_attr[
            is_valid_mirror_edge, :
        ].to(dtype=all_attn_edge_type.dtype)
        return (all_adj, all_attn_edge_type)
