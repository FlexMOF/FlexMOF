"""MOF dataset loading, batching, and deterministic distributed sampling."""

from __future__ import annotations

import math
import pickle
from itertools import permutations
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import lmdb
import numpy as np
import torch
from .model.gaussian_composition import AtomVocabulary
from .ordering import canonical_cell_order, dfs_atom_order
from .block_identity import block_certificate, occurrence_order
from numpy import dot
from numpy.linalg import norm
from torch.utils.data import Dataset, Sampler


CELL_CORNERS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
)


@lru_cache(maxsize=None)
def _open_lmdb(path: str):
    return lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


class MOFDataset(Dataset):
    """MetalOxo LMDB reader for the checkpoint's crystal-generation task."""

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        max_atoms: int = 768,
        composition_config=None,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"MOF dataset directory not found: {self.root}")

        self._environment = _open_lmdb(str(self.root / "MetalOxo.lmdb"))
        self._transaction = self._environment.begin(write=False)
        keys = torch.load(self.root / "index_to_key_map.pt", weights_only=False)
        sizes = torch.load(self.root / "sizes.pt", weights_only=False)
        self.composition_config = composition_config
        overfit_key = getattr(composition_config, "mof_overfit_key", "")
        split = {"val": "valid", "validation": "valid"}.get(split, split)
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown MOF dataset split: {split}")
        split_keys = {int(overfit_key)} if overfit_key != "" else set(
            np.atleast_1d(
                np.genfromtxt(
                    self.root / f"{split}_split_filtered.txt",
                    delimiter=",",
                    dtype=int,
                )
            ).tolist()
        )
        selected = [
            (key, size)
            for key, size in zip(keys, sizes)
            if int(key) in split_keys and int(size) <= max_atoms
        ]
        self.keys = [key for key, _ in selected]
        self.sizes = [int(size) for _, size in selected]
        if not self.keys:
            raise ValueError(f"No MOFs selected for split={split}, overfit_key={overfit_key!r}, max_atoms={max_atoms}")
        if overfit_key != "":
            repeats = getattr(composition_config, "mof_overfit_repeats", 100000)
            self.keys *= repeats
            self.sizes *= repeats

        offsets = [
            [i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)
        ]
        self.neighbor_cells = torch.tensor(offsets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_environment"] = None
        state["_transaction"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._environment = _open_lmdb(str(self.root / "MetalOxo.lmdb"))
        self._transaction = self._environment.begin(write=False)

    @staticmethod
    def _orient_lattice(pbc: np.ndarray, cell: np.ndarray):
        axes = np.eye(3, dtype=np.float64)
        best_permutation = None
        best_signs = None
        best_score = 0.0
        for permutation in permutations(range(3)):
            score = 0.0
            signs = []
            for output_axis, cell_axis in enumerate(permutation):
                vector = cell[cell_axis]
                cosine = dot(vector, axes[output_axis]) / (
                    norm(vector) * norm(axes[output_axis])
                )
                score += abs(cosine)
                signs.append(-1.0 if cosine < 0.0 else 1.0)
            if score > best_score:
                best_permutation = permutation
                best_signs = signs
                best_score = score
        return (
            pbc[list(best_permutation)],
            cell[list(best_permutation)] * np.asarray(best_signs)[:, None],
        )

    @staticmethod
    def _symmetrize_lattice(cell: np.ndarray, positions: torch.Tensor):
        transposed = cell.T
        left, singular_values, right = np.linalg.svd(transposed)
        rotation = left @ right
        symmetric_cell = (right.T @ np.diag(singular_values) @ right).T
        rotated_positions = (rotation.T @ positions.cpu().numpy().T).T
        return symmetric_cell, rotated_positions

    def _bond_graph(
        self,
        fractional: torch.Tensor,
        record,
        zero_target_cell: bool = True,
    ):
        normalized = fractional % 1.0 % 1.0
        cell = torch.as_tensor(record.cell)
        offsets = self.neighbor_cells.to(dtype=cell.dtype) @ cell
        local_to_global: dict[int, dict[int, int]] = {}
        directed_edges = []
        target_cells = {}

        for block_id, block in enumerate(record.pyg_mols):
            if not record.is_linker[block_id]:
                continue
            block_fractional = block.frac_coords
            normalized_block = block_fractional % 1.0 % 1.0
            distances = (normalized_block[:, None, :] - normalized[None, :, :]).norm(
                dim=-1, p=2
            )
            matched = distances < 1e-6
            if not matched.any(dim=-1).all():
                raise ValueError("Could not map linker atoms into the MOF unit cell")
            matched_indices = matched.long().argmax(dim=-1)
            local_to_global[block_id] = {
                local: int(global_index)
                for local, global_index in enumerate(matched_indices)
            }

            for source, target in zip(block.edge_index[0], block.edge_index[1]):
                source, target = int(source), int(target)
                directed_edges.append(
                    [
                        local_to_global[block_id][source],
                        local_to_global[block_id][target],
                    ]
                )

            cartesian = normalized_block.to(dtype=cell.dtype) @ cell
            source_cartesian = cartesian[block.edge_index[0]]
            target_cartesian = cartesian[block.edge_index[1]]
            closest_targets = (
                (
                    source_cartesian[:, None, :]
                    - (target_cartesian[:, None, :] + offsets[None, :, :])
                )
                .norm(dim=-1, p=2)
                .argmin(dim=-1)
            )
            closest_sources = (
                (
                    target_cartesian[:, None, :]
                    - (source_cartesian[:, None, :] + offsets[None, :, :])
                )
                .norm(dim=-1, p=2)
                .argmin(dim=-1)
            )

            for source, target, cell_index in zip(
                block.edge_index[0], block.edge_index[1], closest_targets
            ):
                source = local_to_global[block_id][int(source)]
                target = local_to_global[block_id][int(target)]
                index = int(cell_index)
                target_cells[(source, target)] = [
                    index // 9 - 1,
                    (index % 9) // 3 - 1,
                    index % 3 - 1,
                ]
            for source, target, cell_index in zip(
                block.edge_index[0], block.edge_index[1], closest_sources
            ):
                source = local_to_global[block_id][int(source)]
                target = local_to_global[block_id][int(target)]
                index = int(cell_index)
                target_cells[(target, source)] = [
                    index // 9 - 1,
                    (index % 9) // 3 - 1,
                    index % 3 - 1,
                ]

        if not directed_edges:
            return (
                torch.zeros([2, 0], dtype=torch.long),
                torch.zeros([0, 3], dtype=torch.long),
            )

        canonical = np.unique([f"{min(a, b)} {max(a, b)}" for a, b in directed_edges])
        edges, edge_cells = [], []
        for value in canonical:
            source, target = (int(part) for part in value.split())
            edges.append([source, target])
            edge_cells.append(target_cells[(source, target)])
        edge_target_cell = torch.tensor(edge_cells, dtype=torch.long)
        if zero_target_cell:
            edge_target_cell.zero_()
        return torch.tensor(edges, dtype=torch.long).T, edge_target_cell

    @staticmethod
    def _canonical_linker_atom_cell_shifts(
        num_atoms: int,
        edge_index: torch.Tensor,
        edge_target_cell: torch.Tensor,
    ) -> torch.Tensor:
        """Select continuous periodic images for all bonded linker atoms."""
        edge_index = torch.as_tensor(edge_index, dtype=torch.long).cpu()
        edge_target_cell = torch.as_tensor(
            edge_target_cell, dtype=torch.long
        ).cpu()

        if edge_index.numel() == 0:
            return torch.zeros([num_atoms, 3], dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "Linker edge_index must have shape [2, num_edges], got "
                f"{tuple(edge_index.shape)}"
            )
        if edge_target_cell.shape != (edge_index.size(1), 3):
            raise ValueError(
                "Linker edge_target_cell must have shape [num_edges, 3], got "
                f"{tuple(edge_target_cell.shape)}"
            )

        adjacency = [[] for _ in range(num_atoms)]
        for edge_id in range(edge_index.size(1)):
            source = int(edge_index[0, edge_id])
            target = int(edge_index[1, edge_id])
            if not (0 <= source < num_atoms and 0 <= target < num_atoms):
                raise ValueError(
                    "Linker edge contains an atom index outside the structure: "
                    f"({source}, {target}) for {num_atoms} atoms"
                )
            offset = edge_target_cell[edge_id]
            adjacency[source].append((target, offset))
            adjacency[target].append((source, -offset))

        shifts = torch.zeros([num_atoms, 3], dtype=torch.long)
        assigned = torch.zeros([num_atoms], dtype=torch.bool)
        linker_atoms = sorted(set(edge_index.reshape(-1).tolist()))

        for root in linker_atoms:
            if assigned[root]:
                continue
            assigned[root] = True
            stack = [root]
            component = []
            while stack:
                source = stack.pop()
                component.append(source)
                for target, offset in adjacency[source]:
                    expected_shift = shifts[source] + offset
                    if assigned[target]:
                        if not torch.equal(shifts[target], expected_shift):
                            raise ValueError(
                                "Cannot unwrap linker because its periodic bond "
                                "offsets are cycle-inconsistent: edge "
                                f"({source}, {target}) expects atom {target} shift "
                                f"{expected_shift.tolist()}, but it was already "
                                f"assigned {shifts[target].tolist()}"
                            )
                        continue
                    shifts[target] = expected_shift
                    assigned[target] = True
                    stack.append(target)

            component_index = torch.tensor(component, dtype=torch.long)
            component_origin = shifts[component_index].median(dim=0).values
            shifts[component_index] -= component_origin

        return shifts

    @staticmethod
    def _record_field(record, name, default=None):
        if isinstance(record, dict):
            return record.get(name, default)
        return getattr(record, name, default)

    @classmethod
    def _block_atom_types(cls, block):
        for name in ("atom_types", "atomic_numbers", "numbers", "x", "z"):
            value = cls._record_field(block, name)
            if value is not None:
                atoms = torch.as_tensor(value, dtype=torch.long).reshape(-1)
                if atoms.numel():
                    return atoms
        return None

    @classmethod
    def _block_edge_index(cls, block):
        for name in ("edge_index", "bond_index", "bonds"):
            value = cls._record_field(block, name)
            if value is None:
                continue
            edges = torch.as_tensor(value, dtype=torch.long)
            if edges.numel() == 0:
                return edges.new_zeros([2, 0])
            if edges.dim() == 2 and edges.size(0) == 2:
                return edges
            if edges.dim() == 2 and edges.size(1) == 2:
                return edges.T.contiguous()
        return None

    def _composition_targets(self, record, atomic_numbers, fractional):
        config = self.composition_config
        blocks = []
        pyg_mols = self._record_field(record, "pyg_mols")
        is_linker = self._record_field(record, "is_linker")
        if pyg_mols is None or is_linker is None:
            raise ValueError(
                "MOF composition diffusion requires pyg_mols and is_linker"
            )
        for block_index, block in enumerate(pyg_mols):
            atoms = self._block_atom_types(block)
            if atoms is None:
                continue
            linker = bool(is_linker[block_index])
            if linker:
                blocks.append((atoms, self._block_edge_index(block), True, block))
            else:
                for atom_index, atom in enumerate(atoms):
                    blocks.append((atom.view(1), None, False, (block, atom_index)))

        ordered_blocks = []
        canonical = getattr(config, "mof_canonical_order", False)
        for atoms, edge_index, linker, source in blocks:
            order = dfs_atom_order(atoms, edge_index) if canonical else torch.arange(atoms.numel())
            if canonical and edge_index is not None:
                inverse = torch.empty_like(order)
                inverse[order] = torch.arange(order.numel())
                edge_index = inverse[edge_index]
            atoms = atoms[order]
            ordered_blocks.append((atoms, edge_index, linker, source, order))
        if canonical:
            # Remaining ties are resolved by the whole DFS element sequence and
            # edge list; Python's stable sort preserves identical occurrences.
            def block_key(block):
                atoms, edges, *_ = block
                edge_key = () if edges is None else tuple(sorted({tuple(sorted(edge)) for edge in edges.T.tolist()}))
                return (atoms.numel(), int(atoms[0]), tuple(atoms.tolist()), edge_key)
            ordered_blocks.sort(key=block_key)

        unique_blocks = []
        repetitions = []
        signature_to_index = {}
        identity_signatures = []
        reference_graphs = []
        canonical_blocks = getattr(config, "mof_joint_canonical_blocks", False)
        occurrences = []
        for atoms, edge_index, linker, source, order in ordered_blocks:
            if int(atoms.max()) >= config.mof_composition_num_atom_classes:
                raise ValueError("MOF atom class exceeds composition vocabulary")
            if atoms.numel() > config.mof_composition_max_atoms_per_bb:
                raise ValueError("MOF building block exceeds the configured atom limit")
            edge_key = ()
            if edge_index is not None and edge_index.numel():
                undirected = torch.sort(edge_index[:2], dim=0).values
                edge_key = tuple(
                    sorted({
                        (int(source_index), int(target_index))
                        for source_index, target_index in undirected.T.tolist()
                        if source_index != target_index
                    })
                )
            signature = (tuple(int(atom) for atom in atoms.tolist()), edge_key)
            graph_signature = signature
            if canonical_blocks:
                signature = block_certificate(*graph_signature, linker)
            if signature not in signature_to_index:
                if len(unique_blocks) >= config.mof_composition_max_unique_bbs:
                    raise ValueError(
                        "MOF exceeds the configured unique building-block limit"
                    )
                signature_to_index[signature] = len(unique_blocks)
                unique_blocks.append((atoms, edge_index, linker))
                repetitions.append(0)
                occurrences.append([])
                reference_graphs.append(graph_signature)
                identity_signatures.append(signature)
            unique_index = signature_to_index[signature]
            if canonical_blocks:
                permutation = occurrence_order(
                    *reference_graphs[unique_index], *graph_signature
                )
                order = order[torch.tensor(permutation, dtype=torch.long)]
            repetitions[unique_index] += 1
            occurrences[unique_index].append((linker, source, order))

        if canonical_blocks:
            # Match the historical whole-MOF preparation's certificate order.
            indices = sorted(range(len(unique_blocks)), key=identity_signatures.__getitem__)
            unique_blocks = [unique_blocks[i] for i in indices]
            repetitions = [repetitions[i] for i in indices]
            occurrences = [occurrences[i] for i in indices]

        if not unique_blocks:
            raise ValueError("MOF has no composition building-block targets")
        max_atoms = max(int(atoms.numel()) for atoms, _, _ in unique_blocks)
        max_repetitions = max(repetitions)
        if max_repetitions > config.mof_composition_max_repetitions:
            raise ValueError("MOF exceeds the configured repetition limit")
        atom_types = torch.zeros(len(unique_blocks), max_atoms, dtype=torch.long)
        atom_mask = torch.zeros_like(atom_types, dtype=torch.bool)
        bond_adj = torch.zeros(
            len(unique_blocks), max_atoms, max_atoms, dtype=torch.long
        )
        occurrence_index = torch.full(
            (len(unique_blocks), max_repetitions, max_atoms), -1, dtype=torch.long
        )
        occurrence_mask = torch.zeros(
            len(unique_blocks), max_repetitions, dtype=torch.bool
        )
        normalized = fractional.float().remainder(1.0)

        for bb_index, (atoms, edge_index, linker) in enumerate(unique_blocks):
            atom_types[bb_index, : atoms.numel()] = atoms
            atom_mask[bb_index, : atoms.numel()] = True
            if linker and edge_index is not None and edge_index.numel():
                valid = (
                    (edge_index[0] >= 0)
                    & (edge_index[1] >= 0)
                    & (edge_index[0] < atoms.numel())
                    & (edge_index[1] < atoms.numel())
                    & (edge_index[0] != edge_index[1])
                )
                edges = edge_index[:, valid]
                bond_adj[bb_index, edges[0], edges[1]] = 1
                bond_adj[bb_index, edges[1], edges[0]] = 1

            for repetition, (_, source, order) in enumerate(occurrences[bb_index]):
                if linker:
                    block_fractional = self._record_field(source, "frac_coords")
                    local_fractional = torch.as_tensor(block_fractional).float()
                else:
                    block, atom_index = source
                    block_fractional = self._record_field(block, "frac_coords")
                    local_fractional = torch.as_tensor(block_fractional).float()[
                        atom_index : atom_index + 1
                    ]
                local_fractional = local_fractional[order].remainder(1.0)
                distance = (local_fractional[:, None, :] - normalized[None, :, :]).norm(
                    dim=-1
                )
                matched = distance < 1e-6
                if not matched.any(dim=-1).all():
                    raise ValueError("Could not map a building block into the MOF cell")
                indices = matched.long().argmax(dim=-1)
                if not torch.equal(atomic_numbers[indices], atoms):
                    raise ValueError("Building-block elements disagree with matched cell atoms")
                length = min(indices.numel(), max_atoms)
                occurrence_index[bb_index, repetition, :length] = indices[:length]
                occurrence_mask[bb_index, repetition] = True

        remapped = config.mof_composition_diffusion_strategy in {
            "gaussian_ddpm",
            "lgflow",
            "joint_lgflow",
        }
        if remapped:
            atom_types = AtomVocabulary(config.mof_composition_atom_vocab).encode(atom_types)
        return {
            "mof_composition_remapped": remapped,
            "mof_total_num_atoms": torch.tensor(atomic_numbers.numel()),
            "mof_num_unique_bbs": torch.tensor(len(unique_blocks)),
            "mof_has_composition_targets": torch.tensor(True),
            "mof_bb_repetitions": torch.tensor(repetitions, dtype=torch.long),
            "mof_bb_atom_types": atom_types,
            "mof_bb_atom_mask": atom_mask,
            "mof_bb_bond_adj": bond_adj,
            "mof_bb_occ_atom_index": occurrence_index,
            "mof_bb_occ_mask": occurrence_mask,
            "mof_bb_mask": torch.ones(len(unique_blocks), dtype=torch.bool),
            # The denoiser conditions on roles, and generation has to supply
            # the same roles from its request rather than infer them from a
            # graph it has not produced yet.
            "mof_bb_is_linker": torch.tensor(
                [bool(linker) for _, _, linker in unique_blocks], dtype=torch.bool
            ),
        }

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        record = pickle.loads(self._transaction.get(key.encode()))
        atomic_numbers = torch.as_tensor(record["atom_types"], dtype=torch.long)
        atom_count = int(atomic_numbers.numel())
        fractional = torch.as_tensor(record["frac_coords"], dtype=torch.float)
        composition = (
            self._composition_targets(record, atomic_numbers, fractional)
            if self.composition_config is not None
            and (self.composition_config.use_mof_composition_diffusion
                 or self.composition_config.mof_canonical_order)
            else {}
        )
        edge_index, original_target_cell_index = self._bond_graph(
            fractional, record, zero_target_cell=False
        )
        target_cell_index = torch.zeros_like(original_target_cell_index)
        normalized_fractional = fractional % 1.0 % 1.0
        unwrap_linkers_and_center = bool(
            getattr(
                self.composition_config,
                "mof_unwrap_linkers_and_center",
                False,
            )
        )
        if unwrap_linkers_and_center:
            atom_cell_shifts = self._canonical_linker_atom_cell_shifts(
                atom_count, edge_index, original_target_cell_index
            )
            normalized_fractional = normalized_fractional + atom_cell_shifts.to(
                dtype=normalized_fractional.dtype
            )
        positions = normalized_fractional @ record["cell"]

        cell = record["cell"].cpu().numpy()
        cell, positions = self._symmetrize_lattice(cell, positions)
        pbc, cell = self._orient_lattice(np.ones(3, dtype=bool), cell)
        cell = torch.as_tensor(cell, dtype=torch.float64)
        positions = torch.as_tensor(positions, dtype=torch.float64)

        if unwrap_linkers_and_center and atom_count > 0:
            positions = positions + (
                0.5 * cell.sum(dim=0) - positions.mean(dim=0)
            )

        if getattr(self.composition_config, "mof_canonical_order", False):
            order = canonical_cell_order(composition, atom_count)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(atom_count)
            atomic_numbers = atomic_numbers[order]
            positions = positions[order]
            edge_index = inverse[edge_index]
            occurrence = composition["mof_bb_occ_atom_index"]
            valid = occurrence.ge(0)
            occurrence[valid] = inverse[occurrence[valid]]

        tokens = torch.cat([atomic_numbers, torch.full([8], 128)])
        positions = torch.cat([positions, CELL_CORNERS.to(cell) @ cell], dim=0)
        record["cell"] = cell
        edge_count = edge_index.size(1)
        edge_attr = torch.ones([edge_count, 1], dtype=torch.long)

        node_count = tokens.numel()
        adjacency = torch.ones([node_count, node_count], dtype=torch.bool)
        adjacency[edge_index[0], edge_index[1]] = True
        adjacency[edge_index[1], edge_index[0]] = True
        attention_edge_type = torch.zeros([node_count, node_count, 1], dtype=torch.long)
        attention_edge_type[edge_index[0], edge_index[1]] = edge_attr
        attention_edge_type[edge_index[1], edge_index[0]] = edge_attr

        item = {
            "idx": index,
            "token_type": tokens,
            "num_atoms": atom_count,
            "coords": positions,
            "cell": cell,
            "pbc": torch.as_tensor(pbc, dtype=torch.bool),
            "node_attr": torch.cat(
                [tokens[:, None], torch.zeros([node_count, 8], dtype=torch.long)],
                dim=-1,
            ),
            "adj": adjacency,
            "attn_bias": torch.zeros([node_count + 1, node_count + 1]),
            "in_degree": adjacency.long().sum(dim=1),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "target_cell_index": target_cell_index,
        }
        if getattr(self.composition_config, "use_mof_composition_diffusion", False):
            item.update(composition)
        return item


def _pad_1d(value: torch.Tensor, length: int, increment: bool = False):
    if increment:
        value = value + 1
    output = value.new_zeros([length])
    output[: value.size(0)] = value
    return output


def _pad_2d(value: torch.Tensor, rows: int, columns: int | None = None):
    columns = value.size(1) if columns is None else columns
    output = value.new_zeros([rows, columns])
    output[: value.size(0), : value.size(1)] = value
    return output


def _pad_square(value: torch.Tensor, length: int):
    output = value.new_zeros([length, length, *value.shape[2:]])
    output[: value.size(0), : value.size(1)] = value
    return output


def _collate_composition(items: list[dict], max_nodes: int) -> dict:
    batch_size = len(items)
    max_bbs = max(item["mof_bb_mask"].size(0) for item in items)
    max_atoms = max(item["mof_bb_atom_types"].size(1) for item in items)
    max_repetitions = max(item["mof_bb_occ_atom_index"].size(1) for item in items)
    atoms = torch.zeros(batch_size, max_bbs, max_atoms, dtype=torch.long)
    atom_mask = torch.zeros_like(atoms, dtype=torch.bool)
    bonds = torch.zeros(batch_size, max_bbs, max_atoms, max_atoms, dtype=torch.long)
    occurrence_index = torch.full(
        (batch_size, max_bbs, max_repetitions, max_atoms), -1, dtype=torch.long
    )
    occurrence_mask = torch.zeros(
        batch_size, max_bbs, max_repetitions, dtype=torch.bool
    )
    bb_mask = torch.zeros(batch_size, max_bbs, dtype=torch.bool)
    repetitions = torch.zeros(batch_size, max_bbs, dtype=torch.long)
    is_linker = torch.zeros(batch_size, max_bbs, dtype=torch.bool)

    for batch_index, item in enumerate(items):
        num_bbs, num_atoms = item["mof_bb_atom_types"].shape
        num_repetitions = item["mof_bb_occ_atom_index"].size(1)
        atoms[batch_index, :num_bbs, :num_atoms] = item["mof_bb_atom_types"]
        sample_atoms = atoms[batch_index, :num_bbs, :num_atoms]
        if not item.get("mof_composition_remapped", False):
            sample_atoms[sample_atoms > 0] += 1
        atom_mask[batch_index, :num_bbs, :num_atoms] = item["mof_bb_atom_mask"]
        bonds[batch_index, :num_bbs, :num_atoms, :num_atoms] = item["mof_bb_bond_adj"]
        occurrence_index[batch_index, :num_bbs, :num_repetitions, :num_atoms] = item[
            "mof_bb_occ_atom_index"
        ]
        occurrence_mask[batch_index, :num_bbs, :num_repetitions] = item[
            "mof_bb_occ_mask"
        ]
        bb_mask[batch_index, :num_bbs] = item["mof_bb_mask"]
        repetitions[batch_index, :num_bbs] = item["mof_bb_repetitions"]
        if "mof_bb_is_linker" in item:
            is_linker[batch_index, :num_bbs] = item["mof_bb_is_linker"]

    atom_bb_id = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    flat_mask = occurrence_mask.reshape(batch_size, -1)
    occurrence_ids = (torch.cumsum(flat_mask.long(), dim=1) * flat_mask).reshape_as(
        occurrence_mask
    )
    expanded_ids = occurrence_ids.unsqueeze(-1).expand_as(occurrence_index)
    valid = (
        occurrence_mask.unsqueeze(-1)
        & occurrence_index.ge(0)
        & occurrence_index.lt(max_nodes)
    )
    for batch_index in range(batch_size):
        atom_bb_id[batch_index, occurrence_index[batch_index][valid[batch_index]]] = (
            expanded_ids[batch_index][valid[batch_index]]
        )

    return {
        "mof_total_num_atoms": torch.stack(
            [item["mof_total_num_atoms"] for item in items]
        ).long(),
        "mof_num_unique_bbs": torch.stack(
            [item["mof_num_unique_bbs"] for item in items]
        ).long(),
        "mof_has_composition_targets": torch.stack(
            [item["mof_has_composition_targets"] for item in items]
        ).bool(),
        "mof_bb_repetitions": repetitions,
        "mof_bb_atom_types": atoms,
        "mof_bb_atom_mask": atom_mask,
        "mof_bb_bond_adj": bonds,
        "mof_bb_occ_atom_index": occurrence_index,
        "mof_bb_occ_mask": occurrence_mask,
        "mof_atom_bb_id": atom_bb_id,
        "mof_bb_mask": bb_mask,
        "mof_bb_is_linker": is_linker,
    }


def collate_mofs(items: list[dict]) -> dict[str, torch.Tensor]:
    """Pad MOFs while retaining only fields consumed by generation."""
    max_nodes = max(item["token_type"].numel() for item in items)
    max_edges = max(item["edge_attr"].size(0) for item in items)
    node_attr = torch.stack(
        [_pad_2d(item["node_attr"] + 1, max_nodes) for item in items]
    )
    batch = {
        "idx": torch.tensor([item["idx"] for item in items], dtype=torch.long),
        "token_id": node_attr[:, :, 0],
        "node_attr": node_attr,
        "pos": torch.stack([_pad_2d(item["coords"], max_nodes, 3) for item in items]),
        "cell": torch.stack([item["cell"] for item in items]),
        "pbc": torch.stack([item["pbc"] for item in items]),
        "num_atoms": torch.tensor(
            [item["num_atoms"] for item in items], dtype=torch.long
        ),
        "is_stable_periodic": torch.ones(len(items), dtype=torch.bool),
        "chain_ids": torch.stack(
            [_pad_1d(torch.ones_like(item["token_type"]), max_nodes) for item in items]
        ),
        "position_ids": torch.stack(
            [
                _pad_1d(
                    torch.arange(item["token_type"].numel(), dtype=torch.long),
                    max_nodes,
                    increment=True,
                )
                for item in items
            ]
        ),
        "adj": torch.stack([_pad_square(item["adj"], max_nodes) for item in items]),
        "attn_bias": torch.stack(
            [_pad_square(item["attn_bias"], max_nodes + 1) for item in items]
        ),
        "in_degree": torch.stack(
            [_pad_1d(item["in_degree"], max_nodes, increment=True) for item in items]
        ),
        "edge_index": torch.stack(
            [_pad_2d(item["edge_index"].T, max_edges, 2) for item in items]
        ),
        "edge_attr": torch.stack(
            [_pad_2d(item["edge_attr"], max_edges, 1) for item in items]
        ),
        "target_cell_index": torch.stack(
            [_pad_2d(item["target_cell_index"], max_edges, 3) for item in items]
        ),
        "num_edges": torch.tensor(
            [item["edge_attr"].size(0) for item in items], dtype=torch.long
        ),
    }
    if "mof_bb_atom_types" in items[0]:
        batch.update(_collate_composition(items, max_nodes))
    return batch


class DistributedMOFBatchSampler(Sampler[list[int]]):
    """Legacy-compatible shuffled and size-bucketed validation order."""

    def __init__(
        self,
        dataset: MOFDataset,
        batch_size: int,
        replicas: int,
        rank: int,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.replicas = replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.sample_count = (
            math.ceil(len(dataset) / (batch_size * replicas)) * batch_size
        )

    def __len__(self) -> int:
        return self.sample_count // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        numpy_generator = np.random.default_rng(self.seed + self.epoch)
        torch_generator = torch.Generator().manual_seed(self.seed + self.epoch)
        padded = self.sample_count * self.replicas
        indices = []
        remaining = padded
        while remaining > len(self.dataset):
            indices.extend(
                torch.randperm(len(self.dataset), generator=torch_generator).tolist()
            )
            remaining -= len(self.dataset)
        indices.extend(
            numpy_generator.choice(len(self.dataset), remaining, replace=False).tolist()
        )
        order = torch.randperm(len(indices), generator=torch_generator).tolist()
        indices = [indices[i] for i in order][self.rank : padded : self.replicas]
        order = torch.randperm(len(indices), generator=torch_generator).tolist()
        indices = [indices[i] for i in order]
        indices.sort(key=lambda index: int(np.asarray(self.dataset.sizes[index]).max()))
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        batch_generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batch_order = torch.randperm(len(batches), generator=batch_generator).tolist()
        return iter([batches[i] for i in batch_order])
