"""ODAC25 -> FlexMOF dataset adapter.

Converts AtomMOF's preprocessed ODAC25 records (`src.data.types.MOFStructure`)
into the item format FlexMOF's `collate_mofs` consumes, reusing FlexMOF's own
unwrap / lattice-symmetrisation / centring so the geometry pipeline stays
identical to the one the checkpoint was pretrained with.

Three things differ from FlexMOF's native `MOFDataset` and are deliberate:

* Block-to-global atom mapping uses a running offset. ODAC25 blocks tile the
  global atom array in element order, but `block.cart_coords` holds a *local
  reference* geometry (deviating by up to ~139 A), so FlexMOF's 1e-6 coordinate
  matching cannot be reused.
* Bonds come from every block that has them (LINKER / SOLVENT / ADSORBATE);
  NODE blocks carry no internal bonds, matching FlexMOF's linker-only graph.
* `chain_ids` encodes the four AtomMOF roles instead of the constant 1 that
  `collate_mofs` hardcodes.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from flexmof.data import (  # noqa: E402
    CELL_CORNERS,
    MOFDataset,
    _pad_1d,
    collate_mofs,
)

# LINKER and NODE keep chain id 1, which is the only value FlexMOF ever saw during
# pretraining. SOLVENT and ADSORBATE get fresh ids whose embedding rows are
# zeroed at finetune init (see zero_new_role_embeddings).
BLOCK_TYPE_TO_CHAIN_ID = {
    "NODE": 1,
    "LINKER": 1,
    "SOLVENT": 2,
    "ADSORBATE": 3,
}
# The author's own ids, kept verbatim so evaluation masks (block_type == 3 for
# adsorbate) transfer without translation.
BLOCK_TYPE_ID = {"NODE": 0, "LINKER": 1, "SOLVENT": 2, "ADSORBATE": 3}
NEW_ROLE_CHAIN_IDS = (2, 3)
CORNER_TOKEN = 128
CORNER_CHAIN_ID = 1


class OdacConversionError(RuntimeError):
    """Raised when a record cannot be represented in FlexMOF's input format."""


def _block_type_name(block) -> str:
    value = block.block_type
    return value.name if hasattr(value, "name") else str(value)


class OdacDataset(Dataset):
    """Reads an ODAC25 LMDB and emits FlexMOF-compatible items.

    Item order follows the metadata pickle, which is what the integer indices in
    `test_system_indices.pkl` refer to.
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        metadata_path: str | Path,
        max_atoms: int = 768,
        subset_indices: Iterable[int] | None = None,
        drop_bonds: bool = False,
        drop_bond_features: bool = False,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.metadata_path = Path(metadata_path)
        self.max_atoms = int(max_atoms)
        self.drop_bonds = bool(drop_bonds)
        self.drop_bond_features = bool(drop_bond_features)

        with self.metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        all_keys = list(metadata.keys())
        # Length of the author's ordering, before any filtering. Prediction
        # files must be this long even when max_atoms drops most records,
        # because the coverage evaluator indexes into them with positions from
        # test_system_indices.pkl.
        self.source_length = len(all_keys)

        # dataset_index is the position in the author's ordering and must be
        # preserved so coverage grouping keeps pointing at the right records.
        candidates = [
            (position, key)
            for position, key in enumerate(all_keys)
            if int(metadata[key]["num_atoms"]) <= self.max_atoms
        ]
        if subset_indices is not None:
            wanted = set(int(i) for i in subset_indices)
            candidates = [item for item in candidates if item[0] in wanted]

        self.dataset_indices = [position for position, _ in candidates]
        self.keys = [key for _, key in candidates]
        # Atom counts drive dynamic batching; read from metadata to avoid
        # deserialising every record up front.
        self.sizes = [int(metadata[key]["num_atoms"]) for _, key in candidates]

        self._environment = None
        self._transaction = None

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

    def _txn(self):
        if self._transaction is None:
            self._environment = lmdb.open(
                str(self.lmdb_path), subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False,
            )
            self._transaction = self._environment.begin(write=False)
        return self._transaction

    def _bond_graph(self, record, cell: torch.Tensor, fractional: torch.Tensor):
        """Global bond graph plus the periodic image each bond points at.

        Local block indices are offset by the running atom count; the periodic
        image is chosen from the 27 neighbours using the *global* coordinates,
        because block coordinates are local reference geometry.
        """
        edges: set[tuple[int, int]] = set()
        offset = 0
        for block in record.blocks:
            count = len(block.atomic_numbers)
            bonds = np.asarray(block.bond_indices)
            if not self.drop_bonds and bonds.size:
                for source, target in zip(bonds[0], bonds[1]):
                    a = int(source) + offset
                    b = int(target) + offset
                    if a != b:
                        edges.add((min(a, b), max(a, b)))
            offset += count

        if offset != int(fractional.size(0)):
            raise OdacConversionError(
                f"blocks cover {offset} atoms but the record has "
                f"{int(fractional.size(0))}"
            )
        if not edges:
            return (
                torch.zeros([2, 0], dtype=torch.long),
                torch.zeros([0, 3], dtype=torch.long),
            )

        ordered = sorted(edges)
        source_index = torch.tensor([a for a, _ in ordered], dtype=torch.long)
        target_index = torch.tensor([b for _, b in ordered], dtype=torch.long)

        normalized = fractional % 1.0 % 1.0
        cartesian = normalized.to(dtype=cell.dtype) @ cell
        image_offsets = self.neighbor_cells.to(dtype=cell.dtype) @ cell

        source_xyz = cartesian[source_index]
        target_xyz = cartesian[target_index]
        distances = (
            source_xyz[:, None, :] - (target_xyz[:, None, :] + image_offsets[None, :, :])
        ).norm(dim=-1, p=2)
        closest = distances.argmin(dim=-1)

        target_cell = torch.stack(
            [closest // 9 - 1, (closest % 9) // 3 - 1, closest % 3 - 1], dim=-1
        ).to(torch.long)
        edge_index = torch.stack([source_index, target_index], dim=0)
        return edge_index, target_cell

    def _chain_ids(self, record, num_atoms: int):
        chain_ids = torch.zeros([num_atoms], dtype=torch.long)
        block_type = torch.zeros([num_atoms], dtype=torch.long)
        offset = 0
        for block in record.blocks:
            count = len(block.atomic_numbers)
            name = _block_type_name(block)
            if name not in BLOCK_TYPE_TO_CHAIN_ID:
                raise OdacConversionError(f"unknown block_type {name!r}")
            chain_ids[offset : offset + count] = BLOCK_TYPE_TO_CHAIN_ID[name]
            block_type[offset : offset + count] = BLOCK_TYPE_ID[name]
            offset += count
        return chain_ids, block_type

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        record = pickle.loads(self._txn().get(key))

        atomic_numbers = torch.as_tensor(
            np.asarray(record.atomic_numbers), dtype=torch.long
        )
        atom_count = int(atomic_numbers.numel())
        fractional = torch.as_tensor(np.asarray(record.frac_coords), dtype=torch.float)
        cell = torch.as_tensor(np.asarray(record.cell), dtype=torch.float)

        edge_index, target_cell_index = self._bond_graph(record, cell, fractional)
        atom_chain_ids, atom_block_type = self._chain_ids(record, atom_count)

        # Same order as FlexMOF MOFDataset.__getitem__: unwrap in fractional
        # space, go cartesian, symmetrise + orient the lattice, then centre the
        # whole system as one rigid body.
        normalized_fractional = fractional % 1.0 % 1.0
        shifts = MOFDataset._canonical_linker_atom_cell_shifts(
            atom_count, edge_index, target_cell_index
        )
        normalized_fractional = normalized_fractional + shifts.to(
            dtype=normalized_fractional.dtype
        )
        positions = normalized_fractional @ cell

        numpy_cell = cell.cpu().numpy().astype(np.float64)
        numpy_cell, positions = MOFDataset._symmetrize_lattice(numpy_cell, positions)
        pbc, numpy_cell = MOFDataset._orient_lattice(
            np.ones(3, dtype=bool), numpy_cell
        )
        oriented_cell = torch.as_tensor(numpy_cell, dtype=torch.float64)
        positions = torch.as_tensor(positions, dtype=torch.float64)
        positions = positions + (
            0.5 * oriented_cell.sum(dim=0) - positions.mean(dim=0)
        )

        tokens = torch.cat([atomic_numbers, torch.full([8], CORNER_TOKEN)])
        positions = torch.cat(
            [positions, CELL_CORNERS.to(oriented_cell) @ oriented_cell], dim=0
        )
        chain_ids = torch.cat(
            [atom_chain_ids, torch.full([8], CORNER_CHAIN_ID, dtype=torch.long)]
        )

        if self.drop_bond_features:
            edge_index = torch.zeros([2, 0], dtype=torch.long)
            target_cell_index = torch.zeros([0, 3], dtype=torch.long)

        edge_count = edge_index.size(1)
        edge_attr = torch.ones([edge_count, 1], dtype=torch.long)
        node_count = tokens.numel()

        # Kept identical to FlexMOF: the dense adjacency is all-ones, so the
        # shortest-path bias is uniform and bonds only enter through
        # attention_edge_type. Changing this would break checkpoint transfer.
        adjacency = torch.ones([node_count, node_count], dtype=torch.bool)
        attention_edge_type = torch.zeros([node_count, node_count, 1], dtype=torch.long)
        if edge_count:
            attention_edge_type[edge_index[0], edge_index[1]] = edge_attr
            attention_edge_type[edge_index[1], edge_index[0]] = edge_attr

        return {
            "idx": self.dataset_indices[index],
            "key": key.decode() if isinstance(key, bytes) else str(key),
            "atomic_numbers": atomic_numbers,
            "block_type": atom_block_type,
            "token_type": tokens,
            "num_atoms": atom_count,
            "coords": positions,
            "cell": oriented_cell,
            "pbc": torch.as_tensor(pbc, dtype=torch.bool),
            "chain_ids": chain_ids,
            "node_attr": torch.cat(
                [tokens[:, None], torch.zeros([node_count, 8], dtype=torch.long)],
                dim=-1,
            ),
            "adj": adjacency,
            "attn_bias": torch.zeros([node_count + 1, node_count + 1]),
            "in_degree": adjacency.long().sum(dim=1),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "target_cell_index": torch.zeros_like(target_cell_index),
        }


def collate_odac(items: list[dict]) -> dict[str, torch.Tensor]:
    """`collate_mofs` plus the role-aware chain ids it would otherwise flatten."""
    batch = collate_mofs(items)
    max_nodes = batch["chain_ids"].size(1)
    batch["chain_ids"] = torch.stack(
        [_pad_1d(item["chain_ids"], max_nodes) for item in items]
    )
    return batch


def zero_new_role_embeddings(module: torch.nn.Module) -> list[int]:
    """Zero the SOLVENT/ADSORBATE rows of the pretrained chain-id embedding.

    Those rows were never updated during FlexMOF pretraining (chain_ids was a
    constant 1 and weight decay was 0), so their random init would inject noise
    into exactly the atoms the finetune is meant to learn.
    """
    embedding = None
    for name, submodule in module.named_modules():
        if name.endswith("embedding.chain_id_proj"):
            embedding = submodule
            break
    if embedding is None:
        raise RuntimeError("chain_id_proj not found; check the module layout")
    with torch.no_grad():
        for chain_id in NEW_ROLE_CHAIN_IDS:
            embedding.weight[chain_id].zero_()
    return list(NEW_ROLE_CHAIN_IDS)
