"""ODAC25 adapter for the FlexMOF joint model's clean graph conditioning.

The original ODAC adapter establishes cell-atom order from block offsets.  The
joint model additionally requires a unique-block graph and an occurrence map.
We invoke its own canonicalisation using synthetic, distinct fractional IDs
only for matching occurrences to offsets; those IDs are not model inputs.
"""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import torch

from flexmof.data import MOFDataset
from flexmof.model.gaussian_composition import AtomVocabulary
from odac_adapter import OdacDataset, collate_odac


BASE_VOCAB = (1, 6, 7, 8, 9, 15, 16, 17, 23, 24, 28, 29, 30, 35, 53, 56)
FULL_VOCAB = BASE_VOCAB + tuple(z for z in range(1, 119) if z not in BASE_VOCAB)


class JointOdacDataset(OdacDataset):
    def __init__(self, *args, composition_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.composition_config = composition_config
        # _composition_targets is independent of the BW LMDB; avoid its init.
        self._composition_builder = object.__new__(MOFDataset)
        self._composition_builder.composition_config = composition_config

    def __getitem__(self, index):
        item = super().__getitem__(index)
        record = pickle.loads(self._txn().get(self.keys[index]))
        numbers = torch.as_tensor(np.asarray(record.atomic_numbers), dtype=torch.long)
        n = int(numbers.numel())
        ids = torch.zeros((n, 3), dtype=torch.float32)
        ids[:, 0] = (torch.arange(n, dtype=torch.float32) + 1) / (n + 1)
        ids[:, 1] = 0.25
        ids[:, 2] = 0.5
        blocks, roles, offset = [], [], 0
        for block in record.blocks:
            atomic = torch.as_tensor(np.asarray(block.atomic_numbers), dtype=torch.long)
            length = atomic.numel()
            if length == 0 or not torch.equal(numbers[offset:offset + length], atomic):
                raise ValueError(f"ODAC block order mismatch at index {index}, offset {offset}")
            name = getattr(block.block_type, "name", str(block.block_type))
            if name not in ("NODE", "LINKER", "SOLVENT", "ADSORBATE"):
                raise ValueError(f"unknown block role: {name}")
            edges = torch.as_tensor(np.asarray(block.bond_indices), dtype=torch.long)
            if edges.numel() == 0:
                edges = torch.zeros((2, 0), dtype=torch.long)
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"invalid bond shape {tuple(edges.shape)}")
            if edges.numel() and (edges.min() < 0 or edges.max() >= length):
                raise ValueError("block bond index out of range")
            blocks.append({"atomic_numbers": atomic, "edge_index": edges,
                           "frac_coords": ids[offset:offset + length]})
            # NODEs are represented as single metal atoms, as in BW training;
            # linker, solvent and guest molecules retain their explicit graphs.
            roles.append(name != "NODE")
            offset += length
        if offset != n:
            raise ValueError(f"ODAC blocks cover {offset} of {n} atoms")
        pseudo = {"pyg_mols": blocks, "is_linker": roles}
        composition = self._composition_builder._composition_targets(pseudo, numbers, ids)
        occurrences = composition["mof_bb_occ_atom_index"]
        present = composition["mof_bb_occ_mask"].unsqueeze(-1)
        atom_mask = composition["mof_bb_atom_mask"].unsqueeze(1)
        valid = present & atom_mask & occurrences.ge(0)
        chosen = occurrences[valid]
        if chosen.numel() != n or not torch.equal(torch.sort(chosen).values, torch.arange(n)):
            raise ValueError("joint occurrence map must cover each cell atom exactly once")
        decoded = AtomVocabulary(self.composition_config.mof_composition_atom_vocab).decode(
            composition["mof_bb_atom_types"]
        ).unsqueeze(1).expand_as(occurrences)[valid]
        if not torch.equal(decoded, numbers[chosen]):
            raise ValueError("joint graph occurrence changes an atom's element")
        item.update(composition)
        return item


def make_config(source, checkpoint_config_factory):
    args = source["args"]
    base = tuple(int(x) for x in args["mof_composition_atom_vocab"].split(","))
    if base != BASE_VOCAB:
        raise ValueError("unexpected pretrained vocabulary; refusing to reinterpret weights")
    # These three limits guard preprocessing only; they are not learned shapes.
    return checkpoint_config_factory(source, False, {
        "mof_composition_atom_vocab": ",".join(map(str, FULL_VOCAB)),
        "mof_composition_max_unique_bbs": 677,
        "mof_composition_max_atoms_per_bb": 677,
        "mof_composition_max_repetitions": 677,
    })


def load_expanded_weights(model, source):
    """Preserve every FlexMOF tensor; append zero rows for unseen elements."""
    old = source["model"]
    new = model.state_dict()
    if old.keys() != new.keys():
        raise ValueError(f"checkpoint keys differ: missing={new.keys()-old.keys()}, extra={old.keys()-new.keys()}")
    changed = {
        "net.mof_joint.autoencoder.atom_embedding.weight",
        "net.mof_joint.autoencoder.decoder.atom_out.weight",
        "net.mof_joint.autoencoder.decoder.atom_out.bias",
    }
    result = {}
    for key, target in new.items():
        value = old[key]
        if key == "net.mof_joint.vocabulary":
            if tuple(value.tolist()) != BASE_VOCAB or tuple(target.tolist()) != FULL_VOCAB:
                raise ValueError("vocabulary identity mismatch")
            result[key] = target
        elif key in changed:
            if value.shape[0] != len(BASE_VOCAB) or target.shape[0] != len(FULL_VOCAB) or value.shape[1:] != target.shape[1:]:
                raise ValueError(f"unexpected expanded tensor shape: {key}")
            expanded = torch.zeros_like(target)
            expanded[:len(BASE_VOCAB)].copy_(value)
            result[key] = expanded
        else:
            if value.shape != target.shape:
                raise ValueError(f"unexpected shape change for {key}: {value.shape} vs {target.shape}")
            result[key] = value
    model.load_state_dict(result, strict=True)
    return sorted(changed)
