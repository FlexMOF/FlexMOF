"""Atom-and-binary-bond linker graphs, with no coordinates or MOF identifiers.

The binary graph certificate deliberately treats every edge as a single bond;
it identifies the representation used by FlexMOF, not chemical bond orders.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset


def graph_certificate(numbers, edges):
    """Canonical certificate retaining explicit H and all binary connectivity."""
    from rdkit import Chem

    molecule = Chem.RWMol()
    for number in numbers:
        atom = Chem.Atom(int(number))
        atom.SetNoImplicit(True)
        molecule.AddAtom(atom)
    for first, second in edges:
        molecule.AddBond(int(first), int(second), Chem.BondType.SINGLE)
    return Chem.MolToSmiles(molecule.GetMol(), canonical=True, allHsExplicit=True)


def identity_split(certificate, seed=20260915):
    digest = hashlib.sha256(f"{seed}:{certificate}".encode()).hexdigest()
    value = int(digest[:16], 16) / 2**64
    return "train" if value < 0.8 else "valid" if value < 0.9 else "test"


def laplacian_features(adjacency, frequencies):
    """Lowest combinatorial Laplacian eigenpairs, including the zero modes.

    Eigensystems are computed on each unpadded graph in float64. The mask
    excludes missing frequencies rather than treating them as zero eigenpairs.
    """
    matrix = adjacency.double()
    values, vectors = torch.linalg.eigh(torch.diag(matrix.sum(-1)) - matrix)
    count = min(len(matrix), frequencies)
    result = torch.zeros(len(matrix), frequencies, 2, device=matrix.device)
    result[:, :count, 0] = vectors[:, :count].float()
    result[:, :count, 1] = values[:count].clamp_min(0).float()
    mask = torch.arange(frequencies, device=matrix.device) < count
    return result, mask


class LinkerDataset(Dataset):
    def __init__(self, path, split, *, vocabulary=None, include_singletons=False, corpus=None):
        self.path = Path(path)
        self.corpus = corpus if corpus is not None else torch.load(self.path, map_location="cpu", weights_only=False)
        if self.corpus["format_version"] != 1:
            raise ValueError("Unsupported linker corpus version")
        if split not in {"train", "valid", "test", "all"}:
            raise ValueError(f"Unknown linker split: {split}")
        self.records = [r for r in self.corpus["graphs"] if split == "all" or r["split"] == split]
        self.vocabulary = vocabulary or self.corpus["vocabulary"]
        if not set(self.corpus["vocabulary"]).issubset(self.vocabulary):
            raise ValueError("Vocabulary does not cover corpus atoms")
        self.lookup = {z: i for i, z in enumerate(self.vocabulary)}
        if include_singletons:
            if split != "train":
                raise ValueError("Auxiliary singleton blocks are training-only")
            # MOF non-linker atoms are singleton compact blocks. Include a small
            # explicit auxiliary set so the same codec also supports metals.
            frequencies = self.corpus["frequencies"]
            spectral, spectral_mask = laplacian_features(torch.zeros(1, 1), frequencies)
            repeats = max(1, len(self.records) // (20 * len(self.vocabulary)))
            for z in self.vocabulary:
                record = {"numbers": torch.tensor([z]), "adjacency": torch.zeros(1,1,dtype=torch.uint8),
                          "spectral": spectral, "spectral_mask": spectral_mask,
                          "certificate": f"auxiliary_singleton_{z}", "split": "train"}
                self.records.extend([record] * repeats)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            **record,
            "atom_labels": torch.tensor([self.lookup[int(z)] for z in record["numbers"]]),
        }


def collate_linkers(records):
    count = len(records)
    length = max(len(r["numbers"]) for r in records)
    frequencies = records[0]["spectral"].size(1)
    atoms = torch.zeros(count, length, dtype=torch.long)
    bonds = torch.zeros(count, length, length, dtype=torch.long)
    mask = torch.zeros(count, length, dtype=torch.bool)
    spectral = torch.zeros(count, length, frequencies, 2)
    spectral_mask = torch.zeros(count, frequencies, dtype=torch.bool)
    for i, record in enumerate(records):
        n = len(record["numbers"])
        atoms[i, :n] = record["atom_labels"]
        bonds[i, :n, :n] = record["adjacency"]
        mask[i, :n] = True
        spectral[i, :n] = record["spectral"]
        spectral_mask[i] = record["spectral_mask"]
    return {"atoms": atoms, "bonds": bonds, "mask": mask, "spectral": spectral,
            "spectral_mask": spectral_mask}
