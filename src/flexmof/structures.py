"""Conversion and evaluation of generated periodic MOF structures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .match_protocol import resolve


def compact_composition_record(numbers, bonds, atom_mask, occurrences, occurrence_mask):
    """JSON-ready generated labels/edges and their count-conditioned cell mapping."""
    numbers, bonds, atom_mask, occurrences, occurrence_mask = (
        value.detach().cpu() for value in
        (numbers, bonds, atom_mask, occurrences, occurrence_mask)
    )
    blocks = []
    for index, mask in enumerate(atom_mask.bool()):
        if not mask.any():
            continue
        adjacency = bonds[index][mask][:, mask]
        blocks.append({
            "atomic_numbers": numbers[index][mask].tolist(),
            "bonds": torch.triu(adjacency, diagonal=1).nonzero().tolist(),
            "repetitions": int(occurrence_mask[index].sum()),
            "cell_atom_indices": occurrences[index][occurrence_mask[index]][:, mask].tolist(),
        })
    return {"schema_version": 1, "blocks": blocks}


def compact_graphs_match(atoms, bonds, target_atoms, target_bonds, atom_mask):
    """Exact element-labelled graph equality, allowing local atom permutations."""
    import networkx as nx

    atoms, bonds, target_atoms, target_bonds, atom_mask = (
        value.detach().cpu()
        for value in (atoms, bonds, target_atoms, target_bonds, atom_mask)
    )
    for block, mask in enumerate(atom_mask.bool()):
        if not mask.any():
            continue
        graphs = []
        for labels, adjacency in ((atoms, bonds), (target_atoms, target_bonds)):
            graph = nx.from_numpy_array(adjacency[block][mask][:, mask].numpy())
            nx.set_node_attributes(
                graph, dict(enumerate(labels[block][mask].tolist())), "atom"
            )
            graphs.append(graph)
        if not nx.is_isomorphic(
            *graphs,
            node_match=nx.algorithms.isomorphism.categorical_node_match("atom", None),
        ):
            return False
    return True


class MOFStructureEvaluator:
    def __init__(self, output_dir: str | Path, protocol: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = resolve(protocol)
        self.matcher = StructureMatcher(**self.protocol.matcher_kwargs())

    @staticmethod
    def _convert(
        batch: dict,
        positions: torch.Tensor,
        sampled_atom_types: torch.Tensor | None = None,
    ) -> list[Atoms | None]:
        atom_counts = batch["num_atoms"].cpu().tolist()
        atomic_numbers = (
            sampled_atom_types.detach().cpu() - 1
            if sampled_atom_types is not None
            else batch["node_attr"][:, :-8, 0].detach().cpu() - 1
        )
        pbcs = batch["pbc"].cpu().numpy()
        structures = []
        for index, atom_count in enumerate(atom_counts):
            try:
                position = positions[index, : atom_count + 8].cpu()
                numbers = atomic_numbers[index, :atom_count].numpy()
                position = position - position[-8]
                cell = torch.stack([position[-4], position[-6], position[-7]])
                structures.append(
                    Atoms(
                        numbers=numbers,
                        positions=position[:-8].numpy(),
                        cell=cell.numpy(),
                        pbc=pbcs[index],
                    )
                )
            except Exception as error:
                print(f"could not convert structure {index}: {error}", flush=True)
                structures.append(None)
        return structures

    @staticmethod
    def _is_p1(structure: Structure) -> float:
        try:
            return float(structure.get_space_group_info(symprec=0.1)[0] == "P1")
        except (TypeError, ValueError):
            return 1.0

    def _match(
        self,
        sampled: Atoms | None,
        original: Atoms | None,
        sample_id: int,
        sample_time_index: int = 0,
    ) -> dict[str, float]:
        if original is not None:
            original.write(
                self.output_dir / f"original_{sample_id}_{sample_time_index}.cif",
                format="cif",
            )
        if sampled is not None:
            sampled.write(
                self.output_dir / f"sampled_{sample_id}_{sample_time_index}.cif",
                format="cif",
            )
        if sampled is None or original is None:
            return {"rmsd": np.nan, "p1": 1.0, "match": 0}

        sampled_structure = Structure.from_ase_atoms(sampled)
        original_structure = Structure.from_ase_atoms(original)
        p1 = self._is_p1(sampled_structure)
        try:
            rmsd = self.matcher.get_rms_dist(sampled_structure, original_structure)
        except Exception as error:
            print(f"could not match structure {sample_id}: {error}", flush=True)
            rmsd = None
        return {
            "rmsd": np.nan if rmsd is None else float(rmsd[0]),
            "p1": p1,
            "match": int(rmsd is not None),
        }

    def convert_and_match(
        self,
        batch: dict,
        original_positions: torch.Tensor,
        sampled_atom_types: torch.Tensor | None = None,
        sample_time_index: int = 0,
    ) -> list[dict[str, float]]:
        sampled = self._convert(batch, batch["pos"], sampled_atom_types)
        original = self._convert(batch, original_positions)
        return [
            self._match(
                generated,
                target,
                int(sample_id),
                sample_time_index=sample_time_index,
            )
            for generated, target, sample_id in zip(
                sampled, original, batch["idx"].cpu().tolist()
            )
        ]
