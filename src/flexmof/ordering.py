"""Deterministic training-target ordering; never run on reverse-diffusion states.

The requested arbitrary leaf root is chosen by (atomic number, original index).
Neighbors use the same tie-break. This is a reproducible DFS convention, not a
graph-isomorphism canonical labelling: changing original indices can break ties
differently in symmetric molecules. Components without leaves start at their
lowest (atomic number, original index) atom. Singleton metal blocks are retained.
"""

from __future__ import annotations

import torch


CANONICAL_ORDER_VERSION = "size_leaf_dfs_first_element_v1"


def dfs_atom_order(atoms: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
    """Return new-to-old indices for an undirected, possibly disconnected graph."""
    numbers = atoms.tolist()
    neighbors = [set() for _ in numbers]
    if edge_index is not None:
        for first, second in edge_index.T.tolist():
            if not (0 <= first < len(numbers) and 0 <= second < len(numbers)):
                raise ValueError("Building-block bond index is outside its atom range")
            if first != second:
                neighbors[first].add(second)
                neighbors[second].add(first)
    key = lambda index: (numbers[index], index)
    ordered_neighbors = [sorted(adjacent, key=key, reverse=True) for adjacent in neighbors]
    remaining = set(range(len(numbers)))
    order = []
    while remaining:
        leaves = [index for index in remaining if len(neighbors[index]) == 1]
        root = min(leaves or remaining, key=key)
        stack = [root]
        while stack:
            index = stack.pop()
            if index not in remaining:
                continue
            remaining.remove(index)
            order.append(index)
            stack.extend(ordered_neighbors[index])
    return torch.tensor(order, dtype=torch.long, device=atoms.device)


def canonical_cell_order(composition: dict, atom_count: int) -> torch.Tensor:
    """Flatten sorted blocks/occurrences and require exactly one copy per atom."""
    occurrence = composition["mof_bb_occ_atom_index"]
    valid = (
        composition["mof_bb_occ_mask"][..., None]
        & composition["mof_bb_atom_mask"][:, None, :]
    )
    order = occurrence[valid]
    if order.numel() != atom_count or not torch.equal(
        order.sort().values, torch.arange(atom_count, device=order.device)
    ):
        raise ValueError("Canonical blocks must cover every cell atom exactly once")
    return order
