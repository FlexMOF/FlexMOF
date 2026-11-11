"""Coordinate operations shared by MOF training and sampling."""

from __future__ import annotations

import torch


CELL_CORNERS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def initialize_lattice(batch: dict, size_factor: float) -> torch.Tensor:
    """Create the centered cubic prior used by periodic DDPM training."""
    stable = batch["is_stable_periodic"]
    atom_count = batch["num_atoms"][stable]
    size = size_factor * atom_count.pow(1 / 3)
    corners = batch["pos"].new_tensor(CELL_CORNERS)
    corners = corners[None] * size[:, None, None] - size[:, None, None] / 2
    initial = torch.zeros_like(batch["pos"])
    scatter = torch.arange(8, device=initial.device).view(1, 8, 1)
    scatter = scatter.repeat(initial[stable].size(0), 1, 3)
    scatter += atom_count[:, None, None]
    initial[stable] = initial[stable].scatter(1, scatter, corners)
    batch["init_pos"] = initial
    return initial


def center_periodic(batch: dict, padding_mask: torch.Tensor) -> torch.Tensor:
    """Center each periodic structure at the midpoint of opposite corners."""
    stable = batch["is_stable_periodic"]
    atom_count = batch["num_atoms"][stable]
    gather = atom_count[:, None, None].repeat(1, 1, 3)
    first = torch.gather(batch["pos"][stable], 1, gather)
    last = torch.gather(batch["pos"][stable], 1, gather + 7)
    batch["pos"][stable] -= (first + last) / 2.0
    batch["pos"] = batch["pos"].masked_fill(padding_mask[..., None], 0.0)
    return batch["pos"]


def complete_lattice(positions: torch.Tensor, batch: dict) -> torch.Tensor:
    """Reconstruct all eight corners from the four independent corner nodes."""
    stable = batch["is_stable_periodic"]
    periodic = positions[stable]
    matrix = periodic.new_tensor(CELL_CORNERS)
    atom_count = batch["num_atoms"][stable]
    gather = periodic.new_tensor([0, 4, 2, 1], dtype=torch.long)
    gather = gather.view(1, 4, 1).repeat(periodic.size(0), 1, 3)
    gather += atom_count[:, None, None]
    lattice_points = torch.gather(periodic, 1, gather)
    corner = lattice_points[:, 0]
    lattice = lattice_points[:, 1:] - corner[:, None]
    batch["cell"][stable] = lattice.to(batch["cell"].dtype)
    cell = torch.matmul(matrix, lattice) + corner[:, None]
    scatter = torch.arange(8, device=periodic.device).view(1, 8, 1)
    scatter = scatter.repeat(periodic.size(0), 1, 3) + atom_count[:, None, None]
    cell -= ((cell[:, 0] + cell[:, 7]) / 2.0)[:, None]
    positions[stable] = periodic.scatter(1, scatter, cell)
    padding_mask = batch["token_id"].eq(0)
    return positions.masked_fill(padding_mask[..., None], 0.0)
