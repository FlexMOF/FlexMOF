"""Batched conversion between explicit-atom graphs and ``terminal_h_counts_v1``.

``hydrogen_data`` factors and expands one graph at a time on CPU, which is the
right shape for corpus preparation but not for a training step that has to
convert every building block of a batch. The functions here are the same
representation -- the collapse rule, the core ordering and the expansion order
are checked against ``factor_terminal_hydrogens`` and
``expand_terminal_hydrogens`` -- applied to padded ``[B, A]`` tensors on
whatever device the batch already lives on.

The fold also returns the index maps the fusion needs. ``core_to_explicit``
and ``explicit_to_core`` relate the two atom orderings, and
``explicit_latent_core`` says which core atom's latent an explicit atom should
read: itself if it survived, its parent if it was collapsed into an H count.
Composed with ``mof_bb_occ_atom_index`` that routes a per-core-atom latent to
every cell atom of every occurrence.
"""

from __future__ import annotations

import torch

from .hydrogen_data import REPRESENTATION


def _degree_and_parent(bonds: torch.Tensor, mask: torch.Tensor):
    valid_pair = mask[:, :, None] & mask[:, None, :]
    adjacency = (bonds > 0) & valid_pair
    degree = adjacency.sum(-1)
    # Matches ``adjacency.long().argmax(-1)``: for a degree-one atom this is
    # its only neighbour, and for every other atom the value is unused.
    parent = adjacency.long().argmax(-1)
    return adjacency, degree, parent


def fold_terminal_hydrogens(
    atom_types: torch.Tensor,
    bond_adj: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    hydrogen_label: int,
    count_classes: int | None = None,
    strict: bool = True,
) -> dict:
    """Collapse degree-one hydrogens on a heavy atom into per-core-atom counts.

    ``atom_types`` holds codec labels, so ``hydrogen_label`` is the position of
    Z=1 in the codec vocabulary rather than the atomic number. Padding is read
    from ``atom_mask`` only; padded label values are never inspected.
    """
    if atom_types.shape != atom_mask.shape:
        raise ValueError("Atom labels and mask must align")
    if bond_adj.shape != (*atom_types.shape, atom_types.shape[-1]):
        raise ValueError("Bond adjacency must be square over the atom axis")
    mask = atom_mask.bool()
    batch, atoms = atom_types.shape
    adjacency, degree, parent = _degree_and_parent(bond_adj, mask)

    is_hydrogen = atom_types == hydrogen_label
    parent_is_hydrogen = is_hydrogen.gather(1, parent)
    collapsed = mask & is_hydrogen & (degree == 1) & ~parent_is_hydrogen
    keep = mask & ~collapsed

    # Core atoms keep their original relative order, as in the single-graph
    # factorization, so a core index is the rank of that atom among survivors.
    rank = keep.long().cumsum(-1) - 1
    core_sizes = keep.sum(-1)
    cores = int(core_sizes.max()) if batch else 0
    cores = max(cores, 1)

    arange_b = torch.arange(batch, device=atom_types.device)[:, None]
    safe_rank = rank.clamp_min(0)

    core_mask = torch.zeros(batch, cores, dtype=torch.bool, device=atom_types.device)
    core_mask[arange_b.expand_as(rank)[keep], safe_rank[keep]] = True
    core_atoms = torch.zeros(batch, cores, dtype=atom_types.dtype, device=atom_types.device)
    core_atoms[arange_b.expand_as(rank)[keep], safe_rank[keep]] = atom_types[keep]

    core_to_explicit = torch.full(
        (batch, cores), -1, dtype=torch.long, device=atom_types.device
    )
    explicit_positions = torch.arange(atoms, device=atom_types.device)[None].expand(batch, atoms)
    core_to_explicit[arange_b.expand_as(rank)[keep], safe_rank[keep]] = explicit_positions[keep]

    explicit_to_core = torch.where(keep, rank, torch.full_like(rank, -1))

    parent_core = rank.gather(1, parent)
    counts = torch.zeros(batch, cores, dtype=torch.long, device=atom_types.device)
    counts.index_put_(
        (arange_b.expand_as(rank)[collapsed], parent_core.clamp_min(0)[collapsed]),
        torch.ones(int(collapsed.sum()), dtype=torch.long, device=atom_types.device),
        accumulate=True,
    )
    counts = counts.masked_fill(~core_mask, 0)
    if count_classes is not None and strict and bool((counts >= count_classes).any()):
        worst = int(counts.max())
        raise ValueError(
            f"Attached-H count {worst} needs at least {worst + 1} classes, "
            f"but the codec provides {count_classes}"
        )

    gather_index = core_to_explicit.clamp_min(0)
    rows = bond_adj.gather(1, gather_index[:, :, None].expand(batch, cores, atoms))
    core_bonds = rows.gather(2, gather_index[:, None, :].expand(batch, cores, cores))
    core_bonds = core_bonds * (core_mask[:, :, None] & core_mask[:, None, :])

    # An explicit atom reads its own latent when it survived and its parent's
    # when it was folded away; padding reads nothing.
    explicit_latent_core = torch.where(collapsed, parent_core, explicit_to_core)
    explicit_latent_core = explicit_latent_core.masked_fill(~mask, -1)

    return {
        "representation": REPRESENTATION,
        "core_atoms": core_atoms,
        "core_bonds": core_bonds,
        "core_mask": core_mask,
        "core_hydrogen_counts": counts,
        "core_sizes": core_sizes,
        "core_to_explicit": core_to_explicit,
        "explicit_to_core": explicit_to_core,
        "explicit_latent_core": explicit_latent_core,
        "collapsed": collapsed,
    }


def expand_terminal_hydrogen_counts(
    core_atoms: torch.Tensor,
    core_bonds: torch.Tensor,
    core_mask: torch.Tensor,
    core_hydrogen_counts: torch.Tensor,
    *,
    hydrogen_label: int,
) -> dict:
    """Inverse of :func:`fold_terminal_hydrogens`, with no chemistry checks.

    Every count is expanded. Hydrogens follow all core atoms, grouped by parent
    in core order, which is the order ``expand_terminal_hydrogens`` produces.
    """
    mask = core_mask.bool()
    counts = core_hydrogen_counts.long().masked_fill(~mask, 0)
    if bool((counts < 0).any()):
        raise ValueError("Attached-H counts must be nonnegative")
    batch, cores = core_atoms.shape
    device = core_atoms.device

    core_sizes = mask.sum(-1)
    totals = core_sizes + counts.sum(-1)
    atoms = max(int(totals.max()) if batch else 1, 1)

    atom_types = torch.zeros(batch, atoms, dtype=core_atoms.dtype, device=device)
    atom_mask = torch.zeros(batch, atoms, dtype=torch.bool, device=device)
    bond_adj = torch.zeros(batch, atoms, atoms, dtype=core_bonds.dtype, device=device)

    core_positions = torch.arange(cores, device=device)[None].expand(batch, cores)
    core_to_explicit = torch.where(mask, core_positions, torch.full_like(core_positions, -1))
    atom_types[:, :cores] = core_atoms.masked_fill(~mask, 0)
    atom_mask[:, :cores] = mask
    bond_adj[:, :cores, :cores] = core_bonds

    # Hydrogen slot h of core atom k lands at core_size + prefix[k] + h.
    prefix = counts.cumsum(-1) - counts
    width = max(int(counts.max()) if batch else 1, 1)
    slots = torch.arange(width, device=device)
    active = slots[None, None, :] < counts[:, :, None]
    target = core_sizes[:, None, None] + prefix[:, :, None] + slots[None, None, :]

    rows, cols = active.nonzero(as_tuple=True)[:2]
    hydrogen_index = target[active]
    atom_types[rows, hydrogen_index] = hydrogen_label
    atom_mask[rows, hydrogen_index] = True
    parent_index = core_positions[rows, cols]
    bond_adj[rows, parent_index, hydrogen_index] = 1
    bond_adj[rows, hydrogen_index, parent_index] = 1

    explicit_latent_core = torch.full((batch, atoms), -1, dtype=torch.long, device=device)
    explicit_latent_core[:, :cores] = core_to_explicit
    explicit_latent_core[rows, hydrogen_index] = parent_index

    return {
        "representation": REPRESENTATION,
        "atom_types": atom_types,
        "bond_adj": bond_adj,
        "atom_mask": atom_mask,
        "atom_sizes": totals,
        "core_to_explicit": core_to_explicit,
        "explicit_latent_core": explicit_latent_core,
    }


def compose_core_occurrence_index(
    occurrence_index: torch.Tensor,
    core_to_explicit: torch.Tensor,
    occurrence_mask: torch.Tensor,
) -> torch.Tensor:
    """Route a per-core-atom quantity to cell atoms.

    ``occurrence_index`` is ``mof_bb_occ_atom_index`` shaped ``[U, R, A]`` in
    explicit-local atom order. The result is ``[U, R, C]`` in core order, with
    ``-1`` wherever the core slot or the occurrence is padding.
    """
    unique, repetitions, _ = occurrence_index.shape
    cores = core_to_explicit.shape[-1]
    gather = core_to_explicit.clamp_min(0)[:, None, :].expand(unique, repetitions, cores)
    routed = occurrence_index.gather(2, gather)
    valid = (core_to_explicit >= 0)[:, None, :] & occurrence_mask[:, :, None]
    return routed.masked_fill(~valid, -1)


def cell_atom_core_index(
    occurrence_index: torch.Tensor,
    explicit_latent_core: torch.Tensor,
    occurrence_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    total_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every cell atom, the building block and core atom it reads from.

    Returns ``(bb_index, core_index)``, both ``[total_atoms]`` and both ``-1``
    for cell atoms no occurrence claims. Explicit hydrogens that the
    representation folded away resolve to their parent core atom.
    """
    unique, repetitions, _ = occurrence_index.shape
    device = occurrence_index.device
    bb = torch.full((total_atoms,), -1, dtype=torch.long, device=device)
    core = torch.full((total_atoms,), -1, dtype=torch.long, device=device)
    claimed = (
        occurrence_mask[:, :, None]
        & atom_mask[:, None, :]
        & (occurrence_index >= 0)
        & (explicit_latent_core >= 0)[:, None, :]
    )
    if not bool(claimed.any()):
        return bb, core
    unique_ids = torch.arange(unique, device=device)[:, None, None].expand_as(claimed)
    cores = explicit_latent_core[:, None, :].expand_as(claimed)
    cells = occurrence_index[claimed]
    bb[cells] = unique_ids[claimed]
    core[cells] = cores[claimed]
    return bb, core
