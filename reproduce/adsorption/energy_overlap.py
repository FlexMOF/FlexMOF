"""Periodic clash screen, preserving both author-code and ASE radii choices.

AtomMOF's overlap guidance uses COVALENT_RADII from its source, whereas the
paper specifies ASE radii for the related random-sampling clash criterion.
We retain both results because the paper does not publish its energy filter.
"""
from itertools import product

import numpy as np
import torch
from ase.data import chemical_symbols, covalent_radii

from author_const import COVALENT_RADII


def radii_lookup(numbers, device):
    z = np.asarray(numbers, dtype=int)
    if z.ndim != 1 or not len(z) or z.min() <= 0 or z.max() >= len(chemical_symbols):
        raise ValueError('unsupported atomic number')
    author = []
    ase = []
    for number in z:
        symbol = chemical_symbols[number]
        if symbol not in COVALENT_RADII or number >= len(covalent_radii):
            raise ValueError(f'unknown covalent radius for {symbol}')
        author.append(COVALENT_RADII[symbol])
        ase.append(covalent_radii[number])
    a = torch.as_tensor(author, dtype=torch.float64, device=device)
    b = torch.as_tensor(ase, dtype=torch.float64, device=device)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or torch.any(a <= 0) or torch.any(b <= 0):
        raise ValueError('invalid covalent radius')
    return a, b


def overlap_flags_extended(numbers, positions, lattice, guest_bonds=(), device='cpu'):
    """Return all-pair and direct-adsorbate-bond-exempt flags for both radii.

    Excludes i==j exactly as AtomMOF's AtomicOverlapEnergy does. Treats a
    distance equal to the threshold as a clash, as the paper specifies.
    """
    pos = torch.as_tensor(np.asarray(positions), dtype=torch.float64, device=device)
    cell = torch.as_tensor(np.asarray(lattice), dtype=torch.float64, device=device)
    if pos.shape != (len(numbers), 3) or cell.shape != (3, 3):
        raise ValueError('invalid structure shape')
    if not torch.isfinite(pos).all() or not torch.isfinite(cell).all():
        raise ValueError('non-finite structure')
    if abs(float(torch.linalg.det(cell))) < 1e-6:
        raise ValueError('degenerate cell')
    ar, br = radii_lookup(numbers, device)
    if len(numbers) < 2:
        return False, False, False, False
    i, j = torch.triu_indices(len(numbers), len(numbers), offset=1, device=device)
    kept = torch.ones(i.shape, dtype=torch.bool, device=device)
    for bond in guest_bonds:
        if len(bond)!=2:
            raise ValueError('invalid guest bond')
        a,b=sorted(map(int,bond))
        if not 0<=a<b<len(numbers):
            raise ValueError('guest bond index outside structure')
        kept &= ~((i==a)&(j==b))
    frac = pos @ torch.linalg.inv(cell)
    delta = frac[i] - frac[j]
    delta -= torch.round(delta)
    threshold_a = (0.8 * (ar[i] + ar[j])).square()
    threshold_b = (0.8 * (br[i] + br[j])).square()
    hit_a = hit_b = hit_a_exempt = hit_b_exempt = False
    shifts = [(0, 0, 0)] + [s for s in product((-1, 0, 1), repeat=3) if s != (0, 0, 0)]
    for shift in shifts:
        shift_t = torch.tensor(shift, dtype=delta.dtype, device=device)
        cart = (delta + shift_t) @ cell
        d2 = torch.sum(cart * cart, dim=1)
        a_overlap = d2 <= threshold_a
        b_overlap = d2 <= threshold_b
        if not hit_a:
            hit_a = bool(torch.any(a_overlap).item())
        if not hit_b:
            hit_b = bool(torch.any(b_overlap).item())
        if not hit_a_exempt:
            hit_a_exempt = bool(torch.any(a_overlap & kept).item())
        if not hit_b_exempt:
            hit_b_exempt = bool(torch.any(b_overlap & kept).item())
        if hit_a and hit_b and hit_a_exempt and hit_b_exempt:
            break
    return hit_a, hit_b, hit_a_exempt, hit_b_exempt


def overlap_flags(numbers, positions, lattice, device='cpu'):
    """Backward-compatible all-pair flags using author and ASE radii."""
    author,ase,_,_=overlap_flags_extended(numbers,positions,lattice,(),device)
    return author,ase
