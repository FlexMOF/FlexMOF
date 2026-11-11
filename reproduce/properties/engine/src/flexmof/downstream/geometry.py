"""CIF and generated-coordinate checks without any reference-geometry input."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii

from .common import digest, file_hash


def elements(atoms):
    return dict(sorted(Counter(atoms.get_chemical_symbols()).items()))


def physical_atoms(positions, token_ids, num_atoms, pbc=(True, True, True)):
    """Collated tokens are Z+1, corners=129, padding=0; corners start at N."""
    p, t, n = np.asarray(positions), np.asarray(token_ids), int(num_atoms)
    if p.shape != (len(t), 3) or n < 1 or len(t) < n + 8:
        raise ValueError("Invalid structure tensor shape")
    if not np.all(t[n:n + 8] == 129) or not np.all(t[n + 8:] == 0):
        raise ValueError("Invalid virtual-corner/padding layout")
    z = t[:n] - 1
    if np.any(z < 1) or np.any(z > 118):
        raise ValueError("Nonphysical atom token")
    origin = p[n]
    atoms = Atoms(numbers=z, positions=p[:n] - origin,
                  cell=p[n + np.array([4, 2, 1])] - origin, pbc=pbc)
    validate_atoms(atoms)
    return atoms


def validate_atoms(atoms, expected=None, min_volume=0.1):
    cell = np.asarray(atoms.cell)
    if not len(atoms) or not np.all(atoms.pbc):
        raise ValueError("empty_or_nonperiodic")
    if not np.isfinite(cell).all() or not np.isfinite(atoms.positions).all():
        raise ValueError("nonfinite_geometry")
    if abs(np.linalg.det(cell)) < min_volume or np.linalg.svd(cell, compute_uv=False).min() < 1e-6:
        raise ValueError("degenerate_cell")
    if np.any(atoms.numbers < 1) or np.any(atoms.numbers > 118):
        raise ValueError("illegal_element")
    if expected is not None and elements(atoms) != expected:
        raise ValueError("composition_mismatch")
    for occ in atoms.info.get("occupancy", {}).values():
        if len(occ) != 1 or abs(sum(occ.values()) - 1) > 1e-8:
            raise ValueError("partial_or_disordered_occupancy")
    return atoms


def read_cif(path, expected=None):
    """ASE expands symmetry; reject implicit occupancy/disorder and multiple frames."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frames = read(str(path), index=":", format="cif", store_tags=True)
    if len(frames) != 1:
        raise ValueError("multiple_cif_blocks")
    atoms = frames[0]
    occ = atoms.info.get("_atom_site_occupancy", [])
    if any(abs(float(x) - 1.0) > 1e-8 for x in np.atleast_1d(occ)):
        raise ValueError("partial_or_disordered_occupancy")
    validate_atoms(atoms, expected)
    return atoms, [str(w.message) for w in caught]


def right_handed(atoms):
    """Unimodular basis swap preserves Cartesian geometry, handedness and lattice set."""
    result = atoms.copy()
    if np.linalg.det(result.cell) < 0:
        result.set_cell(np.asarray(result.cell)[[1, 0, 2]], scale_atoms=False)
    return result


def geometry_hash(atoms):
    # Exact float identities for DFT/MLIP comparisons; NOT a symmetry matcher.
    return digest(dict(numbers=atoms.numbers.tolist(), positions=atoms.positions.tolist(),
                       cell=np.asarray(atoms.cell).tolist(), pbc=atoms.pbc.tolist()))


def periodic_graph_diagnostic(atoms, radius_factor=1.25):
    """Rank of lattice winding cycles in a distance-inferred graph; NOT bond truth."""
    cutoffs=covalent_radii[atoms.numbers]*radius_factor
    i,j,shifts=neighbor_list("ijS",atoms,cutoffs,self_interaction=False)
    adjacent=[[] for _ in atoms]
    for a,b,s in zip(i,j,shifts):adjacent[int(a)].append((int(b),s))
    offsets={};components=[]
    for start in range(len(atoms)):
        if start in offsets:continue
        offsets[start]=np.zeros(3,dtype=int);queue=[start];cycles=[];size=0
        while queue:
            a=queue.pop();size+=1
            for b,s in adjacent[a]:
                expected=offsets[a]+s
                if b not in offsets:offsets[b]=expected;queue.append(b)
                else:
                    winding=expected-offsets[b]
                    if np.any(winding):cycles.append(winding)
        components.append(dict(atoms=size,periodic_rank=int(np.linalg.matrix_rank(np.asarray(cycles))) if cycles else 0))
    return dict(method="covalent radius distance graph, winding-cycle rank; geometry diagnostic only",radius_factor=radius_factor,
                components=components,max_periodic_rank=max(c['periodic_rank'] for c in components))


def diagnostics(atoms, bonds=(), collision_A=0.5, bond_ratio_bounds=(0.55, 1.8)):
    validate_atoms(atoms)
    i, j, d, shifts = neighbor_list("ijdS", atoms, cutoff=max(3.5, collision_A), self_interaction=False)
    collision_count = int(np.count_nonzero(d < collision_A) // 2)
    lengths = [float(atoms.get_distance(int(a), int(b), mic=True)) for a, b in bonds]
    ratios = [length / (covalent_radii[atoms.numbers[a]] + covalent_radii[atoms.numbers[b]])
              for (a, b), length in zip(bonds, lengths)]
    bad = int(sum(x < bond_ratio_bounds[0] or x > bond_ratio_bounds[1] for x in ratios))
    # These are distance-shell diagnostics, not oxidation-state or bond-order inference.
    from pymatgen.core import Element
    metal = np.array([Element(s).is_metal for s in atoms.get_chemical_symbols()])
    donor = np.isin(atoms.numbers, [7, 8, 16])
    shell = {str(a): int(np.count_nonzero((i == a) & donor[j] & (d <= 3.0)))
             for a in np.flatnonzero(metal)}
    return dict(volume_A3=float(atoms.get_volume()), density_g_cm3=float(atoms.get_masses().sum() * 1.66053906660 / atoms.get_volume()),
                collision_count=collision_count, min_distance_within_3p5_A=float(d.min()) if len(d) else None,
                linker_bond_lengths_A=lengths, linker_bond_bad_count=bad,
                metal_donor_counts_within_3_A=shell, geometry_pass=(collision_count == 0 and bad == 0),
                periodic_connectivity=periodic_graph_diagnostic(atoms))


def displacement(before, after):
    if not np.array_equal(before.numbers, after.numbers):
        raise ValueError("atom_order_changed")
    # Remove affine cell deformation first; then remove a common periodic translation.
    df = after.get_scaled_positions(wrap=False) - before.get_scaled_positions(wrap=False)
    v, _ = find_mic(df @ after.cell, after.cell, pbc=True)
    shift = v.mean(axis=0)
    for _ in range(3):
        internal, _ = find_mic(v - shift, after.cell, pbc=True)
        shift += internal.mean(axis=0)
    internal, distances = find_mic(v - shift, after.cell, pbc=True)
    return dict(internal_rms_displacement_A=float(np.sqrt(np.mean(distances**2))),
                internal_max_displacement_A=float(distances.max()),
                translation_A=shift.tolist(), deformation_gradient=(np.linalg.solve(before.cell, after.cell)).tolist(),
                volume_ratio=float(after.get_volume() / before.get_volume()),
                alignment="fixed atom identities; affine deformation removed; periodic translation fitted; no atom permutation")


def audit_candidate(candidate, config):
    result = dict(candidate_id=candidate["candidate_id"], condition_id=candidate["condition_id"], status="ok")
    try:
        if candidate["status"] != "present":
            result.update(status=candidate["status"], geometry_pass=False)
            return result
        if file_hash(candidate["cif_path"]) != candidate["cif_sha256"]:
            raise ValueError("cif_hash_changed")
        atoms, notes = read_cif(candidate["cif_path"], candidate["expected_elements"])
        result.update(diagnostics(atoms, candidate.get("linker_bonds", []), config["collision_A"], config["bond_ratio_bounds"]))
        result.update(n_atoms=len(atoms), elements=elements(atoms), warnings=notes,
                      geometry_hash=geometry_hash(atoms), handedness="left_basis" if np.linalg.det(atoms.cell)<0 else "right_basis",
                      hydrogen_present="H" in elements(atoms), chemistry_status=candidate["chemistry"]["status"])
    except Exception as exc:
        result.update(status="invalid", error=f"{type(exc).__name__}: {exc}", geometry_pass=False)
    return result
