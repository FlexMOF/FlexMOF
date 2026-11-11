"""Evaluate adsorption candidates with eSEN relaxation and UMA energies."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from ase import Atoms
from ase.optimize import FIRE

from energy_overlap import overlap_flags_extended


CHECKPOINT_SHA = 'f37e315fd4545d4ee31a42f8050bcf3298594a38c5e89d33dd5a502ed47829e7'
UMA_SHA = 'ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b'
ESEN_SHA = '48fbbd851f73e82a5b2d0cb89e946cf3331d8e6734ddfe3db4112b8636b2010e'
ESEN = Path(os.environ.get('FLEXMOF_ESEN_CHECKPOINT', 'checkpoints/esen_sm_odac25_full.pt'))
VERSION = 'odac-energy-v1'
PROFILE = 'unsteered'
EXPECTED_STEERING = 'none'
FMAX = 0.05
STEPS = 300
_code_hash = hashlib.sha256()
for _filename in ('energy_eval.py', 'energy_overlap.py', 'author_const.py'):
    _code_hash.update(_filename.encode() + b'\0')
    _code_hash.update((Path(__file__).resolve().parent/_filename).read_bytes())
ENERGY_CODE_SHA = _code_hash.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def new_file(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # Never silently replace existing evidence.
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    encoded = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    new_file(path, lambda out: out.write(encoded))


def write_structure(path, atoms):
    value = {'atomic_numbers': torch.as_tensor(atoms.numbers.copy(), dtype=torch.long),
             'positions': torch.as_tensor(atoms.positions.copy(), dtype=torch.float64),
             'cell': torch.as_tensor(np.asarray(atoms.cell).copy(), dtype=torch.float64)}
    new_file(path, lambda out: torch.save(value, out))
    return sha256(path)


def identity(**extra):
    return {'protocol': VERSION, 'checkpoint_sha256': CHECKPOINT_SHA,
            'uma_sha256': UMA_SHA, 'esen_sha256': ESEN_SHA,
            'energy_code_sha256': ENERGY_CODE_SHA, 'sampling_profile': PROFILE, **extra}


def resume(path, expected):
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f'persisted energy result identity mismatch: {path}, {key}')
    if row.get('structure_path'):
        structure = Path(row['structure_path'])
        if not structure.is_file() or sha256(structure) != row['structure_sha256']:
            raise ValueError(f'persisted relaxed structure mismatch: {path}')
    return row


def make_atoms(numbers, positions, lattice):
    numbers = np.asarray(numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if (positions.shape != (len(numbers), 3) or lattice.shape != (3, 3) or
        not np.isfinite(positions).all() or not np.isfinite(lattice).all() or
        abs(np.linalg.det(lattice)) < 1e-6):
        raise ValueError('invalid structure geometry')
    if not len(numbers) or numbers.min() < 1 or numbers.max() > 100:
        raise ValueError('element outside MLIP support')
    return Atoms(numbers=numbers, positions=positions, cell=lattice, pbc=True,
                 info={'charge': 0, 'spin': 0})


def calculate_energy(atoms, calculator):
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    if not math.isfinite(energy):
        raise ValueError('nonfinite UMA energy')
    return energy


def relax(atoms, calculator):
    atoms.calc = calculator
    optimizer = FIRE(atoms, logfile=None)
    optimizer.run(fmax=FMAX, steps=STEPS)
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError('nonfinite eSEN forces')
    max_force = float(np.linalg.norm(forces, axis=1).max())
    if not np.isfinite(atoms.positions).all():
        raise ValueError('nonfinite relaxed geometry')
    return {'converged': max_force < FMAX, 'max_force_eV_per_A': max_force,
            'steps': int(optimizer.nsteps)}


def error_dict(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
    return {'error_type': type(exc).__name__, 'error': str(exc)[:1000]}


def host_result(root, host, esen, uma):
    host_id = int(host['host_id'])
    path = root/'energy/hosts'/f'{host_id:04d}.json'
    base = identity(kind='host', host_id=host_id, input_sha256=host['geometry_sha256'])
    existing = resume(path, base)
    if existing is not None:
        return existing
    start = time.monotonic()
    result = dict(base)
    try:
        source = Path(host['geometry_path'])
        if sha256(source) != host['geometry_sha256']:
            raise ValueError('host input hash mismatch')
        geometry = torch.load(source, map_location='cpu', weights_only=True)
        atoms = make_atoms(geometry['atomic_numbers'].numpy(), geometry['positions'].numpy(), geometry['cell'].numpy())
        result.update(relax(atoms, esen))
        structure_path = root/'energy/relaxed_hosts'/f'{host_id:04d}_{uuid.uuid4().hex}.pt'
        result.update(structure_path=str(structure_path), structure_sha256=write_structure(structure_path, atoms))
        if result['converged']:
            result.update(status='complete', uma_energy_eV=calculate_energy(atoms, uma))
        else:
            result.update(status='relax_not_converged', uma_energy_eV=None)
    except Exception as exc:
        result.update(status='error', **error_dict(exc))
    result['seconds'] = time.monotonic() - start
    write_json(path, result)
    return result


def reference_result(root, index, source_sha, record, uma):
    path = root/'energy/references'/f'{index:05d}.json'
    base = identity(kind='reference', dataset_index=index, source_sha256=source_sha)
    existing = resume(path, base)
    if existing is not None:
        return existing
    result = dict(base)
    start = time.monotonic()
    try:
        ref = record['reference']
        atoms = make_atoms(ref['atom_types'].numpy(), ref['true_coords'].numpy(), ref['true_lattice'].numpy())
        result.update(status='complete', uma_energy_eV=calculate_energy(atoms, uma), atoms=len(atoms))
    except Exception as exc:
        result.update(status='error', **error_dict(exc))
    result['seconds'] = time.monotonic() - start
    write_json(path, result)
    return result


def candidate_result(root, index, sample_index, source_sha, record, esen, uma, device, guest_bonds, guest_bonds_sha):
    path = root/'energy/candidates'/f'{index:05d}_{sample_index:02d}.json'
    sample = record['samples'][sample_index]
    base = identity(kind='candidate', dataset_index=index, sample_index=sample_index,
                    seed=int(sample['seed']), source_sha256=source_sha, guest_bonds_sha256=guest_bonds_sha)
    existing = resume(path, base)
    if existing is not None:
        return existing
    result = dict(base)
    start = time.monotonic()
    try:
        if sample['coords'] is None or sample['lattice'] is None:
            result.update(status='generation_failed', generation_error=sample.get('error_type'))
        else:
            numbers = record['reference']['atom_types'].numpy()
            pos = sample['coords'].numpy()
            cell = sample['lattice'].numpy()
            flags=overlap_flags_extended(numbers,pos,cell,guest_bonds,device=device)
            result.update(overlap_author_radii=flags[0],overlap_ase_radii=flags[1],
                overlap_author_guest_bond_exempt_radii=flags[2],overlap_ase_guest_bond_exempt_radii=flags[3])
            if all(flags):
                result['status'] = 'overlap_filtered_all'
            else:
                atoms = make_atoms(numbers, pos, cell)
                result.update(relax(atoms, esen))
                if result['converged']:
                    structure_path = root/'energy/relaxed_candidates'/f'{index:05d}_{sample_index:02d}_{uuid.uuid4().hex}.pt'
                    result.update(structure_path=str(structure_path), structure_sha256=write_structure(structure_path, atoms))
                    result.update(status='complete', uma_energy_eV=calculate_energy(atoms, uma))
                else:
                    result['status'] = 'relax_not_converged'
    except Exception as exc:
        result.update(status='error', **error_dict(exc))
    result['seconds'] = time.monotonic() - start
    write_json(path, result)
    return result


def make_calculators(root):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    settings = InferenceSettings(tf32=False, activation_checkpointing=False, merge_mole=False,
        compile=False, external_graph_gen=False, internal_graph_gen_version=1,
        base_precision_dtype='float32', execution_mode='general')
    e = load_predict_unit(ESEN, device='cuda', inference_settings=settings)
    u = load_predict_unit(root/'weights/uma-s-1p2.pt', device='cuda', inference_settings=settings)
    return FAIRChemCalculator(e), FAIRChemCalculator(u, task_name='odac')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world < 1:
        raise ValueError('world size must be positive')
    root = args.root
    attempt = os.environ.get('ENERGY_ATTEMPT', '')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', attempt):
        raise ValueError('ENERGY_ATTEMPT must identify this launch')
    pilot_dir = root/'energy/pilots'/attempt
    failure_dir = root/'energy/failures'/attempt
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=datetime.timedelta(days=7))
    try:
        inputs_path=root/'energy/inputs/manifest.json'
        inputs_sha=sha256(inputs_path)
        if inputs_sha!=os.environ.get('ENERGY_INPUTS_MANIFEST_SHA256'):
            raise ValueError('energy input manifest differs from the submitted input')
        inputs = json.loads(inputs_path.read_text())
        guest_bond_path=root/'energy/inputs/guest_bonds.json'
        guest_bond_data=json.loads(guest_bond_path.read_text())
        guest_bonds_sha=sha256(guest_bond_path)
        if guest_bonds_sha!=os.environ.get('ENERGY_GUEST_BONDS_SHA256'):
            raise ValueError('guest bond file differs from the submitted input')
        if (len(guest_bond_data['records'])!=3582 or guest_bond_data['dataset_sha256']!=inputs['dataset_sha256'] or
            guest_bond_data['metadata_sha256']!=inputs['metadata_sha256'] or
            [r['dataset_index'] for r in guest_bond_data['records']]!=list(range(3582))):
            raise ValueError('guest bond input identity mismatch')
        generation = json.loads((root/'generation/manifest.json').read_text())
        if (len(inputs['records']) != 3582 or len(inputs['systems']) != 683 or
            generation['status'] != 'completed' or generation['records'] != 3582 or generation['samples_per_record'] != 10 or
            generation['checkpoint_sha256'] != CHECKPOINT_SHA or generation['steering'] != EXPECTED_STEERING or
            generation['sampling_calls'] != 5000):
            raise ValueError('input identity mismatch')
        host_by_index = {r['dataset_index']:r['host_id'] for r in inputs['records']}
        required_hosts = set()
        for system in inputs['systems']:
            group = {host_by_index[i] for i in system['indices']}
            if len(group) > 1:
                required_hosts.update(group)
        selected_hosts = [h for h in inputs['hosts'] if h['host_id'] in required_hosts]
        if len(selected_hosts) != 71:
            raise ValueError('unexpected number of surrogate host references')
        if rank == 0:
            if sha256(ESEN) != ESEN_SHA or sha256(root/'weights/uma-s-1p2.pt') != UMA_SHA:
                raise ValueError('MLIP checkpoint hash mismatch')
        dist.barrier()
        esen, uma = make_calculators(root)
        pilot_index = max(inputs['records'], key=lambda v: v['atoms'])['dataset_index'] if rank == 0 else (rank * 97) % 3582
        pilot_record = torch.load(root/'generation/records'/f'{pilot_index:05d}.pt', map_location='cpu', weights_only=True)
        ref = pilot_record['reference']
        test_atoms = make_atoms(ref['atom_types'].numpy(), ref['true_coords'].numpy(), ref['true_lattice'].numpy())
        test_atoms.calc = esen
        forces = np.asarray(test_atoms.get_forces())
        if forces.shape != (len(test_atoms), 3) or not np.isfinite(forces).all():
            raise ValueError('eSEN GPU pilot produced invalid forces')
        pilot_energy = calculate_energy(test_atoms, uma)
        pilot = identity(rank=rank, dataset_index=pilot_index, atoms=len(test_atoms),
                         uma_energy_eV=pilot_energy, esen_force_max=float(np.linalg.norm(forces,axis=1).max()),
                         status='passed')
        if rank == world - 1:
            # Exercise the complete optimizer/energy path on an isolated copy
            # of a real, previously screened candidate before formal scoring.
            selected = None
            for index in range(3582):
                sample_record=torch.load(root/'generation/records'/f'{index:05d}.pt',map_location='cpu',weights_only=True)
                numbers=sample_record['reference']['atom_types'].numpy()
                for candidate, sample in enumerate(sample_record['samples']):
                    if sample['coords'] is None:
                        continue
                    flags=overlap_flags_extended(numbers,sample['coords'].numpy(),sample['lattice'].numpy(),
                          guest_bond_data['records'][index]['exclude_adsorbate_bonds'],device=device)
                    if not flags[0] and not flags[1]:
                        selected=(index,candidate,sample,numbers)
                        break
                if selected is not None:
                    break
            if selected is None:
                raise ValueError('no non-overlapping candidate is available for the energy pilot')
            index,candidate,sample,numbers=selected
            real=make_atoms(numbers,sample['coords'].numpy(),sample['lattice'].numpy())
            trial_start=time.monotonic()
            try:
                trial=relax(real,esen)
                trial.update(status='complete',uma_energy_eV=calculate_energy(real,uma))
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                # A single generated geometry can be physically pathological;
                # retain that diagnosis rather than failing all test systems.
                trial={'status':'candidate_error',**error_dict(exc)}
            trial.update(dataset_index=index,sample_index=candidate,seconds=time.monotonic()-trial_start,
                         note='diagnostic only; scored independently after all pilots pass')
            pilot['real_candidate_relaxation']=trial
        write_json(pilot_dir/f'rank{rank:02d}.json', pilot)
        dist.barrier()
        if rank == 0:
            reports = [json.loads((pilot_dir/f'rank{i:02d}.json').read_text()) for i in range(world)]
            if len(reports) != world or any(v['status'] != 'passed' for v in reports):
                raise ValueError('MLIP pilot incomplete')
            write_json(pilot_dir/'passed.json', identity(status='passed', ranks=world, reports=reports))
        dist.barrier()
        started = time.monotonic()
        for host in selected_hosts[rank::world]:
            result = host_result(root, host, esen, uma)
            print(json.dumps({'rank':rank,'host_id':host['host_id'],'status':result['status'],
                              'seconds':result['seconds']}), flush=True)
        nref_errors = 0
        for index in range(rank, 3582, world):
            source = root/'generation/records'/f'{index:05d}.pt'
            source_sha = sha256(source)
            record = torch.load(source, map_location='cpu', weights_only=True)
            if (record['dataset_index'] != index or record['checkpoint_sha256'] != CHECKPOINT_SHA or
                len(record['samples']) != 10 or record['steering'] != EXPECTED_STEERING):
                raise ValueError('source candidate identity mismatch')
            ref = reference_result(root, index, source_sha, record, uma)
            nref_errors = nref_errors + 1 if ref['status'] == 'error' else 0
            if nref_errors >= 10:
                raise RuntimeError('ten consecutive UMA reference errors; stop instead of silently scoring failures')
            bonds=guest_bond_data['records'][index]['exclude_adsorbate_bonds']
            results = [candidate_result(root, index, k, source_sha, record, esen, uma, device,
                                        bonds, guest_bonds_sha) for k in range(10)]
            print(json.dumps({'rank':rank,'dataset_index':index,'reference':ref['status'],
                              'candidates':[r['status'] for r in results],
                              'elapsed_seconds':time.monotonic()-started}), flush=True)
        dist.barrier()
        if rank == 0:
            paths = {'hosts':root/'energy/hosts','references':root/'energy/references',
                     'candidates':root/'energy/candidates'}
            counts = {k:len(list(v.glob('*.json'))) for k,v in paths.items()}
            if counts != {'hosts':len(selected_hosts), 'references':3582, 'candidates':35820}:
                raise ValueError(f'energy output incomplete: {counts}')
            write_json(root/'energy/evaluation_manifest.json', identity(status='completed',counts=counts,
                relaxation={'model':'eSEN-ODAC25-Full','optimizer':'ASE FIRE','max_steps':STEPS,
                            'convergence_max_force_eV_per_A':FMAX,'fixed_cell':True},
                evaluation={'model':'UMA-S-1.2','task':'odac','reference_geometry':'dataset, unrelaxed'},
                generation={'samples_per_record':10,'steering':EXPECTED_STEERING},
                inputs_manifest_sha256=inputs_sha,
                guest_bonds_sha256=guest_bonds_sha,
                author_radii_source_sha256='f01eae267b02b37280b5e209fa3c16eb638364cea0e37fad48ef653ad29ee9d3',
                notes=['Relaxed bare hosts are reconstructed surrogates.',
                       'Both author-source and ASE overlap radii flags are recorded.',
                       'Exempting directly bonded adsorbate atoms is a separate sensitivity analysis only.']))
            print('ENERGY_EVALUATION_COMPLETE',json.dumps(counts),flush=True)
        dist.barrier()
    except Exception as exc:
        failure = failure_dir/f'rank{rank:02d}.json'
        if not failure.exists():
            write_json(failure, {'rank':rank,'error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
