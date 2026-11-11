"""Calculate system-macro coverage of ODAC25 adsorption candidates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from prepare_run import CHECKPOINT_SHA256, DATA_SHA256, sha


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def structure(numbers, coordinates, lattice):
    numbers = np.asarray(numbers, dtype=int)
    coordinates, lattice = np.asarray(coordinates, dtype=float), np.asarray(lattice, dtype=float)
    if (coordinates.shape != (len(numbers), 3) or lattice.shape != (3, 3)
            or not np.isfinite(coordinates).all() or not np.isfinite(lattice).all()
            or abs(np.linalg.det(lattice)) < 1e-6):
        raise ValueError('invalid structure geometry')
    return AseAtomsAdaptor.get_structure(Atoms(numbers=numbers, positions=coordinates,
                                               cell=lattice, pbc=True))


def score_system(task):
    root_string, system_index, system_id, indices = task
    root = Path(root_string)
    paths = [root / 'generation/records' / f'{i:05d}.pt' for i in indices]
    hashes = [sha(path) for path in paths]
    destination = root / 'coverage/systems' / f'{system_index:04d}.json'
    if destination.exists():
        previous = json.loads(destination.read_text())
        if (previous['indices'] != indices or previous['checkpoint_sha256'] != CHECKPOINT_SHA256
                or previous['record_sha256'] != hashes):
            raise ValueError('existing coverage result no longer matches input records')
        return previous
    references, predictions, first, failures = [], [], [], []
    for index, path in zip(indices, paths):
        record = torch.load(path, map_location='cpu', weights_only=True)
        if (record['dataset_index'] != index or record['checkpoint_sha256'] != CHECKPOINT_SHA256
                or record['sampling_calls'] != 5000 or record['steering'] != 'none'
                or len(record['samples']) != 10):
            raise ValueError(f'invalid candidate record: {index}')
        ref = record['reference']
        numbers = ref['atom_types'].numpy()
        references.append(structure(numbers, ref['true_coords'].numpy(), ref['true_lattice'].numpy()))
        for candidate, sample in enumerate(record['samples']):
            if sample['seed'] != 202611110 + index * 10 + candidate:
                raise ValueError('candidate seed changed')
            if sample['coords'] is None:
                failures.append(dict(dataset_index=index, candidate=candidate,
                                     reason=sample.get('error_type', 'missing')))
                continue
            try:
                prediction = structure(numbers, sample['coords'].numpy(), sample['lattice'].numpy())
                if candidate == 0:
                    first.append(len(predictions))
                predictions.append(prediction)
            except (ValueError, TypeError) as exc:
                failures.append(dict(dataset_index=index, candidate=candidate,
                                     reason=type(exc).__name__))
    matcher = StructureMatcher(stol=.5, ltol=.3, angle_tol=10)
    matrix = np.full((len(predictions), len(references)), np.inf)
    errors = 0
    for i, prediction in enumerate(predictions):
        for j, reference in enumerate(references):
            try:
                match = matcher.get_rms_dist(prediction, reference)
                if match is not None:
                    matrix[i, j] = float(match[0])
            except Exception:
                errors += 1
    covered10 = np.isfinite(matrix).any(axis=0) if predictions else np.zeros(len(references), dtype=bool)
    covered1 = np.isfinite(matrix[first, :]).any(axis=0) if first else np.zeros(len(references), dtype=bool)
    result = dict(system_index=system_index, system_id=str(system_id), indices=indices,
                  checkpoint_sha256=CHECKPOINT_SHA256, record_sha256=hashes,
                  references=len(references), valid_predictions=len(predictions),
                  failed_candidates=len(failures), matching_errors=errors,
                  covered_k1=int(covered1.sum()), covered_k10=int(covered10.sum()),
                  cov_k1=float(covered1.mean()), cov_k10=float(covered10.mean()), failures=failures)
    atomic_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    root = args.run_dir.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    system_path = Path(protocol['data_dir']) / 'test_system_indices.pkl'
    if (protocol['checkpoint_sha256'] != CHECKPOINT_SHA256 or
            sha(system_path) != DATA_SHA256[system_path.name]):
        raise ValueError('checkpoint or system index changed')
    with system_path.open('rb') as stream:
        systems = pickle.load(stream)
    flattened = [int(i) for indices in systems.values() for i in indices]
    if len(systems) != 683 or sorted(flattened) != list(range(3582)):
        raise ValueError('system indices do not cover the test split exactly once')
    missing = [i for i in range(3582) if not (root / 'generation/records' / f'{i:05d}.pt').is_file()]
    if missing:
        raise ValueError(f'generation is incomplete: {len(missing)} records missing')
    tasks = [(str(root), j, key, [int(i) for i in values])
             for j, (key, values) in enumerate(systems.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(score_system, tasks))
    if len(rows) != 683 or sum(row['references'] for row in rows) != 3582:
        raise ValueError('coverage denominator changed')
    result = dict(complete=True, checkpoint_sha256=CHECKPOINT_SHA256,
                  systems=len(rows), references=sum(row['references'] for row in rows),
                  valid_predictions=sum(row['valid_predictions'] for row in rows),
                  failed_candidates=sum(row['failed_candidates'] for row in rows),
                  matching_errors=sum(row['matching_errors'] for row in rows),
                  cov_macro_k1=sum(row['cov_k1'] for row in rows) / len(rows),
                  cov_macro_k10=sum(row['cov_k10'] for row in rows) / len(rows),
                  cov_micro_k1=sum(row['covered_k1'] for row in rows) / 3582,
                  cov_micro_k10=sum(row['covered_k10'] for row in rows) / 3582,
                  matcher=dict(stol=.5, ltol=.3, angle_tol=10))
    atomic_json(root / 'coverage/summary.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
