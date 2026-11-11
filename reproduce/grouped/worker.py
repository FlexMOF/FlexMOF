"""Conditional DDPM sampling with fixed seeds and evaluation records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback

FULL_COUNT = 30880
SELECTED_COUNT = 19792
MODE_RANKS = 32
CANDIDATES = 5
ACTIVE_CANDIDATES = (0, 1, 2, 3, 4)
VARIANT_DIR = Path(__file__).resolve().parent
MATCHER = dict(ltol=.3, stol=.5, angle_tol=10, primitive_cell=True,
               scale=False, attempt_supercell=True, allow_subset=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def choose_missing_smoke_pair(context):
    """Select only a missing candidate; never resample a committed record."""
    ordered = sorted(context['selected'], key=lambda i: (context['manifest'][i]['num_atoms'], i))
    records = context['run'] / 'csp' / 'records'
    pair = next(((i, c) for i in ordered for c in ACTIVE_CANDIDATES
                 if not (records / f'{i:05d}_{c}.json').exists()), None)
    require(pair is not None, 'No missing candidate is available for the smoke test')
    return pair


def load_contract(run_dir, attempt_id, *, verify_checkpoint=True):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', attempt_id),
            'Unsafe or missing attempt_id')
    run = Path(run_dir).resolve()
    protocol = load_json(run / 'protocol.json')
    manifest = load_json(run / 'csp_manifest.json')
    selected = load_json(run / 'selected_ids.json')
    assignments = load_json(VARIANT_DIR / 'assignments32.json')
    require(protocol['mode'] in ('on', 'off'), 'Unknown PBC mode')
    require(protocol['mode'] == run.name, 'Mode directory/protocol mismatch')
    require(len(manifest) == FULL_COUNT, 'Full manifest must retain 30,880 rows')
    require(isinstance(selected, list) and len(selected) == SELECTED_COUNT,
            'The evaluation denominator must be the fixed 19,792 structures')
    require(all(type(i) is int and 0 <= i < FULL_COUNT for i in selected), 'Invalid selected ID')
    require(len(set(selected)) == SELECTED_COUNT, 'Duplicate selected IDs')
    require(protocol['selected_count'] == SELECTED_COUNT, 'Protocol cohort count mismatch')
    require(protocol.get('candidates', CANDIDATES) == CANDIDATES, 'Five candidates are required')
    require(protocol['matcher_kwargs'] == MATCHER, 'Matcher protocol mismatch')
    require(len(assignments) == MODE_RANKS, 'Exactly 32 ranks are required for PBC on')
    flattened = [i for group in assignments for i in group]
    require(all(type(i) is int for i in flattened), 'Assignment IDs must be integers')
    require(len(flattened) == SELECTED_COUNT and sorted(flattened) == sorted(selected),
            'Assignments must cover the selected cohort exactly once')
    for i, entry in enumerate(manifest):
        require(entry['sample_id'] == i, 'Sample ID ordering changed')
        require(entry['candidate_seeds'] == [12347 + entry['dataset_index'] + 100003*j
                                            for j in range(CANDIDATES)],
                f'Candidate seed scheme changed at {i}')
    require(len({str(entry['data_key']) for entry in manifest}) == FULL_COUNT,
            'Duplicate data keys in full manifest')
    require(max(manifest[i]['num_atoms'] for i in selected) <= 4096,
            'Selected structure exceeds supported max_atoms=4096')
    identity = {
        'protocol_sha256': sha(run / 'protocol.json'),
        'manifest_sha256': sha(run / 'csp_manifest.json'),
        'selected_ids_sha256': sha(run / 'selected_ids.json'),
        'checkpoint_sha256': protocol['checkpoint_sha256'],
    }
    for field in ('manifest_sha256', 'selected_ids_sha256'):
        require(identity[field] == protocol[field], f'{field} mismatch')
    if verify_checkpoint:
        require(sha(protocol['checkpoint']) == identity['checkpoint_sha256'], 'Checkpoint hash mismatch')
    receipt = load_json(run / 'validated_resume.json')
    require(receipt.get('schema_version') == 1 and receipt.get('valid') is True,
            'A successful validate_resume receipt is required')
    require(receipt.get('attempt_id') == attempt_id, 'Resume receipt belongs to another attempt')
    require(receipt.get('mode') == protocol['mode'], 'Resume receipt PBC mode mismatch')
    for key, value in identity.items():
        require(receipt.get(key) == value, f'Resume receipt {key} mismatch')
    require(isinstance(receipt.get('records'), dict), 'Resume receipt lacks record index')
    selected_set = set(selected)
    for name, approval in receipt['records'].items():
        require(approval['sample_id'] in selected_set, 'Receipt contains an unselected sample')
        require(name == f"{approval['sample_id']:05d}_{approval['candidate_id']}.json",
                'Resume receipt filename/identity mismatch')
    return dict(run=run, attempt_id=attempt_id, protocol=protocol, manifest=manifest,
                selected=selected, assignments=assignments, identity=identity, resume=receipt,
                attempt=run / 'attempts' / attempt_id,
                global_attempt=run.parent / 'attempts' / attempt_id)


def failures(context):
    directory = context['global_attempt'] / 'failures'
    return sorted(p.name for p in directory.glob('*.json')) if directory.exists() else []


def check_failures(context):
    names = failures(context)
    require(not names, 'Another task process failed: ' + ', '.join(names))


def write_failure(run_dir, attempt_id, component, error):
    run = Path(run_dir).resolve()
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', attempt_id):
        atomic(run.parent / 'attempts' / attempt_id / 'failures' / f'{run.name}_{component}.json',
               dict(mode=run.name, attempt_id=attempt_id, component=component,
                    error=str(error), traceback=traceback.format_exc(), time=time.time()))


def validate_committed(context, entry, candidate, *, output=None, allow_imported=True):
    """Verify an imported approval or a record produced by this exact attempt."""
    i = entry['sample_id']
    output = Path(output) if output is not None else context['run'] / 'csp'
    name = f'{i:05d}_{candidate}.json'
    path = output / 'records' / name
    row = load_json(path)
    require(row.get('sample_id') == i and row.get('candidate_id') == candidate,
            f'Record identity mismatch: {name}')
    require(str(row.get('data_key')) == str(entry['data_key']) and
            row.get('seed') == entry['candidate_seeds'][candidate], f'Record key/seed mismatch: {name}')
    require(row.get('status') in ('complete', 'generation_error'), f'Invalid terminal status: {name}')
    approval = context['resume']['records'].get(name) if allow_imported else None
    if approval is not None:
        require(sha(path) == approval['record_sha256'], f'Imported record changed: {name}')
        for key in ('sample_id', 'candidate_id', 'seed', 'status'):
            require(approval[key] == row[key], f'Approval identity mismatch: {name}: {key}')
        require(str(approval['data_key']) == str(row['data_key']), f'Approval key mismatch: {name}')
        hashes = approval
    else:
        require(row.get('producer_attempt_id') == context['attempt_id'],
                f'Unvalidated record from another attempt: {name}')
        require(row.get('evaluation_identity') == context['identity'], f'Record protocol mismatch: {name}')
        hashes = row
    if row['status'] == 'complete':
        require(isinstance(row.get('metrics'), dict) and row['metrics'].get('match') in (0, 1),
                f'Invalid sampling metrics: {name}')
        for kind, field in (('original', 'original_sha256'), ('sampled', 'sampled_sha256')):
            cif = output / 'structures' / f'{kind}_{i}_{candidate}.cif'
            require(cif.is_file() and cif.stat().st_size > 0, f'Missing committed CIF: {cif}')
            require(sha(cif) == hashes[field], f'Committed CIF changed: {cif}')
    else:
        require(isinstance(row.get('error'), str) and row['error'].strip(),
                f'Generation failure lacks its original explanation: {name}')
    return row


def completed_counts(context, rank):
    counts = {'complete': 0, 'generation_error': 0}
    for i in context['assignments'][rank]:
        for candidate in ACTIVE_CANDIDATES:
            row = validate_committed(context, context['manifest'][i], candidate)
            counts[row['status']] += 1
    return counts


def batch_cap(context, lower, upper, baseline):
    """Adopt only a measured, identity-bound benchmark choice; otherwise use baseline."""
    path = context['attempt'] / 'benchmark' / 'batch_policy.json'
    if not path.is_file():
        return baseline
    policy = load_json(path)
    require(policy.get('attempt_id') == context['attempt_id'] and
            policy.get('evaluation_identity') == context['identity'], 'Benchmark policy identity mismatch')
    cap = policy.get('choices', {}).get(f'{lower}-{upper}', baseline)
    require(type(cap) is int and baseline <= cap <= 2 * baseline, 'Invalid measured batch cap')
    return cap


def run_sampling(args, context):
    from group_worker_sampling import run
    return run(args, context)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--rank', type=int, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-five', action='store_true')
    parser.add_argument('--local-gpu', type=int)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    require(0 <= args.rank < MODE_RANKS, 'Rank must be in [0, 31]')
    require(not args.pilot and not args.smoke_five,
            'Only the missing-candidate smoke or normal five-candidate run is supported')
    return args


def main():
    args = parse_args()
    component = ('pilot' if args.pilot else 'smoke' if args.smoke else 'rank') + str(args.rank)
    try:
        context = load_contract(args.run_dir, args.attempt_id)
        check_failures(context)
        if args.validate_only:
            print(json.dumps(dict(valid=True, rank=args.rank, assigned=len(context['assignments'][args.rank]))))
            return
        import torch.distributed as dist
        with tempfile.TemporaryDirectory() as temporary:
            dist.init_process_group('gloo', init_method=f'file://{temporary}/init', rank=0, world_size=1)
            try:
                run_sampling(args, context)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
    except Exception as error:
        write_failure(args.run_dir, args.attempt_id, component, error)
        raise


if __name__ == '__main__':
    main()
