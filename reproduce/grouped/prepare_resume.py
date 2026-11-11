"""Validate independently generated first candidates for grouped sampling."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--first-attempt', required=True)
    parser.add_argument('--grouped-attempt', required=True)
    args = parser.parse_args()
    if args.first_attempt == args.grouped_attempt:
        parser.error('the two attempt IDs must differ')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', args.grouped_attempt):
        parser.error('invalid grouped attempt ID')
    run = args.run_dir.resolve()
    spec = importlib.util.spec_from_file_location('first_candidate_worker', run/'worker.py')
    first = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(first)
    context = first.load_contract(run, args.first_attempt)
    first.require(context['protocol']['mode'] == 'on', 'Grouped sampling requires PBC on')
    first.check_failures(context)
    first.require(not (run/'attempts'/args.grouped_attempt).exists(), 'Grouped attempt already exists')
    approvals = {}
    for rank, assigned in enumerate(context['assignments']):
        done = first.load_json(context['attempt']/'done'/f'rank{rank}.json')
        first.require(done.get('rank') == rank and done.get('attempt_id') == args.first_attempt,
                      'First-candidate completion marker identity mismatch')
        first.require(done.get('evaluation_identity') == context['identity'] and
                      done.get('assignment_sha256') == first.ids_sha(assigned),
                      'First-candidate completion marker inputs mismatch')
        first.require(done.get('assigned') == len(assigned) and done.get('records') == len(assigned)
                      and done.get('candidate_count') == 1, 'First-candidate rank is incomplete')
        for sample_id in assigned:
            entry = context['manifest'][sample_id]
            row = first.validate_committed(context, entry, 0)
            name = f'{sample_id:05d}_0.json'
            approval = {key: row[key] for key in ('sample_id','candidate_id','seed','status','data_key')}
            approval['record_sha256'] = first.sha(run/'csp/records'/name)
            if row['status'] == 'complete':
                approval.update(original_sha256=row['original_sha256'], sampled_sha256=row['sampled_sha256'])
            approvals[name] = approval
    first.require(len(approvals) == len(context['selected']), 'First-candidate coverage mismatch')
    backup = run/'validated_resume_first.json'
    with backup.open('x', encoding='utf-8') as stream:
        json.dump(context['resume'], stream, indent=2)
        stream.write('\n')
    receipt = dict(schema_version=1, valid=True, attempt_id=args.grouped_attempt, mode='on',
                   **context['identity'], records=approvals)
    first.atomic(run/'validated_resume.json', receipt)
    print(json.dumps(dict(valid=True, approved=len(approvals), attempt_id=args.grouped_attempt)))


if __name__ == '__main__':
    main()
