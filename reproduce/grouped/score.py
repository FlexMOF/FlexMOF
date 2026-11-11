"""Independently score the five saved CIF candidates."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import multiprocessing
from pathlib import Path
import signal
import time

from worker import (CANDIDATES, MODE_RANKS, SELECTED_COUNT, atomic, check_failures,
                    ids_sha, load_contract, load_json, require, sha,
                    validate_committed, write_failure)

_CONTEXT = None


def termination_signal(signum, _frame):
    # Let the active pool's BaseException cleanup terminate only our own workers.
    raise SystemExit(128 + signum)


def verify_done(context, rank):
    marker = context['attempt'] / 'done' / f'rank{rank}.json'
    if not marker.exists():
        return False
    row = load_json(marker)
    assigned = context['assignments'][rank]
    require(row.get('attempt_id') == context['attempt_id'] and row.get('rank') == rank,
            'Completion marker belongs to another attempt/rank')
    require(row.get('evaluation_identity') == context['identity'], 'Completion protocol mismatch')
    require(row.get('assignment_sha256') == ids_sha(assigned), 'Completion assignment mismatch')
    require(row.get('assigned') == len(assigned) and row.get('records') == len(assigned)*CANDIDATES,
            'Completion candidate count mismatch')
    return True


def wait_for_ranks(context, timeout, poll=10):
    deadline = time.monotonic() + timeout
    while True:
        check_failures(context)
        if all(verify_done(context, rank) for rank in range(MODE_RANKS)):
            return
        require(time.monotonic() < deadline, 'Timed out waiting for all 32 PBC-on ranks')
        time.sleep(min(poll, max(0, deadline-time.monotonic())))


def initialize_pool(context):
    global _CONTEXT
    _CONTEXT = context


def score_entry(entry):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    context = _CONTEXT
    check_failures(context)
    i = entry['sample_id']
    output = context['run'] / 'csp'
    matcher = StructureMatcher(**context['protocol']['matcher_kwargs'])
    candidates = []
    for candidate, seed in enumerate(entry['candidate_seeds']):
        record = validate_committed(context, entry, candidate)
        reuse = record.get('seq_forward_reuse')
        result = dict(candidate_id=candidate, seed=seed, match=False, status=record['status'],
                      record_sha256=sha(output / 'records' / f'{i:05d}_{candidate}.json'),
                      producer_attempt_id=record.get('producer_attempt_id'),
                      seq_reuse_applied=bool(reuse.get('applied')) if isinstance(reuse, dict) else False,
                      seq_reuse_anchor_candidate_id=reuse.get('anchor_candidate_id') if isinstance(reuse, dict) else None)
        if record['status'] == 'complete':
            original = output / 'structures' / f'original_{i}_{candidate}.cif'
            sampled = output / 'structures' / f'sampled_{i}_{candidate}.cif'
            result.update(ground_sha256=sha(original), prediction_sha256=sha(sampled))
            try:
                reference = Structure.from_file(str(original))
                prediction = Structure.from_file(str(sampled))
            except Exception as error:
                result.update(status='cif_parse_error', error=f'{type(error).__name__}: {error}')
            else:
                try:
                    rms = matcher.get_rms_dist(reference, prediction)
                    if rms is not None:
                        require(math.isfinite(float(rms[0])), 'Matcher returned a nonfinite RMSD')
                    result.update(match=rms is not None, rmsd=float(rms[0]) if rms is not None else None)
                except Exception as error:
                    result.update(match=False, rmsd=None, status='matcher_error',
                                  error=f'{type(error).__name__}: {error}')
        candidates.append(result)
    row = dict(sample_id=i, data_key=str(entry['data_key']), candidates=candidates,
               attempt_id=context['attempt_id'], evaluation_identity=context['identity'],
               matcher_kwargs=context['protocol']['matcher_kwargs'], pymatgen_version='2026.5.4')
    atomic(context['attempt'] / 'scores' / f'{i:05d}.json', row)
    return row


def aggregate(context, rows):
    require(len(rows) == SELECTED_COUNT, 'Incomplete fixed-denominator cohort')
    require({row['sample_id'] for row in rows} == set(context['selected']), 'Score cohort IDs differ')
    require(len({row['data_key'] for row in rows}) == SELECTED_COUNT, 'Duplicate score keys')
    for row in rows:
        entry = context['manifest'][row['sample_id']]
        require(row['data_key'] == str(entry['data_key']), 'Scored key differs from manifest')
        require([c['candidate_id'] for c in row['candidates']] == list(range(CANDIDATES)),
                'Missing candidate positions in score')
        require([c['seed'] for c in row['candidates']] == entry['candidate_seeds'], 'Scored seeds changed')
        require(row['evaluation_identity'] == context['identity'] and
                row['attempt_id'] == context['attempt_id'], 'Stale score identity')
    hits = {k: sum(any(c['match'] for c in row['candidates'][:k]) for row in rows)
            for k in range(1, CANDIDATES+1)}
    errors = {status: sum(c['status'] == status for row in rows for c in row['candidates'])
              for status in ('generation_error', 'cif_parse_error', 'matcher_error')}
    producers = Counter(str(c.get('producer_attempt_id') or 'unknown') for row in rows for c in row['candidates'])
    reuse_counts = Counter('applied' if c.get('seq_reuse_applied') else 'not_applied'
                           for row in rows for c in row['candidates'])
    return dict(complete=True, mode=context['protocol']['mode'], attempt_id=context['attempt_id'],
                structures=SELECTED_COUNT, candidates=CANDIDATES*SELECTED_COUNT,
                hits_at_k=hits, mr_at_k_percent={k: 100*value/SELECTED_COUNT for k, value in hits.items()},
                errors=errors, producer_attempt_counts=dict(producers),
                seq_forward_reuse_counts=dict(reuse_counts),
                mixed_sampling_methods=len(reuse_counts)>1,
                **context['identity'], sampler='DDPM 5000 adjacent steps',
                matcher_kwargs=context['protocol']['matcher_kwargs'], pymatgen_version='2026.5.4',
                denominator='all requested test structures; failed candidates are nonmatches')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--wait-timeout', type=int, default=604800)
    parser.add_argument('--processes', type=int, default=16)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, termination_signal)
    signal.signal(signal.SIGINT, termination_signal)
    try:
        require(args.wait_timeout > 0 and 1 <= args.processes <= 64, 'Invalid scoring limits')
        context = load_contract(args.run_dir, args.attempt_id)
        require(importlib.metadata.version('pymatgen') == '2026.5.4', 'pymatgen version changed')
        wait_for_ranks(context, args.wait_timeout)
        (context['attempt'] / 'scores').mkdir(parents=True, exist_ok=False)  # Never reuse an older score cache.
        entries = [context['manifest'][i] for i in context['selected']]
        deadline = time.monotonic() + args.wait_timeout
        rows = []
        pool = multiprocessing.Pool(args.processes, initializer=initialize_pool, initargs=(context,))
        try:
            results = pool.imap_unordered(score_entry, entries, chunksize=1)
            while len(rows) < SELECTED_COUNT:
                check_failures(context)
                require(time.monotonic() < deadline, 'Scoring timed out')
                try:
                    rows.append(results.next(timeout=min(10, max(.01, deadline-time.monotonic()))))
                except multiprocessing.TimeoutError:
                    continue
                if len(rows) % 100 == 0:
                    atomic(context['attempt'] / 'score_progress.json',
                           dict(scored=len(rows), total=SELECTED_COUNT, attempt_id=args.attempt_id))
            pool.close()
            pool.join()
        except BaseException:
            pool.terminate()  # Only the CPU scoring processes created by this process.
            pool.join()
            raise
        result = aggregate(context, rows)
        atomic(context['attempt'] / 'joint_results.json', result)
        atomic(context['run'] / 'csp' / f'joint_results.{args.attempt_id}.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        write_failure(args.run_dir, args.attempt_id, 'score', error)
        raise


if __name__ == '__main__':
    main()
