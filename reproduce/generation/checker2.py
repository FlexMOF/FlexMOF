"""Apply the 15-rule MOFChecker 2.0 scorer."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal

from checker2_reference import CHECKS, check_rules, init, sha

ROOT = Path(os.environ['FLEXMOF_EVAL_ROOT'])
OUT = ROOT / 'denovo/checker2'


def one(task):
    i, cif, expected_sha = task
    row = dict(sample_id=i, cif=cif, valid=False, status='checker_error')
    def timeout(*_):
        raise TimeoutError('180-second per-structure timeout')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(180)
    try:
        from mofchecker import MOFChecker
        assert sha(cif) == expected_sha
        checker = MOFChecker.from_cif(cif)
        descriptors = {}
        for key in CHECKS:
            if key == 'has_high_charges':
                value = checker.checks['no_high_charges'].is_ok
                if value is None:
                    raise RuntimeError('Charge check unavailable')
                descriptors[key] = not bool(value)
            else:
                value = getattr(checker, key)
                if value is None:
                    raise RuntimeError(f'Check unavailable: {key}')
                descriptors[key] = bool(value)
        row.update(status='checked', checks=descriptors,
                   valid=all(descriptors[k] == v for k, v in CHECKS.items()),
                   cif_sha256=expected_sha)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        row['status'] = 'timeout' if isinstance(exc, TimeoutError) else 'checker_error'
    finally:
        signal.alarm(0)
    path = OUT / 'scores' / f'{i:05d}.json'
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(row) + '\n')
    tmp.replace(path)
    return row


def main():
    check_rules()
    init()
    (OUT / 'scores').mkdir(parents=True, exist_ok=True)
    originals = []
    tasks = []
    for i in range(10000):
        r = json.loads((ROOT / 'denovo/scores' / f'{i:05d}.json').read_text())
        assert r['sample_id'] == i
        originals.append(r)
        cached = OUT / 'scores' / f'{i:05d}.json'
        if cached.exists():
            continue
        if not r.get('cif_sha256'):
            row = dict(sample_id=i, cif='', valid=False, status='generation_error')
            cached.write_text(json.dumps(row) + '\n')
            continue
        tasks.append((i, r['cif'], r['cif_sha256']))
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('spawn'),
                             initializer=init) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 100 == 0 or count == len(tasks):
                print(json.dumps(dict(checked=count, remaining=len(tasks)-count)), flush=True)
    rows = [json.loads((OUT / 'scores' / f'{i:05d}.json').read_text())
            for i in range(10000)]
    counts = dict(exists=sum(r['exists'] for r in originals),
                  valid=sum(r['valid'] for r in rows),
                  novel=sum(r['novel'] for r in originals),
                  unique=sum(r['unique'] for r in originals),
                  evnu=sum(s['valid'] and r['exists'] and r['novel'] and r['unique']
                           for s, r in zip(rows, originals)))
    result = dict(complete=True, requested=10000, counts=counts,
                  percent_of_all={k: 100*v/10000 for k, v in counts.items()},
                  checker_status=dict(Counter(r['status'] for r in rows)),
                  checker='MOFChecker 2.0, 15 rules',
                  novelty='Complete CIF-derived MOFid versus 247062 training keys',
                  uniqueness='MOFid singleton in all 10000 attempts')
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
