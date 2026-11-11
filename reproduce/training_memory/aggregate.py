"""Summarize completed, failed, and missing paired training-memory trials."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads((HERE / 'cohort.json').read_text())['samples']
    results = args.run_dir / 'results'
    records = []
    for sample in samples:
        for batch in (4, 1):
            for model in ('flex', 'atommof'):
                path = results / f"train_{model}_key{sample['data_key']}_b{batch}.json"
                if not path.is_file():
                    records.append(dict(size_bin=sample['size_bin'], model=model,
                                        batch_size=batch, status='missing'))
                    continue
                result = json.loads(path.read_text())
                if (result['key'] != sample['data_key'] or result['batch_size'] != batch or
                        (result.get('sample') and result['sample']['data_key'] != sample['data_key'])):
                    raise ValueError(f'trial identity does not match cohort: {path.name}')
                if result['status'] == 'ok' and len(result.get('steps', [])) != 3:
                    raise ValueError(f'expected three measured updates: {path.name}')
                records.append(dict(size_bin=sample['size_bin'], model=model, batch_size=batch,
                                    status=result['status'],
                                    peak_gib=result.get('peak_allocated_bytes', 0) / 2**30,
                                    wall_seconds=median(x['wall_seconds'] for x in result['steps'])
                                    if result.get('steps') else None))
    summary = []
    for size_bin in ('1-96', '97-192', '193-384', '385-512', '513-768'):
        for batch in (4, 1):
            for model in ('flex', 'atommof'):
                group = [r for r in records if r['size_bin'] == size_bin and
                         r['batch_size'] == batch and r['model'] == model]
                ok = [r for r in group if r['status'] == 'ok']
                summary.append(dict(size_bin=size_bin, batch_size=batch, model=model,
                                    status_counts=dict(Counter(r['status'] for r in group)),
                                    median_peak_gib=median(r['peak_gib'] for r in ok) if ok else None,
                                    median_step_seconds=median(r['wall_seconds'] for r in ok) if ok else None))
    print(json.dumps({'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
