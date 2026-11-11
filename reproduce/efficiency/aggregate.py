"""Summarize paired periodic-attention quality, time and peak memory."""
import argparse
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.results_dir.resolve()
    cohort = json.loads((HERE / 'cohort.json').read_text())
    rows = {}
    checkpoint = None
    for sample in cohort['samples']:
        for candidate in range(5):
            for mode in ('aggregated', 'explicit'):
                path = directory / f"{mode}_{sample['sample_id']:05d}_{candidate}.json"
                row = json.loads(path.read_text())
                if (row['mode'] != mode or row['sample_id'] != sample['sample_id'] or
                        row['candidate_id'] != candidate or row['seed'] != sample['candidate_seeds'][candidate]
                        or row['size_bin'] != sample['size_bin'] or row['smoke']
                        or row['sampling_calls'] != 5000 or row['elapsed_seconds'] <= 0):
                    raise ValueError(f'invalid trial identity: {path}')
                checkpoint = checkpoint or row['checkpoint_sha256']
                if checkpoint != row['checkpoint_sha256']:
                    raise ValueError('trials use different checkpoints')
                rows[(sample['sample_id'], candidate, mode)] = row
    summaries = {}
    population = sum(group['population'] for group in cohort['groups'].values())
    weighted = {mode: {'mr1': 0.0, 'mr5': 0.0} for mode in ('aggregated', 'explicit')}
    for size_bin, group in cohort['groups'].items():
        samples = [s for s in cohort['samples'] if s['size_bin'] == size_bin]
        if len(samples) != group['selected']:
            raise ValueError('stratum selection changed')
        item = {'selected': len(samples), 'population': group['population']}
        for mode in ('aggregated', 'explicit'):
            first = [rows[(s['sample_id'], 0, mode)]['match'] for s in samples]
            any_five = [any(rows[(s['sample_id'], k, mode)]['match'] for k in range(5)) for s in samples]
            selected = [rows[(s['sample_id'], k, mode)] for s in samples for k in range(5)]
            mr1, mr5 = sum(first) / len(samples), sum(any_five) / len(samples)
            weighted[mode]['mr1'] += group['population'] / population * mr1
            weighted[mode]['mr5'] += group['population'] / population * mr5
            item[mode] = dict(mr1=mr1, mr5=mr5,
                              median_seconds=statistics.median(r['elapsed_seconds'] for r in selected),
                              median_peak_allocated_bytes=statistics.median(r['peak_allocated_bytes'] for r in selected))
        ratios = [rows[(s['sample_id'], k, 'explicit')]['elapsed_seconds'] /
                  rows[(s['sample_id'], k, 'aggregated')]['elapsed_seconds'] for s in samples for k in range(5)]
        item['median_paired_time_ratio'] = statistics.median(ratios)
        summaries[size_bin] = item
    result = dict(checkpoint_sha256=checkpoint, strata=summaries, weighted=weighted)
    output = directory / 'summary.json'
    if output.exists():
        raise FileExistsError('summary already exists')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
