"""Validate the complete ODAC25 candidate set before scoring."""
import argparse
import json
import os
from pathlib import Path
import torch
from prepare_run import CHECKPOINT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('checkpoint identity changed')
    directory = root / 'generation/records'
    expected = {f'{i:05d}.pt' for i in range(3582)}
    if {p.name for p in directory.glob('*.pt')} != expected:
        raise ValueError('generation records do not exactly cover the test split')
    success = failure = 0
    for index in range(3582):
        row = torch.load(directory / f'{index:05d}.pt', map_location='cpu', weights_only=True)
        if (row['dataset_index'] != index or row['checkpoint_sha256'] != CHECKPOINT_SHA256
                or row['sampling_calls'] != 5000 or row['steering'] != 'none'
                or len(row['samples']) != 10):
            raise ValueError(f'invalid generation record: {index}')
        for candidate, sample in enumerate(row['samples']):
            if sample['seed'] != protocol['seed_base'] + index * 10 + candidate:
                raise ValueError('candidate seed changed')
            success += int(sample['coords'] is not None)
            failure += int(sample['coords'] is None)
    output = root / 'generation/manifest.json'
    if output.exists():
        raise FileExistsError('generation manifest already exists')
    value = dict(status='completed', records=3582, samples_per_record=10,
                 valid_candidates=success, failed_candidates=failure,
                 checkpoint_sha256=CHECKPOINT_SHA256, sampling_calls=5000, steering='none')
    temporary = output.with_name(output.name + f'.tmp.{os.getpid()}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps(value))


if __name__ == '__main__':
    main()
