"""Prepare an ODAC25 adsorption evaluation with the released checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

CHECKPOINT_SHA256 = 'f37e315fd4545d4ee31a42f8050bcf3298594a38c5e89d33dd5a502ed47829e7'
DATA_SHA256 = {
    'odac25_test.lmdb': 'eac018779608b3fceb6ebd31a2c63e63fddfecc1fb82a56be6743c7711736c64',
    'odac25_test_metadata.pkl': 'd5f40dd368522f46728b85d705bf49c364180a3fc2bc06e9946a89c1ab41834c',
    'test_system_indices.pkl': 'bbd83de7f79d546898f53c2055b423220cd89d8f87c6baaddffb577f2895fdd9',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    run, data, checkpoint = (args.run_dir.resolve(), args.data_dir.resolve(),
                             args.checkpoint.resolve())
    if run.exists():
        parser.error('run-dir must not already exist')
    if not checkpoint.is_file() or sha(checkpoint) != CHECKPOINT_SHA256:
        parser.error('checkpoint does not match the released model')
    for name, expected in DATA_SHA256.items():
        if not (data / name).is_file() or sha(data / name) != expected:
            parser.error(f'missing or incorrect ODAC25 file: {name}')
    run.mkdir(parents=True)
    protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA256,
                    data_dir=str(data), data_sha256=DATA_SHA256,
                    coordinate_sampler='ddpm', num_timesteps=5000,
                    num_timesteps_stepsize=-1, candidates=10, seed_base=202611110)
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print('Prepared:', run)


if __name__ == '__main__':
    main()
