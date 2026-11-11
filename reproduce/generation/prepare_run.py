"""Prepare an isolated, fresh generation run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
EXPECTED = '554503150202a309d906078479eeeb9662dddb0af9c1fe5e0681eea27954ea23'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    a = p.parse_args()
    run, checkpoint, data_root = a.run_dir.resolve(), a.checkpoint.resolve(), a.data_root.resolve()
    if run.exists():
        p.error('run-dir must not exist')
    if not checkpoint.is_file() or sha(checkpoint) != EXPECTED:
        p.error('checkpoint is missing or does not match the published checksum')
    if not data_root.is_dir():
        p.error('data-root does not exist')
    (run / 'model').mkdir(parents=True)
    shutil.copy2(HERE / 'metadata.json', run / 'metadata.json')
    protocol = json.loads((HERE / 'protocol.json').read_text())
    protocol.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), data_root=str(data_root))
    (run / 'model/protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print('Prepared:', run)

if __name__ == '__main__':
    main()
