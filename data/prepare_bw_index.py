"""Install the fixed index files into a BW database directory."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent / 'bw_index'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.digest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bw-db', type=Path, required=True)
    a = p.parse_args()
    dest = a.bw_db.resolve()
    if not (dest / 'MetalOxo.lmdb').is_file():
        p.error('the BW directory must contain MetalOxo.lmdb')
    for source in sorted(HERE.iterdir()):
        target = dest / source.name
        if target.exists():
            if not target.is_file() or sha(target) != sha(source):
                raise FileExistsError('Different file already exists: ' + str(target))
        else:
            shutil.copy2(source, target)
    print('BW index ready:', dest)

if __name__ == '__main__':
    main()
