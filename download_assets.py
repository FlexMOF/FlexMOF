"""Download and verify the released model and evaluation assets."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = 'https://github.com/FlexMOF/FlexMOF/releases/download/v1.0/'
ASSETS = {
    'structure': ('checkpoints/structure_ddpm_e370_inference.pt', '351404ef67ab445591c5639f550176ca14caf565272ddda40decf1a03b4f35e6'),
    'generation': ('checkpoints/joint_generation_e226_inference.pt', '554503150202a309d906078479eeeb9662dddb0af9c1fe5e0681eea27954ea23'),
    'ablation': ('checkpoints/structure_ddpm_pbc_off_inference.pt', 'a43c45d92bd5b83533ff6c2f821796a30509101473f708124f6e4a4db3fde800'),
    'adsorption': ('checkpoints/adsorption_inference.pt', 'f37e315fd4545d4ee31a42f8050bcf3298594a38c5e89d33dd5a502ed47829e7'),
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(name):
    relative, expected = ASSETS[name]
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f'Existing asset has an incorrect SHA256: {path}')
        print(f'{name}: verified {path}')
        return
    partial = path.with_name(path.name + '.download')
    if partial.exists():
        raise FileExistsError(f'Remove an interrupted download before retrying: {partial}')
    try:
        request = urllib.request.Request(BASE + path.name, headers={'User-Agent': 'FlexMOF-reproduction'})
        with urllib.request.urlopen(request) as response, partial.open('xb') as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
        if digest(partial) != expected:
            raise ValueError(f'Download SHA256 mismatch: {path.name}')
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f'{name}: downloaded and verified {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', nargs='+', choices=[*ASSETS, 'all'])
    args = parser.parse_args()
    names = list(ASSETS) if 'all' in args.assets else list(dict.fromkeys(args.assets))
    for name in names:
        download(name)


if __name__ == '__main__':
    main()
