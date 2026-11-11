"""Prepare an isolated, fresh conditional DDPM sampling run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPECTED = {
    'on': '351404ef67ab445591c5639f550176ca14caf565272ddda40decf1a03b4f35e6',
    'off': 'a43c45d92bd5b83533ff6c2f821796a30509101473f708124f6e4a4db3fde800',
}

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
    p.add_argument('--attempt-id', default='review')
    p.add_argument('--mode', choices=('on', 'off'), default='on')
    a = p.parse_args()
    run, checkpoint, data_root = a.run_dir.resolve(), a.checkpoint.resolve(), a.data_root.resolve()
    if run.name != a.mode or run.exists():
        p.error('run-dir must be a new directory named on or off, matching --mode')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', a.attempt_id):
        p.error('invalid attempt ID')
    if not checkpoint.is_file() or sha(checkpoint) != EXPECTED[a.mode]:
        p.error('checkpoint is missing or does not match the published checksum')
    bw = data_root / 'bw_db'
    for name in ('MetalOxo.lmdb', 'index_to_key_map.pt', 'sizes.pt', 'test_split_filtered.txt'):
        if not (bw / name).is_file():
            p.error('missing data file: ' + str(bw / name))
    run.mkdir(parents=True)
    for name in ('worker.py', 'noise.py', 'score.py', 'csp_manifest.json', 'selected_ids.json', 'assignments.json'):
        shutil.copy2(HERE / name, run / name)
    shutil.copytree(REPO / 'src', run / 'source/src')
    protocol = json.loads((HERE / 'protocol.json').read_text())
    protocol.update(mode=a.mode, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), data_root=str(data_root))
    protocol['overrides']['mof_disable_pbc'] = a.mode == 'off'
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    receipt = {'schema_version': 1, 'valid': True, 'attempt_id': a.attempt_id, 'mode': a.mode,
               'protocol_sha256': sha(run / 'protocol.json'), 'manifest_sha256': sha(run / 'csp_manifest.json'),
               'selected_ids_sha256': sha(run / 'selected_ids.json'), 'checkpoint_sha256': sha(checkpoint),
               'records': {}}
    (run / 'validated_resume.json').write_text(json.dumps(receipt, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('conditional_worker', run / 'worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_contract(run, a.attempt_id)
    print('Prepared:', run)

if __name__ == '__main__':
    main()
