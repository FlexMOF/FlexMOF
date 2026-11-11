"""Measure source-atom and explicit-image attention with the same sample."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'conditional'))
from noise import PerStructureNoise
from expanded_attention import install
from flexmof.data import MOFDataset, collate_mofs
from flexmof.model.diffusion import DDPM
from flexmof.runtime import build_config
from flexmof.sampler import MOFSampler
import flexmof.sampler as sampler_module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def move(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if isinstance(value, dict):
        return {k: move(v) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-id', type=int, required=True)
    parser.add_argument('--candidate-id', type=int, choices=range(5), required=True)
    parser.add_argument('--mode', choices=('aggregated', 'explicit'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    protocol = json.loads((run / 'protocol.json').read_text())
    manifest = json.loads((run / 'csp_manifest.json').read_text())
    cohort = json.loads((HERE / 'cohort.json').read_text())
    matches = [row for row in cohort['samples'] if row['sample_id'] == args.sample_id]
    if len(matches) != 1 or protocol['mode'] != 'on' or sha(protocol['checkpoint']) != protocol['checkpoint_sha256']:
        raise ValueError('invalid sample, PBC mode, or checkpoint')
    sample = matches[0]
    entry = manifest[args.sample_id]
    if (entry['dataset_index'] != sample['dataset_index'] or
            str(entry['data_key']) != str(sample['data_key']) or
            entry['candidate_seeds'] != sample['candidate_seeds']):
        raise ValueError('cohort and run manifest differ')
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{args.mode}_{args.sample_id:05d}_{args.candidate_id}.json"
    if result_path.exists():
        raise FileExistsError('result already exists')
    if torch.cuda.device_count() != 1:
        raise ValueError('expose exactly one GPU per worker')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    overrides = protocol['overrides'].copy()
    overrides.update(coordinate_sampler='ddpm', num_timesteps=5000, num_timesteps_stepsize=-1)
    config = build_config(protocol['checkpoint'], protocol['data_root'], str(output),
                          overrides=overrides, seed=12347)
    dataset = MOFDataset(Path(protocol['data_root']) / 'bw_db', split='test', max_atoms=4096,
                         composition_config=config)
    if str(dataset.keys[entry['dataset_index']]) != str(entry['data_key']):
        raise ValueError('dataset key ordering changed')
    payload = dataset._transaction.get(str(entry['data_key']).encode())
    if not payload or hashlib.sha256(payload).hexdigest() != sample['input_sha256']:
        raise ValueError('input structure differs from the fixed cohort')
    item = dataset[entry['dataset_index']]
    sampler_module.tqdm = lambda iterable, **kwargs: iterable
    sampler = MOFSampler(config).cuda().eval().requires_grad_(False)
    if not isinstance(sampler.diffusion_process, DDPM) or len(sampler.diffusion_process.timesteps) != 5000:
        raise ValueError('expected 5000-step DDPM')
    state = install(sampler.net, backend='sdpa_ckpt') if args.mode == 'explicit' else None
    schedule = sampler.diffusion_process.timesteps.clone()
    seed = int(entry['candidate_seeds'][args.candidate_id])

    def sample_run(length):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sampler.diffusion_process.timesteps = schedule[:length]
        batch = move(collate_mofs([copy.deepcopy(item)]))
        batch['idx'] = torch.tensor([args.sample_id], device='cuda')
        reference = batch['pos'].clone()
        batch['pos'].zero_()
        batch['cell'].zero_()
        noise = PerStructureNoise(sampler, [dict(num_atoms=entry['num_atoms'], csp_seed=seed)])
        try:
            with torch.inference_mode():
                prediction = sampler.sample(batch)
            if noise.calls != length + 1:
                raise ValueError('noise stream length changed')
            return batch, reference, prediction
        finally:
            noise.remove()
            sampler.diffusion_process.timesteps = schedule

    warmup = sample_run(2)
    del warmup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    length = 2 if args.smoke else 5000
    started = time.perf_counter()
    batch, reference, prediction = sample_run(length)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    sampled = sampler.converter._convert(batch, prediction['pred_pos'],
                                         prediction.get('pred_mof_cell_atom_types'))[0]
    original = sampler.converter._convert(batch, reference)[0]
    if sampled is None or original is None:
        raise ValueError('structure conversion failed')
    matcher = StructureMatcher(ltol=.3, stol=.5, angle_tol=10, primitive_cell=True,
                               scale=False, attempt_supercell=True, allow_subset=False)
    match = matcher.get_rms_dist(AseAtomsAdaptor.get_structure(original),
                                 AseAtomsAdaptor.get_structure(sampled))
    result = dict(mode=args.mode, sample_id=args.sample_id, candidate_id=args.candidate_id,
                  size_bin=sample['size_bin'], seed=seed, smoke=args.smoke,
                  sampling_calls=length, checkpoint_sha256=protocol['checkpoint_sha256'],
                  match=match is not None, rmsd=float(match[0]) if match is not None else None,
                  elapsed_seconds=elapsed, peak_allocated_bytes=peak_allocated,
                  attention_calls=state['attention_calls'] if state else None,
                  gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='flexmof-efficiency-') as tmp:
        dist.init_process_group('gloo', init_method=f'file://{tmp}/init', rank=0, world_size=1)
        try:
            main()
        finally:
            dist.destroy_process_group()
