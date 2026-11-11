"""Unconditional generation with a fixed per-sample RNG protocol."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

from flexmof.model.diffusion import DDPM
from flexmof.random_streams import auxiliary_noise_seed
from flexmof.runtime import build_config
from flexmof.sampler import MOFSampler
from flexmof.structures import MOFStructureEvaluator
from flexmof.two_stage import generate_unconditional


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def seed(value):
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--full-pilot', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.rank < args.world or args.limit < 0:
        raise ValueError('rank must be within world')
    if args.full_pilot and not args.smoke:
        raise ValueError('full pilot must write only to the smoke directory')
    protocol = json.loads((args.root / 'protocol.json').read_text())
    entries = json.loads((args.root.parent / 'metadata.json').read_text())
    assert len(entries) == protocol['metadata_count'] == 10000
    phase = 'denovo_pilot5000' if args.full_pilot else ('denovo_smoke' if args.smoke else 'denovo')
    out = args.root / phase
    for name in ['records', 'structures', 'coordinates', 'metrics']:
        (out / name).mkdir(parents=True, exist_ok=True)
    overrides = protocol['sampling_overrides'].copy()
    if args.smoke and not args.full_pilot:
        overrides['num_timesteps_stepsize'] = -100
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = build_config(protocol['checkpoint'], protocol['data_root'], str(out / 'structures'),
                          micro_batch_size=1, seed=12347, num_samples=1, overrides=overrides)
    config.metric_path = str(out / 'metrics' / f'rank{args.rank}')
    assert config.coordinate_sampler == 'ddpm' and config.num_timesteps == 5000
    assert config.num_timesteps_stepsize == (-100 if args.smoke and not args.full_pilot else -1)
    assert config.use_per_atom_outcell_index and config.pbc_cutoff == 40.0
    assert not config.mof_disable_pbc and config.num_recycles == 0
    sampler = MOFSampler(config).to('cuda:0').eval()
    sampler.requires_grad_(False)
    assert isinstance(sampler.diffusion_process, DDPM)
    assert len(sampler.diffusion_process.timesteps) == (50 if args.smoke and not args.full_pilot else 5000)
    assert sampler.net.joint_composition and sampler.net.mof_joint.latent_normalization.ready
    import flexmof.sampler as sampler_module
    sampler_module.tqdm = lambda it, **kwargs: it
    # Unconditional samples must not read a source LMDB or cached training latents.
    import sys
    def forbid_source_reads(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            name = str(arguments[0])
            if name.endswith('.lmdb') or Path(name).name == 'latents.pt':
                raise RuntimeError('de novo sampling attempted to read source data')
    sys.addaudithook(forbid_source_reads)
    assigned = list(range(args.rank, len(entries), args.world))
    if args.limit:
        assigned = assigned[:args.limit]
    completed = generated = 0
    for i in assigned:
        path = out / 'records' / f'{i:05d}.json'
        prior = json.loads(path.read_text()) if path.exists() else None
        if prior and prior.get('status') in {'generated', 'generation_error'}:
            assert prior['sample_id'] == i
            assert prior['metadata'] == entries[i]
            assert prior['composition_seed'] == protocol['composition_seed_base'] + i
            assert prior['csp_seed'] == protocol['csp_denovo_seed_base'] + i
            if prior['status'] == 'generated':
                assert Path(prior['cif']) == out / 'structures' / f'sampled_{i:05d}.cif'
                assert Path(prior['cif']).is_file() and (out / 'coordinates' / f'{i:05d}.pt').is_file()
            completed += 1
            continue
        if prior and prior.get('status') != 'running':
            raise RuntimeError(f'Unexpected existing record: {path}')
        started = time.monotonic()
        row = dict(sample_id=i, rank=args.rank, status='running', metadata=entries[i],
                   composition_seed=protocol['composition_seed_base'] + i,
                   csp_seed=protocol['csp_denovo_seed_base'] + i,
                   started_at=time.time(), resumed_interrupted=bool(prior))
        atomic(path, row)
        seed(row['csp_seed'])
        sampler.diffusion_noise.torch_generator = torch.Generator(device='cuda:0').manual_seed(
            auxiliary_noise_seed(row['csp_seed'], 'csp_diffusion', protocol['noise_stream_version']))
        try:
            with torch.inference_mode():
                result = generate_unconditional(sampler, [entries[i]], start_index=i,
                    steps=protocol['composition_steps'], churn=protocol['composition_churn'],
                    seed=row['composition_seed'])
            if not torch.isfinite(result['pred_pos']).all():
                raise FloatingPointError('Nonfinite generated coordinates')
            structure = MOFStructureEvaluator._convert(result['stage_two_batch'], result['pred_pos'])[0]
            if structure is None or len(structure) != int(result['stage_two_batch']['num_atoms'][0]):
                raise ValueError('CIF conversion changed atom count')
            cif = out / 'structures' / f'sampled_{i:05d}.cif'
            cif_tmp = cif.with_name(cif.name + f'.{os.getpid()}.tmp')
            structure.write(cif_tmp, format='cif')
            os.replace(cif_tmp, cif)
            coord = out / 'coordinates' / f'{i:05d}.pt'
            coord_tmp = coord.with_name(coord.name + f'.{os.getpid()}.tmp')
            torch.save(dict(positions=result['pred_pos'].cpu(), blocks=result['blocks'][0],
                            metadata=entries[i]), coord_tmp)
            os.replace(coord_tmp, coord)
            row.update(status='generated', cif=str(cif), atoms=len(structure),
                       formula=structure.get_chemical_formula())
            generated += 1
            del result
        except (torch.cuda.OutOfMemoryError, FloatingPointError, ValueError) as exc:
            row.update(status='generation_error', cif='', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc())
            torch.cuda.empty_cache()
        row.update(seconds=time.monotonic() - started, finished_at=time.time())
        atomic(path, row)
        completed += 1
        if completed % 10 == 0 or args.smoke:
            print(json.dumps(dict(rank=args.rank, completed=completed, generated=generated,
                                  sample_id=i, seconds=row['seconds'], status=row['status'])), flush=True)
            gc.collect()
    if args.smoke and generated == 0 and not any(
            json.loads((out / 'records' / f'{i:05d}.json').read_text()).get('status') == 'generated'
            for i in assigned):
        raise RuntimeError('Smoke produced no structures')
    atomic(out / f'worker{args.rank}.done.json', dict(completed=completed, generated_this_run=generated,
                                                       assigned=len(assigned), smoke=args.smoke))


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='flexmof-denovo-') as tmp:
        dist.init_process_group('gloo', init_method=f'file://{tmp}/init', rank=0, world_size=1)
        try:
            main()
        finally:
            dist.destroy_process_group()
