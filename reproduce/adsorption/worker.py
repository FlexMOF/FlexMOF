"""Generate ODAC25 adsorption candidates with the released DDPM model."""
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'atommof_compat'))

import numpy as np
import torch
import torch.distributed as dist

from evaluation_dataset import EvaluationJointOdacDataset
from odac_multigraph import collate_odac
from prepare_run import CHECKPOINT_SHA256, DATA_SHA256, sha
from flexmof.config import config_from_checkpoint
from flexmof.model.diffusion import DDPM
from flexmof.sampler import MOFSampler
import flexmof.sampler as sampler_module


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def clean_batch(item, device):
    batch = {k: v.to(device).clone() if torch.is_tensor(v) else v
             for k, v in collate_odac([item]).items()}
    batch['pos'].zero_()
    batch['cell'].zero_()
    if torch.count_nonzero(batch['target_cell_index']).item():
        raise ValueError('unexpected reference-derived periodic offsets')
    return batch


def sample_one(sampler, item, seed, device):
    batch = clean_batch(item, device)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    sampler.diffusion_noise.torch_generator = torch.Generator(device=device).manual_seed(seed)
    started = time.monotonic()
    with torch.inference_mode():
        output = sampler.sample(batch)
    n = int(item['num_atoms'])
    frame = output['pred_pos'][0, :n + 8].detach().cpu().float()
    frame = frame - frame[-8]
    cell = torch.stack((frame[-4], frame[-6], frame[-7]))
    if not torch.isfinite(frame).all() or not torch.isfinite(cell).all():
        raise ValueError('nonfinite generated coordinates or cell')
    if abs(float(torch.linalg.det(cell))) < 1e-6:
        raise ValueError('degenerate generated cell')
    return dict(coords=frame[:n].clone(), lattice=cell.clone(), seed=seed,
                seconds=time.monotonic() - started)


def raw_reference(dataset, index):
    record = pickle.loads(dataset._txn().get(dataset.keys[index]))
    atoms = torch.as_tensor(np.asarray(record.atomic_numbers), dtype=torch.long).clone()
    frac = torch.as_tensor(np.asarray(record.frac_coords), dtype=torch.float64)
    cell = torch.as_tensor(np.asarray(record.cell), dtype=torch.float64).reshape(3, 3)
    roles = {'NODE': 0, 'LINKER': 1, 'SOLVENT': 2, 'ADSORBATE': 3}
    block_ids = []
    for block in record.blocks:
        name = getattr(block.block_type, 'name', str(block.block_type))
        block_ids.extend([roles[name]] * len(block.atomic_numbers))
    if len(block_ids) != len(atoms) or frac.shape != (len(atoms), 3):
        raise ValueError('invalid reference structure')
    key = dataset.keys[index]
    return dict(atom_types=atoms, atom_block_type=torch.tensor(block_ids, dtype=torch.long),
                true_coords=(frac @ cell).float(), true_lattice=cell.float(),
                key=key.decode() if isinstance(key, bytes) else str(key))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--full-pilot', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if (not 0 <= args.rank < args.world or args.limit < 0 or (args.limit and not args.smoke)
            or (args.full_pilot and not args.smoke)):
        parser.error('invalid rank, world, or smoke limit')
    run = args.run_dir.resolve()
    protocol = json.loads((run / 'protocol.json').read_text())
    if (protocol['checkpoint_sha256'] != CHECKPOINT_SHA256 or
            protocol['data_sha256'] != DATA_SHA256 or
            sha(protocol['checkpoint']) != CHECKPOINT_SHA256):
        raise ValueError('checkpoint or data protocol changed')
    if (protocol['coordinate_sampler'], protocol['num_timesteps'],
            protocol['num_timesteps_stepsize'], protocol['candidates']) != ('ddpm', 5000, -1, 10):
        raise ValueError('sampling protocol changed')
    data = Path(protocol['data_dir'])
    for name, expected in DATA_SHA256.items():
        if not (data / name).is_file() or sha(data / name) != expected:
            raise ValueError(f'missing or incorrect ODAC25 file: {name}')
    out = run / ('pilot' if args.full_pilot else 'smoke' if args.smoke else 'generation')
    (out / 'records').mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1:
        raise ValueError('expose exactly one GPU per worker')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    overrides = dict(coordinate_sampler='ddpm', num_timesteps=5000,
                     num_timesteps_stepsize=-100 if args.smoke and not args.full_pilot else -1,
                     num_recycles=0)
    config = config_from_checkpoint(source['args'], protocol['checkpoint'], str(data),
                                    str(out / 'structures'), 'odac25', 1, -1,
                                    protocol['seed_base'], overrides=overrides)
    del source
    sampler_module.tqdm = lambda iterable, **kwargs: iterable
    sampler = MOFSampler(config).cuda().eval().requires_grad_(False)
    steps = 50 if args.smoke and not args.full_pilot else 5000
    if not isinstance(sampler.diffusion_process, DDPM) or len(sampler.diffusion_process.timesteps) != steps:
        raise ValueError('incorrect coordinate sampler')
    dataset = EvaluationJointOdacDataset(data / 'odac25_test.lmdb', data / 'odac25_test_metadata.pkl',
                                         max_atoms=1000, composition_config=config)
    if len(dataset) != 3582 or dataset.source_length != 3582:
        raise ValueError('unexpected ODAC25 test split')
    indices = list(range(args.rank, len(dataset), args.world))
    if args.limit:
        indices = indices[:args.limit]
    count = 1 if args.smoke else protocol['candidates']
    generated = 0
    for index in indices:
        path = out / 'records' / f'{index:05d}.pt'
        if path.exists():
            previous = torch.load(path, map_location='cpu', weights_only=True)
            if (previous['dataset_index'] != index or previous['checkpoint_sha256'] != CHECKPOINT_SHA256
                    or len(previous['samples']) != count or previous['sampling_calls'] != steps):
                raise ValueError('existing record does not match this run')
            continue
        reference = raw_reference(dataset, index)
        try:
            item = dataset[index]
            if not torch.equal(item['atomic_numbers'], reference['atom_types']):
                raise ValueError('conditioning atom order differs from reference')
            samples = []
            for candidate in range(count):
                seed = protocol['seed_base'] + index * protocol['candidates'] + candidate
                try:
                    samples.append(sample_one(sampler, item, seed, 'cuda:0'))
                except (torch.cuda.OutOfMemoryError, FloatingPointError, ValueError) as exc:
                    torch.cuda.empty_cache()
                    samples.append(dict(coords=None, lattice=None, seed=seed,
                                        error_type=type(exc).__name__, error=str(exc)[:600]))
        except (ValueError, KeyError) as exc:
            samples = [dict(coords=None, lattice=None,
                            seed=protocol['seed_base'] + index * protocol['candidates'] + candidate,
                            error_type=type(exc).__name__, error=str(exc)[:600])
                       for candidate in range(count)]
        atomic_save(path, dict(dataset_index=index, checkpoint_sha256=CHECKPOINT_SHA256,
                               reference=reference, samples=samples, sampling_calls=steps,
                               steering='none'))
        generated += 1
        if args.smoke and not any(s['coords'] is not None for s in samples):
            raise RuntimeError('smoke did not generate a structure')
        if generated % 10 == 0 or args.smoke:
            print(json.dumps(dict(rank=args.rank, generated=generated, sample_id=index)), flush=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='flexmof-adsorption-') as tmp:
        dist.init_process_group('gloo', init_method=f'file://{tmp}/init', rank=0, world_size=1)
        try:
            main()
        finally:
            dist.destroy_process_group()
