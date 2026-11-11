"""Checkpoint-derived MOF sampling runtime.

The model configuration is read from the checkpoint, then only sampling and
I/O settings are overridden. This keeps old PSM checkpoints compatible while
removing the old launcher's hundreds of user-facing arguments.
"""

from __future__ import annotations

import gc
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from flexmof.checkpoint import load_checkpoint
from flexmof.config import MOFConfig, config_from_checkpoint
from flexmof.data import DistributedMOFBatchSampler, MOFDataset, collate_mofs
from flexmof.logging import log_info, log_metrics
from flexmof.performance import DistributedThroughputMeter
from flexmof.sampler import MOFSampler


def build_config(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    dataset_subdir: str = "bw_db",
    micro_batch_size: int = 4,
    max_batches: int = -1,
    seed: int = 12347,
    overrides: dict | None = None,
    num_samples: int = 1,
) -> MOFConfig:
    checkpoint = load_checkpoint(checkpoint_path)
    if "model" not in checkpoint or "args" not in checkpoint:
        raise ValueError("Expected a PSM checkpoint containing 'model' and 'args'.")
    checkpoint_args = dict(checkpoint["args"])
    del checkpoint
    gc.collect()

    return config_from_checkpoint(
        checkpoint_args,
        checkpoint_path,
        data_root,
        output_dir,
        dataset_subdir,
        micro_batch_size,
        max_batches,
        seed,
        overrides,
        num_samples,
    )


def _initialize_process(config: MOFConfig) -> torch.device:
    config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.rank = int(os.environ.get("RANK", 0))
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(config.local_rank)
    if not torch.distributed.is_initialized():
        timeout = timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MINUTES", 3000)))
        torch.distributed.init_process_group("nccl", timeout=timeout)
    return torch.device("cuda", config.local_rank)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _match_metric_totals(metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pack matched count, sample count, finite RMSD sum, and RMSD count."""
    matches = metrics["match"].detach().float().reshape(-1)
    rmsds = metrics["rmsd"].detach().float().reshape(-1)
    finite_rmsds = rmsds[matches.bool() & torch.isfinite(rmsds)]
    return torch.stack(
        (
            matches.sum(),
            matches.new_tensor(matches.numel()),
            finite_rmsds.sum(),
            matches.new_tensor(finite_rmsds.numel()),
        )
    )


def _match_metric_means(totals: torch.Tensor) -> tuple[float, float]:
    """Return matched rate and mean RMSD over matched structures."""
    matched_rate = float(totals[0]) / max(float(totals[1]), 1.0)
    rmsd = (
        float(totals[2]) / float(totals[3])
        if float(totals[3]) > 0
        else float("nan")
    )
    return matched_rate, rmsd


def sample(config: MOFConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    device = _initialize_process(config)
    _seed_everything(config.seed)

    dataset = MOFDataset(
        Path(config.data_root) / config.dataset_subdir,
        split="test",
        max_atoms=config.mof_composition_max_total_atoms,
        composition_config=config,
    )
    sampler = DistributedMOFBatchSampler(
        dataset,
        batch_size=config.micro_batch_size,
        replicas=config.world_size,
        rank=config.rank,
        seed=config.seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_mofs)

    model = MOFSampler(config).to(device)
    # Sampling determinism should not depend on how many unused modules a
    # checkpoint happened to contain.
    _seed_everything(config.seed + config.rank)
    model.eval()
    meter = DistributedThroughputMeter(device)
    accumulated_match_totals = torch.zeros(4, device=device)
    total_batches = len(loader)
    if config.max_batches > 0:
        total_batches = min(total_batches, config.max_batches)
    if config.rank == 0:
        log_info("Start sampling")
    for batch_index, batch in enumerate(loader):
        if 0 < config.max_batches <= batch_index:
            break
        batch = _move_to_device(batch, device)
        sample_metrics = []
        with torch.inference_mode():
            for sample_time_index in range(config.num_samples):
                sample_metrics.append(
                    model.sample_and_match(batch, sample_time_index)
                )
        matches = torch.cat(
            [metrics["match"] for metrics in sample_metrics], dim=1
        )
        rmsds = torch.cat(
            [metrics["rmsd"] for metrics in sample_metrics], dim=1
        )
        best_match = matches.amax(dim=1, keepdim=True)
        best_rmsd = rmsds.masked_fill(~torch.isfinite(rmsds), torch.inf).amin(
            dim=1, keepdim=True
        )
        best_rmsd = best_rmsd.masked_fill(~torch.isfinite(best_rmsd), torch.nan)
        metrics = {"match": best_match, "rmsd": best_rmsd}
        local_candidates = matches.numel()
        meter.update(local_candidates)
        throughput = meter.measure()
        batch_match_totals = _match_metric_totals(metrics)
        torch.distributed.all_reduce(batch_match_totals)
        accumulated_match_totals += batch_match_totals
        matched_rate, rmsd = _match_metric_means(batch_match_totals)
        accumulated_matched_rate, accumulated_rmsd = _match_metric_means(
            accumulated_match_totals
        )
        if config.rank == 0:
            log_metrics(
                {
                    "sampling_batch": batch_index + 1,
                    "sampling_batches": total_batches,
                    "total_samples": throughput.samples,
                    "matched_rate": matched_rate,
                    "accumulated_matched_rate": accumulated_matched_rate,
                    "rmsd": rmsd,
                    "accumulated_rmsd": accumulated_rmsd,
                    "samples_per_second": throughput.samples_per_second,
                    "seconds": throughput.seconds,
                }
            )
    if config.rank == 0:
        log_info("Finished sampling")
