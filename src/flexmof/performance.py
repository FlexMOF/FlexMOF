"""Distributed wall-clock throughput measurement."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Throughput:
    samples: int
    seconds: float
    samples_per_second: float


def global_sample_count(local_samples: int, device: torch.device) -> int:
    """Sum a local structure count over the complete DDP world."""
    samples = torch.tensor(local_samples, device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(samples, op=dist.ReduceOp.SUM)
    return int(samples)


class DistributedThroughputMeter:
    """Measure global samples over the slowest rank's elapsed wall time."""

    def __init__(
        self,
        device: torch.device,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.device = device
        self.clock = clock
        self.local_samples = 0
        self.started = clock()

    def update(self, local_samples: int) -> None:
        self.local_samples += int(local_samples)

    def exclude(self, seconds: float) -> None:
        """Exclude validation, checkpointing, or other paused work."""
        self.started += max(0.0, float(seconds))

    def measure(self, *, reset: bool = False) -> Throughput:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(0.0, self.clock() - self.started)
        seconds = torch.tensor(elapsed, device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        global_samples = global_sample_count(self.local_samples, self.device)
        global_seconds = float(seconds)
        rate = global_samples / global_seconds if global_seconds > 0.0 else 0.0
        if reset:
            self.local_samples = 0
            self.started = self.clock()
        return Throughput(global_samples, global_seconds, rate)
