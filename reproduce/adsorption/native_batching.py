"""Deterministic, lossless dynamic batches independent of Torch/CUDA.

Indices are positions in the metadata-ordered dataset. Every real position is
used exactly once each epoch. Missing DDP microbatches repeat a real microbatch
only to run the same collectives; their objective weight is zero.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Microbatch:
    indices: tuple[int, ...]
    padding: bool = False

    @property
    def objective_count(self):
        return 0 if self.padding else len(self.indices)


def epoch_batches(sizes, seed, epoch, max_square, max_batch, bucket_size=1024):
    """Shuffle all records, bucket locally by size, then shuffle packed batches.

    A singleton exceeding the declared capacity is an error, never a dropped
    structure. Padding cost includes the eight native lattice corner tokens.
    """
    sizes = [int(n) for n in sizes]
    if not sizes or min(sizes) <= 0:
        raise ValueError("sizes must contain positive atom counts")
    if min(int(max_square), int(max_batch), int(bucket_size)) <= 0:
        raise ValueError("batch limits must be positive")
    largest = max(sizes)
    if (largest + 8) ** 2 > max_square:
        raise ValueError(f"largest {largest}-atom singleton costs {(largest + 8) ** 2}, above max_square={max_square}; refusing to filter")
    rng = random.Random(int(seed) + int(epoch) * 1_000_003)
    order = list(range(len(sizes)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), int(bucket_size)):
        bucket = sorted(order[start:start + int(bucket_size)], key=lambda i: sizes[i])
        current = []
        largest = 0
        for index in bucket:
            next_largest = max(largest, sizes[index])
            if current and (len(current) >= max_batch or (len(current) + 1) * (next_largest + 8) ** 2 > max_square):
                batches.append(tuple(current))
                current, largest = [], 0
            current.append(index)
            largest = max(largest, sizes[index])
        if current:
            batches.append(tuple(current))
    rng.shuffle(batches)
    return batches


def epoch_fingerprint(batches):
    return fingerprint(batches)


def update_window(batches, cursor, world_size, accumulation_steps):
    """Return rounds x ranks, next cursor, and the global real structure count."""
    if world_size < 1 or accumulation_steps < 1:
        raise ValueError("world_size and accumulation_steps must be positive")
    if not 0 <= cursor < len(batches):
        raise ValueError("cursor must identify a pending microbatch")
    end = min(len(batches), cursor + world_size * accumulation_steps)
    actual = [Microbatch(tuple(indices)) for indices in batches[cursor:end]]
    objective_count = sum(m.objective_count for m in actual)
    # Only as many rounds as needed; every rank still executes the same count.
    while len(actual) % world_size:
        actual.append(Microbatch(actual[0].indices, padding=True))
    rounds = [actual[i:i + world_size] for i in range(0, len(actual), world_size)]
    return rounds, end, objective_count


def ddp_loss_scale(local_count, global_count, world_size):
    """Scale a microbatch mean so DDP averaging yields a structure mean."""
    if global_count <= 0 or not 0 <= local_count <= global_count:
        raise ValueError("invalid structure counts")
    return float(local_count) * int(world_size) / int(global_count)


def exposure_update(rounds, objective_exposure, compute_exposure):
    for row in rounds:
        for microbatch in row:
            for index in microbatch.indices:
                compute_exposure[index] += 1
                if not microbatch.padding:
                    objective_exposure[index] += 1


def resume_order_check(batches, saved_fingerprint, cursor):
    if epoch_fingerprint(batches) != saved_fingerprint:
        raise ValueError("data order changed since checkpoint")
    if not 0 <= cursor <= len(batches):
        raise ValueError("saved data cursor is outside epoch")
