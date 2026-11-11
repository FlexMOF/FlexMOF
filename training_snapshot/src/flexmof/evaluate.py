"""MatterGen-compatible structure matching for FlexMOF samples."""

from __future__ import annotations

import json
import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable


def _sample_identity(path: str | Path) -> tuple[str, int]:
    stem = Path(path).stem.removeprefix("sampled_").removeprefix("original_")
    structure_id, separator, sample_index = stem.rpartition("_")
    if not separator:
        return stem, 0
    try:
        return structure_id, int(sample_index)
    except ValueError:
        return stem, 0


def _match_pair(pair: tuple[str, int, str, str] | tuple[str, int, str, str, str]) -> dict:
    from pymatgen.core import Structure

    from .match_protocol import build_matcher

    structure_id, sample_index, sampled_path, target_path, *rest = pair
    protocol = rest[0] if rest else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampled = Structure.from_file(sampled_path)
            target = Structure.from_file(target_path)
        distance = build_matcher(protocol).get_rms_dist(sampled, target)
        return {
            "structure_id": structure_id,
            "sample_index": sample_index,
            "sampled": sampled_path,
            "target": target_path,
            "match": int(distance is not None),
            "rmsd": None if distance is None else float(distance[0]),
        }
    except Exception as exc:  # malformed output should count as a failed match
        return {
            "structure_id": structure_id,
            "sample_index": sample_index,
            "sampled": sampled_path,
            "target": target_path,
            "match": 0,
            "rmsd": None,
            "error": str(exc),
        }


def find_pairs(
    directory: str | Path, num_samples: int | None = None
) -> list[tuple[str, int, str, str]]:
    root = Path(directory)
    pairs: list[tuple[str, int, str, str]] = []
    if num_samples is not None:
        originals: dict[str, dict[int, Path]] = {}
        for target in root.glob("original_*.cif"):
            structure_id, sample_index = _sample_identity(target)
            originals.setdefault(structure_id, {})[sample_index] = target
        for structure_id, targets in sorted(originals.items()):
            fallback = targets.get(0, targets[min(targets)])
            for sample_index in range(num_samples):
                target = targets.get(sample_index, fallback)
                sampled = root / f"sampled_{structure_id}_{sample_index}.cif"
                pairs.append(
                    (structure_id, sample_index, str(sampled), str(target))
                )
        return pairs

    for sampled in sorted(root.glob("sampled_*.cif")):
        structure_id, sample_index = _sample_identity(sampled)
        target = root / sampled.name.replace("sampled_", "original_", 1)
        if not target.is_file():
            target = root / f"original_{structure_id}_0.cif"
        if target.is_file():
            pairs.append(
                (structure_id, sample_index, str(sampled), str(target))
            )
    return pairs


def summarize(records: Iterable[dict], protocol: str | None = None) -> dict:
    from .match_protocol import resolve

    rows = list(records)
    groups: dict[Any, list[dict]] = {}
    for fallback_index, row in enumerate(rows):
        structure_id = row.get(
            "structure_id", row.get("sample_id", fallback_index)
        )
        groups.setdefault(structure_id, []).append(row)

    best_rmsds = []
    matched = 0
    for candidates in groups.values():
        rmsds = [
            float(row["rmsd"])
            for row in candidates
            if row.get("match", 0)
            and row.get("rmsd") is not None
            and math.isfinite(float(row["rmsd"]))
        ]
        if any(int(row.get("match", 0)) for row in candidates):
            matched += 1
        if rmsds:
            best_rmsds.append(min(rmsds))
    return {
        "num_structures": len(groups),
        "num_candidates": len(rows),
        "max_samples_per_structure": max(map(len, groups.values()), default=0),
        "num_matched": matched,
        "matched_rate": matched / len(groups) if groups else None,
        "mean_matched_rmsd": (
            sum(best_rmsds) / len(best_rmsds) if best_rmsds else None
        ),
        "num_errors": sum("error" in row for row in rows),
        "matcher": resolve(protocol).record(),
    }


def evaluate_directory(
    directory: str | Path,
    workers: int = 0,
    output: str | Path | None = None,
    num_samples: int | None = None,
    protocol: str | None = None,
) -> dict:
    pairs = [
        pair + (protocol,)
        for pair in find_pairs(directory, num_samples=num_samples)
    ]
    if workers == 1 or len(pairs) < 2:
        records = [_match_pair(pair) for pair in pairs]
    else:
        max_workers = workers or min(32, len(pairs))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            records = list(pool.map(_match_pair, pairs, chunksize=16))
    result = summarize(records, protocol)
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize_metric_files(paths: Iterable[str | Path]) -> dict:
    records = []
    for path in paths:
        with Path(path).open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    result = summarize(records)
    result["source"] = "sampling metric JSONL"
    return result
