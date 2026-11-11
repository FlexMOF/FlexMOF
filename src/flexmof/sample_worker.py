"""Internal torchrun entry point for FlexMOF sampling."""

from __future__ import annotations

import argparse
import json

import torch

from .runtime import build_config, sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-subdir", default="bw_db")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12347)
    parser.add_argument("--mof-overrides-json", default="{}")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    try:
        sample(
            build_config(
                checkpoint_path=args.checkpoint,
                data_root=args.data_root,
                output_dir=args.output,
                dataset_subdir=args.dataset_subdir,
                micro_batch_size=args.micro_batch_size,
                max_batches=args.max_batches,
                num_samples=args.num_samples,
                seed=args.seed,
                overrides=json.loads(args.mof_overrides_json),
            )
        )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
