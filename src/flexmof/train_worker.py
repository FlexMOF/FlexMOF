"""Internal torchrun entry point for native FlexMOF training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch.distributed as dist

from .training import TrainingConfig, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, dest="output_dir")
    parser.add_argument("--dataset-subdir", default="bw_db")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--max-atoms", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=4_375_000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--composition-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--learning-rate-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=262_500)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=2500)
    parser.add_argument("--validate-interval", type=int, default=0)
    parser.add_argument("--validation-batches", type=int, default=-1)
    parser.add_argument("--validation-log-interval", type=int, default=1)
    parser.add_argument("--validation-seed", type=int, default=-1)
    parser.add_argument("--validate-at-start", action="store_true")
    parser.add_argument("--run-until-stopped", action="store_true")
    parser.add_argument(
        "--validate-at-epoch-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-at-epoch-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12347)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-log-checkpoints", action="store_true")
    parser.add_argument("--composition-diffusion", action="store_true")
    parser.add_argument("--mof-overrides-json", default="{}")
    return parser


def main() -> None:
    try:
        values = vars(build_parser().parse_args())
        mof_overrides = json.loads(values.pop("mof_overrides_json"))
        train(TrainingConfig(**values), mof_overrides)
    except BaseException:
        # Graceful NCCL teardown can wait forever for peers already waiting on
        # this failed rank. Surface the original exception and let torchrun
        # terminate the worker group immediately instead of hiding it behind
        # a later collective timeout.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
