"""Command-line interface for FlexMOF."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from .evaluate import evaluate_directory, summarize_metric_files
from .match_protocol import DEFAULT_PROTOCOL, PROTOCOLS
from .config import MOFConfig
from .logging import log_metrics


_REQUIRED_ARGUMENTS = {
    "sample": ("checkpoint", "data_root", "output"),
    "train": ("data_root", "output"),
}
_MOF_PUBLIC_ARGUMENTS = {
    "sample": {
        "checkpoint_path",
        "data_root",
        "dataset_subdir",
        "output_dir",
        "micro_batch_size",
        "max_batches",
        "num_samples",
        "seed",
    },
    "train": {
        "checkpoint_path",
        "data_root",
        "dataset_subdir",
        "output_dir",
        "micro_batch_size",
        "seed",
    },
}
_CONFIG_KEY_ALIASES = {
    "sample": {"checkpoint_path": "checkpoint", "output_dir": "output"},
    "train": {
        "checkpoint_path": "init_checkpoint",
        "output_dir": "output",
    },
}


def _nonnegative_integer(value: Any) -> int:
    # YAML values must not silently turn floats or booleans into recycle counts.
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("expected a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("expected a non-negative integer")
    return result


def _add_mof_arguments(
    parser: argparse.ArgumentParser, excluded: set[str] | None = None
) -> None:
    """Expose otherwise-unhandled MOFConfig fields without overriding defaults."""
    existing = {action.dest for action in parser._actions}
    types = get_type_hints(MOFConfig)
    group = parser.add_argument_group("MOF model configuration")
    for field in fields(MOFConfig):
        if field.name in existing or field.name in (excluded or set()):
            continue
        option = f"--{field.name.replace('_', '-')}"
        value_type = types[field.name]
        if field.name == "num_recycles":
            group.add_argument(
                option,
                type=_nonnegative_integer,
                default=argparse.SUPPRESS,
                help="additional denoising passes from the previous sample, without re-noising (default: 0)",
            )
        elif value_type is bool:
            group.add_argument(
                option,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
        else:
            group.add_argument(
                option, type=value_type, default=argparse.SUPPRESS
            )


def _command_parser(
    parser: argparse.ArgumentParser, command: str
) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[command]
    raise ValueError(f"No parser registered for command {command!r}")


def _load_config(
    path: str | Path, command: str, command_parser: argparse.ArgumentParser
) -> dict[str, Any]:
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text())
    except OSError as error:
        raise ValueError(f"Could not read config {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")

    command_sections = {"train", "sample"}.intersection(loaded)
    if command_sections:
        unexpected = set(loaded) - {"train", "sample"}
        if unexpected:
            names = ", ".join(sorted(map(str, unexpected)))
            raise ValueError(
                "A config with train/sample sections cannot also contain "
                f"top-level parameters: {names}"
            )
        loaded = loaded.get(command, {})
        if not isinstance(loaded, Mapping):
            raise ValueError(f"The {command!r} config section must be a mapping")

    actions = {
        action.dest: action
        for action in command_parser._actions
        if action.dest not in {"help", "config"}
    }
    aliases = _CONFIG_KEY_ALIASES.get(command, {})
    normalized = {
        aliases.get(str(key).replace("-", "_"), str(key).replace("-", "_")): value
        for key, value in loaded.items()
    }
    unknown = set(normalized) - set(actions)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown {command} config parameter(s): {names}")

    values: dict[str, Any] = {}
    for name, value in normalized.items():
        action = actions[name]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"Config parameter {name!r} must be true or false")
        elif action.type is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value for config parameter {name!r}: {value!r}"
                ) from error
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(map(str, action.choices))
            raise ValueError(
                f"Invalid value for config parameter {name!r}: {value!r}; "
                f"choose from {choices}"
            )
        values[name] = value
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        return build_parser().parse_args(arguments)
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("command", choices=("train", "sample", "evaluate"))
    probe.add_argument("--config")
    preliminary, _ = probe.parse_known_args(arguments)

    parser = build_parser()
    command_parser = _command_parser(parser, preliminary.command)
    if preliminary.config:
        try:
            config = _load_config(
                preliminary.config, preliminary.command, command_parser
            )
        except ValueError as error:
            parser.error(str(error))
        command_parser.set_defaults(**config)
    args = parser.parse_args(arguments)
    provided_options = {
        action.dest
        for action in command_parser._actions
        for option in action.option_strings
        if any(item.split("=", 1)[0] == option for item in arguments)
    }
    if args.command == "train":
        if (
            "resume" in provided_options
            and "init_checkpoint" not in provided_options
        ):
            args.init_checkpoint = ""
        elif (
            "init_checkpoint" in provided_options and "resume" not in provided_options
        ):
            args.resume = ""
        if args.init_checkpoint and args.resume:
            parser.error("use either --init-checkpoint or --resume, not both")
    missing = [
        name
        for name in _REQUIRED_ARGUMENTS.get(args.command, ())
        if not getattr(args, name, None)
    ]
    if missing:
        options = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        parser.error(f"the following arguments are required: {options}")
    public_arguments = _MOF_PUBLIC_ARGUMENTS.get(args.command, set())
    args.mof_overrides = {
        field.name: getattr(args, field.name)
        for field in fields(MOFConfig)
        if field.name not in public_arguments
        and hasattr(args, field.name)
    }
    return args


def _sample(args: argparse.Namespace) -> int:
    if args.num_samples < 1:
        print("error: --num-samples must be positive", file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.devices),
        "--module",
        "flexmof.sample_worker",
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--data-root",
        str(Path(args.data_root).resolve()),
        "--output",
        str(output),
        "--dataset-subdir",
        args.dataset_subdir,
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--max-batches",
        str(args.max_batches),
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--mof-overrides-json",
        json.dumps(args.mof_overrides),
    ]
    environment = os.environ.copy()
    environment.setdefault("DDP_TIMEOUT_MINUTES", "3000")
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode:
        return completed.returncode
    metric_files = sorted(output.glob("metrics.rank*.jsonl"))
    if metric_files:
        summary = summarize_metric_files(metric_files)
        (output / "sampling_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        log_metrics(
            {
                key: summary[key]
                for key in (
                    "num_structures",
                    "num_matched",
                    "matched_rate",
                    "mean_matched_rmsd",
                    "num_errors",
                )
            }
        )
    return 0


def _train(args: argparse.Namespace) -> int:
    torchrun_options = ["--standalone"]
    nnodes = os.environ.get("FLEXMOF_NNODES")
    if nnodes:
        rendezvous = {
            "FLEXMOF_NODE_RANK": "--node_rank",
            "FLEXMOF_MASTER_ADDR": "--master_addr",
            "FLEXMOF_MASTER_PORT": "--master_port",
        }
        missing = [name for name in rendezvous if not os.environ.get(name)]
        if missing:
            raise ValueError(
                "Multi-node launch requires " + ", ".join(missing)
            )
        torchrun_options = ["--nnodes", nnodes]
        for name, option in rendezvous.items():
            torchrun_options.extend((option, os.environ[name]))

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *torchrun_options,
        "--nproc_per_node",
        str(args.devices),
        "--module",
        "flexmof.train_worker",
    ]
    for name in (
        "data_root",
        "output",
        "dataset_subdir",
        "micro_batch_size",
        "accumulation_steps",
        "max_atoms",
        "max_steps",
        "epochs",
        "learning_rate",
        "composition_lr_multiplier",
        "ema_decay",
        "warmup_steps",
        "weight_decay",
        "gradient_clip",
        "precision",
        "log_interval",
        "save_interval",
        "validate_interval",
        "validation_batches",
        "validation_log_interval",
        "validation_seed",
        "num_workers",
        "seed",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "wandb_group",
        "wandb_tags",
        "wandb_mode",
    ):
        command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    for name in ("validate_at_epoch_end", "save_at_epoch_end"):
        option = f"--{name.replace('_', '-')}"
        command.append(option if getattr(args, name) else f"--no-{option[2:]}")
    for name in ("init_checkpoint", "resume"):
        value = getattr(args, name)
        if value:
            command.extend([f"--{name.replace('_', '-')}", str(Path(value).resolve())])
    if args.wandb_log_checkpoints:
        command.append("--wandb-log-checkpoints")
    if args.composition_diffusion:
        command.append("--composition-diffusion")
    if args.validate_at_start:
        command.append("--validate-at-start")
    command.extend(("--mof-overrides-json", json.dumps(args.mof_overrides)))
    environment = os.environ.copy()
    environment.setdefault("DDP_TIMEOUT_MINUTES", "3000")
    return subprocess.run(command, env=environment, check=False).returncode


def _evaluate(args: argparse.Namespace) -> int:
    if args.num_samples is not None and args.num_samples < 1:
        print("error: --num-samples must be positive", file=sys.stderr)
        return 2
    output = args.output or str(Path(args.directory) / "evaluation.json")
    result = evaluate_directory(
        args.directory,
        args.workers,
        output,
        num_samples=args.num_samples,
        protocol=getattr(args, "match_protocol", None),
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flexmof")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser(
        "sample", help="sample MOF structures from a checkpoint"
    )
    sample_parser.add_argument("--config", help="YAML configuration file")
    sample_parser.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint")
    sample_parser.add_argument("--data-root")
    sample_parser.add_argument("--output", "--output-dir", dest="output")
    sample_parser.add_argument("--dataset-subdir", default="bw_db")
    sample_parser.add_argument("--devices", type=int, default=1)
    sample_parser.add_argument("--micro-batch-size", type=int, default=4)
    sample_parser.add_argument(
        "--max-batches", type=int, default=-1, help="-1 samples the full test split"
    )
    sample_parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="independent samples per test structure (default: 1)",
    )
    sample_parser.add_argument("--seed", type=int, default=12347)
    _add_mof_arguments(sample_parser, _MOF_PUBLIC_ARGUMENTS["sample"])
    sample_parser.set_defaults(func=_sample)

    train_parser = subparsers.add_parser(
        "train", help="train the MOF denoiser with native PyTorch DDP"
    )
    train_parser.add_argument("--config", help="YAML configuration file")
    train_parser.add_argument("--data-root")
    train_parser.add_argument("--output", "--output-dir", dest="output")
    train_parser.add_argument("--dataset-subdir", default="bw_db")
    source = train_parser.add_mutually_exclusive_group()
    source.add_argument(
        "--init-checkpoint", "--checkpoint-path", dest="init_checkpoint", default=""
    )
    source.add_argument("--resume", default="")
    train_parser.add_argument("--devices", type=int, default=1)
    train_parser.add_argument("--micro-batch-size", type=int, default=4)
    train_parser.add_argument("--accumulation-steps", type=int, default=32)
    train_parser.add_argument("--max-atoms", type=int, default=768)
    train_parser.add_argument("--max-steps", type=int, default=4_375_000)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--composition-lr-multiplier", type=float, default=1.0)
    train_parser.add_argument("--ema-decay", type=float, default=0.0)
    train_parser.add_argument("--warmup-steps", type=int, default=262_500)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="fp32"
    )
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.add_argument("--save-interval", type=int, default=2500)
    train_parser.add_argument("--validate-interval", type=int, default=0)
    train_parser.add_argument("--validation-batches", type=int, default=-1)
    train_parser.add_argument("--validation-log-interval", type=int, default=1)
    train_parser.add_argument("--validation-seed", type=int, default=-1)
    train_parser.add_argument("--validate-at-start", action="store_true")
    train_parser.add_argument(
        "--validate-at-epoch-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument(
        "--save-at-epoch-end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=12347)
    train_parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", ""),
        help="enable W&B logging to this project (or set WANDB_PROJECT)",
    )
    train_parser.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY", "")
    )
    train_parser.add_argument("--wandb-name", default="")
    train_parser.add_argument("--wandb-group", default="")
    train_parser.add_argument(
        "--wandb-tags", default="", help="comma-separated W&B run tags"
    )
    train_parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    train_parser.add_argument(
        "--wandb-log-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="upload saved training checkpoints as W&B artifacts",
    )
    train_parser.add_argument(
        "--composition-diffusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable original DiGress atom and bond composition diffusion",
    )
    _add_mof_arguments(train_parser, _MOF_PUBLIC_ARGUMENTS["train"])
    train_parser.set_defaults(func=_train)

    eval_parser = subparsers.add_parser(
        "evaluate", help="compute matched_rate for sampled/original CIF pairs"
    )
    eval_parser.add_argument("directory")
    eval_parser.add_argument("--workers", type=int, default=0)
    eval_parser.add_argument(
        "--num-samples",
        type=int,
        help="require sample indices 0..K-1 and report best-of-K",
    )
    eval_parser.add_argument("--output")
    eval_parser.add_argument(
        "--match-protocol",
        default=DEFAULT_PROTOCOL,
        choices=sorted(PROTOCOLS),
        help="StructureMatcher parameter set; recorded in the summary",
    )
    eval_parser.set_defaults(func=_evaluate)
    return parser


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
