"""Checkpoint loading for FlexMOF."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .logging import log_info


class _LegacyUnpickler(pickle.Unpickler):
    """Deserialize obsolete enum metadata as its string value."""

    def find_class(self, module: str, name: str):
        if module in {
            "sfm.models.psm.psm_config",
            "sfm.pipeline.accelerator.dataclasses",
        }:
            return str
        return super().find_class(module, name)


_PICKLE_MODULE = SimpleNamespace(
    __name__="flexmof_legacy_pickle",
    Unpickler=_LegacyUnpickler,
    load=pickle.load,
    loads=pickle.loads,
    dump=pickle.dump,
    dumps=pickle.dumps,
)


def load_checkpoint(path: str | Path, *, mmap: bool = True) -> dict:
    """Load a checkpoint without importing the old ``sfm`` package tree."""
    try:
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
            mmap=mmap,
            pickle_module=_PICKLE_MODULE,
        )
    except RuntimeError as error:
        # Network filesystems can occasionally make PyTorch's initial zip
        # signature probe fail even for a valid zip checkpoint. Retrying
        # without mmap is also required for genuinely legacy torch.save files.
        if not mmap or "mmap can only be used with files saved" not in str(error):
            raise
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
            mmap=False,
            pickle_module=_PICKLE_MODULE,
        )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not a FlexMOF/PSM checkpoint: {path}")
    return checkpoint


def load_model_weights(
    module: nn.Module,
    checkpoint: dict,
    source: str | Path,
    *,
    allow_missing_prefixes: tuple[str, ...] = (),
) -> int:
    """Load exactly the tensors used by ``module`` and reject missing weights."""
    saved = checkpoint["model"]
    expected = module.state_dict()
    useful = {key: saved[key] for key in expected if key in saved}
    missing = sorted(set(expected) - set(useful))
    unexpected_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allow_missing_prefixes)
    ]
    if unexpected_missing:
        raise ValueError(
            f"Checkpoint is missing {len(unexpected_missing)} required tensors: "
            f"{unexpected_missing}"
        )
    module.load_state_dict(useful, strict=not missing)
    ignored = len(saved) - len(useful)
    is_primary = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    if is_primary:
        log_info(
            f"loaded {len(useful)} MOF tensors from {source}; "
            f"ignored {ignored} unrelated tensors"
            + (f"; initialized {len(missing)} new tensors" if missing else "")
        )
    return ignored
