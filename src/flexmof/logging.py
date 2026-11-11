"""Legacy PSM-style console logging without its logging dependencies."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Mapping


def _prefix() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"[{timestamp}][INFO]:"


def _is_global_rank_zero() -> bool:
    """Log normally outside torchrun and only from global rank zero inside it."""
    return os.environ.get("RANK", "0") == "0"


def log_info(message: str) -> None:
    """Print an INFO message using the original repository's console style."""
    if not _is_global_rank_zero():
        return
    print(f"{_prefix()} {message}", flush=True)


def log_metrics(metrics: Mapping[str, Any]) -> None:
    """Print pipe-separated key/value metrics like the old MetricLogger."""
    if not _is_global_rank_zero():
        return
    fields = "".join(
        f" | {name}={value} " for name, value in metrics.items() if value is not None
    )
    print(f"{_prefix()}{fields}", flush=True)
