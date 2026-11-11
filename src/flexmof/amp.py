"""Autocast compatibility across supported PyTorch versions."""

import torch


def is_autocast_enabled(device_type: str) -> bool:
    """Query the device's ambient autocast state, including older CPU/CUDA APIs."""
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        # Older releases expose a no-argument CUDA query and a separate CPU one.
        if device_type == "cuda":
            return torch.is_autocast_enabled()
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        raise
