"""Opportunistic device selection, shared by every backbone: `cuda` if
available, else `cpu`, never a hard CUDA dependency. CI installs CPU-only
torch, so this path is the one that's actually exercised there."""

from __future__ import annotations

import torch


def select_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
