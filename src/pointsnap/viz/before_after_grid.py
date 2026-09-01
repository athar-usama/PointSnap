"""Grid compositors for before/after comparison figures -- plain numpy
stacking with consistent padding, no plotting-library chart chrome."""

from __future__ import annotations

import numpy as np


def hstack_panels(panels: list[np.ndarray], pad: int = 4, pad_value: int = 255) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    out = []
    for i, p in enumerate(panels):
        if p.shape[0] != h:
            p = np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0)), constant_values=pad_value)
        out.append(p)
        if i < len(panels) - 1:
            out.append(np.full((h, pad, 3), pad_value, dtype=p.dtype))
    return np.hstack(out)


def vstack_panels(panels: list[np.ndarray], pad: int = 4, pad_value: int = 255) -> np.ndarray:
    w = max(p.shape[1] for p in panels)
    out = []
    for i, p in enumerate(panels):
        if p.shape[1] != w:
            p = np.pad(p, ((0, 0), (0, w - p.shape[1]), (0, 0)), constant_values=pad_value)
        out.append(p)
        if i < len(panels) - 1:
            out.append(np.full((pad, w, 3), pad_value, dtype=p.dtype))
    return np.vstack(out)
