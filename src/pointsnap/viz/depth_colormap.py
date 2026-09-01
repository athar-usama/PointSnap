"""Consistent depth colorization used across every figure in this project:
percentile-clipped normalization (robust to a handful of extreme outliers)
and nearer = brighter/warmer throughout."""

from __future__ import annotations

import cv2
import numpy as np

from pointsnap.viz.style import DEPTH_COLORMAP


def normalize_depth(
    depth: np.ndarray, valid_mask: np.ndarray | None = None, low_pct: float = 1, high_pct: float = 99
) -> np.ndarray:
    valid = depth[valid_mask] if valid_mask is not None else depth
    lo, hi = np.percentile(valid, [low_pct, high_pct])
    hi = max(hi, lo + 1e-6)
    return np.clip((depth - lo) / (hi - lo), 0, 1)


def colorize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    colormap: int = DEPTH_COLORMAP,
    invert: bool = True,
) -> np.ndarray:
    norm = normalize_depth(depth, valid_mask)
    if invert:
        norm = 1.0 - norm
    gray = (norm * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, colormap)
    rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    if valid_mask is not None:
        rgb = rgb.copy()
        rgb[~valid_mask] = 20
    return rgb
