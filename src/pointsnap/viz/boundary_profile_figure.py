"""Renders the bridging-rate concept as a small-multiple of colored pixel
strips (raw vs refined, both against ground truth) rather than a line chart:
each strip is the actual depth profile sampled across a real boundary
crossing, colorized with the same depth colormap used everywhere else in
this project. A crisp silhouette shows as a single hard color transition; a
bridged one shows as a visible gradient smear across several pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from pointsnap.viz.style import DEPTH_COLORMAP


def _profile_to_strip(profile: np.ndarray, lo: float, hi: float, height: int = 40, scale: int = 24) -> np.ndarray:
    norm = np.clip((profile - lo) / max(hi - lo, 1e-9), 0, 1)
    gray = ((1.0 - norm) * 255).astype(np.uint8)[None, :]
    strip = cv2.applyColorMap(gray, DEPTH_COLORMAP)
    strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    strip = cv2.resize(strip, (profile.shape[0] * scale, height), interpolation=cv2.INTER_NEAREST)
    return strip


def boundary_profile_panel(
    profile_gt: np.ndarray, profile_raw: np.ndarray, profile_refined: np.ndarray, scale: int = 24
) -> np.ndarray:
    """Three stacked strips (ground truth / raw / refined) sampled across the
    same boundary crossing, all normalized to the same depth range so the
    smear is visually comparable."""
    all_vals = np.concatenate([profile_gt, profile_raw, profile_refined])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    strips = [
        _profile_to_strip(profile_gt, lo, hi, scale=scale),
        _profile_to_strip(profile_raw, lo, hi, scale=scale),
        _profile_to_strip(profile_refined, lo, hi, scale=scale),
    ]
    gap = np.full((6, strips[0].shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([strips[0], gap, strips[1], gap, strips[2]])
