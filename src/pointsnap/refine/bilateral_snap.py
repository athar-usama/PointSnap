"""Sub-pixel boundary snap: a narrow-band joint-bilateral pass, restricted to
pixels within a couple of pixels of a label transition, for anti-aliased
final boundaries. Uses the same color bandwidth as the MRF smoothness term
so the two stages stay consistent.

A boundary is O(perimeter), not O(area) -- typically a small fraction of an
image's pixels -- so a direct per-pixel loop over just the narrow band is
fine even unvectorized, and removes any hard dependency on opencv's contrib
`ximgproc` module (not guaranteed to be present in every opencv-python
build).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def label_transition_band(label: np.ndarray, radius: int = 2) -> np.ndarray:
    edges = np.zeros_like(label, dtype=bool)
    edges[:-1, :] |= label[:-1, :] != label[1:, :]
    edges[1:, :] |= label[:-1, :] != label[1:, :]
    edges[:, :-1] |= label[:, :-1] != label[:, 1:]
    edges[:, 1:] |= label[:, :-1] != label[:, 1:]
    return ndimage.binary_dilation(edges, iterations=radius)


def joint_bilateral_snap(
    depth: np.ndarray,
    gray: np.ndarray,
    label: np.ndarray,
    sigma_c: float = 0.15,
    sigma_s: float = 2.0,
    window: int = 5,
    band_radius: int = 2,
) -> np.ndarray:
    band = label_transition_band(label, radius=band_radius)
    out = depth.copy()
    h, w = depth.shape
    ys, xs = np.where(band)
    half = window // 2

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        patch_depth = depth[y0:y1, x0:x1]
        patch_gray = gray[y0:y1, x0:x1]

        dy, dx = np.mgrid[y0 - y : y1 - y, x0 - x : x1 - x]
        spatial_w = np.exp(-(dy**2 + dx**2) / (2 * sigma_s**2))
        color_w = np.exp(-((patch_gray - gray[y, x]) ** 2) / (2 * sigma_c**2))
        weight = spatial_w * color_w
        weight_sum = weight.sum()
        if weight_sum > 1e-8:
            out[y, x] = np.sum(weight * patch_depth) / weight_sum

    return out
