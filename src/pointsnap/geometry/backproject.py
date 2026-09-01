"""Pinhole back-projection / re-projection and point-cloud assembly.

Every array in this module is (H, W, ...) with pixel coordinates u=column,
v=row, matching OpenCV/numpy image-array convention throughout the project.
"""

from __future__ import annotations

import numpy as np

from pointsnap.geometry.intrinsics import Intrinsics


def pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    v, u = np.mgrid[0:height, 0:width]
    return u.astype(np.float64), v.astype(np.float64)


def backproject(depth: np.ndarray, k: Intrinsics) -> np.ndarray:
    """depth: (H, W) metric or affine-ambiguous depth -> (H, W, 3) camera-space points."""
    u, v = pixel_grid(*depth.shape)
    z = depth
    x = (u - k.cx) / k.fx * z
    y = (v - k.cy) / k.fy * z
    return np.stack([x, y, z], axis=-1)


def project(points: np.ndarray, k: Intrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(..., 3) camera-space points -> (u, v, z), the inverse of backproject."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k.fx * x / z + k.cx
        v = k.fy * y / z + k.cy
    return u, v, z


def to_point_cloud(
    depth: np.ndarray,
    k: Intrinsics,
    rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Flatten a depth map to an (N, 3) point cloud, optionally colored and masked."""
    points = backproject(depth, k).reshape(-1, 3)
    colors = rgb.reshape(-1, rgb.shape[-1]) if rgb is not None else None
    if mask is not None:
        flat_mask = mask.reshape(-1).astype(bool)
        points = points[flat_mask]
        if colors is not None:
            colors = colors[flat_mask]
    return points, colors
