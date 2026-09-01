"""Two novel evaluation metrics quantifying fly-point artifacts directly, in
ways standard 2D depth metrics (AbsRel, delta-accuracy) cannot see. Both are
designed as scale-invariant ratios, so the backbone's unknown affine
inverse-depth ambiguity never leaks into the numbers (see PLAN.md / README).

  - Point-Isolation Score (PIS): a boundary point's real 3D k-NN spacing
    divided by its local surface's expected spacing. PIS >> 1 is the direct
    3D signature of a point floating in space that's locally almost empty.
  - Bridging Rate (BR): across a profile perpendicular to a boundary
    crossing, a crossing is "bridged" if no single step carries most of the
    total jump -- i.e. the transition is smeared rather than a hard step.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from pointsnap.geometry.backproject import backproject
from pointsnap.geometry.intrinsics import Intrinsics


def point_isolation_scores(
    depth: np.ndarray,
    boundary_mask: np.ndarray,
    intrinsics: Intrinsics,
    k: int = 5,
    window_radius: int = 10,
) -> np.ndarray:
    """Returns one PIS value per True pixel in `boundary_mask`, in the same
    order as `np.where(boundary_mask)`.

    Vectorized in two passes rather than rebuilding a fresh local KD-tree per
    boundary pixel (which is correct but does not scale past small synthetic
    images -- it makes a real-photo-resolution benchmark run take minutes per
    image): (1) one batched query for every boundary point's own k-NN
    distance against the whole point cloud, and (2) one pass over the
    *confident* points computing each of their own k-NN spacings against
    the confident-only cloud, so a boundary pixel's `d_expected` becomes a
    cheap array lookup (median over its window) instead of a tree rebuild.
    """
    h, w = depth.shape
    points = backproject(depth, intrinsics).reshape(-1, 3)
    full_tree = cKDTree(points)

    ys, xs = np.where(boundary_mask)
    if len(ys) == 0:
        return np.array([])
    boundary_flat = ys * w + xs
    d_knn_all, _ = full_tree.query(points[boundary_flat], k=k + 1)
    d_knn_all = d_knn_all[:, 1:].mean(axis=1)

    confident_flat = np.flatnonzero(~boundary_mask.reshape(-1))
    density_map = np.full(h * w, np.nan)
    if len(confident_flat) > k:
        confident_points = points[confident_flat]
        confident_tree = cKDTree(confident_points)
        conf_dists, _ = confident_tree.query(confident_points, k=k + 1)
        density_map[confident_flat] = conf_dists[:, 1:].mean(axis=1)

    scores = np.full(len(ys), np.nan)
    for i, (y, x) in enumerate(zip(ys, xs)):
        y0, y1 = max(0, y - window_radius), min(h, y + window_radius + 1)
        x0, x1 = max(0, x - window_radius), min(w, x + window_radius + 1)
        window_flat = (np.arange(y0, y1)[:, None] * w + np.arange(x0, x1)[None, :]).ravel()
        local_densities = density_map[window_flat]
        local_densities = local_densities[~np.isnan(local_densities)]
        if len(local_densities) == 0:
            continue
        d_expected = float(np.median(local_densities))
        scores[i] = float(d_knn_all[i]) / max(d_expected, 1e-8)

    return scores


def summarize_point_isolation(scores: np.ndarray, isolation_threshold: float = 2.0) -> dict:
    valid = scores[~np.isnan(scores)]
    if len(valid) == 0:
        return {"median_pis": float("nan"), "isolation_rate": float("nan"), "n": 0}
    return {
        "median_pis": float(np.median(valid)),
        "isolation_rate": float(np.mean(valid > isolation_threshold)),
        "n": len(valid),
    }


def _bilinear_sample(depth: np.ndarray, y: float, x: float) -> float:
    return float(map_coordinates(depth, [[y], [x]], order=1, mode="nearest")[0])


def bridging_rate(
    depth: np.ndarray,
    boundary_mask: np.ndarray,
    profile_halfwidth: int = 3,
    gamma: float = 0.8,
    smooth_sigma: float = 1.0,
) -> float:
    smoothed = gaussian_filter(depth.astype(np.float64), sigma=smooth_sigma)
    gy, gx = np.gradient(smoothed)
    ys, xs = np.where(boundary_mask)
    offsets = np.arange(-profile_halfwidth, profile_halfwidth + 1)

    outcomes = []
    for y, x in zip(ys, xs):
        grad = np.array([gy[y, x], gx[y, x]])
        norm = float(np.linalg.norm(grad))
        if norm < 1e-8:
            continue
        direction = grad / norm
        profile = np.array(
            [_bilinear_sample(depth, y + o * direction[0], x + o * direction[1]) for o in offsets]
        )
        total_jump = abs(profile[-1] - profile[0])
        if total_jump < 1e-8:
            continue
        max_step = float(np.max(np.abs(np.diff(profile))))
        outcomes.append(1.0 if max_step < gamma * total_jump else 0.0)

    return float(np.mean(outcomes)) if outcomes else 0.0
