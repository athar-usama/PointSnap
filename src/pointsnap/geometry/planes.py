"""Inverse-depth plane fitting.

A real 3D plane n.X = d, back-projected through a pinhole model, satisfies

    1/Z(u, v) = A*u + B*v + C

for constants (A, B, C) that fold in the plane's normal, offset, and the
camera intrinsics (see derivation in PLAN.md / README). This holds whether Z
is metric or an unknown affine transform of the true depth, because an affine
function of an affine field is still affine, so every function here operates
directly on a monocular backbone's raw output with no calibration step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlaneFit:
    coeffs: np.ndarray  # (A, B, C)
    inlier_mask: np.ndarray
    rmse: float
    n_inliers: int

    def eval(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return eval_plane(self.coeffs, u, v)


def eval_plane(coeffs: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    a, b, c = coeffs
    return a * u + b * v + c


def fit_plane_exact(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Solve the 3x3 linear system for (A, B, C) from exactly 3 points."""
    m = np.stack([u, v, np.ones_like(u)], axis=1)
    try:
        return np.linalg.solve(m, w)
    except np.linalg.LinAlgError:
        return None


def fit_plane_lsq(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Least-squares refit of (A, B, C) over an arbitrary-size point set."""
    m = np.stack([u, v, np.ones_like(u)], axis=1)
    coeffs, *_ = np.linalg.lstsq(m, w, rcond=None)
    return coeffs


def ransac_plane(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_iters: int = 200,
    threshold: float | None = None,
    min_inliers: int = 6,
    early_exit_ratio: float = 0.8,
    rng: np.random.Generator | None = None,
) -> PlaneFit | None:
    """RANSAC-fit an inverse-depth plane, robust to the confident-neighbor set
    still containing a few points from the *other* side of a boundary."""
    rng = rng or np.random.default_rng()
    n = len(u)
    if n < 3:
        return None
    if threshold is None:
        threshold = 0.02 * max(np.median(np.abs(w)), 1e-8)

    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(n_iters):
        idx = rng.choice(n, size=3, replace=False)
        coeffs = fit_plane_exact(u[idx], v[idx], w[idx])
        if coeffs is None:
            continue
        residuals = np.abs(eval_plane(coeffs, u, v) - w)
        inliers = residuals < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            if count / n > early_exit_ratio:
                break

    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None

    coeffs = fit_plane_lsq(u[best_inliers], v[best_inliers], w[best_inliers])
    residuals = eval_plane(coeffs, u[best_inliers], v[best_inliers]) - w[best_inliers]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return PlaneFit(coeffs=coeffs, inlier_mask=best_inliers, rmse=rmse, n_inliers=int(best_inliers.sum()))


def kmeans_1d(values: np.ndarray, k: int = 2, n_iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 1D k-means (percentile-seeded, so no RNG dependence)."""
    values = np.asarray(values, dtype=np.float64)
    centers = np.percentile(values, np.linspace(10, 90, k))
    labels = np.zeros(len(values), dtype=int)
    for _ in range(n_iters):
        dists = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            in_cluster = labels == i
            if np.any(in_cluster):
                new_centers[i] = values[in_cluster].mean()
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def is_bimodal(values: np.ndarray, min_gap_ratio: float = 5.0) -> bool:
    """A window is bimodal (i.e. genuinely straddles a depth boundary) only if
    the two 1D-kmeans cluster centers separate by more than `min_gap_ratio`
    times the *within-cluster* spread. This is the classical detector's
    false-positive safety valve: a smooth, unimodal window is never split into
    two planes.

    Comparing the gap to overall spread (e.g. IQR of the whole window) would
    fail here: 2-means always finds *some* split even in genuinely unimodal
    data, and for a unimodal Gaussian forced into two halves the gap-to-
    within-cluster-std ratio converges to a fixed value around 2.6 regardless
    of scale. Requiring the gap to exceed several within-cluster standard
    deviations (rather than several fractions of the total spread) is what
    actually separates "two real surfaces" from "one noisy surface that
    2-means arbitrarily cut in half". The ratio's sampling noise is
    non-trivial at small n (Monte Carlo calibration: at n=20 unimodal
    Gaussian windows this ratio exceeds 5.0 about 1% of the time, and never
    at n>=40), which is why windows below `min_points` are rejected outright
    rather than trusted at a lenient threshold.
    """
    min_points = 12
    if len(values) < min_points:
        return False
    values = np.asarray(values, dtype=np.float64)
    labels, centers = kmeans_1d(values, k=2)
    gap = abs(centers[1] - centers[0])
    stds = []
    for i in range(2):
        cluster_vals = values[labels == i]
        if len(cluster_vals) < 2:
            return False
        stds.append(cluster_vals.std())
    pooled_std = max(float(np.mean(stds)), 1e-8)
    return gap > min_gap_ratio * pooled_std
