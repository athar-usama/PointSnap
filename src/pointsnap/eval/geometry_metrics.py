"""Standard monocular depth evaluation metrics, Eigen et al. (2014) convention."""

from __future__ import annotations

import numpy as np


def align_median_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Median-ratio scale alignment for metric-ish predictions that only need
    a scale correction (not a full affine remap)."""
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    return float(np.median(gt[mask] / np.clip(pred[mask], 1e-8, None)))


def align_affine(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Least-squares (scale, shift) alignment for genuinely affine-ambiguous
    predictions (e.g. raw MiDaS-style disparity)."""
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    p, g = pred[mask].ravel(), gt[mask].ravel()
    a_mat = np.stack([p, np.ones_like(p)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(a_mat, g, rcond=None)
    return float(scale), float(shift)


def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / np.clip(gt[mask], 1e-8, None)))


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    return float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))


def delta_accuracy(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None, threshold: float = 1.25
) -> float:
    if mask is None:
        mask = np.ones_like(pred, dtype=bool)
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / np.clip(g, 1e-8, None), g / np.clip(p, 1e-8, None))
    return float(np.mean(ratio < threshold))
