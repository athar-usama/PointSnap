"""Fast, whole-image classical candidate detector for fly-point artifacts.

Two complementary signals, both robust-normalized (median/MAD) and combined:

  1. "unexplained depth edge" — a depth discontinuity with no corresponding
     RGB edge nearby. Catches artifacts that spill into flat-colored regions
     (a bridge extending past the true silhouette, an isolated fly-point far
     from any real edge).
  2. "bridging distance" — how close a pixel's inverse depth sits to the
     midpoint between the local min/max, inside windows where that local
     range is large enough to be a genuine boundary rather than noise. A
     pixel sitting cleanly at one extreme scores ~0; a pixel smeared exactly
     between two surfaces scores ~0.5. This is a fast morphological
     (erode/dilate) proxy for "does this pixel fail to belong to either
     local plane" — the exact, expensive version (full RANSAC bi-planar fit)
     is reserved for `refine/ransac_biplanar.py`, run only on the pixels
     this screening pass flags, never on the whole image.

Combined via elementwise max into a single continuous score, thresholded
into a boolean candidate mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def inverse_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return 1.0 / np.clip(depth, eps, None)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.dtype == np.uint8 else rgb.mean(axis=-1)
    return gray.astype(np.float64)


def sobel_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2).astype(np.float64)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def bridging_distance(w: np.ndarray, ksize: int = 15) -> np.ndarray:
    """0 at either local extreme, 0.5 exactly between them; zero wherever the
    local range is within the map's own noise floor (a flat region)."""
    w32 = w.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    local_min = cv2.erode(w32, kernel)
    local_max = cv2.dilate(w32, kernel)
    local_range = (local_max - local_min).astype(np.float64)

    noise_floor = 3.0 * (np.median(np.abs(w - np.median(w))) + 1e-8)
    significant = local_range > noise_floor

    dist_to_min = w - local_min
    dist_to_max = local_max - w
    closeness = np.minimum(dist_to_min, dist_to_max)
    score = np.divide(closeness, local_range, out=np.zeros_like(w), where=local_range > 1e-12)
    return np.where(significant, score, 0.0)


@dataclass
class ClassicalScore:
    zscore: np.ndarray
    candidate_mask: np.ndarray


def classical_candidate_score(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    color_edge_z: float = 1.0,
    threshold: float = 2.5,
) -> ClassicalScore:
    w = inverse_depth(depth_raw)
    depth_grad_z = robust_z(sobel_gradient_magnitude(w))
    rgb_grad_z = robust_z(sobel_gradient_magnitude(to_gray(rgb)))

    color_edge_present = rgb_grad_z > color_edge_z
    unexplained_depth_edge = np.clip(depth_grad_z, 0, None) * (~color_edge_present)

    bridge_z = robust_z(bridging_distance(w))

    zscore = np.maximum(unexplained_depth_edge, bridge_z)
    candidate_mask = zscore > threshold
    return ClassicalScore(zscore=zscore, candidate_mask=candidate_mask)
