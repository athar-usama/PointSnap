"""Assembles the 8-channel feature stack fed to the confidence CNN:
R, G, B, inverse-depth, d(inverse-depth)/du, d(inverse-depth)/dv,
RGB-gradient magnitude, classical-detector z-score.

The classical z-score is included as an explicit input feature (not just an
ensemble partner downstream) because it gives a tiny network trained on a
modest dataset a useful head start rather than having to rediscover the same
signal from raw pixels."""

from __future__ import annotations

import cv2
import numpy as np

from pointsnap.detect.classical import (
    classical_candidate_score,
    inverse_depth,
    robust_z,
    sobel_gradient_magnitude,
    to_gray,
)

NUM_CHANNELS = 8


def compute_feature_stack(rgb: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
    """rgb: (H, W, 3) uint8. depth_raw: (H, W) float. Returns (8, H, W) float32."""
    rgb_f = rgb.astype(np.float32) / 255.0
    w = inverse_depth(depth_raw).astype(np.float32)
    gx = cv2.Sobel(w, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(w, cv2.CV_32F, 0, 1, ksize=3)
    rgb_grad = sobel_gradient_magnitude(to_gray(rgb)).astype(np.float32)
    classical = classical_candidate_score(rgb, depth_raw)

    def norm(x: np.ndarray) -> np.ndarray:
        return robust_z(x).astype(np.float32)

    stack = np.stack(
        [
            rgb_f[..., 0],
            rgb_f[..., 1],
            rgb_f[..., 2],
            norm(w),
            norm(gx),
            norm(gy),
            norm(rgb_grad),
            np.clip(classical.zscore, -5, 5).astype(np.float32) / 5.0,
        ],
        axis=0,
    )
    return stack.astype(np.float32)
