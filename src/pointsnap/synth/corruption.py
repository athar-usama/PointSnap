"""Depth corruption model: simulates what a real monocular depth network's
raw output looks like near occlusion boundaries.

Applied ONLY to depth: RGB stays the true composite. That asymmetry is the
whole premise of the downstream RGB-edge-aware relabeling: it needs a channel
that still tells the truth about where the real edge is, exactly like a real
photo does even when a depth network's prediction blurs across it.

Four independent effects, at three severities:
  1. boundary jitter: the whole depth field is shifted by a small random
                       offset before blurring, decoupling "wrong location"
                       from "wrong sharpness".
  2. Gaussian depth blur: the base ramp artifact.
  3. fly-pixel injection: sparse pixels within a band around the true
                       boundary set to a random convex combination of the
                       two neighboring surfaces' depths, the direct
                       simulacrum of a floating point in the 3D cloud.
  4. global multiplicative noise: unrelated to edges, so a detector can't
                       cheat by flagging every local depth gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SEVERITY_PARAMS = {
    "easy": {
        "blur_sigma": (0.5, 1.0), "jitter_px": (0.0, 1.0),
        "fly_pixel_rate": (0.01, 0.03), "noise_pct": (0.5, 1.0),
    },
    "moderate": {
        "blur_sigma": (1.0, 2.0), "jitter_px": (1.0, 2.0),
        "fly_pixel_rate": (0.03, 0.06), "noise_pct": (1.0, 1.5),
    },
    "hard": {
        "blur_sigma": (2.0, 3.0), "jitter_px": (2.0, 3.0),
        "fly_pixel_rate": (0.06, 0.10), "noise_pct": (1.5, 2.0),
    },
}


@dataclass
class CorruptionResult:
    depth_raw: np.ndarray
    artifact_mask: np.ndarray


def _depth_by_instance(instance_mask: np.ndarray, depth_gt: np.ndarray) -> dict[int, float]:
    ids = np.unique(instance_mask)
    out = {}
    for i in ids:
        vals = depth_gt[instance_mask == i]
        out[int(i)] = float(vals[0]) if len(vals) else 0.0
    return out


def _local_two_depths(instance_mask: np.ndarray, depth_by_id: dict[int, float], y: int, x: int,
                       radius: int = 3):
    y0, y1 = max(0, y - radius), min(instance_mask.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(instance_mask.shape[1], x + radius + 1)
    patch = instance_mask[y0:y1, x0:x1]
    ids, counts = np.unique(patch, return_counts=True)
    if len(ids) < 2:
        return None
    top = ids[np.argsort(-counts)[:2]]
    return depth_by_id[int(top[0])], depth_by_id[int(top[1])]


def corrupt_depth(
    depth_gt: np.ndarray,
    instance_mask: np.ndarray,
    boundary_mask: np.ndarray,
    severity: str,
    rng: np.random.Generator,
) -> CorruptionResult:
    params = SEVERITY_PARAMS[severity]
    depth_by_id = _depth_by_instance(instance_mask, depth_gt)

    jitter_px = float(rng.uniform(*params["jitter_px"]))
    dx, dy = (rng.normal(scale=jitter_px, size=2) if jitter_px > 1e-6 else (0.0, 0.0))
    warp = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        depth_gt.astype(np.float32), warp, depth_gt.shape[::-1],
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blur_sigma = float(rng.uniform(*params["blur_sigma"]))
    blurred = cv2.GaussianBlur(shifted, ksize=(0, 0), sigmaX=blur_sigma) if blur_sigma > 1e-6 else shifted
    depth_raw = blurred.astype(np.float64)

    band = cv2.dilate(boundary_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.where(band)
    fly_rate = float(rng.uniform(*params["fly_pixel_rate"]))
    n_fly = int(fly_rate * len(ys))
    if n_fly > 0 and len(ys) > 0:
        chosen = rng.choice(len(ys), size=min(n_fly, len(ys)), replace=False)
        for i in chosen:
            y, x = int(ys[i]), int(xs[i])
            pair = _local_two_depths(instance_mask, depth_by_id, y, x)
            if pair is None:
                continue
            alpha = rng.uniform(0.2, 0.8)
            depth_raw[y, x] = alpha * pair[0] + (1 - alpha) * pair[1]

    # Ground truth is derived from the boundary corruption alone, *before* the
    # unrelated noise pass: the noise is a distractor the detector must
    # learn to see through, not a phenomenon it should learn to flag. Adding
    # it before labeling would leak ~1-3% of flat, non-boundary background
    # pixels into "artifact" purely from Gaussian tail crossings, which both
    # corrupts CNN training labels and inflates every downstream benchmark
    # number with noise that has nothing to do with fly-points.
    depth_values = np.array(list(depth_by_id.values()))
    depth_range = max(float(depth_values.max() - depth_values.min()), 1e-6)
    artifact_mask = np.abs(depth_raw - depth_gt) > 0.05 * depth_range

    noise_pct = float(rng.uniform(*params["noise_pct"]))
    noise = 1.0 + rng.normal(scale=noise_pct / 100.0, size=depth_raw.shape)
    depth_raw = depth_raw * noise

    return CorruptionResult(depth_raw=depth_raw, artifact_mask=artifact_mask)
