"""End-to-end refinement pipeline: candidate detection -> bi-planar RANSAC ->
RGB-edge-aware MRF relabeling -> sub-pixel bilateral snap.

Model-agnostic: takes any monocular depth model's raw output as `depth_raw`
and refines it directly, with no dependency on which backbone produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pointsnap.detect.classical import classical_candidate_score, inverse_depth, to_gray
from pointsnap.detect.ensemble import classical_score_to_prob, ensemble_probability
from pointsnap.refine.bilateral_snap import joint_bilateral_snap
from pointsnap.refine.mrf_icm import MRFResult, relax_labels
from pointsnap.refine.ransac_biplanar import fit_biplanar_field


@dataclass
class RefinementResult:
    depth_refined: np.ndarray
    candidate_mask: np.ndarray
    label: np.ndarray
    energy_trace: list[float]
    n_iterations: int
    fraction_candidates: float


def refine_depth(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    cnn_prob: np.ndarray | None = None,
    candidate_threshold: float = 0.5,
    margin: int = 20,
    apply_bilateral_snap: bool = True,
    mrf_kwargs: dict | None = None,
) -> RefinementResult:
    mrf_kwargs = dict(mrf_kwargs or {})
    gray = to_gray(rgb) / 255.0
    w_raw = inverse_depth(depth_raw)

    classical = classical_candidate_score(rgb, depth_raw)
    if cnn_prob is not None:
        prob = ensemble_probability(classical.zscore, cnn_prob, mode="noisy_or")
    else:
        prob = classical_score_to_prob(classical.zscore)
    candidate_mask = prob > candidate_threshold

    field = fit_biplanar_field(w_raw, candidate_mask, margin=margin)

    initial_label = np.zeros_like(candidate_mask)
    initial_label[field.confident_side == 1] = True
    initial_label[field.confident_side == -1] = False
    valid_candidates = candidate_mask & field.valid
    dist_fg = np.abs(w_raw - field.fg_value)
    dist_bg = np.abs(w_raw - field.bg_value)
    initial_label[valid_candidates] = dist_fg[valid_candidates] < dist_bg[valid_candidates]

    mrf_result: MRFResult = relax_labels(
        w_raw=w_raw,
        gray=gray,
        candidate_mask=candidate_mask,
        fg_value=field.fg_value,
        bg_value=field.bg_value,
        fg_rmse=field.fg_rmse,
        bg_rmse=field.bg_rmse,
        valid=field.valid,
        initial_label=initial_label,
        **mrf_kwargs,
    )

    w_refined = w_raw.copy()
    w_refined[valid_candidates] = np.where(
        mrf_result.labels[valid_candidates],
        field.fg_value[valid_candidates],
        field.bg_value[valid_candidates],
    )
    depth_refined = 1.0 / np.clip(w_refined, 1e-8, None)

    if apply_bilateral_snap:
        depth_refined = joint_bilateral_snap(depth_refined, gray, mrf_result.labels)

    return RefinementResult(
        depth_refined=depth_refined,
        candidate_mask=candidate_mask,
        label=mrf_result.labels,
        energy_trace=mrf_result.energy_trace,
        n_iterations=mrf_result.n_iterations,
        fraction_candidates=float(candidate_mask.mean()),
    )
