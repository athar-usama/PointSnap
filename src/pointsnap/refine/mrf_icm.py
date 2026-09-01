"""RGB-edge-aware relaxation labeling via checkerboard-parity ICM.

Binary MRF over candidate pixels only (label FG=True / BG=False). Confident
(non-candidate) pixels are clamped to an externally supplied effective label
and are never updated, but still contribute to the smoothness term.

Energy:
    E(l) = sum_p D_p(l_p) + lambda * sum_{(p,q) in N} V_pq(l_p, l_q)

    D_p(k) = (w_raw(p)-w_k(p))^2 / (2*sigma_d^2) + beta*rmse_k(p)/sigma_r   [+inf if hypothesis invalid]
    V_pq   = 1[l_p != l_q] * exp(-(I_p-I_q)^2 / (2*sigma_c^2))    (contrast-sensitive Potts, Boykov & Jolly 2001)

Solved with a checkerboard 2-coloring of the pixel grid: on each half-step
only one color class is updated, using the *other* class's labels held
fixed. Because simultaneously-updated pixels are never 4-adjacent to each
other, this is exactly sequential ICM restricted to an independent set,
which provably cannot increase E on that half-step (verified directly in
tests/test_mrf_icm_energy_nonincreasing.py) -- unlike a naive fully
synchronous update over the whole field, which can oscillate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FG = True
BG = False


def data_term(
    w_raw: np.ndarray,
    fg_value: np.ndarray,
    bg_value: np.ndarray,
    fg_rmse: np.ndarray,
    bg_rmse: np.ndarray,
    valid: np.ndarray,
    sigma_d: float,
    beta: float,
    sigma_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    d_fg = (w_raw - fg_value) ** 2 / (2 * sigma_d**2) + beta * np.clip(fg_rmse, 0, None) / sigma_r
    d_bg = (w_raw - bg_value) ** 2 / (2 * sigma_d**2) + beta * np.clip(bg_rmse, 0, None) / sigma_r
    d_fg = np.where(valid, d_fg, np.inf)
    d_bg = np.where(valid, d_bg, np.inf)
    return d_fg, d_bg


def _pairwise_weights(gray: np.ndarray, sigma_c: float):
    padded = np.pad(gray, 1, mode="edge")
    w_up = np.exp(-((gray - padded[:-2, 1:-1]) ** 2) / (2 * sigma_c**2))
    w_down = np.exp(-((gray - padded[2:, 1:-1]) ** 2) / (2 * sigma_c**2))
    w_left = np.exp(-((gray - padded[1:-1, :-2]) ** 2) / (2 * sigma_c**2))
    w_right = np.exp(-((gray - padded[1:-1, 2:]) ** 2) / (2 * sigma_c**2))
    return w_up, w_down, w_left, w_right


def _neighbor_labels(label: np.ndarray):
    padded = np.pad(label, 1, mode="edge")
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return up, down, left, right


def total_energy(
    label: np.ndarray,
    d_fg: np.ndarray,
    d_bg: np.ndarray,
    valid: np.ndarray,
    gray: np.ndarray,
    sigma_c: float,
    lam: float,
) -> float:
    """The actual global objective E(l): each edge counted exactly once
    (down + right only), each data term counted exactly once (at valid
    positions only) -- this is what the ICM update is proven not to
    increase, and what test_mrf_icm_energy_nonincreasing.py checks directly.
    """
    d_eff = np.where(label, d_fg, d_bg)
    data_energy = float(np.sum(d_eff[valid]))

    diff_down = gray[:-1, :] - gray[1:, :]
    w_down = np.exp(-(diff_down**2) / (2 * sigma_c**2))
    mismatch_down = (label[:-1, :] != label[1:, :]).astype(np.float64)

    diff_right = gray[:, :-1] - gray[:, 1:]
    w_right = np.exp(-(diff_right**2) / (2 * sigma_c**2))
    mismatch_right = (label[:, :-1] != label[:, 1:]).astype(np.float64)

    smooth_energy = float(np.sum(w_down * mismatch_down) + np.sum(w_right * mismatch_right))
    return data_energy + lam * smooth_energy


@dataclass
class MRFResult:
    labels: np.ndarray
    energy_trace: list[float]
    n_iterations: int


def relax_labels(
    w_raw: np.ndarray,
    gray: np.ndarray,
    candidate_mask: np.ndarray,
    fg_value: np.ndarray,
    bg_value: np.ndarray,
    fg_rmse: np.ndarray,
    bg_rmse: np.ndarray,
    valid: np.ndarray,
    initial_label: np.ndarray,
    sigma_d: float = 0.05,
    sigma_c: float = 0.15,
    sigma_r: float = 1.0,
    beta: float = 1.0,
    lam: float = 1.0,
    max_iters: int = 50,
    tol_frac: float = 0.001,
) -> MRFResult:
    d_fg, d_bg = data_term(w_raw, fg_value, bg_value, fg_rmse, bg_rmse, valid, sigma_d, beta, sigma_r)
    w_up, w_down, w_left, w_right = _pairwise_weights(gray, sigma_c)

    label = initial_label.copy()
    updatable = candidate_mask & valid
    n_updatable = int(updatable.sum())

    u_idx, v_idx = np.meshgrid(np.arange(label.shape[1]), np.arange(label.shape[0]))
    parity_map = (u_idx + v_idx) % 2

    energy_trace = [total_energy(label, d_fg, d_bg, valid, gray, sigma_c, lam)]
    iterations_run = 0

    for t in range(max_iters):
        iterations_run = t + 1
        up, down, left, right = _neighbor_labels(label)
        cost_fg = d_fg + lam * (
            w_up * (~up) + w_down * (~down) + w_left * (~left) + w_right * (~right)
        )
        cost_bg = d_bg + lam * (
            w_up * up + w_down * down + w_left * left + w_right * right
        )
        new_label = cost_fg < cost_bg

        mask = updatable & (parity_map == (t % 2))
        n_flipped = int(np.sum(new_label[mask] != label[mask]))
        label = label.copy()
        label[mask] = new_label[mask]

        energy_trace.append(total_energy(label, d_fg, d_bg, valid, gray, sigma_c, lam))

        if n_updatable == 0 or n_flipped <= max(0, int(tol_frac * n_updatable)):
            break

    return MRFResult(labels=label, energy_trace=energy_trace, n_iterations=iterations_run)
