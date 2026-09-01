"""Combines the classical detector's continuous z-score with the CNN's
predicted probability into a single fly-point confidence map.

Default combination is noisy-OR: either detector firing is enough to flag a
candidate, since each is expected to catch different failure modes (the
classical detector: bridging in regions the CNN wasn't trained to expect;
the CNN: subtle statistical patterns no hand-written rule captures). A
tunable weighted blend is kept as an alternative, grid-searched against
validation F1 as its own ablation row.
"""

from __future__ import annotations

import numpy as np


def classical_score_to_prob(zscore: np.ndarray, midpoint: float = 2.5, scale: float = 1.5) -> np.ndarray:
    """Maps the unbounded classical z-score to a (0, 1) probability via a
    logistic centered at the detector's own decision threshold, so it
    combines meaningfully with the CNN's probability output."""
    return 1.0 / (1.0 + np.exp(-(zscore - midpoint) / scale))


def noisy_or(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - p_a) * (1.0 - p_b)


def weighted_blend(p_a: np.ndarray, p_b: np.ndarray, weight_b: float) -> np.ndarray:
    return (1.0 - weight_b) * p_a + weight_b * p_b


def ensemble_probability(
    classical_zscore: np.ndarray,
    cnn_prob: np.ndarray,
    mode: str = "noisy_or",
    weight_b: float = 0.5,
) -> np.ndarray:
    p_classical = classical_score_to_prob(classical_zscore)
    if mode == "noisy_or":
        return noisy_or(p_classical, cnn_prob)
    if mode == "weighted":
        return weighted_blend(p_classical, cnn_prob, weight_b)
    raise ValueError(f"unknown ensemble mode: {mode}")
