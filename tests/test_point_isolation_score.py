import numpy as np

from pointsnap.eval.boundary_metrics import point_isolation_scores, summarize_point_isolation
from pointsnap.geometry.intrinsics import Intrinsics


def test_hand_placed_flying_point_scores_high_pis():
    size = 30
    depth = np.full((size, size), 3.0)
    depth[15, 15] = 0.1  # ejected far from its neighborhood in 3D

    boundary_mask = np.zeros((size, size), dtype=bool)
    boundary_mask[15, 15] = True
    boundary_mask[5, 5] = True  # a normal, non-anomalous point for comparison

    k_intr = Intrinsics.rule_of_thumb(width=size, height=size)
    scores = point_isolation_scores(depth, boundary_mask, k_intr, k=5, window_radius=8)

    coords = list(zip(*np.where(boundary_mask)))
    idx_fly = coords.index((15, 15))
    idx_normal = coords.index((5, 5))

    assert scores[idx_fly] > 5.0
    assert scores[idx_normal] < 1.5


def test_summarize_point_isolation_reports_median_and_rate():
    scores = np.array([1.0, 1.1, 0.9, 3.0, 5.0])
    summary = summarize_point_isolation(scores, isolation_threshold=2.0)
    assert summary["n"] == 5
    assert summary["isolation_rate"] == 2 / 5
    assert 0.9 <= summary["median_pis"] <= 1.1


def test_summarize_point_isolation_handles_all_nan():
    scores = np.array([np.nan, np.nan])
    summary = summarize_point_isolation(scores)
    assert summary["n"] == 0
    assert np.isnan(summary["median_pis"])
