import numpy as np

from pointsnap.eval.boundary_metrics import bridging_rate


def test_bridging_rate_zero_for_a_hard_single_pixel_step():
    size = 20
    depth = np.zeros((size, size))
    depth[:, : size // 2] = 1.0
    depth[:, size // 2 :] = 5.0

    boundary_mask = np.zeros((size, size), dtype=bool)
    boundary_mask[5:15, size // 2 - 1] = True

    br = bridging_rate(depth, boundary_mask, profile_halfwidth=3, gamma=0.8, smooth_sigma=0.0)
    assert br < 0.05


def test_bridging_rate_high_for_a_smooth_ramp_spanning_the_profile():
    size = 20
    ramp_width = 7
    start_col = size // 2 - ramp_width // 2
    depth = np.zeros((size, size))
    depth[:, :start_col] = 1.0
    depth[:, start_col + ramp_width :] = 5.0
    for i in range(ramp_width):
        depth[:, start_col + i] = 1.0 + (5.0 - 1.0) * i / (ramp_width - 1)

    boundary_mask = np.zeros((size, size), dtype=bool)
    boundary_mask[5:15, start_col + ramp_width // 2] = True

    br = bridging_rate(depth, boundary_mask, profile_halfwidth=3, gamma=0.8, smooth_sigma=0.0)
    assert br > 0.95


def test_bridging_rate_is_between_zero_and_one():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1, 5, size=(30, 30))
    boundary_mask = rng.uniform(size=(30, 30)) > 0.9
    br = bridging_rate(depth, boundary_mask)
    assert 0.0 <= br <= 1.0


def test_bridging_rate_zero_boundary_pixels_returns_zero():
    depth = np.ones((10, 10))
    boundary_mask = np.zeros((10, 10), dtype=bool)
    assert bridging_rate(depth, boundary_mask) == 0.0
