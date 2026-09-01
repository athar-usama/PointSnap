import numpy as np

from pointsnap.geometry.planes import (
    eval_plane,
    is_bimodal,
    kmeans_1d,
    ransac_plane,
)


def test_ransac_recovers_known_plane_without_outliers():
    rng = np.random.default_rng(1)
    a_true, b_true, c_true = 0.01, -0.02, 0.4
    u = rng.uniform(0, 200, size=200)
    v = rng.uniform(0, 150, size=200)
    w = a_true * u + b_true * v + c_true

    fit = ransac_plane(u, v, w, rng=rng)
    assert fit is not None
    assert np.allclose(fit.coeffs, [a_true, b_true, c_true], atol=1e-6)
    assert fit.n_inliers == 200
    assert fit.rmse < 1e-8


def test_ransac_is_robust_to_a_minority_of_far_side_points():
    rng = np.random.default_rng(2)
    a_true, b_true, c_true = 0.005, 0.01, 0.5
    n_inlier = 180
    u_in = rng.uniform(0, 200, size=n_inlier)
    v_in = rng.uniform(0, 150, size=n_inlier)
    w_in = a_true * u_in + b_true * v_in + c_true

    # a different plane's points contaminating the same local window
    n_outlier = 20
    u_out = rng.uniform(0, 200, size=n_outlier)
    v_out = rng.uniform(0, 150, size=n_outlier)
    w_out = 0.2 * u_out + 0.3 * v_out + 5.0

    u = np.concatenate([u_in, u_out])
    v = np.concatenate([v_in, v_out])
    w = np.concatenate([w_in, w_out])

    fit = ransac_plane(u, v, w, rng=rng)
    assert fit is not None
    assert np.allclose(fit.coeffs, [a_true, b_true, c_true], atol=1e-4)
    assert fit.n_inliers == n_inlier


def test_ransac_returns_none_for_degenerate_input():
    u = np.array([1.0, 2.0])
    v = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0])
    assert ransac_plane(u, v, w) is None


def test_eval_plane_matches_definition():
    coeffs = np.array([2.0, -3.0, 1.0])
    u = np.array([1.0, 4.0])
    v = np.array([2.0, 0.5])
    result = eval_plane(coeffs, u, v)
    assert np.allclose(result, [2 * 1 - 3 * 2 + 1, 2 * 4 - 3 * 0.5 + 1])


def test_kmeans_1d_separates_two_well_separated_clusters():
    values = np.concatenate([np.full(10, 1.0), np.full(10, 5.0)])
    labels, centers = kmeans_1d(values, k=2)
    assert set(np.unique(labels)) == {0, 1}
    assert np.allclose(sorted(centers), [1.0, 5.0])


def test_is_bimodal_true_for_two_separated_clusters():
    values = np.concatenate([np.full(20, 0.2), np.full(20, 0.8)])
    assert bool(is_bimodal(values)) is True


def test_is_bimodal_false_for_a_single_noisy_cluster():
    # Monte Carlo calibrated (see planes.is_bimodal docstring): at n=40 a
    # unimodal Gaussian window never exceeds the ratio-5.0 threshold, so this
    # is a stable check, not a lucky seed.
    for seed in range(20):
        rng = np.random.default_rng(seed)
        values = 0.5 + rng.normal(scale=0.01, size=40)
        assert bool(is_bimodal(values)) is False


def test_is_bimodal_false_for_too_few_points():
    assert bool(is_bimodal(np.array([0.1, 0.9, 0.1]))) is False
