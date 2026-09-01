import numpy as np

from pointsnap.detect.ensemble import (
    classical_score_to_prob,
    ensemble_probability,
    noisy_or,
    weighted_blend,
)


def test_noisy_or_is_certain_if_either_input_is_certain():
    p_a = np.array([1.0, 0.3])
    p_b = np.array([0.2, 1.0])
    result = noisy_or(p_a, p_b)
    assert np.allclose(result, [1.0, 1.0])


def test_noisy_or_is_zero_only_if_both_inputs_are_zero():
    p_a = np.array([0.0, 0.0, 0.4])
    p_b = np.array([0.0, 0.5, 0.0])
    result = noisy_or(p_a, p_b)
    assert result[0] == 0.0
    assert result[1] == 0.5
    assert result[2] == 0.4


def test_noisy_or_is_monotonically_nondecreasing_in_each_input():
    rng = np.random.default_rng(0)
    p_a = rng.uniform(0, 1, size=50)
    p_b = rng.uniform(0, 1, size=50)
    baseline = noisy_or(p_a, p_b)
    bumped = noisy_or(np.clip(p_a + 0.1, 0, 1), p_b)
    assert np.all(bumped >= baseline - 1e-12)


def test_weighted_blend_matches_endpoints():
    p_a = np.array([0.2, 0.8])
    p_b = np.array([0.9, 0.1])
    assert np.allclose(weighted_blend(p_a, p_b, weight_b=0.0), p_a)
    assert np.allclose(weighted_blend(p_a, p_b, weight_b=1.0), p_b)


def test_classical_score_to_prob_is_bounded_and_monotonic():
    z = np.linspace(-10, 10, 100)
    p = classical_score_to_prob(z)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= 0)


def test_ensemble_probability_dispatches_by_mode():
    z = np.array([2.5, 2.5])
    cnn = np.array([0.5, 0.9])
    noisy = ensemble_probability(z, cnn, mode="noisy_or")
    weighted = ensemble_probability(z, cnn, mode="weighted", weight_b=1.0)
    assert np.allclose(weighted, cnn)
    assert not np.allclose(noisy, weighted)
