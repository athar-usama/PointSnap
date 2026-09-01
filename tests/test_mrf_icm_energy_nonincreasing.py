import numpy as np

from pointsnap.refine.mrf_icm import data_term, relax_labels


def _toy_problem(seed, size=24, adversarial_init=True):
    rng = np.random.default_rng(seed)
    h = w = size

    # a vertical silhouette: left half is "near" (larger inverse depth), right is "far"
    gray = np.zeros((h, w))
    gray[:, : w // 2] = 0.2
    gray[:, w // 2 :] = 0.8
    gray += rng.normal(scale=0.02, size=(h, w))  # texture noise, still a clean edge

    w_raw = np.where(np.arange(w)[None, :] < w // 2, 0.9, 0.1) * np.ones((h, w))
    w_raw += rng.normal(scale=0.01, size=(h, w))

    fg_value = np.full((h, w), 0.9)
    bg_value = np.full((h, w), 0.1)
    fg_rmse = np.full((h, w), 0.01)
    bg_rmse = np.full((h, w), 0.01)

    candidate_mask = np.zeros((h, w), dtype=bool)
    band = slice(w // 2 - 3, w // 2 + 3)
    candidate_mask[:, band] = True
    valid = candidate_mask.copy()

    if adversarial_init:
        # deliberately wrong: candidates start entirely mislabeled
        initial_label = np.zeros((h, w), dtype=bool)
        initial_label[:, : w // 2] = False  # should be True (FG/near)
        initial_label[:, w // 2 :] = True  # should be False (BG/far)
    else:
        initial_label = w_raw > 0.5

    return {
        "w_raw": w_raw, "gray": gray, "candidate_mask": candidate_mask,
        "fg_value": fg_value, "bg_value": bg_value,
        "fg_rmse": fg_rmse, "bg_rmse": bg_rmse, "valid": valid, "initial_label": initial_label,
    }


def test_energy_never_increases_across_iterations():
    for seed in range(10):
        problem = _toy_problem(seed)
        result = relax_labels(**problem, max_iters=40)
        trace = np.array(result.energy_trace)
        diffs = np.diff(trace)
        assert np.all(diffs <= 1e-9), f"seed={seed} energy increased: {trace}"


def test_relaxation_converges_to_the_correct_silhouette_from_adversarial_init():
    problem = _toy_problem(seed=1, adversarial_init=True)
    result = relax_labels(**problem, max_iters=100, lam=2.0)
    w = problem["w_raw"].shape[1]
    # after relaxation, the left half of the candidate band should be FG (True),
    # the right half BG (False) -- i.e. it recovered the true silhouette despite
    # starting fully inverted
    band_labels = result.labels[:, w // 2 - 3 : w // 2 + 3]
    left_half = band_labels[:, :3]
    right_half = band_labels[:, 3:]
    assert left_half.mean() > 0.9
    assert right_half.mean() < 0.1


def test_confident_pixels_are_never_relabeled():
    problem = _toy_problem(seed=2)
    original_confident = problem["initial_label"][~problem["candidate_mask"]].copy()
    result = relax_labels(**problem, max_iters=40)
    assert np.array_equal(result.labels[~problem["candidate_mask"]], original_confident)


def test_invalid_candidates_are_excluded_from_energy_and_never_updated():
    problem = _toy_problem(seed=3)
    # mark half the candidate band as invalid (no plane hypothesis available)
    problem["valid"] = problem["candidate_mask"].copy()
    problem["valid"][:, : problem["w_raw"].shape[1] // 2] = False
    original = problem["initial_label"].copy()
    result = relax_labels(**problem, max_iters=40)
    invalid_region = problem["candidate_mask"] & ~problem["valid"]
    assert np.array_equal(result.labels[invalid_region], original[invalid_region])


def test_data_term_is_infinite_wherever_invalid():
    h, w = 5, 5
    valid = np.zeros((h, w), dtype=bool)
    valid[2, 2] = True
    d_fg, d_bg = data_term(
        w_raw=np.zeros((h, w)), fg_value=np.zeros((h, w)), bg_value=np.zeros((h, w)),
        fg_rmse=np.zeros((h, w)), bg_rmse=np.zeros((h, w)), valid=valid, sigma_d=1.0, beta=1.0, sigma_r=1.0,
    )
    assert np.isinf(d_fg[0, 0]) and np.isinf(d_bg[0, 0])
    assert np.isfinite(d_fg[2, 2]) and np.isfinite(d_bg[2, 2])
