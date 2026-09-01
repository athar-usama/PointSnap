import numpy as np

from pointsnap.refine.ransac_biplanar import fit_biplanar_field


def _two_plane_scene(size=60, band=3, seed=0):
    """A clean vertical silhouette: left half at inverse-depth 0.9, right at
    0.1, with a thin band of candidate pixels straddling the seam."""
    rng = np.random.default_rng(seed)
    inv_depth = np.where(np.arange(size)[None, :] < size // 2, 0.9, 0.1) * np.ones((size, size))
    inv_depth += rng.normal(scale=0.005, size=(size, size))

    candidate_mask = np.zeros((size, size), dtype=bool)
    candidate_mask[:, size // 2 - band : size // 2 + band] = True
    return inv_depth, candidate_mask


def test_recovers_correct_fg_bg_values_for_a_clean_silhouette():
    inv_depth, candidate_mask = _two_plane_scene()
    field = fit_biplanar_field(inv_depth, candidate_mask)

    assert field.valid[candidate_mask].mean() > 0.9
    # near/far assignment: fg (near) should be close to 0.9, bg (far) to 0.1
    valid_candidates = candidate_mask & field.valid
    assert np.allclose(field.fg_value[valid_candidates], 0.9, atol=0.05)
    assert np.allclose(field.bg_value[valid_candidates], 0.1, atol=0.05)


def test_confident_side_is_tagged_for_neighbor_training_points():
    inv_depth, candidate_mask = _two_plane_scene()
    field = fit_biplanar_field(inv_depth, candidate_mask)

    tagged = field.confident_side != 0
    assert tagged.sum() > 0
    # every tagged pixel must be a confident (non-candidate) pixel
    assert not np.any(tagged & candidate_mask)


def test_many_scattered_tiny_components_all_get_processed():
    """Regression test for the local-crop optimization: a mask fragmented
    into many small, disjoint components (as a noisy real-world ensemble
    candidate mask often is) must still be handled -- one hypothesis per
    component, none silently dropped."""
    size = 80
    inv_depth, _ = _two_plane_scene(size=size, band=6, seed=1)

    # fragment the band into disjoint single-pixel specks (a checkerboard
    # subset of the band, so no two candidates are 8-connected) instead of
    # one contiguous block, mimicking a scattered ensemble mask
    candidate_mask = np.zeros((size, size), dtype=bool)
    rows = np.arange(0, size, 3)
    cols = np.arange(size // 2 - 10, size // 2 + 10, 3)
    for r in rows:
        for c in cols:
            candidate_mask[r, c] = True

    field = fit_biplanar_field(inv_depth, candidate_mask)
    n_components_seen = len(field.hypotheses)
    assert n_components_seen >= 100  # every scattered speck must get its own hypothesis
    assert field.valid[candidate_mask].mean() > 0.5


def test_unimodal_window_yields_no_valid_hypothesis():
    size = 40
    rng = np.random.default_rng(2)
    inv_depth = 0.5 + rng.normal(scale=0.01, size=(size, size))  # a single flat surface
    candidate_mask = np.zeros((size, size), dtype=bool)
    candidate_mask[15:25, 15:25] = True

    field = fit_biplanar_field(inv_depth, candidate_mask)
    assert field.valid[candidate_mask].sum() == 0
