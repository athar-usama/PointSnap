import numpy as np

from pointsnap.synth.corruption import SEVERITY_PARAMS, corrupt_depth
from pointsnap.synth.scene import generate_scene


def test_scene_has_consistent_instance_and_depth_labeling():
    scene = generate_scene(seed=1, size=96)
    ids = np.unique(scene.instance_mask)
    assert ids[0] == 0  # background always present
    # every instance id maps to exactly one depth value (a flat synthetic plane)
    for i in ids:
        depths_for_id = np.unique(scene.depth_gt[scene.instance_mask == i])
        assert len(depths_for_id) == 1


def test_foreground_shapes_are_strictly_nearer_than_background():
    scene = generate_scene(seed=2, size=96)
    ids = np.unique(scene.instance_mask)
    bg_depth = scene.depth_gt[scene.instance_mask == 0][0]
    for i in ids[ids != 0]:
        fg_depth = scene.depth_gt[scene.instance_mask == i][0]
        assert fg_depth < bg_depth


def test_boundary_mask_is_nonempty_and_thin():
    scene = generate_scene(seed=3, size=96)
    assert scene.boundary_mask.sum() > 0
    # boundary should be a small fraction of the image (a contour, not a region)
    assert scene.boundary_mask.mean() < 0.15


def test_rgb_is_smoothly_antialiased_while_depth_gt_is_crisp():
    scene = generate_scene(seed=4, size=96, supersample=4)
    # depth_gt must take on exactly len(unique instance ids) distinct values,
    # no blending, unlike the RGB channel which is downsampled with area averaging
    n_ids = len(np.unique(scene.instance_mask))
    n_depth_values = len(np.unique(scene.depth_gt))
    assert n_depth_values == n_ids


def test_corruption_artifact_mask_concentrates_near_boundary():
    scene = generate_scene(seed=5, size=128)
    rng = np.random.default_rng(0)
    result = corrupt_depth(scene.depth_gt, scene.instance_mask, scene.boundary_mask, "hard", rng)

    dilated_boundary = scene.boundary_mask.copy()
    from scipy.ndimage import binary_dilation

    band = binary_dilation(dilated_boundary, iterations=8)
    frac_in_band = result.artifact_mask[band].sum() / max(result.artifact_mask.sum(), 1)
    assert frac_in_band > 0.9


def test_corruption_does_not_touch_rgb():
    scene = generate_scene(seed=6, size=96)
    rgb_before = scene.rgb.copy()
    rng = np.random.default_rng(0)
    corrupt_depth(scene.depth_gt, scene.instance_mask, scene.boundary_mask, "moderate", rng)
    assert np.array_equal(rgb_before, scene.rgb)


def test_severity_ordering_increases_expected_artifact_rate():
    scene = generate_scene(seed=8, size=160)
    counts = {}
    for severity in SEVERITY_PARAMS:
        rng = np.random.default_rng(42)
        result = corrupt_depth(scene.depth_gt, scene.instance_mask, scene.boundary_mask, severity, rng)
        counts[severity] = result.artifact_mask.sum()
    assert counts["easy"] <= counts["moderate"] <= counts["hard"]
