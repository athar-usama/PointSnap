import numpy as np

from pointsnap.geometry.backproject import backproject, project, to_point_cloud
from pointsnap.geometry.intrinsics import Intrinsics


def test_backproject_project_roundtrip():
    k = Intrinsics.exact(fx=500.0, fy=480.0, cx=64.0, cy=48.0, width=128, height=96)
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 5.0, size=(96, 128))

    points = backproject(depth, k)
    u, v, z = project(points, k)

    u_grid, v_grid = np.meshgrid(np.arange(128, dtype=np.float64), np.arange(96, dtype=np.float64))
    assert np.allclose(u, u_grid, atol=1e-8)
    assert np.allclose(v, v_grid, atol=1e-8)
    assert np.allclose(z, depth, atol=1e-8)


def test_to_point_cloud_shapes_and_masking():
    k = Intrinsics.rule_of_thumb(width=8, height=6)
    depth = np.full((6, 8), 2.0)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    points, colors = to_point_cloud(depth, k, rgb=rgb)
    assert points.shape == (48, 3)
    assert colors.shape == (48, 3)
    assert np.all(colors[:, 0] == 255)

    mask = np.zeros((6, 8), dtype=bool)
    mask[0, 0] = True
    mask[2, 3] = True
    points_masked, colors_masked = to_point_cloud(depth, k, rgb=rgb, mask=mask)
    assert points_masked.shape == (2, 3)
    assert colors_masked.shape == (2, 3)


def test_rule_of_thumb_intrinsics_centered():
    k = Intrinsics.rule_of_thumb(width=100, height=50)
    assert k.cx == 50.0
    assert k.cy == 25.0
    assert k.fx == k.fy == 120.0


def test_exif_focal_length_scales_with_width():
    k_narrow = Intrinsics.from_exif_focal_35mm(focal_35mm=50.0, width=1000, height=750)
    k_wide = Intrinsics.from_exif_focal_35mm(focal_35mm=50.0, width=2000, height=1500)
    assert k_wide.fx == 2 * k_narrow.fx
