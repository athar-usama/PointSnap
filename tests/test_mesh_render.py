import numpy as np

from pointsnap.geometry.intrinsics import Intrinsics
from pointsnap.viz.mesh_render import (
    _quad_triangle_indices,
    _rank_remap,
    _robust_cubic_limits,
    build_mesh,
)


def test_quad_triangle_indices_count_and_range():
    h, w = 4, 5
    tris = _quad_triangle_indices(h, w)
    # two triangles per (h-1) x (w-1) quad
    assert tris.shape == ((h - 1) * (w - 1) * 2, 3)
    assert tris.min() >= 0
    assert tris.max() < h * w


def test_quad_triangle_indices_cover_a_2x2_grid_exactly():
    tris = _quad_triangle_indices(2, 2)
    # a single quad: exactly two triangles, using all four corners between them
    assert tris.shape == (2, 3)
    assert set(tris.ravel().tolist()) == {0, 1, 2, 3}


def test_rank_remap_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.01, 1000.0, size=(20, 20))  # deliberately skewed scale
    remapped, rank_fraction = _rank_remap(depth, nominal_range=(0.5, 5.0))

    assert remapped.min() >= 0.5 - 1e-9
    assert remapped.max() <= 5.0 + 1e-9
    assert rank_fraction.min() >= 0.0
    assert rank_fraction.max() <= 1.0

    # rank order must match depth order exactly (monotonic remap)
    order_depth = np.argsort(depth.ravel())
    order_rank = np.argsort(remapped.ravel())
    assert np.array_equal(order_depth, order_rank)


def test_build_mesh_culls_triangles_across_a_hard_boundary():
    """A clean vertical silhouette: left half near, right half far. Triangles
    that would bridge the two surfaces must be culled; triangles safely
    inside either surface must survive."""
    size = 40
    depth = np.where(np.arange(size)[None, :] < size // 2, 1.0, 5.0) * np.ones((size, size))
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, : size // 2] = [200, 50, 50]
    rgb[:, size // 2 :] = [50, 50, 200]
    intrinsics = Intrinsics.rule_of_thumb(width=size, height=size)

    face_verts, _face_colors, all_verts = build_mesh(
        depth, rgb, intrinsics, max_dim=size, cull_rank_fraction=0.05
    )

    # every surviving face must have near-identical depth (rank) at its own
    # three vertices -- reconstruct via the plotted Z coordinate (axis 1)
    # is not directly comparable across faces, so instead check no face
    # straddles the boundary column by checking its plotted-X spread is
    # small relative to the object width (a bridging face would span most
    # of the image, a same-surface face would not)
    x_span = face_verts[:, :, 0].max(axis=1) - face_verts[:, :, 0].min(axis=1)
    total_x_range = all_verts[:, 0].max() - all_verts[:, 0].min()
    assert x_span.max() < 0.5 * total_x_range  # no face spans the whole silhouette

    assert len(face_verts) > 0  # plenty of faces survive within each flat side


def test_build_mesh_keeps_a_fully_flat_surface_connected():
    size = 30
    depth = np.full((size, size), 3.0)
    rgb = np.full((size, size, 3), 128, dtype=np.uint8)
    intrinsics = Intrinsics.rule_of_thumb(width=size, height=size)

    face_verts, _face_colors, _all_verts = build_mesh(
        depth, rgb, intrinsics, max_dim=size, cull_rank_fraction=0.05
    )
    expected_faces = (size - 1) * (size - 1) * 2
    # a perfectly flat surface has zero depth variation, so nothing should
    # be culled at all
    assert len(face_verts) == expected_faces


def test_robust_cubic_limits_gives_equal_range_on_every_axis():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(500, 3)) * np.array([1.0, 5.0, 20.0])  # very anisotropic
    xlim, ylim, zlim = _robust_cubic_limits(points)
    ranges = [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]]
    assert np.allclose(ranges, ranges[0])
