"""Mesh-based 3D rendering for the point-cloud GIFs.

A scattered point cloud from a single camera view is a thin wedge fanning
out from the camera origin: it has almost no extent along the one axis
(vertical) that would make a rotation look like a solid object, and a
sparse dot scatter reads as noise rather than a surface at any angle. This
was confirmed directly on a real scene (Motorcycle): X/Y/Z percentile
spread of roughly 3.0 / 1.5 / 4.4, with the smallest axis being the one
that would show "volume" under rotation, and a full 360-degree sweep
across that scene showed no angle where the scatter looked like anything
but a flattened streak or, at 90 degrees, a squashed bar.

The fix used throughout the rest of this project for a much older problem,
model-agnostic post-hoc depth refinement, is worth reusing here too:
connect the depth grid into a real triangulated surface (each pixel joined
to its immediate neighbors), and cull any triangle that bridges a real
depth discontinuity. An uncalled bridging triangle is the mesh-rendering
equivalent of a fly-point: a fake surface connecting foreground to
background. Culling it produces a genuinely legible relief surface with
real gaps at object boundaries, the same "parallax depth photo" look used
by consumer depth-based 3D photo effects, rather than a spray of dust.
"""

from __future__ import annotations

import io

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from pointsnap.geometry.backproject import backproject
from pointsnap.geometry.intrinsics import Intrinsics
from pointsnap.viz.style import FIGURE_DPI


def _decimate(depth: np.ndarray, rgb: np.ndarray, intrinsics: Intrinsics, max_dim: int):
    h, w = depth.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return depth, rgb, intrinsics
    new_w, new_h = max(2, round(w * scale)), max(2, round(h * scale))
    # nearest-neighbor for depth: averaging would blend real discontinuities
    # into fake intermediate values, recreating exactly the bridging
    # artifact this project exists to remove
    depth_small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_intrinsics = Intrinsics.exact(
        fx=intrinsics.fx * scale, fy=intrinsics.fy * scale,
        cx=intrinsics.cx * scale, cy=intrinsics.cy * scale,
        width=new_w, height=new_h,
    )
    return depth_small, rgb_small, scaled_intrinsics


def _rank_remap(depth: np.ndarray, nominal_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (remapped_depth, rank_fraction), both (H, W). Rank position,
    not raw value, is what gets remapped: monocular depth's own scale is
    often wildly non-uniform (one real case: 99% of a scene's depth
    compressed into 5% of its own value range), and a linear remap
    preserves that skew while a rank remap does not."""
    flat = depth.ravel()
    ranks = np.argsort(np.argsort(flat)).astype(np.float64)
    rank_fraction = (ranks / max(depth.size - 1, 1)).reshape(depth.shape)
    remapped = nominal_range[0] + rank_fraction * (nominal_range[1] - nominal_range[0])
    return remapped, rank_fraction


def _quad_triangle_indices(h: int, w: int) -> np.ndarray:
    """Two triangles per 2x2 pixel quad, vectorized. Returns (N, 3) vertex
    index triples into a flattened (h*w) vertex array."""
    r, c = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing="ij")
    tl = r * w + c
    tr = r * w + (c + 1)
    bl = (r + 1) * w + c
    br = (r + 1) * w + (c + 1)
    tri_a = np.stack([tl, tr, bl], axis=-1).reshape(-1, 3)
    tri_b = np.stack([tr, br, bl], axis=-1).reshape(-1, 3)
    return np.concatenate([tri_a, tri_b], axis=0)


def build_mesh(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    max_dim: int = 180,
    nominal_range: tuple[float, float] = (0.5, 5.0),
    cull_rank_fraction: float = 0.05,
):
    """Builds a culled triangle mesh from a depth map: vertices in the
    plotted (X, Z, -Y) convention used throughout viz/, matching face
    colors, and the rank-fraction span used for culling. A triangle is
    dropped whenever its three vertices span more than `cull_rank_fraction`
    of the image's own depth-rank distribution, the mesh equivalent of a
    fly-point bridging two real surfaces.
    """
    depth_small, rgb_small, intr = _decimate(depth, rgb, intrinsics, max_dim)
    remapped, rank_fraction = _rank_remap(depth_small, nominal_range)

    points = backproject(remapped, intr)  # (h, w, 3): X, Y, Z
    h, w = remapped.shape
    verts_plot = np.stack([points[..., 0], points[..., 2], -points[..., 1]], axis=-1).reshape(-1, 3)
    colors_flat = (rgb_small.reshape(-1, 3).astype(np.float64) / 255.0)
    rank_flat = rank_fraction.ravel()

    tris = _quad_triangle_indices(h, w)
    span = rank_flat[tris].max(axis=1) - rank_flat[tris].min(axis=1)
    keep = span <= cull_rank_fraction
    tris = tris[keep]

    face_verts = verts_plot[tris]  # (M, 3, 3)
    face_colors = colors_flat[tris].mean(axis=1)  # (M, 3)
    return face_verts, face_colors, verts_plot[np.unique(tris)]


def _robust_cubic_limits(points: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0):
    lo = np.percentile(points, low_pct, axis=0)
    hi = np.percentile(points, high_pct, axis=0)
    center = (lo + hi) / 2
    half_range = max(float(np.max(hi - lo)) / 2, 1e-6) * 1.1
    xlim = (center[0] - half_range, center[0] + half_range)
    ylim = (center[1] - half_range, center[1] + half_range)
    zlim = (center[2] - half_range, center[2] + half_range)
    return xlim, ylim, zlim


def render_mesh_frame(
    face_verts: np.ndarray,
    face_colors: np.ndarray,
    limits,
    elev: float,
    azim: float,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = FIGURE_DPI,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(face_verts, facecolors=face_colors, edgecolors=face_colors, linewidths=0.1)
    ax.add_collection3d(collection)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    xlim, ylim, zlim = limits
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_rotating_mesh_gif(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    out_path: str,
    n_frames: int = 24,
    elev: float = 15.0,
    azim_center: float = 40.0,
    azim_sweep: float = 25.0,
    max_dim: int = 180,
    cull_rank_fraction: float = 0.05,
    duration: float = 0.08,
) -> str:
    face_verts, face_colors, all_verts = build_mesh(
        depth, rgb, intrinsics, max_dim=max_dim, cull_rank_fraction=cull_rank_fraction
    )
    limits = _robust_cubic_limits(all_verts)
    angles = azim_center + azim_sweep * np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
    frames = [
        render_mesh_frame(face_verts, face_colors, limits, elev=elev, azim=float(a)) for a in angles
    ]
    imageio.mimsave(out_path, frames, duration=duration, loop=0)
    return out_path
