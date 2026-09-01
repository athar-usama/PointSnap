"""Point-cloud rendering via matplotlib's software 3D scatter -- deliberately
not trimesh/pyrender's OpenGL rasterizer, which has no reliable headless
backend on Windows (OSMesa/EGL both have limited support there, and the one
Windows-friendly pyrender backend needs an active window, defeating
"headless"). matplotlib's rasterizer works identically headless on any
platform, which is all a rotating turntable GIF actually needs.
"""

from __future__ import annotations

import io

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pointsnap.geometry.backproject import to_point_cloud
from pointsnap.geometry.intrinsics import Intrinsics
from pointsnap.viz.style import FIGURE_DPI


def render_point_cloud_frame(
    points: np.ndarray,
    colors: np.ndarray,
    elev: float = 15,
    azim: float = 0,
    point_size: float = 1.5,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = FIGURE_DPI,
    limits: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    # plot (X, Z, -Y): Z (depth-into-scene) as the horizontal "away" axis,
    # -Y (image-down negated) as the vertical axis -- azim then orbits
    # around the true vertical, giving a natural turntable rotation
    ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=colors / 255.0, s=point_size, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if limits is not None:
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


def _robust_cubic_limits(points: np.ndarray, low_pct: float, high_pct: float):
    """A single equal-range box around the percentile-clipped extent of the
    points on every axis -- a handful of far-background outliers (the sky
    behind foliage, a distant wall through a doorway) would otherwise blow
    out matplotlib's auto-scaled axes and shrink the actual subject to a
    few invisible pixels in a corner of the frame."""
    lo = np.percentile(points, low_pct, axis=0)
    hi = np.percentile(points, high_pct, axis=0)
    center = (lo + hi) / 2
    half_range = max(float(np.max(hi - lo)) / 2, 1e-6) * 1.1
    xlim = (center[0] - half_range, center[0] + half_range)
    ylim = (center[2] - half_range, center[2] + half_range)  # plotted as the "Z" axis
    zlim = (-center[1] - half_range, -center[1] + half_range)  # plotted as "-Y"
    return xlim, ylim, zlim


def render_rotating_gif(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    out_path: str,
    mask: np.ndarray | None = None,
    n_frames: int = 24,
    elev: float = 20,
    azim_center: float = 40.0,
    azim_sweep: float = 30.0,
    subsample: int = 3,
    duration: float = 0.08,
    nominal_range: tuple[float, float] = (0.5, 5.0),
) -> str:
    # A non-metric backbone's raw output is disparity-like (1/Z), so a
    # narrow range of small disparities near the far background inverts
    # into a hugely stretched depth tail -- most of a real scene ends up
    # compressed into a sliver near the camera while a handful of far
    # points drag the axis limits out (confirmed directly: one Middlebury
    # scene's depth had p1=0.11, p50=0.16, but p99=1.41 and max=2.63 -- 99%
    # of the scene living in 5% of its own depth range). A *linear*
    # percentile remap preserves that same skew; a *rank* remap doesn't --
    # it spreads points by percentile position instead of raw value, so the
    # rendered cloud fills its nominal range evenly regardless of how
    # skewed the source depth distribution is. Display-only: this never
    # touches the refinement/metrics pipeline, only how the point cloud
    # looks when rendered.
    ranks = np.argsort(np.argsort(depth.ravel())).reshape(depth.shape).astype(np.float64)
    ranks /= max(depth.size - 1, 1)
    depth_for_render = nominal_range[0] + ranks * (nominal_range[1] - nominal_range[0])

    points, colors = to_point_cloud(depth_for_render, intrinsics, rgb=rgb, mask=mask)

    if subsample > 1:
        points, colors = points[::subsample], colors[::subsample]

    limits = _robust_cubic_limits(points, 1.0, 99.0)
    # A single-camera point cloud is a wedge fanning out from the camera
    # origin, not a solid object -- a full 360-degree orbit inevitably swings
    # through an edge-on view of that wedge (a near-flat sliver) for at least
    # part of the rotation, no matter which point cloud it is. A small
    # back-and-forth wobble around a good oblique angle stays legible for
    # every frame instead of trading half the loop away to a bad angle.
    angles = azim_center + azim_sweep * np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
    frames = [
        render_point_cloud_frame(points, colors, elev=elev, azim=float(a), limits=limits) for a in angles
    ]
    imageio.mimsave(out_path, frames, duration=duration, loop=0)
    return out_path
