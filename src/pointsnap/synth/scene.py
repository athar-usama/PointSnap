"""Procedural scene compositing for the controlled synthetic stress test.

Every shape is defined once, in normalized [0, 1] canvas coordinates, and
rasterized at two different resolutions from the same parameters:

  - at `size * supersample` for the RGB render, which is then downsampled
    with area averaging -> naturally anti-aliased edges, like a real camera.
  - directly at `size` for the instance/depth ground truth -> a crisp,
    un-blurred label at every pixel, like a laser scanner.

Both share the same underlying geometric contour; only the sampling
resolution differs. That is what lets the corruption model (`corruption.py`)
inject a *depth*-only artifact near a boundary that RGB still describes
truthfully: the premise the whole refinement algorithm depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Shape:
    kind: str
    params: dict = field(default_factory=dict)


def _perlin_like_noise(height: int, width: int, rng: np.random.Generator, octaves: int = 4,
                        base_freq: int = 4) -> np.ndarray:
    """Cheap multi-octave value noise (dependency-free stand-in for Perlin
    noise): random low-res grids upsampled with cubic interpolation and
    summed across octaves, then normalized to [0, 1]."""
    canvas = np.zeros((height, width), dtype=np.float64)
    amplitude, total_amp = 1.0, 0.0
    for o in range(octaves):
        freq = base_freq * (2**o)
        small = rng.uniform(0, 1, size=(freq, freq))
        layer = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        canvas += amplitude * layer
        total_amp += amplitude
        amplitude *= 0.5
    canvas /= total_amp
    canvas -= canvas.min()
    canvas /= max(canvas.max(), 1e-8)
    return canvas


def _textured_layer(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    noise = _perlin_like_noise(height, width, rng)
    tint = rng.uniform(40, 220, size=3)
    rgb = np.stack([tint[c] * (0.5 + 0.5 * noise) for c in range(3)], axis=-1)
    return np.clip(rgb, 0, 255)


def _random_shape(rng: np.random.Generator) -> Shape:
    kind = rng.choice(["ellipse", "rect", "polygon"])
    cx, cy = rng.uniform(0.2, 0.8, size=2)
    if kind == "ellipse":
        rx, ry = rng.uniform(0.08, 0.25, size=2)
        return Shape("ellipse", {"cx": cx, "cy": cy, "rx": rx, "ry": ry, "angle": float(rng.uniform(0, 180))})
    if kind == "rect":
        w, h = rng.uniform(0.15, 0.4, size=2)
        return Shape("rect", {"cx": cx, "cy": cy, "w": w, "h": h, "angle": float(rng.uniform(0, 90))})
    n = int(rng.integers(3, 7))
    radius = rng.uniform(0.1, 0.25)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n))
    radii = radius * (0.7 + 0.3 * rng.uniform(size=n))
    verts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
    return Shape("polygon", {"vertices": verts})


def rasterize_shape(shape: Shape, resolution: int) -> np.ndarray:
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    s = resolution
    p = shape.params
    if shape.kind == "ellipse":
        center = (int(p["cx"] * s), int(p["cy"] * s))
        axes = (max(int(p["rx"] * s), 1), max(int(p["ry"] * s), 1))
        cv2.ellipse(mask, center, axes, p["angle"], 0, 360, 1, -1, lineType=cv2.LINE_8)
    elif shape.kind == "rect":
        rect = ((p["cx"] * s, p["cy"] * s), (max(p["w"] * s, 1), max(p["h"] * s, 1)), p["angle"])
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(mask, box, 1, lineType=cv2.LINE_8)
    else:
        verts = (p["vertices"] * s).astype(np.int32)
        cv2.fillPoly(mask, [verts], 1, lineType=cv2.LINE_8)
    return mask.astype(bool)


@dataclass
class SyntheticScene:
    rgb: np.ndarray            # (H, W, 3) uint8, anti-aliased render
    depth_gt: np.ndarray       # (H, W) float64, crisp ground-truth depth
    boundary_mask: np.ndarray  # (H, W) bool, true occlusion contour, ~2px thick
    instance_mask: np.ndarray  # (H, W) int32, 0 = background, 1..k = foreground shapes
    seed: int
    nominal_focal: float       # a plausible fx=fy for this scene's synthetic camera


def generate_scene(seed: int, size: int = 256, supersample: int = 4) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    big = size * supersample

    bg_depth = float(rng.uniform(3.0, 8.0))
    n_shapes = int(rng.integers(1, 5))
    min_gap = 0.4
    shapes: list[Shape] = []
    depths = [bg_depth]
    prev_depth = bg_depth
    for _ in range(n_shapes):
        max_depth = max(0.6, prev_depth - min_gap)
        shape_depth = float(rng.uniform(0.5, max_depth))
        shapes.append(_random_shape(rng))
        depths.append(shape_depth)
        prev_depth = min(prev_depth, shape_depth)

    # painter's algorithm: farthest foreground shape first, nearest drawn last so it occludes
    order = list(np.argsort(depths[1:])[::-1]) if shapes else []

    def build_instances(resolution: int) -> np.ndarray:
        instance = np.zeros((resolution, resolution), dtype=np.int32)
        for idx in order:
            instance[rasterize_shape(shapes[idx], resolution)] = idx + 1
        return instance

    instance_big = build_instances(big)
    instance_final = build_instances(size)

    rgb_big = _textured_layer(big, big, rng)
    for idx in range(len(shapes)):
        tex = _textured_layer(big, big, rng)
        m = instance_big == (idx + 1)
        rgb_big[m] = tex[m]
    rgb = np.clip(cv2.resize(rgb_big, (size, size), interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)

    depth_gt = np.full((size, size), bg_depth, dtype=np.float64)
    for idx, d in enumerate(depths[1:]):
        depth_gt[instance_final == (idx + 1)] = d

    inst_u8 = instance_final.astype(np.uint8)
    dilated = cv2.dilate(inst_u8, np.ones((3, 3), np.uint8))
    eroded = cv2.erode(inst_u8, np.ones((3, 3), np.uint8))
    boundary_mask = dilated != eroded

    return SyntheticScene(
        rgb=rgb, depth_gt=depth_gt, boundary_mask=boundary_mask, instance_mask=instance_final,
        seed=seed, nominal_focal=1.2 * size,
    )
