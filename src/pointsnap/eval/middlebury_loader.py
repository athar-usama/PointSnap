"""Loader for the Middlebury Stereo 2014 dataset -- PLAN.md's documented
fallback for the public-benchmark leg, used instead of iBims-1 because
iBims-1's MediaTUM host serves its file listing through an interactive
share portal (ownCloud-style JS app) with no stable anonymous FTP or API
path, which this build actually tried and confirmed fails (anonymous FTP:
"530 Login incorrect"). Middlebury 2014 has an explicit academic license,
direct HTTPS zip downloads per scene, and -- more directly relevant to this
project than iBims-1's general boundary masks -- ships ground-truth
disparity with occluded pixels marked as infinite, which is a genuine,
real-world occlusion-boundary signal.

Each scene's `disp0.pfm` stores the true disparity for `im0.png`, with
occluded pixels set to +inf. Depth is recovered via the standard stereo
relation depth = baseline * fx / (disparity + doffs), using `calib.txt`'s
own camera parameters -- so, unlike the synthetic leg or the real-photo
gallery, no intrinsics heuristic is needed here at all.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy import ndimage

BASE_URL = "https://vision.middlebury.edu/stereo/data/scenes2014/zip"

DEFAULT_SCENES = [
    "Motorcycle-perfect",
    "Piano-perfect",
    "Backpack-perfect",
    "Playtable-perfect",
    "Adirondack-perfect",
    "Recycle-perfect",
]


def download_scene(scene: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{scene}.zip"
    scene_dir = out_dir / scene
    if scene_dir.exists():
        return scene_dir

    url = f"{BASE_URL}/{scene}.zip"
    urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    return scene_dir


def read_pfm(path: str | Path) -> tuple[np.ndarray, float]:
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"not a PFM file: {path}")
        color = header == "PF"

        dims_line = f.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = f.readline().decode("ascii").strip()
        width, height = (int(v) for v in dims_line.split())

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data, abs(scale)


def parse_calib(path: str | Path) -> dict:
    text = Path(path).read_text()
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()

    cam0_match = re.findall(r"[-\d.]+", values["cam0"])
    fx = float(cam0_match[0])
    cx0 = float(cam0_match[2])
    fy = float(cam0_match[4])
    cy0 = float(cam0_match[5])

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx0,
        "cy": cy0,
        "baseline_mm": float(values["baseline"]),
        "doffs": float(values["doffs"]),
        "width": int(values["width"]),
        "height": int(values["height"]),
    }


@dataclass
class MiddleburyScene:
    rgb: np.ndarray
    depth_gt: np.ndarray
    valid_mask: np.ndarray
    boundary_mask: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


def load_scene(scene_dir: str | Path, boundary_band: int = 2, max_dim: int = 640) -> MiddleburyScene:
    import cv2
    from PIL import Image

    scene_dir = Path(scene_dir)
    rgb = np.array(Image.open(scene_dir / "im0.png").convert("RGB"))
    disp, _ = read_pfm(scene_dir / "disp0.pfm")
    calib = parse_calib(scene_dir / "calib.txt")

    valid_mask = np.isfinite(disp) & (disp > 0)
    baseline_m = calib["baseline_mm"] / 1000.0
    depth_gt = np.zeros_like(disp)
    depth_gt[valid_mask] = (baseline_m * calib["fx"]) / (disp[valid_mask] + calib["doffs"])

    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]

    # full Middlebury 2014 resolution is far larger than needed (and far
    # slower to run a ViT backbone + the refinement pipeline over) -- rescale
    # everything consistently, using nearest-neighbor for depth/masks so
    # occlusion boundaries stay crisp rather than being blurred by resampling
    h, w = rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_size = (round(w * scale), round(h * scale))
        rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
        depth_gt = cv2.resize(depth_gt, new_size, interpolation=cv2.INTER_NEAREST)
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0
        fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale

    invalid = ~valid_mask
    near_invalid = ndimage.binary_dilation(invalid, iterations=boundary_band)
    boundary_mask = near_invalid & valid_mask

    return MiddleburyScene(
        rgb=rgb, depth_gt=depth_gt, valid_mask=valid_mask, boundary_mask=boundary_mask,
        fx=fx, fy=fy, cx=cx, cy=cy,
    )
