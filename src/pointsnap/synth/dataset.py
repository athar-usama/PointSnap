"""Loader for the pregenerated synthetic dataset — used by both CNN training
(`detect/train_cnn.py`) and the synthetic benchmark runner (`eval/`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class SyntheticSample:
    rgb: np.ndarray
    depth_gt: np.ndarray
    depth_raw: np.ndarray
    boundary_mask: np.ndarray
    artifact_mask: np.ndarray
    severity: str
    nominal_focal: float


class SyntheticDataset:
    def __init__(self, root: str | Path, split: str):
        self.dir = Path(root) / split
        if not self.dir.exists():
            raise FileNotFoundError(
                f"{self.dir} not found — run scripts/generate_synthetic_dataset.py first"
            )
        self.indices = sorted(p.name for p in self.dir.iterdir() if p.is_dir())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> SyntheticSample:
        scene_dir = self.dir / self.indices[idx]
        rgb = np.array(Image.open(scene_dir / "rgb.png").convert("RGB"))
        depth_gt = np.load(scene_dir / "depth_gt.npy").astype(np.float64)
        depth_raw = np.load(scene_dir / "depth_raw.npy").astype(np.float64)
        boundary_mask = np.array(Image.open(scene_dir / "boundary_mask.png")) > 127
        artifact_mask = np.array(Image.open(scene_dir / "artifact_mask.png")) > 127
        meta = json.loads((scene_dir / "meta.json").read_text())
        return SyntheticSample(
            rgb=rgb,
            depth_gt=depth_gt,
            depth_raw=depth_raw,
            boundary_mask=boundary_mask,
            artifact_mask=artifact_mask,
            severity=meta["severity"],
            nominal_focal=meta["nominal_focal"],
        )
