"""Generate and persist the synthetic stress-test dataset to disk.

Each scene folder holds exactly what PLAN.md specifies: `rgb.png`,
`depth_gt.npy`, `depth_raw.npy`, `artifact_mask.png`, `boundary_mask.png`,
`meta.json`. Regenerable from a fixed seed range — never committed to git.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from pointsnap.synth.corruption import SEVERITY_PARAMS, corrupt_depth
from pointsnap.synth.scene import generate_scene

SEVERITIES = list(SEVERITY_PARAMS.keys())


def generate_dataset(
    out_dir: str | Path,
    n_train: int = 3200,
    n_val: int = 400,
    n_test: int = 400,
    size: int = 256,
    seed_offset: int = 0,
) -> list[dict]:
    out_dir = Path(out_dir)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    seed = seed_offset
    manifest: list[dict] = []

    for split, n in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(n), desc=f"generating {split}"):
            scene = generate_scene(seed=seed, size=size)
            severity = SEVERITIES[i % len(SEVERITIES)]
            corruption_rng = np.random.default_rng(seed + 1_000_000)
            corruption = corrupt_depth(
                scene.depth_gt, scene.instance_mask, scene.boundary_mask, severity, corruption_rng
            )

            scene_dir = split_dir / f"{i:05d}"
            scene_dir.mkdir(exist_ok=True)
            Image.fromarray(scene.rgb).save(scene_dir / "rgb.png")
            np.save(scene_dir / "depth_gt.npy", scene.depth_gt.astype(np.float32))
            np.save(scene_dir / "depth_raw.npy", corruption.depth_raw.astype(np.float32))
            Image.fromarray((scene.boundary_mask * 255).astype(np.uint8)).save(
                scene_dir / "boundary_mask.png"
            )
            Image.fromarray((corruption.artifact_mask * 255).astype(np.uint8)).save(
                scene_dir / "artifact_mask.png"
            )
            meta = {
                "seed": seed, "severity": severity, "size": size, "nominal_focal": scene.nominal_focal,
            }
            (scene_dir / "meta.json").write_text(json.dumps(meta))
            manifest.append({"split": split, "index": i, **meta})
            seed += 1

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
