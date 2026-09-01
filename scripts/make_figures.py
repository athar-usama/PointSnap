"""Assembles every figure/table the README embeds, from actual run outputs
only -- never mocked. Run after generate_synthetic_dataset.py,
train_confidence_cnn.py, run_synthetic_benchmark.py, run_public_benchmark.py,
and run_real_gallery.py have all completed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from pointsnap.detect.cnn import ConfidenceUNet, predict_probability
from pointsnap.refine.pipeline import refine_depth
from pointsnap.synth.corruption import corrupt_depth
from pointsnap.synth.scene import generate_scene
from pointsnap.viz.before_after_grid import hstack_panels, vstack_panels
from pointsnap.viz.boundary_profile_figure import boundary_profile_panel
from pointsnap.viz.depth_colormap import colorize_depth
from pointsnap.viz.energy_trace import format_energy_trace

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
RESULTS = ROOT / "assets" / "results"


def _load_cnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = ROOT / "weights" / "confidence_cnn.pt"
    if not weights_path.exists():
        return None, device
    model = ConfidenceUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def make_severity_grid(seed: int = 100) -> None:
    model, device = _load_cnn()
    scene = generate_scene(seed=seed, size=256)
    rows = []
    for severity in ["easy", "moderate", "hard"]:
        rng = np.random.default_rng(seed * 7 + hash(severity) % 1000)
        corruption = corrupt_depth(scene.depth_gt, scene.instance_mask, scene.boundary_mask, severity, rng)
        cnn_prob = None
        if model is not None:
            cnn_prob = predict_probability(model, scene.rgb, corruption.depth_raw, device=device)
        result = refine_depth(scene.rgb, corruption.depth_raw, cnn_prob=cnn_prob)
        row = hstack_panels(
            [
                colorize_depth(corruption.depth_raw),
                colorize_depth(result.depth_refined),
            ]
        )
        rows.append(row)
    grid = vstack_panels(rows)
    header = hstack_panels([scene.rgb, colorize_depth(scene.depth_gt)])
    full = vstack_panels([header, grid])
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURES / "severity_grid.png"), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    print("wrote severity_grid.png")


def make_boundary_profile(seed: int = 7, severity: str = "hard") -> None:
    model, device = _load_cnn()
    scene = generate_scene(seed=seed, size=256)
    rng = np.random.default_rng(seed * 13)
    corruption = corrupt_depth(scene.depth_gt, scene.instance_mask, scene.boundary_mask, severity, rng)
    cnn_prob = None
    if model is not None:
        cnn_prob = predict_probability(model, scene.rgb, corruption.depth_raw, device=device)
    result = refine_depth(scene.rgb, corruption.depth_raw, cnn_prob=cnn_prob)

    ys, xs = np.where(scene.boundary_mask)
    if len(ys) == 0:
        print("no boundary pixels found, skipping boundary profile")
        return
    mid = len(ys) // 2
    y = ys[mid]
    x0 = max(0, xs[mid] - 8)
    profile_gt = scene.depth_gt[y, x0 : x0 + 16]
    profile_raw = corruption.depth_raw[y, x0 : x0 + 16]
    profile_refined = result.depth_refined[y, x0 : x0 + 16]

    panel = boundary_profile_panel(profile_gt, profile_raw, profile_refined)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURES / "boundary_profile.png"), cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    print("wrote boundary_profile.png")


def make_energy_trace_text(seed: int = 2, severity: str = "hard") -> None:
    scene = generate_scene(seed=seed, size=256)
    rng = np.random.default_rng(seed * 17)
    corruption = corrupt_depth(scene.depth_gt, scene.instance_mask, scene.boundary_mask, severity, rng)
    result = refine_depth(scene.rgb, corruption.depth_raw)
    text = format_energy_trace(result.energy_trace)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "energy_trace.txt").write_text(text)
    print("wrote energy_trace.txt")
    print(text)


def make_markdown_tables() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ["synthetic_benchmark_summary", "detector_ablation_summary", "public_benchmark_summary"]:
        csv_path = RESULTS / f"{name}.csv"
        if not csv_path.exists():
            print(f"skip {name}: not found")
            continue
        df = pd.read_csv(csv_path)
        (RESULTS / f"{name}.md").write_text(df.to_markdown(index=False))
        print(f"wrote {name}.md")


def main() -> None:
    make_severity_grid()
    make_boundary_profile()
    make_energy_trace_text()
    make_markdown_tables()


if __name__ == "__main__":
    main()
