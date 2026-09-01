"""Leg 2: public-benchmark validation on Middlebury Stereo 2014 (see
middlebury_loader.py for why this replaces iBims-1). Runs every registered
backbone on each curated scene's im0.png, refines the raw output, and
reports standard + novel metrics raw vs refined against real stereo ground
truth and real occlusion-boundary structure -- the actual proof that the
refinement is model-agnostic, not just synthetic-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from pointsnap.backbones import REGISTRY
from pointsnap.detect.cnn import ConfidenceUNet, predict_probability
from pointsnap.eval.boundary_metrics import (
    bridging_rate,
    point_isolation_scores,
    summarize_point_isolation,
)
from pointsnap.eval.geometry_metrics import abs_rel, align_affine, delta_accuracy
from pointsnap.eval.middlebury_loader import load_scene
from pointsnap.geometry.intrinsics import Intrinsics
from pointsnap.refine.pipeline import refine_depth

ROOT = Path(__file__).resolve().parents[3]


def evaluate_backbone_on_scene(
    backbone, scene_dir: str | Path, cnn_model=None, device: str = "cpu"
) -> dict:
    scene_dir = Path(scene_dir)
    scene = load_scene(scene_dir)
    depth_pred = backbone.predict(scene.rgb)

    # Depth Anything V2 / MiDaS (non-metric checkpoints) predict disparity up
    # to an unknown *affine* transform (scale AND shift), not just an unknown
    # scale -- so the alignment has to happen in disparity (inverse-depth)
    # space via a full affine fit, not as a single scale factor applied to
    # depth directly (a scale-only fit in depth space silently assumes zero
    # shift and produces wildly wrong absolute depths whenever that's false,
    # which it usually is). Fit on confidently valid, non-boundary pixels
    # only (the standard Eigen et al. evaluation convention).
    align_mask = scene.valid_mask & ~scene.boundary_mask
    inv_pred = 1.0 / np.clip(depth_pred, 1e-8, None)
    inv_gt = 1.0 / np.clip(scene.depth_gt, 1e-8, None)
    scale, shift = align_affine(inv_pred, inv_gt, mask=align_mask)
    inv_aligned = np.clip(scale * inv_pred + shift, 1e-6, None)
    depth_pred_scaled = 1.0 / inv_aligned

    intrinsics = Intrinsics.exact(
        fx=scene.fx, fy=scene.fy, cx=scene.cx, cy=scene.cy,
        width=scene.rgb.shape[1], height=scene.rgb.shape[0],
    )

    cnn_prob = None
    if cnn_model is not None:
        cnn_prob = predict_probability(cnn_model, scene.rgb, depth_pred_scaled, device=device)
    # bilateral snap is a purely cosmetic anti-aliasing pass for the visual
    # gallery; see run_synthetic_benchmark.py's identical note
    result = refine_depth(scene.rgb, depth_pred_scaled, cnn_prob=cnn_prob, apply_bilateral_snap=False)

    mask = scene.valid_mask
    pis_raw = summarize_point_isolation(
        point_isolation_scores(depth_pred_scaled, scene.boundary_mask, intrinsics)
    )
    pis_refined = summarize_point_isolation(
        point_isolation_scores(result.depth_refined, scene.boundary_mask, intrinsics)
    )

    return {
        "backbone": backbone.name,
        "scene": scene_dir.name,
        "abs_rel_raw": abs_rel(depth_pred_scaled, scene.depth_gt, mask=mask),
        "abs_rel_refined": abs_rel(result.depth_refined, scene.depth_gt, mask=mask),
        "delta1_raw": delta_accuracy(depth_pred_scaled, scene.depth_gt, mask=mask),
        "delta1_refined": delta_accuracy(result.depth_refined, scene.depth_gt, mask=mask),
        "pis_raw": pis_raw["median_pis"],
        "pis_refined": pis_refined["median_pis"],
        "br_raw": bridging_rate(depth_pred_scaled, scene.boundary_mask),
        "br_refined": bridging_rate(result.depth_refined, scene.boundary_mask),
    }


def run_public_benchmark(
    scene_root: str | Path,
    backbone_names: list[str],
    cnn_weights_path: str | Path | None = None,
    scenes: list[str] | None = None,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model = None
    if cnn_weights_path is not None and Path(cnn_weights_path).exists():
        cnn_model = ConfidenceUNet().to(device)
        cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=device))
        cnn_model.eval()

    if scenes is not None:
        scene_dirs = [Path(scene_root) / name for name in scenes]
    else:
        scene_dirs = sorted(p for p in Path(scene_root).iterdir() if p.is_dir())
    rows = []
    for backbone_name in backbone_names:
        backbone = REGISTRY[backbone_name]()
        for scene_dir in tqdm(scene_dirs, desc=backbone_name):
            rows.append(evaluate_backbone_on_scene(backbone, scene_dir, cnn_model=cnn_model, device=device))
    return pd.DataFrame(rows)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "public_benchmark" / "middlebury.yaml").read_text())
    scene_root = ROOT / "data" / "raw" / "middlebury"
    df = run_public_benchmark(
        scene_root,
        backbone_names=config["backbones"],
        cnn_weights_path=ROOT / "weights" / "confidence_cnn.pt",
        scenes=config["scenes"],
    )
    out_dir = ROOT / "assets" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "public_benchmark.csv", index=False)

    summary = df.groupby("backbone").mean(numeric_only=True)
    print(summary)
    summary.to_csv(out_dir / "public_benchmark_summary.csv")


if __name__ == "__main__":
    main()
