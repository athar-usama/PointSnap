"""Leg 1: the controlled synthetic stress test. For every scene in the
synthetic test split, refines depth_raw and compares PIS / bridging-rate /
standard metrics before vs after -- with the trained confidence CNN in the
ensemble when available, classical-only otherwise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

from pointsnap.detect.cnn import ConfidenceUNet, predict_probability
from pointsnap.eval.boundary_metrics import (
    bridging_rate,
    point_isolation_scores,
    summarize_point_isolation,
)
from pointsnap.eval.geometry_metrics import abs_rel, delta_accuracy
from pointsnap.geometry.intrinsics import Intrinsics
from pointsnap.refine.pipeline import refine_depth
from pointsnap.synth.dataset import SyntheticDataset, SyntheticSample

ROOT = Path(__file__).resolve().parents[3]


def _non_boundary_mask(boundary_mask: np.ndarray, dilate_iters: int = 3) -> np.ndarray:
    return ~ndimage.binary_dilation(boundary_mask, iterations=dilate_iters)


def evaluate_sample(
    sample: SyntheticSample, cnn_model: ConfidenceUNet | None = None, device: str = "cpu"
) -> dict:
    intrinsics = Intrinsics.rule_of_thumb(width=sample.rgb.shape[1], height=sample.rgb.shape[0])
    nonb_mask = _non_boundary_mask(sample.boundary_mask)

    cnn_prob = None
    if cnn_model is not None:
        cnn_prob = predict_probability(cnn_model, sample.rgb, sample.depth_raw, device=device)

    # bilateral snap is a purely cosmetic anti-aliasing pass for the visual
    # gallery -- it deliberately softens the boundary over a couple of
    # pixels, which is the opposite of what the bridging-rate metric
    # rewards, so every quantitative benchmark measures the MRF-only output
    result = refine_depth(sample.rgb, sample.depth_raw, cnn_prob=cnn_prob, apply_bilateral_snap=False)

    pis_raw = summarize_point_isolation(
        point_isolation_scores(sample.depth_raw, sample.boundary_mask, intrinsics)
    )
    pis_refined = summarize_point_isolation(
        point_isolation_scores(result.depth_refined, sample.boundary_mask, intrinsics)
    )
    br_raw = bridging_rate(sample.depth_raw, sample.boundary_mask)
    br_refined = bridging_rate(result.depth_refined, sample.boundary_mask)

    return {
        "severity": sample.severity,
        "pis_raw": pis_raw["median_pis"],
        "pis_refined": pis_refined["median_pis"],
        "isolation_rate_raw": pis_raw["isolation_rate"],
        "isolation_rate_refined": pis_refined["isolation_rate"],
        "br_raw": br_raw,
        "br_refined": br_refined,
        "abs_rel_nonboundary_raw": abs_rel(sample.depth_raw, sample.depth_gt, mask=nonb_mask),
        "abs_rel_nonboundary_refined": abs_rel(result.depth_refined, sample.depth_gt, mask=nonb_mask),
        "delta1_nonboundary_raw": delta_accuracy(sample.depth_raw, sample.depth_gt, mask=nonb_mask),
        "delta1_nonboundary_refined": delta_accuracy(
            result.depth_refined, sample.depth_gt, mask=nonb_mask
        ),
        "fraction_candidates": result.fraction_candidates,
    }


def run_benchmark(
    data_root: str | Path,
    cnn_weights_path: str | Path | None = None,
    split: str = "test",
    limit: int | None = None,
) -> pd.DataFrame:
    dataset = SyntheticDataset(data_root, split)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cnn_model = None
    if cnn_weights_path is not None and Path(cnn_weights_path).exists():
        cnn_model = ConfidenceUNet().to(device)
        cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=device))
        cnn_model.eval()

    n = len(dataset) if limit is None else min(limit, len(dataset))
    rows = [evaluate_sample(dataset[i], cnn_model=cnn_model, device=device) for i in tqdm(range(n))]
    return pd.DataFrame(rows)


def main() -> None:
    data_root = ROOT / "data" / "synthetic"
    cnn_weights = ROOT / "weights" / "confidence_cnn.pt"
    df = run_benchmark(data_root, cnn_weights_path=cnn_weights)

    out_dir = ROOT / "assets" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "synthetic_benchmark.csv", index=False)

    summary = df.groupby("severity").mean(numeric_only=True)
    print(summary)
    summary.to_csv(out_dir / "synthetic_benchmark_summary.csv")


if __name__ == "__main__":
    main()
