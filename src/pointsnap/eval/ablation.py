"""Ablation: classical-only vs CNN-only vs ensemble candidate-detection
precision/recall/F1 against the synthetic ground-truth artifact mask --
isolating what the trained CNN actually contributes over the classical
heuristic alone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pointsnap.detect.classical import classical_candidate_score
from pointsnap.detect.cnn import ConfidenceUNet, predict_probability
from pointsnap.detect.ensemble import ensemble_probability
from pointsnap.synth.dataset import SyntheticDataset

ROOT = Path(__file__).resolve().parents[3]


def _prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def run_ablation(
    data_root: str | Path,
    cnn_weights_path: str | Path,
    split: str = "test",
    limit: int | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    dataset = SyntheticDataset(data_root, split)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConfidenceUNet().to(device)
    model.load_state_dict(torch.load(cnn_weights_path, map_location=device))
    model.eval()

    n = len(dataset) if limit is None else min(limit, len(dataset))
    rows = []
    for i in tqdm(range(n), desc="detector ablation"):
        sample = dataset[i]
        classical = classical_candidate_score(sample.rgb, sample.depth_raw)
        cnn_prob = predict_probability(model, sample.rgb, sample.depth_raw, device=device)
        ensemble = ensemble_probability(classical.zscore, cnn_prob, mode="noisy_or")
        gt = sample.artifact_mask

        for method_name, pred_mask in [
            ("classical", classical.candidate_mask),
            ("cnn", cnn_prob > threshold),
            ("ensemble", ensemble > threshold),
        ]:
            precision, recall, f1 = _prf(pred_mask, gt)
            rows.append(
                {
                    "severity": sample.severity, "method": method_name,
                    "precision": precision, "recall": recall, "f1": f1,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = run_ablation(ROOT / "data" / "synthetic", ROOT / "weights" / "confidence_cnn.pt")
    out_dir = ROOT / "assets" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "detector_ablation.csv", index=False)

    summary = df.groupby(["severity", "method"]).mean(numeric_only=True)
    print(summary)
    summary.to_csv(out_dir / "detector_ablation_summary.csv")


if __name__ == "__main__":
    main()
