"""Train the confidence CNN on the pregenerated synthetic dataset.
Usage: python scripts/train_confidence_cnn.py"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from pointsnap.detect.train_cnn import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "confidence_cnn" / "train.yaml").read_text())
    data_root = ROOT / "data" / "synthetic"
    out_path = ROOT / "weights" / "confidence_cnn.pt"
    history = train(
        data_root=data_root,
        out_path=out_path,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        n_train=config["n_train"],
        n_val=config["n_val"],
        crop_size=config["crop_size"],
        num_threads=config["num_threads"],
    )
    results_dir = ROOT / "assets" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cnn_training_history.json").write_text(json.dumps(history, indent=2))
    print(f"saved weights -> {out_path}")


if __name__ == "__main__":
    main()
