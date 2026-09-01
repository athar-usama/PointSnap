"""Generate the synthetic stress-test dataset used for CNN training and the
quantitative leg-1 benchmark. Usage: python scripts/generate_synthetic_dataset.py"""

from __future__ import annotations

from pathlib import Path

import yaml

from pointsnap.synth.generate import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "synthetic" / "dataset.yaml").read_text())
    out_dir = ROOT / "data" / "synthetic"
    manifest = generate_dataset(
        out_dir=out_dir,
        n_train=config["n_train"],
        n_val=config["n_val"],
        n_test=config["n_test"],
        size=config["size"],
        seed_offset=config["seed_offset"],
    )
    print(f"generated {len(manifest)} scenes -> {out_dir}")


if __name__ == "__main__":
    main()
