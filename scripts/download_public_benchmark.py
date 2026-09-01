"""Downloads the curated Middlebury Stereo 2014 scene subset used for the
public-benchmark leg. Usage: python scripts/download_public_benchmark.py"""

from __future__ import annotations

from pathlib import Path

import yaml

from pointsnap.eval.middlebury_loader import download_scene

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "public_benchmark" / "middlebury.yaml").read_text())
    out_dir = ROOT / "data" / "raw" / "middlebury"
    for scene in config["scenes"]:
        print(f"downloading {scene} ...")
        download_scene(scene, out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
