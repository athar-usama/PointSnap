"""Pre-warms the (D:-redirected, see pointsnap/hf_cache.py) Hugging Face
cache for every backbone in the registry. Usage:
python scripts/download_backbone_weights.py"""

from __future__ import annotations

from pointsnap.backbones import REGISTRY


def main() -> None:
    for name, factory in REGISTRY.items():
        print(f"loading {name} ...")
        factory()
        print("  ok")


if __name__ == "__main__":
    main()
