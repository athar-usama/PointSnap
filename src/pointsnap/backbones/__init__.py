"""Backbone registry -- adding a new monocular depth model to benchmark
against means adding one entry here, nothing in the refinement pipeline
changes (that's the entire point of a model-agnostic post-hoc plug-in)."""

from __future__ import annotations

from pointsnap.backbones.depth_anything import DepthAnythingBackbone
from pointsnap.backbones.midas import MiDaSBackbone

REGISTRY = {
    "depth_anything_v2_small": lambda: DepthAnythingBackbone(size="small"),
    "depth_anything_v2_base": lambda: DepthAnythingBackbone(size="base"),
    "midas_dpt_hybrid": lambda: MiDaSBackbone(),
}


def load_backbone(name: str):
    if name not in REGISTRY:
        raise ValueError(f"unknown backbone '{name}', choose from {list(REGISTRY)}")
    return REGISTRY[name]()
