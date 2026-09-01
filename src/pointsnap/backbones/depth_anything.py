"""Depth Anything V2 backbone via `transformers`.

These checkpoints are trained to predict an inverse-depth-like quantity
(disparity: larger raw output = nearer). We invert to this project's
depth_raw convention (larger = farther) so every backbone wrapper produces a
consistent value regardless of the underlying model family.
"""

from __future__ import annotations

from pointsnap.hf_cache import configure_hf_cache

configure_hf_cache()

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from pointsnap.backbones.loader import select_device

CHECKPOINTS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
}


class DepthAnythingBackbone:
    name = "depth_anything_v2"

    def __init__(self, size: str = "small", device: str | None = None):
        checkpoint = CHECKPOINTS[size]
        self.device = select_device(device)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predicted = outputs.predicted_depth
        predicted = (
            F.interpolate(predicted.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False)
            .squeeze()
            .cpu()
            .numpy()
        )
        predicted = np.clip(predicted, 1e-6, None)
        return 1.0 / predicted
