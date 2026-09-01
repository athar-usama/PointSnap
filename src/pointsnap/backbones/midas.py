"""MiDaS DPT-Hybrid backbone via `transformers`, used to prove the
refinement pipeline is model-agnostic: a genuinely different architecture
family (ViT-Hybrid vs Depth Anything's DINOv2) and training recipe. Same
disparity-like native convention as Depth Anything, inverted the same way.
"""

from __future__ import annotations

from pointsnap.hf_cache import configure_hf_cache

configure_hf_cache()

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

from pointsnap.backbones.loader import select_device

CHECKPOINT = "Intel/dpt-hybrid-midas"


class MiDaSBackbone:
    name = "midas_dpt_hybrid"

    def __init__(self, device: str | None = None):
        self.device = select_device(device)
        self.processor = DPTImageProcessor.from_pretrained(CHECKPOINT)
        self.model = DPTForDepthEstimation.from_pretrained(CHECKPOINT).to(self.device)
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
