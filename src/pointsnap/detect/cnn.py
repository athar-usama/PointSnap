"""A small U-Net-lite (~0.5M params) that predicts per-pixel fly-point
confidence from the 8-channel feature stack in `features.py`. Small and fast
enough to train on CPU in minutes; trivially faster if CUDA happens to be
available (see `pointsnap.backbones.loader` for the shared device policy)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pointsnap.detect.features import NUM_CHANNELS, compute_feature_stack


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConfidenceUNet(nn.Module):
    def __init__(self, in_channels: int = NUM_CHANNELS, widths: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.pool = nn.MaxPool2d(2)

        self.enc0 = ConvBlock(in_channels, w0)
        self.enc1 = ConvBlock(w0, w1)
        self.enc2 = ConvBlock(w1, w2)
        self.bottleneck = ConvBlock(w2, w3)

        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = ConvBlock(2 * w2, w2)
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = ConvBlock(2 * w1, w1)
        self.up0 = nn.ConvTranspose2d(w1, w0, 2, stride=2)
        self.dec0 = ConvBlock(2 * w0, w0)

        self.head = nn.Conv2d(w0, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        b = self.bottleneck(self.pool(x2))

        d2 = self.dec2(torch.cat([self.up2(b), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), x0], dim=1))

        return self.head(d0)  # raw logits; callers apply sigmoid / BCEWithLogits as needed

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _pad_to_multiple(x: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    pad_width = [(0, 0)] * (x.ndim - 2) + [(0, pad_h), (0, pad_w)]
    return np.pad(x, pad_width, mode="edge"), (h, w)


@torch.no_grad()
def predict_probability(
    model: ConfidenceUNet, rgb: np.ndarray, depth_raw: np.ndarray, device: str | None = None
) -> np.ndarray:
    """Runs the confidence CNN on a single (rgb, depth_raw) pair, handling
    the padding-to-a-multiple-of-8 the U-Net's three pooling stages require,
    and cropping the output back to the original resolution."""
    device = device or next(model.parameters()).device
    features = compute_feature_stack(rgb, depth_raw)
    padded, (h, w) = _pad_to_multiple(features, multiple=8)
    x = torch.from_numpy(padded[None]).to(device)
    model.eval()
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return prob[:h, :w]
