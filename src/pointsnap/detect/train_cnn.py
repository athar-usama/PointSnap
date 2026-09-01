"""Training loop for the confidence CNN.

Feature stacks are precomputed once up front and cached in memory, rather
than recomputed on the fly every epoch: `compute_feature_stack` runs the
full classical detector (Sobel gradients, morphological erode/dilate,
robust z-scoring) per image, which is cheap once but far too expensive to
repeat 40 times over the training set. Augmentation is then reduced to
flips applied directly to the cached feature tensors (cheap array
reversals, exactly equivalent to re-extracting features from a flipped
image, since every feature here is either pointwise or a small local
convolution/morphology op, all of which commute with a flip).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pointsnap.detect.cnn import ConfidenceUNet
from pointsnap.detect.features import compute_feature_stack
from pointsnap.synth.dataset import SyntheticDataset


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()


def _sample_crop_origin(
    target: np.ndarray, crop_size: int, rng: np.random.Generator, positive_bias: float = 0.7
) -> tuple[int, int]:
    """Boundary-artifact pixels are a small minority of any scene, so a
    purely uniform-random crop would often land entirely on background --
    the same class-imbalance problem the focal loss addresses, but at the
    sampling level instead. With `positive_bias` probability, center the
    crop (jittered by the RNG's own pick among positive pixels) on a real
    artifact pixel; otherwise sample uniformly, so the network still sees
    plenty of true-negative background."""
    h, w = target.shape[-2], target.shape[-1]
    cy, cx = rng.integers(0, h), rng.integers(0, w)
    if rng.random() < positive_bias:
        positive = np.argwhere(target[0] > 0.5)
        if len(positive) > 0:
            cy, cx = positive[rng.integers(len(positive))]
    y0 = int(np.clip(cy - crop_size // 2, 0, max(h - crop_size, 0)))
    x0 = int(np.clip(cx - crop_size // 2, 0, max(w - crop_size, 0)))
    return y0, x0


class FeatureCache(Dataset):
    """Precomputes and caches the 8-channel feature stack + target mask for
    a (possibly subsampled) split, once, up front, then serves random
    boundary-biased crops -- training at a smaller crop than the full
    256x256 scene is ~(256/crop)^2 cheaper per batch on CPU, and since the
    confidence CNN only needs a small local receptive field to recognize a
    fly-point's signature, this loses essentially nothing: inference still
    runs at whatever resolution the input image actually is (the network is
    fully convolutional -- see `cnn.predict_probability`)."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        n_samples: int | None = None,
        augment: bool = False,
        crop_size: int | None = None,
        seed: int = 0,
    ):
        base = SyntheticDataset(root, split)
        indices = np.arange(len(base))
        if n_samples is not None and n_samples < len(indices):
            rng = np.random.default_rng(seed)
            indices = rng.choice(indices, size=n_samples, replace=False)

        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        for i in tqdm(indices, desc=f"precomputing features ({split})"):
            sample = base[int(i)]
            self.features.append(compute_feature_stack(sample.rgb, sample.depth_raw))
            self.targets.append(sample.artifact_mask.astype(np.float32)[None, ...])
        self.augment = augment
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        features = self.features[idx]
        target = self.targets[idx]
        rng = np.random.default_rng()

        if self.crop_size is not None:
            y0, x0 = _sample_crop_origin(target, self.crop_size, rng)
            features = features[:, y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]
            target = target[:, y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

        if self.augment:
            if rng.random() < 0.5:
                features = features[:, :, ::-1]
                target = target[:, :, ::-1]
            if rng.random() < 0.5:
                features = features[:, ::-1, :]
                target = target[:, ::-1, :]
        return torch.from_numpy(np.ascontiguousarray(features)), torch.from_numpy(np.ascontiguousarray(target))


def _epoch_prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def train(
    data_root: str | Path,
    out_path: str | Path,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_train: int | None = 1000,
    n_val: int | None = 300,
    crop_size: int | None = 64,
    num_threads: int | None = 4,
    device: str | None = None,
) -> list[dict]:
    # this machine's default thread count causes CPU op-dispatch overhead to
    # dominate over actual compute for a model this small -- capping it is
    # consistently *faster* wall-clock, not just more polite to share
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = FeatureCache(data_root, "train", n_samples=n_train, augment=True, crop_size=crop_size)
    # val crops at the same size for a fast, consistent model-selection
    # signal during training; the number that actually matters for the
    # project's claims is the full-resolution ablation in eval/ablation.py
    val_ds = FeatureCache(data_root, "val", n_samples=n_val, augment=False, crop_size=crop_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ConfidenceUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = FocalLoss()

    best_f1 = -1.0
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)
        sched.step()

        model.eval()
        val_loss, tp, fp, fn = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                tp += (preds * y).sum().item()
                fp += (preds * (1 - y)).sum().item()
                fn += ((1 - preds) * y).sum().item()
        val_loss /= len(val_ds)
        precision, recall, f1 = _epoch_prf(tp, fp, fn)
        history.append(
            {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_precision": precision, "val_recall": recall, "val_f1": f1,
            }
        )
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={f1:.4f}",
            flush=True,
        )

        if f1 > best_f1:
            best_f1 = f1
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)

    return history
