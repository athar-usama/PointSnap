"""Shared GIF export helper, used wherever a sequence of RGB frames needs to
become a looping GIF for the README (before/after depth colorization
sweeps, point-cloud turntables)."""

from __future__ import annotations

import imageio.v2 as imageio
import numpy as np


def save_gif(frames: list[np.ndarray], out_path: str, duration: float = 0.08, loop: int = 0) -> str:
    imageio.mimsave(out_path, frames, duration=duration, loop=loop)
    return out_path
