"""Common interface for monocular depth backbones.

Every backbone's `predict` returns depth_raw in the convention used
everywhere else in this project: larger value = farther away, whether or
not the values are metric. PointSnap's refinement algorithm and novel
metrics are designed to be invariant to an unknown affine transform of this
quantity (see geometry/planes.py's module docstring), so getting the exact
scale right is never required -- only the orientation convention (larger =
farther), which each wrapper is responsible for normalizing to.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DepthBackbone(Protocol):
    name: str

    def predict(self, rgb: np.ndarray) -> np.ndarray: ...
