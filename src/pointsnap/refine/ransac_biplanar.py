"""RANSAC bi-planar fitting, applied once per connected candidate component.

Fitting per connected component (a whole boundary segment) rather than per
pixel is both far cheaper and more physically sensible: pixels along the
same silhouette edge share the same two local surfaces, so there is no
reason to re-derive independent hypotheses at every single one of them, and
a per-pixel independent RANSAC pass would be too slow to run as part of an
image-level pipeline in pure Python/NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pointsnap.geometry.planes import PlaneFit, is_bimodal, kmeans_1d, ransac_plane


@dataclass
class BiplanarHypothesis:
    fg: PlaneFit | None
    bg: PlaneFit | None
    valid: bool


@dataclass
class BiplanarField:
    labels: np.ndarray  # (H, W) int, 0 = not a candidate component
    hypotheses: dict[int, BiplanarHypothesis]
    fg_value: np.ndarray
    bg_value: np.ndarray
    fg_rmse: np.ndarray
    bg_rmse: np.ndarray
    valid: np.ndarray
    # for confident pixels that were used as near/far training points for some
    # component's fit, their known side -- used to seed the MRF's clamped labels
    confident_side: np.ndarray  # 0 = unset, 1 = near/fg, -1 = far/bg


def fit_biplanar_field(
    inv_depth: np.ndarray,
    candidate_mask: np.ndarray,
    margin: int = 20,
    min_neighbors: int = 12,
    min_cluster_points: int = 6,
) -> BiplanarField:
    h, w = inv_depth.shape
    labels, n = ndimage.label(candidate_mask, structure=np.ones((3, 3)))
    bounding_boxes = ndimage.find_objects(labels)

    fg_value = np.zeros((h, w))
    bg_value = np.zeros((h, w))
    fg_rmse = np.full((h, w), np.inf)
    bg_rmse = np.full((h, w), np.inf)
    valid = np.zeros((h, w), dtype=bool)
    confident_side = np.zeros((h, w), dtype=np.int8)

    hypotheses: dict[int, BiplanarHypothesis] = {}

    for label_id in range(1, n + 1):
        # Each component's search region is cropped to its own bounding box
        # plus `margin`, not dilated over the whole image: a real scene's
        # candidate mask commonly fragments into hundreds of small
        # components (classical + CNN detectors flagging slightly different
        # pixels), and a full-image dilation per component turns an
        # otherwise-cheap per-component fit into the pipeline's dominant
        # cost. Cropping first makes each component's cost scale with its
        # own local neighborhood, not the image size.
        box = bounding_boxes[label_id - 1]
        if box is None:
            continue
        y_slice, x_slice = box
        y0, y1 = max(0, y_slice.start - margin), min(h, y_slice.stop + margin)
        x0, x1 = max(0, x_slice.start - margin), min(w, x_slice.stop + margin)

        local_labels = labels[y0:y1, x0:x1]
        local_candidate = candidate_mask[y0:y1, x0:x1]
        local_inv_depth = inv_depth[y0:y1, x0:x1]
        comp_mask_local = local_labels == label_id

        dilated_local = ndimage.binary_dilation(comp_mask_local, iterations=margin)
        neighbor_mask_local = dilated_local & ~local_candidate

        if int(neighbor_mask_local.sum()) < min_neighbors:
            hypotheses[label_id] = BiplanarHypothesis(None, None, False)
            continue

        local_uu, local_vv = np.meshgrid(
            np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64)
        )
        nu = local_uu[neighbor_mask_local]
        nv = local_vv[neighbor_mask_local]
        nw = local_inv_depth[neighbor_mask_local]
        if not is_bimodal(nw):
            hypotheses[label_id] = BiplanarHypothesis(None, None, False)
            continue

        cluster_labels, centers = kmeans_1d(nw, k=2)
        near_cluster = int(np.argmax(centers))  # larger inverse depth = nearer surface
        far_cluster = int(np.argmin(centers))
        near_mask = cluster_labels == near_cluster
        far_mask = cluster_labels == far_cluster

        fg_fit = (
            ransac_plane(nu[near_mask], nv[near_mask], nw[near_mask])
            if int(near_mask.sum()) >= min_cluster_points
            else None
        )
        bg_fit = (
            ransac_plane(nu[far_mask], nv[far_mask], nw[far_mask])
            if int(far_mask.sum()) >= min_cluster_points
            else None
        )
        hyp = BiplanarHypothesis(fg=fg_fit, bg=bg_fit, valid=(fg_fit is not None and bg_fit is not None))
        hypotheses[label_id] = hyp

        if not hyp.valid:
            continue

        comp_u, comp_v = local_uu[comp_mask_local], local_vv[comp_mask_local]
        fg_value[y0:y1, x0:x1][comp_mask_local] = hyp.fg.eval(comp_u, comp_v)
        bg_value[y0:y1, x0:x1][comp_mask_local] = hyp.bg.eval(comp_u, comp_v)
        fg_rmse[y0:y1, x0:x1][comp_mask_local] = hyp.fg.rmse
        bg_rmse[y0:y1, x0:x1][comp_mask_local] = hyp.bg.rmse
        valid[y0:y1, x0:x1][comp_mask_local] = True

        # tag the actual confident neighbor pixels that trained this component's
        # planes, so the MRF can clamp them to the correct side instead of an
        # arbitrary default
        neighbor_idx = np.argwhere(neighbor_mask_local)
        near_points = neighbor_idx[near_mask]
        far_points = neighbor_idx[far_mask]
        confident_side_local = confident_side[y0:y1, x0:x1]
        confident_side_local[near_points[:, 0], near_points[:, 1]] = 1
        confident_side_local[far_points[:, 0], far_points[:, 1]] = -1

    return BiplanarField(
        labels=labels, hypotheses=hypotheses, fg_value=fg_value, bg_value=bg_value,
        fg_rmse=fg_rmse, bg_rmse=bg_rmse, valid=valid, confident_side=confident_side,
    )
