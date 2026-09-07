"""Pinhole camera intrinsics and the fallback heuristics used when a real
calibration isn't available (only the real-photo gallery leg needs the
fallback path; synthetic scenes and iBims-1 both ship exact intrinsics)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def exact(cls, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> Intrinsics:
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    @classmethod
    def from_exif_focal_35mm(cls, focal_35mm: float, width: int, height: int) -> Intrinsics:
        """35mm-equivalent focal length maps to pixel focal length via the sensor's
        36mm reference width: fx = (focal_35mm / 36mm) * image_width. This assumes
        square pixels and no lens distortion: a reasonable approximation for
        rectilinear lenses, explicitly not for the fisheye gallery photo (see
        README's "what this doesn't solve")."""
        f = (focal_35mm / 36.0) * width
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    @classmethod
    def rule_of_thumb(cls, width: int, height: int) -> Intrinsics:
        """Used only when no EXIF focal length is present. fx=fy=1.2*max(W,H) is a
        documented, unvalidated guess; never treated as ground truth."""
        f = 1.2 * max(width, height)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, width=width, height=height)
