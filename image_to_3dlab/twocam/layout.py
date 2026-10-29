"""The two-camera frame: where each view sits, and how pixels map to metres and back.

Front view (left, `w_front` px) looks along +Y at the creature's face: u grows with X.
Side view (right, `w_side` px) looks along +X at its right flank: u grows with -Y, so the
head (-Y) is on the right. In both, v grows downward with -Z, the ground (z = 0) is row
`ground_v`, and one pixel is `m_per_px` metres. A joint at height z sits on the same row
in both halves; that shared row is what makes two views enough for 3D.

The layout is part of the rig: AnyMove's `build_rig.py` writes it into `rig_info.json`
and every stage must read it from there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

BACKDROP = (128, 130, 134)
FLOOR = (152, 154, 158)
DIVIDER = (34, 34, 38)


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class Layout:
    h: int = 720
    w_front: int = 420
    w_side: int = 860
    m_per_px: float = 1.9 / 720
    ground_v: float = 630

    def __post_init__(self) -> None:
        if min(self.h, self.w_front, self.w_side) <= 0 or self.m_per_px <= 0:
            raise LayoutError(f"layout sizes must be positive: {self}")

    @property
    def w(self) -> int:
        return self.w_front + self.w_side

    @property
    def z_center(self) -> float:
        """Height in metres of the frame's middle row."""
        return (self.ground_v - self.h / 2) * self.m_per_px

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> Layout:
        try:
            return cls(
                h=int(source.get("H", 720)),
                w_front=int(source["W_FRONT"]),
                w_side=int(source["W_SIDE"]),
                m_per_px=float(source["M_PER_PX"]),
                ground_v=float(source["GROUND_V"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LayoutError(f"layout needs W_FRONT, W_SIDE, M_PER_PX and GROUND_V: {exc}") from exc

    @classmethod
    def from_rig_info(cls, info: Mapping[str, Any]) -> Layout:
        if "layout" not in info:
            raise LayoutError(
                "rig_info.json has no 'layout': rebuild the rig, or the capture would "
                "silently use another creature's scale"
            )
        return cls.from_dict(info["layout"])

    def to_dict(self) -> dict[str, float]:
        return {"H": self.h, "W_FRONT": self.w_front, "W_SIDE": self.w_side,
                "M_PER_PX": self.m_per_px, "GROUND_V": self.ground_v, "Z_CENTER": self.z_center}

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (..., 3) in metres to (front uv, side uv) in layout pixels."""
        p = np.asarray(p, np.float64)
        v = self.h / 2 - (p[..., 2] - self.z_center) / self.m_per_px
        front = np.stack([self.w_front / 2 + p[..., 0] / self.m_per_px, v], -1)
        side = np.stack([self.w_side / 2 - p[..., 1] / self.m_per_px, v], -1)
        return front, side

    def unproject_front(self, uv: np.ndarray) -> np.ndarray:
        """Front-view pixels to (x, z) in metres."""
        uv = np.asarray(uv, np.float64)
        return np.stack([(uv[..., 0] - self.w_front / 2) * self.m_per_px,
                         self.z_center + (self.h / 2 - uv[..., 1]) * self.m_per_px], -1)

    def unproject_side(self, uv: np.ndarray) -> np.ndarray:
        """Side-view pixels to (y, z) in metres."""
        uv = np.asarray(uv, np.float64)
        return np.stack([-(uv[..., 0] - self.w_side / 2) * self.m_per_px,
                         self.z_center + (self.h / 2 - uv[..., 1]) * self.m_per_px], -1)

    def split(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A layout-sized frame (H x W x C) into its front and side views."""
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            raise LayoutError(f"frame is {frame.shape[1]}x{frame.shape[0]}, layout is {self.w}x{self.h}")
        return frame[:, :self.w_front], frame[:, self.w_front:]

    def compose(self, front_rgba: np.ndarray, side_rgba: np.ndarray) -> np.ndarray:
        """Alpha-composite two RGBA renders over the studio backdrop (wall, floor band, divider)."""
        out = np.empty((self.h, self.w, 3), np.float32)
        out[:] = BACKDROP
        out[int(round(self.ground_v)):] = FLOOR
        for img, x0 in ((front_rgba, 0), (side_rgba, self.w_front)):
            img = np.asarray(img, np.float32)
            a = img[..., 3:4] / 255.0
            h, w = img.shape[:2]
            out[:h, x0:x0 + w] = img[..., :3] * a + out[:h, x0:x0 + w] * (1 - a)
        out[:, self.w_front - 2:self.w_front + 2] = DIVIDER
        return out.clip(0, 255).astype(np.uint8)
