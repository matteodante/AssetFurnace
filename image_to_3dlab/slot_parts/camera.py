"""Pixal3D's single-view camera, and the map from its input picture to the source picture.

`scripts/pixal3d_generate.py` turns the model to face +Z (glTF's front) and its saved
camera with it, so in the glTF frame the camera sits on the +Z side, image x is +X and
image up is +Y. Pixal3D crops the matted source to the subject and resizes it, so input
pixels map to source pixels by one scale and one offset, found from the two subject boxes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    fov: float                       # horizontal field of view, radians
    dist: float                      # camera distance from the origin
    size: int                        # Pixal3D input picture, pixels a side
    scale: float = 1.0               # source px = input px * scale + offset
    offset: tuple[float, float] = (0.0, 0.0)

    def to_source(self, uv: np.ndarray) -> np.ndarray:
        return np.asarray(uv, float) * self.scale + np.asarray(self.offset, float)

    def project(self, points: np.ndarray) -> np.ndarray:
        """glTF points (N, 3) to source-picture pixels (N, 2)."""
        p = np.asarray(points, float).reshape(-1, 3)
        depth = self.dist - p[:, 2]
        f = (self.size / 2) / np.tan(self.fov / 2)
        uv = np.stack([self.size / 2 + f * p[:, 0] / depth, self.size / 2 - f * p[:, 1] / depth], 1)
        return self.to_source(uv)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["offset"] = list(self.offset)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> Camera:
        return cls(float(data["fov"]), float(data["dist"]), int(data["size"]), float(data.get("scale", 1.0)),
                   tuple(float(v) for v in data.get("offset", (0.0, 0.0))))


def subject_alpha(picture: Path, matte: Path | None = None) -> np.ndarray:
    """The subject Pixal3D saw in a picture: `matte` when given, else Pixal3D's cut-out beside
    the picture (`<stem>__matted.png`), else the picture's own transparency. A picture that
    arrives already cut out gets no `__matted.png` from Pixal3D."""
    from PIL import Image

    from image_to_3dlab.matte import is_matted

    picture = Path(picture)
    found = Path(matte) if matte else picture.with_name(f"{picture.stem}__matted.png")
    if found.is_file():
        return np.asarray(Image.open(found).convert("RGBA"))[..., 3] > 127
    with Image.open(picture) as image:
        if matte is None and is_matted(image):
            return np.asarray(image.convert("RGBA"))[..., 3] > 127
    raise ValueError(f"no cut-out of {picture.name} at {found}: run Pixal3D on it, or pass --matte")


def alpha_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), inclusive, of the True pixels."""
    ys, xs = np.where(np.asarray(mask))
    if not len(xs):
        raise ValueError("the subject mask is empty: nothing to measure")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def read_views(views: Path) -> tuple[float, float, np.ndarray]:
    """(fov, distance, input alpha) from a Pixal3D `<model>.svviews` folder."""
    # here, not at the top: Blender's Python has no PIL and imports this module
    from PIL import Image

    views = Path(views)
    meta = json.loads((views / "transforms.json").read_text())
    frame = meta["frames"][0]
    matrix = np.asarray(frame["transform_matrix"], float)
    picture = np.asarray(Image.open(views / frame["file_path"]).convert("RGBA"))
    return float(meta["camera_angle_x"]), float(np.linalg.norm(matrix[:3, 3])), picture[..., 3] > 127


def from_views(views: Path, source_alpha: np.ndarray) -> Camera:
    """The camera of a Pixal3D run, mapped onto the source picture whose subject mask is given."""
    fov, dist, alpha = read_views(views)
    a, b = alpha_box(alpha), alpha_box(source_alpha)
    scale = ((b[2] - b[0]) / max(a[2] - a[0], 1) + (b[3] - b[1]) / max(a[3] - a[1], 1)) / 2
    return Camera(fov, dist, int(alpha.shape[0]), float(scale), (b[0] - a[0] * scale, b[1] - a[1] * scale))
