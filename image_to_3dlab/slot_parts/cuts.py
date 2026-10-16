"""One cut-out per slot, flattened on grey, and the way back from that canvas to the concept.

Route C sends each flattened cut to an edit ("complete this piece") and makes the piece in
3D from the clean picture it gets back. The edit leaves the piece where it found it, so the
piece's box in that picture, mapped back through the same crop and scale, says where it
sits on the concept and hence on the body. A transparent cut must be flattened first: an
edit model sees the whole picture under a transparent alpha.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from image_to_3dlab.concept.maps import GREY
from image_to_3dlab.slot_parts import slots

SHARE = 0.73  # how much of the canvas the piece's longer side fills: room for the edit to complete it


@dataclass(frozen=True)
class Placement:
    """canvas px = (concept px - box origin) * scale + pad."""
    box: tuple[int, int, int, int]
    scale: float
    pad: tuple[int, int]
    size: int

    def to_concept(self, box) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = box
        bx, by = self.box[0], self.box[1]
        return (bx + (x0 - self.pad[0]) / self.scale, by + (y0 - self.pad[1]) / self.scale,
                bx + (x1 - self.pad[0]) / self.scale, by + (y1 - self.pad[1]) / self.scale)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> Placement:
        return cls(tuple(int(v) for v in data["box"]), float(data["scale"]),
                   tuple(int(v) for v in data["pad"]), int(data["size"]))


def cut_slot(rgb: np.ndarray, labels: np.ndarray, slot: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """(RGBA crop, its box in the concept). A paired slot keeps the character's right side,
    which is the image-left of the two largest pieces; specks under 2% of the largest go."""
    mask = np.asarray(labels) == slots.label(slot)
    if not mask.any():
        raise ValueError(f"no {slot} in the slot map: the concept does not show it where the body's bones are")
    pieces, count = ndimage.label(ndimage.binary_closing(mask, iterations=2))
    sizes = np.bincount(pieces.ravel())[1:]
    order = np.argsort(sizes)[::-1] + 1
    if slots.CATALOGUE[slot].paired and count >= 2:
        two = order[:2]
        keep = np.isin(pieces, [min(two, key=lambda p: np.nonzero(pieces == p)[1].mean())])
    else:
        keep = np.isin(pieces, [p for p in order if sizes[p - 1] >= 0.02 * sizes.max()])
    keep = ndimage.binary_fill_holes(keep)
    ys, xs = np.nonzero(keep)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    x0, y0, x1, y1 = box
    rgba = np.dstack([np.asarray(rgb, np.uint8)[..., :3], keep.astype(np.uint8) * 255])
    return rgba[y0:y1 + 1, x0:x1 + 1].copy(), box


def flatten(piece: np.ndarray, box, size: int = 1024, share: float = SHARE) -> tuple[np.ndarray, Placement]:
    """The RGBA piece scaled to fill `share` of a square grey canvas, centred; and its Placement."""
    from PIL import Image

    h, w = piece.shape[:2]
    scale = share * size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    pad = ((size - nw) // 2, (size - nh) // 2)
    canvas = Image.new("RGB", (size, size), GREY)
    image = Image.fromarray(np.asarray(piece, np.uint8), "RGBA").resize((nw, nh), Image.LANCZOS)
    canvas.paste(image, pad, image)
    return np.asarray(canvas), Placement(tuple(int(v) for v in box), float(scale), pad, int(size))
