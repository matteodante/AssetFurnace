"""Silhouette against a reference: similar, but not identical.

Shape only, never size. Each view is drawn orthographically (no perspective), then fitted
to the frame and centred on its own, so a model twice as big casts the same silhouette.
Size is `measure.height`'s job; this one judges proportion and outline.

Four views that differ: front, side, three-quarter (45 degrees) and top. Back and left
would add nothing, because an orthographic back view is the front one mirrored.

The score per view is IoU (intersection over union): how much of the two outlines
overlaps. 1 is identical, 0 is no overlap. The check passes inside a band: under `min`
the reference is not recognisable, over `max` the asset is practically a copy. Any one
view under `min_view` warns even when the mean is fine: a wrong arm shows from the side.

A picture as reference (a concept, a cut-out) gives only the front view.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from image_to_3dlab.checks import geometry
from image_to_3dlab.checks.registry import grade_band, register, result

VIEWS = {"front": (0.0, 0.0), "side": (90.0, 0.0), "three_quarter": (45.0, 0.0), "top": (0.0, 90.0)}
FILL = 0.9            # share of the frame the outline's longer side fills
# Overlay colours, as in scripts/compare_to_source.py: reference only, asset only, both.
REFERENCE_ONLY, ASSET_ONLY, BOTH, NEITHER = (220, 0, 220), (0, 200, 0), (255, 255, 255), (24, 24, 24)


def rotation(azimuth: float, elevation: float) -> np.ndarray:
    """Turn the model so the camera, looking down -Z, sees it from (azimuth, elevation)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    turn = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    tilt = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return tilt @ turn


def fit(points: np.ndarray, resolution: int) -> np.ndarray:
    """Scale 2D points uniformly so their box fills the frame, and centre them."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = float((hi - lo).max()) or 1.0
    scale = FILL * resolution / span
    return (points - (lo + hi) / 2) * scale + resolution / 2


def mask_of_mesh(vertices: np.ndarray, faces: np.ndarray, view: str, resolution: int = 256) -> np.ndarray:
    """Boolean silhouette of a mesh from one view, fitted to a square frame."""
    turned = vertices @ rotation(*VIEWS[view]).T
    points = fit(turned[:, :2], resolution)
    points[:, 1] = resolution - points[:, 1]  # image rows grow downward
    return geometry.rasterize(points, faces, resolution, resolution)


def subject_of_image(path: Path, threshold: int = 127) -> np.ndarray:
    """Boolean subject mask of a picture: alpha when it has some, else differs from the corners."""
    from PIL import Image

    with Image.open(path) as image:
        if "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 255:
            return np.array(image.getchannel("A")) > threshold
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    background = np.median(corners, axis=0)
    return np.abs(rgb - background).max(axis=2) > 30


def fit_mask(mask: np.ndarray, resolution: int = 256) -> np.ndarray:
    """Crop a mask to its subject, scale it uniformly to fill the frame, and centre it."""
    import cv2

    rows, cols = np.nonzero(mask)
    if not len(rows):
        return np.zeros((resolution, resolution), dtype=bool)
    crop = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.uint8) * 255
    scale = FILL * resolution / max(crop.shape)
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA) > 127
    out = np.zeros((resolution, resolution), dtype=bool)
    top, left = (resolution - small.shape[0]) // 2, (resolution - small.shape[1]) // 2
    out[top:top + small.shape[0], left:left + small.shape[1]] = small
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 1.0


def overlay(reference: np.ndarray, asset: np.ndarray) -> np.ndarray:
    image = np.empty(reference.shape + (3,), dtype=np.uint8)
    image[...] = NEITHER
    image[asset] = ASSET_ONLY
    image[reference] = REFERENCE_ONLY
    image[reference & asset] = BOTH
    return image


def compare(asset, resolution: int = 256) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """IoU per view between the asset and its reference, and the overlay images."""
    vertices, faces = asset.positions
    if asset.reference_kind == "image":
        views = {"front": fit_mask(subject_of_image(asset.reference), resolution)}
    else:
        ref_vertices, ref_faces = asset.reference_asset.positions
        views = {v: mask_of_mesh(ref_vertices, ref_faces, v, resolution) for v in VIEWS}
    scores, images = {}, {}
    for view, reference in views.items():
        mine = mask_of_mesh(vertices, faces, view, resolution)
        scores[view] = round(iou(reference, mine), 4)
        images[view] = overlay(reference, mine)
    return scores, images


def fit_band(labelled: list[tuple[float, bool]]) -> tuple[float, float]:
    """The band [low, high] that best separates approved scores from rejected ones.

    `labelled` holds (score, approved) pairs, for example from the pilot's review queue.
    Edges are tried at midpoints between neighbouring scores; the band that classifies the
    most pairs correctly wins, the narrowest one on a tie.
    """
    if not any(ok for _, ok in labelled):
        raise ValueError("need at least one approved score to fit a band")
    scores = sorted({s for s, _ in labelled})
    edges = [scores[0] - 1e-6] + [(a + b) / 2 for a, b in itertools.pairwise(scores)] + [scores[-1] + 1e-6]
    best = None
    for i, low in enumerate(edges):
        for high in edges[i + 1:]:
            right = sum((low <= s <= high) == ok for s, ok in labelled)
            key = (right, -(high - low))
            if best is None or key > best[0]:
                best = (key, (round(low, 4), round(high, 4)))
    return best[1]


@register("silhouette", needs=("glb", "reference_shape"))
def silhouette(asset, config):
    from PIL import Image

    resolution = int(config.get("resolution", 256))
    band = {"min": config.get("min", 0.65), "max": config.get("max", 0.95),
            "min_view": config.get("min_view", 0.5)}
    scores, images = compare(asset, resolution)
    mean = round(float(np.mean(list(scores.values()))), 4)
    status = grade_band(mean, band["min"], band["max"])
    weakest = min(scores, key=scores.get)
    message = f"silhouette overlap {mean:.2f} (wanted {band['min']}-{band['max']})"
    if mean > band["max"]:
        message += ": practically a copy of the reference"
    elif mean < band["min"]:
        message += ": the reference is not recognisable"
    if status == "pass" and scores[weakest] < band["min_view"]:
        status = "warn"
        message += f"; the {weakest} view is off ({scores[weakest]:.2f})"
    folder = asset.glb.with_name(asset.glb.stem + ".checks")
    folder.mkdir(exist_ok=True)
    written = {}
    for view, image in images.items():
        path = folder / f"silhouette_{view}.png"
        Image.fromarray(image).save(path)
        written[view] = f"{folder.name}/{path.name}"
    return result("silhouette", status, mean, band, message, {"views": scores, "overlays": written})
