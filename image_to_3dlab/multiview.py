"""See an asset from many sides, and bring what a 2D model says about each view back to its atlas.

The material steps all work the same way: Blender renders the albedo unlit from an orbit of
views (``scripts/blender_render_views.py``), a 2D model looks at each view (a segmenter, a
delighter, a material estimator), and its answer is carried back to the texels the view
showed. Each pixel of a view knows its texel from a UV pass, so nothing is guessed:

    plan = render_plan(asset, folder)        # cameras, and a copy with materials tagged
    views = render_views(asset, folder)      # Blender, headless, CPU
    acc = reproject(views, surface, lambda view: model(view.color))
    mean, seen = acc.mean()                  # or vote(acc) for class scores

A texel counts each view by how squarely it was seen; one no view saw takes the value of
the nearest texel that was seen. The parts here are pure and tested; only the render runs
in Blender.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from image_to_3dlab.glb_turn import build_glb, parse_glb
from image_to_3dlab.materials import (
    REPEAT,
    Surface,
    island_labels,
    transform_uv,
    wrap_uv,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "blender_render_views.py"
TAG = "afmat_"


# --------------------------------------------------------------------------- cameras

def orbit_directions(around: int = 16, elevation: float = 20.0) -> np.ndarray:
    """Directions the cameras look along (Blender axes, Z up): a ring, then straight down and up.

    Ring views alternate between looking slightly down and slightly up, so undersides and
    tops both get seen at a useful angle; the last two look straight down and straight up.
    """
    rise = np.radians(elevation)
    out = []
    for k in range(around):
        azimuth = np.radians(k * 360.0 / around)
        tilt = -rise if k % 2 == 0 else rise
        out.append((np.cos(tilt) * np.cos(azimuth), np.cos(tilt) * np.sin(azimuth), np.sin(tilt)))
    out += [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
    return np.array(out)


# --------------------------------------------------------------------------- the copy Blender renders

def tag_materials(data: bytes) -> bytes:
    """The GLB with every material renamed ``afmat_<glTF index>``.

    Blender keeps no glTF index on what it imports and renames clashes (``arm.001``), so
    the render reads the index back from the name instead.
    """
    document, binary = parse_glb(data)
    for index, material in enumerate(document.get("materials", [])):
        material["name"] = f"{TAG}{index}"
    return build_glb(document, binary)


def material_id(name: str) -> int | None:
    """The glTF material index a tagged Blender material stands for, or None."""
    match = re.fullmatch(rf"{TAG}(\d+)(?:\.\d{{3}})?", name or "")
    return int(match.group(1)) if match else None


def render_plan(asset: Path, folder: Path, size: int = 512, around: int = 16,
                elevation: float = 20.0, lit: bool = False, light=None) -> dict:
    """Write the tagged copy and ``plan.json`` that the Blender script reads."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tagged.glb").write_bytes(tag_materials(Path(asset).read_bytes()))
    plan = {"glb": "tagged.glb", "size": int(size),
            "directions": orbit_directions(around, elevation).round(6).tolist()}
    if lit:
        plan["lit"] = True
    if light is not None:
        plan["light"] = [round(float(v), 6) for v in light]
    (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def render_views(asset: Path, folder: Path, *, size: int = 512, around: int = 16, lit: bool = False,
                 light=None, blender: Path | None = None, run: Callable = subprocess.run) -> list[View]:
    """Render the orbit with Blender, headless and on the CPU, and read the views back.

    `lit` adds a picture of each view under a neutral studio light (``View.lit``); `light`, the
    direction it travels in Blender's world axes, fixes it in the world instead of to the camera.
    """
    from image_to_3dlab.blender import find_blender

    folder = Path(folder).resolve()     # Blender runs from the repo, not from the caller's folder
    render_plan(asset, folder, size, around, lit=lit, light=light)
    binary = blender or find_blender()
    if binary is None:
        raise RuntimeError("Blender was not found; install it or set I2L_BLENDER")
    run([str(binary), "-b", "--factory-startup", "-P", str(SCRIPT), "--", str(Path(folder) / "plan.json")],
        check=True, capture_output=True, cwd=REPO)
    return load_views(folder)


# --------------------------------------------------------------------------- views

@dataclass(frozen=True)
class View:
    index: int
    direction: np.ndarray        # where the camera looks, Blender axes
    uv: np.ndarray               # HxWx2, Blender UV (v upward)
    material: np.ndarray         # HxW glTF material index, -1 for background
    normal: np.ndarray           # HxWx3 world normal, Blender axes
    color: np.ndarray | None     # HxWx3 uint8: the albedo as shown, unlit
    lit: np.ndarray | None = None  # HxWx3 uint8: under a neutral studio light, when asked for


def load_views(folder: Path) -> list[View]:
    """The views ``blender_render_views.py`` wrote, as arrays (top row first)."""
    from PIL import Image

    folder = Path(folder)
    plan = json.loads((folder / "views.json").read_text(encoding="utf-8"))
    views = []
    for entry in plan["views"]:
        stem = folder / f"view_{entry['index']:02d}"
        try:
            uv = np.load(str(stem) + "_uv.npz")["rgba"]
            normal = np.load(str(stem) + "_normal.npz")["rgba"]
        except (OSError, KeyError) as exc:
            raise ValueError(f"view {entry['index']} is missing its UV or normal pass in {folder}") from exc
        colour_path = Path(str(stem) + "_color.png")
        colour = np.asarray(Image.open(colour_path).convert("RGB")) if colour_path.is_file() else None
        lit_path = Path(str(stem) + "_lit.png")
        lit = np.asarray(Image.open(lit_path).convert("RGB")) if lit_path.is_file() else None
        views.append(View(entry["index"], np.asarray(entry["direction"], dtype=np.float64),
                          uv[..., :2].astype(np.float64), np.rint(uv[..., 2]).astype(np.int64) - 1,
                          normal[..., :3].astype(np.float64), colour, lit))
    return views


def view_texels(view: View, material: int, size: tuple[int, int], transform: dict | None = None,
                wrap: tuple = (REPEAT, REPEAT)) -> tuple[np.ndarray, np.ndarray]:
    """(pixel, texel) flat indices for every pixel of `view` that shows `material`.

    `size` is the atlas (height, width). Blender's v runs upward; glTF's, and the atlas
    rows, run downward. The texture transform and the sampler's wrap are applied as the
    renderer applies them.
    """
    height, width = size
    pixels = np.flatnonzero(view.material.ravel() == material)
    uv = view.uv.reshape(-1, 2)[pixels]
    uv = transform_uv(np.c_[uv[:, 0], 1.0 - uv[:, 1]], transform)
    cols = np.clip((wrap_uv(uv[:, 0], wrap[0]) * width).astype(np.int64), 0, width - 1)
    rows = np.clip((wrap_uv(uv[:, 1], wrap[1]) * height).astype(np.int64), 0, height - 1)
    return pixels, rows * width + cols


def facing_weight(normals: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """How squarely each surface faced the camera: 1 head-on, 0 edge-on or turned away."""
    direction = np.asarray(direction, dtype=np.float64)
    return np.clip(np.asarray(normals) @ -(direction / np.linalg.norm(direction)), 0.0, 1.0)


class Accumulator:
    """Weighted sums per texel of an atlas, `channels` values each."""

    def __init__(self, size: tuple[int, int], channels: int):
        self.size = size
        self.sums = np.zeros((size[0] * size[1], channels))
        self.weights = np.zeros(size[0] * size[1])

    def add(self, texels: np.ndarray, values: np.ndarray, weights: np.ndarray) -> None:
        np.add.at(self.sums, texels, np.asarray(values, dtype=np.float64) * np.asarray(weights)[:, None])
        np.add.at(self.weights, texels, weights)

    def mean(self) -> tuple[np.ndarray, np.ndarray]:
        """(HxWxC weighted mean, HxW seen) where seen means some view saw the texel."""
        seen = self.weights > 0
        mean = np.zeros_like(self.sums)
        mean[seen] = self.sums[seen] / self.weights[seen, None]
        return mean.reshape(*self.size, -1), seen.reshape(self.size)


def vote(acc: Accumulator) -> tuple[np.ndarray, np.ndarray]:
    """(HxW winning class, HxW its share of the weight) from per-class scores; -1 unseen."""
    mean, seen = acc.mean()
    labels = np.where(seen, mean.argmax(axis=-1), -1)
    return labels, np.where(seen, mean.max(axis=-1), 0.0)


def fill_unseen(values: np.ndarray, seen: np.ndarray, coverage: np.ndarray,
                islands: np.ndarray | None = None) -> np.ndarray:
    """Give covered texels no view saw the value of the nearest seen texel of their own island.

    Atlas neighbours are not surface neighbours, so a fill never crosses into another UV
    island; an island no view saw at all takes the nearest seen texel anywhere.
    """
    if not seen.any():
        return values
    if islands is None:
        islands = island_labels(coverage)
    out = values.copy()
    gap = coverage & ~seen
    _, (rows, cols) = ndimage.distance_transform_edt(~seen, return_indices=True)
    out[gap] = values[rows, cols][gap]                       # fallback for unseen islands
    for number, box in enumerate(ndimage.find_objects(islands), start=1):
        if box is None:
            continue
        mine = islands[box] == number
        own = seen[box] & mine
        holes = gap[box] & mine
        if not own.any() or not holes.any():
            continue
        _, (r, c) = ndimage.distance_transform_edt(~own, return_indices=True)
        part = out[box]
        part[holes] = values[box][r, c][holes]
    return out


def reproject(views: Iterable[View], surface: Surface, values: Callable[[View], np.ndarray],
              channels: int | None = None) -> Accumulator:
    """Carry what `values(view)` says per pixel back to `surface`'s atlas.

    `values(view)` returns HxWxC (or HxW) for a view; each pixel that shows `surface`
    adds its value to its texel, weighted by how squarely the surface faced the camera.
    """
    size = surface.image.shape[:2]
    acc = None
    for view in views:
        pixels, texels = view_texels(view, surface.material, size, surface.transform, surface.wrap)
        if not len(pixels):
            continue
        per_pixel = np.asarray(values(view), dtype=np.float64)
        per_pixel = per_pixel.reshape(per_pixel.shape[0] * per_pixel.shape[1], -1)
        if acc is None:
            acc = Accumulator(size, channels or per_pixel.shape[1])
        acc.add(texels, per_pixel[pixels], facing_weight(view.normal.reshape(-1, 3)[pixels], view.direction))
    return acc if acc is not None else Accumulator(size, channels or 1)
