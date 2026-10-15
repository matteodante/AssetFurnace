"""Budget: triangles, materials and draw calls, texture size, texel density.

Counts come from the glTF itself (accessors, materials, images), read without a mesh
library, so they are what an engine will import. `gltfpack -v`, when it is installed,
adds a second opinion (draw calls, vertices after its own welding, bytes) in `details`.
"""

from __future__ import annotations

import io
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from image_to_3dlab.checks import geometry
from image_to_3dlab.checks.asset import REPO, base_colour
from image_to_3dlab.checks.registry import (
    grade_band,
    grade_max,
    register,
    result,
    worst,
)

TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN = 4, 5, 6


def _scene_nodes(document: dict) -> list[int]:
    """Every node reachable from the default scene (or every root when there is none)."""
    nodes = document.get("nodes", [])
    scenes = document.get("scenes") or []
    if scenes:
        roots = list(scenes[document.get("scene", 0)].get("nodes", []))
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    seen, stack = [], list(roots)
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.append(index)
        stack.extend(nodes[index].get("children", []))
    return seen


def drawn_primitives(document: dict) -> list[dict]:
    """One entry per primitive per node that draws it: what an engine will submit."""
    meshes = document.get("meshes", [])
    out = []
    for node in _scene_nodes(document):
        mesh = document["nodes"][node].get("mesh")
        if mesh is not None:
            out.extend(meshes[mesh].get("primitives", []))
    return out


def triangle_count(document: dict) -> int:
    accessors = document.get("accessors", [])
    total = 0
    for primitive in drawn_primitives(document):
        mode = primitive.get("mode", TRIANGLES)
        if "indices" in primitive:
            count = accessors[primitive["indices"]]["count"]
        else:
            count = accessors[primitive["attributes"]["POSITION"]]["count"]
        if mode == TRIANGLES:
            total += count // 3
        elif mode in (TRIANGLE_STRIP, TRIANGLE_FAN):
            total += max(count - 2, 0)
    return total


def image_sizes(document: dict, binary: bytes, base: Path) -> tuple[list[tuple[int, int]], list[str]]:
    """(width and height of every image, the images that could not be read).

    Read from each header, embedded or beside the file. A glTF URI is percent-encoded
    (`large%20texture.png` is `large texture.png`), so it is decoded before it is looked up.
    """
    from urllib.parse import unquote

    from PIL import Image

    sizes, unreadable = [], []
    views = document.get("bufferViews", [])
    for image in document.get("images", []):
        if "bufferView" in image:
            view = views[image["bufferView"]]
            start = view.get("byteOffset", 0)
            data = binary[start:start + view["byteLength"]]
        elif image.get("uri", "").startswith("data:"):
            import base64

            data = base64.b64decode(image["uri"].split(",", 1)[1])
        else:
            path = base / unquote(image.get("uri", ""))
            data = path.read_bytes() if path.is_file() else b""
        try:
            with Image.open(io.BytesIO(data)) as opened:
                sizes.append(opened.size)
        except Exception:  # noqa: BLE001 - any unreadable image is reported, whatever the reason
            unreadable.append(image.get("uri") or image.get("name") or f"bufferView {image.get('bufferView')}")
    return sizes, unreadable


def gltfpack_stats(glb: Path, binary: Path | None = None, timeout: int = 120) -> dict | None:
    """Parse `gltfpack -v` (meshoptimizer): its input triangles, vertices, draw calls, bytes."""
    binary = binary or next((p for p in (REPO / "vendor" / "gltfpack" / "gltfpack.exe",
                                         REPO / "vendor" / "gltfpack" / "gltfpack") if p.is_file()), None)
    if binary is None:
        return None
    with tempfile.TemporaryDirectory() as scratch:
        try:
            done = subprocess.run([str(binary), "-v", "-i", str(glb), "-o", str(Path(scratch) / "out.glb")],
                                  capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
    return parse_gltfpack(done.stdout + done.stderr)


def parse_gltfpack(text: str) -> dict | None:
    stats: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("input:"):
            prims = re.search(r"\((\d+) triangles, (\d+) vertices\)", line)
            calls = re.search(r"(\d+) draw calls", line)
            if prims:
                stats["triangles"], stats["vertices"] = int(prims.group(1)), int(prims.group(2))
            if calls:
                stats["draw_calls"] = int(calls.group(1))
        elif line.startswith("output: JSON"):
            size = re.search(r"JSON (\d+) bytes, buffers (\d+) bytes", line)
            if size:
                stats["output_bytes"] = int(size.group(1)) + int(size.group(2))
    return stats or None


@register("budget.faces")
def faces(asset, config):
    budget = int(config.get("max", 40000))
    ratio = float(config.get("fail_ratio", 1.25))
    count = triangle_count(asset.document)
    limits = {"warn": budget, "fail": round(budget * ratio)}
    details = {}
    if config.get("gltfpack", True):
        stats = gltfpack_stats(asset.glb)
        if stats:
            details["gltfpack"] = stats
    if count == 0:
        # An empty scene, or points and lines only: nothing for an engine to draw as a surface.
        return result("budget.faces", "fail", 0, limits, "the model draws no triangles at all", details)
    return result("budget.faces", grade_max(count, **limits), count, limits,
                  f"{count:,} triangles against a budget of {budget:,}", details)


@register("budget.materials")
def materials(asset, config):
    """Materials and draw calls: one material reused on many primitives is still many draws."""
    count = len(asset.document.get("materials", []))
    draws = len(drawn_primitives(asset.document))
    limits = {"warn": config.get("warn", 2), "fail": config.get("fail", 4),
              "draw_calls_warn": config.get("draw_calls_warn", 4), "draw_calls_fail": config.get("draw_calls_fail", 8)}
    status = worst([grade_max(count, limits["warn"], limits["fail"]),
                    grade_max(draws, limits["draw_calls_warn"], limits["draw_calls_fail"])])
    return result("budget.materials", status, count, limits,
                  f"{count} material(s), {draws} draw call(s)", {"draw_calls": draws})


@register("budget.textures")
def textures(asset, config):
    sizes, unreadable = image_sizes(asset.document, asset.binary, asset.glb.parent)
    largest = max((max(s) for s in sizes), default=0)
    limits = {"warn": config.get("warn", 2048), "fail": config.get("fail", 4096)}
    details = {"sizes": [list(s) for s in sizes], "unreadable": unreadable}
    if unreadable:
        return result("budget.textures", "fail", largest, limits,
                      f"{len(unreadable)} declared image(s) could not be read, so their size is unknown: "
                      f"{', '.join(unreadable[:3])}", details)
    return result("budget.textures", grade_max(largest, **limits), largest, limits,
                  f"largest texture side {largest} px across {len(sizes)} image(s)", details)


def texel_density(width: int, height: int, uv_area: float, surface_area: float) -> float:
    """Texture pixels per metre of surface: sqrt(pixels x UV share used / surface in m^2)."""
    if surface_area <= 0:
        return 0.0
    return math.sqrt(width * height * uv_area / surface_area)


@register("budget.texel_density", needs=("glb", "texture"))
def density(asset, config):
    weighted, area_total, per_part = 0.0, 0.0, []
    for mesh in asset.primitives:
        image = base_colour(mesh)
        uv = getattr(mesh.visual, "uv", None)
        if image is None or uv is None:
            continue
        surface = float(geometry.triangle_areas(np.asarray(mesh.vertices), np.asarray(mesh.faces)).sum())
        uv_area = float(geometry.triangle_areas(np.asarray(uv, dtype=np.float64), np.asarray(mesh.faces)).sum())
        value = texel_density(image.size[0], image.size[1], uv_area, surface)
        per_part.append(round(value, 1))
        weighted += value * surface
        area_total += surface
    mean = weighted / area_total if area_total else 0.0
    band = {"min": config.get("min", 256), "max": config.get("max", 2048)}
    return result("budget.texel_density", grade_band(mean, band["min"], band["max"]), round(mean, 1), band,
                  f"{mean:.0f} texture pixels per metre (wanted {band['min']}-{band['max']})",
                  {"per_part": per_part})
