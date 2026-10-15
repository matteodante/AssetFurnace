"""Game-ready materials for a generated GLB: tint mask, baked-light checks, maps, ORM.

A game ships a few master materials and thousands of instances. A generated asset fits
that only if its colours can be changed from parameters and its albedo holds up under
the game's own lights. This module gives it both, or sends it back saying why:

- **tint mask**: three recolourable zones (R, G, B) found from the albedo's colour
  regions, drawn texel by texel in UV space so borders follow the paint;
- **baked-light checks**: light painted into the albedo is measured, never corrected.
  Records use the shared check contract (``check``, ``status``, ``value``,
  ``threshold``, ``message``, optional ``details``);
- **normal and roughness from colour** where the asset has none, the Materialize idea
  (height from luminance in frequency bands) rebuilt here: Materialize is GPL-3;
- **ORM**: occlusion, roughness and metallic in one linear image, glTF's own layout.

The GLB is edited in its JSON, never re-exported, so skins and animations survive.
The design and its calibration are in ``docs/plans/2026-10-15-game-materials-design.md``.
"""

from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import color

from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb

SIDECAR_SCHEMA = "assetfurnace.materials/1"
CHANNELS = ("R", "G", "B")
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)

# Colour regions. Lightness counts half in zone distance, so a shadow does not make a
# zone. The light checks use the same groups: at a quarter, black and grey cloth merged
# and their real difference read as painted light (1.73 on a clean warrior, 1.22 here).
ZONE_LIGHTNESS_WEIGHT = 0.5
MAX_CLUSTERS = 12           # 6 merged a warrior's skin into the red cloth beside it
MERGE_DISTANCE = 10.0
MIN_ZONE_SHARE = 0.05
MIN_GROUP_SAMPLES = 200
# Shade lowers lightness and chroma together and leaves hue alone, so two groups with
# the same hue and the same chroma-to-lightness ratio are one pigment, lit and shaded.
PIGMENT_MIN_CHROMA = 8.0
PIGMENT_HUE_DEGREES = 12.0
PIGMENT_RATIO = 1.25

# Thresholds, calibrated on existing outputs (design doc). Re-tune on the pilot.
# Light ratio: clean generated albedos measure 1.00-1.22; a key light baked into a real
# warrior's albedo on purpose (35% ambient) measures only 1.55, because cast shadows and
# mixed colours dilute what a lone sphere shows (2.7). Photo-projected finishes carry
# the concept's light: one girl went from 1.02 raw to 1.52 finished.
LIGHT_RATIO = {"warn": 1.3, "fail": 1.45}
OCCLUSION_SLOPE = {"warn": 0.3, "fail": 0.6}
HIGHLIGHT_SHARE = {"warn": 0.005, "fail": 0.02}
HIGHLIGHT_GAIN = 2.5          # brighter than this many times the local median
HIGHLIGHT_MIN_L = 70.0        # and actually bright, not just a light speck in black
HIGHLIGHT_MAX_CHROMA = 15.0   # and nearly colourless, as a specular reflection is
HIGHLIGHT_NEIGHBOURS = 100    # out of a 5,000-point subsample: about 2% of the surface
HIGHLIGHT_CONE_DEGREES = 45.0
HIGHLIGHT_CONCENTRATION = (0.4, 0.7)

DEFAULT_ROUGHNESS = 0.8
ROUGHNESS_DETAIL = 0.15
HEIGHT_SIGMAS = (1, 2, 4, 8, 16)
HEIGHT_WEIGHTS = (0.5, 0.8, 1.0, 0.7, 0.4)
GUTTER_PX = 8


# --------------------------------------------------------------------------- colour

def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Relative luminance Y of linear RGB (last axis)."""
    return np.asarray(linear_rgb, dtype=np.float64) @ np.array(LUMINANCE_WEIGHTS)


def zone_features(lab: np.ndarray, lightness_weight: float = ZONE_LIGHTNESS_WEIGHT) -> np.ndarray:
    """CIELAB with lightness down-weighted: the space zones are measured in."""
    lab = np.asarray(lab, dtype=np.float64)
    return np.c_[lab[:, 0] * lightness_weight, lab[:, 1], lab[:, 2]]


@dataclass
class Clusters:
    centres: np.ndarray   # (K, 3) in feature space, largest share first
    shares: np.ndarray    # (K,)
    labels: np.ndarray    # (N,) index into centres


def _kmeans(features: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25):
    """k-means++ seeding then Lloyd iterations; small, deterministic for a given rng."""
    centres = [features[rng.integers(len(features))]]
    for _ in range(1, k):
        d2 = np.min([((features - c) ** 2).sum(1) for c in centres], axis=0)
        if d2.sum() <= 0:
            break
        centres.append(features[rng.choice(len(features), p=d2 / d2.sum())])
    centres = np.array(centres)
    for _ in range(iters):
        labels = ((features[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        moved = np.array([features[labels == i].mean(0) if (labels == i).any() else centres[i]
                          for i in range(len(centres))])
        if np.allclose(moved, centres):
            break
        centres = moved
    labels = ((features[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
    return centres, labels


def same_pigment(lab_a, lab_b) -> bool:
    """True when two CIELAB colours look like one pigment under different light."""
    (la, aa, ba), (lb, ab, bb) = lab_a, lab_b
    ca, cb = np.hypot(aa, ba), np.hypot(ab, bb)
    if min(ca, cb) < PIGMENT_MIN_CHROMA or min(la, lb) <= 0:
        return False
    hue = abs((np.degrees(np.arctan2(ba, aa) - np.arctan2(bb, ab)) + 180) % 360 - 180)
    ratio = (ca / la) / (cb / lb)
    return bool(hue <= PIGMENT_HUE_DEGREES and 1 / PIGMENT_RATIO <= ratio <= PIGMENT_RATIO)


def cluster_colours(features: np.ndarray, max_clusters: int = MAX_CLUSTERS,
                    merge_distance: float = MERGE_DISTANCE, seed: int = 0,
                    lightness_weight: float = ZONE_LIGHTNESS_WEIGHT) -> Clusters:
    """Group colours, then merge groups that are close or are one pigment in shade.

    k-means over-splits on purpose (one material with a lightness gradient becomes
    several slices); merging the closest pair, centroid to centroid, puts the slices
    back together. A pair merges when it is nearer than `merge_distance` or when
    ``same_pigment`` says it is one colour lit and shaded; the nearest such pair goes
    first, until none is left.
    """
    features = np.asarray(features, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centres, labels = _kmeans(features, min(max_clusters, len(features)), rng)
    counts = np.bincount(labels, minlength=len(centres)).astype(float)
    groups = [[i] for i in range(len(centres)) if counts[i] > 0]
    cents = [centres[i] for i in range(len(centres)) if counts[i] > 0]
    weights = [counts[i] for i in range(len(centres)) if counts[i] > 0]
    def lab(feature):
        return feature[0] / lightness_weight, feature[1], feature[2]

    while len(cents) > 1:
        pairs = [(np.linalg.norm(cents[a] - cents[b]), a, b)
                 for a in range(len(cents)) for b in range(a + 1, len(cents))
                 if np.linalg.norm(cents[a] - cents[b]) < merge_distance
                 or same_pigment(lab(cents[a]), lab(cents[b]))]
        if not pairs:
            break
        _, a, b = min(pairs)
        total = weights[a] + weights[b]
        cents[a] = (cents[a] * weights[a] + cents[b] * weights[b]) / total
        weights[a] = total
        groups[a] += groups.pop(b)
        cents.pop(b)
        weights.pop(b)
    order = np.argsort(weights)[::-1]
    remap = np.empty(len(centres), dtype=int)
    for new, old in enumerate(order):
        remap[groups[old]] = new
    merged = remap[labels]
    shares = np.bincount(merged, minlength=len(order)) / len(merged)
    return Clusters(np.array([cents[i] for i in order]), shares, merged)


def nearest_cluster(features: np.ndarray, centres: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    out = np.empty(len(features), dtype=np.int16)
    for start in range(0, len(features), chunk):
        block = features[start:start + chunk]
        out[start:start + chunk] = ((block[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
    return out


def apply_tint(albedo_linear: np.ndarray, mask: np.ndarray, means: np.ndarray,
               tints: np.ndarray) -> np.ndarray:
    """The tint formula, in linear light: ``out = albedo + m * g * (t - c)``.

    ``g = luminance(albedo) / luminance(c)`` is the zone's brightness detail, 1 on
    average, so a zone's mean lands exactly on its tint and ``t = c`` changes nothing.
    Unreal reproduces this same formula from the sidecar.
    """
    albedo = np.asarray(albedo_linear, dtype=np.float64)
    out = albedo.copy()
    y = luminance(albedo)
    for i in range(min(3, len(means))):
        yc = float(luminance(np.asarray(means[i])))
        g = y / yc if yc > 1e-6 else np.ones_like(y)
        delta = np.asarray(tints[i], dtype=np.float64) - np.asarray(means[i], dtype=np.float64)
        out += mask[..., i:i + 1] * g[..., None] * delta
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- check records

def check_record(check: str, status: str, value, threshold, message: str, details=None) -> dict:
    record = {"check": check, "status": status, "value": value,
              "threshold": threshold, "message": message}
    if details is not None:
        record["details"] = details
    return json.loads(json.dumps(record, default=_plain))


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _grade(value: float, threshold: dict) -> str:
    if value > threshold["fail"]:
        return "fail"
    if value > threshold["warn"]:
        return "warn"
    return "pass"


def worst(statuses) -> str:
    order = {"skipped": 0, "pass": 1, "warn": 2, "fail": 3}
    present = [s for s in statuses if s in order]
    picked = max(present, key=order.__getitem__) if present else "pass"
    return "pass" if picked == "skipped" else picked


def light_direction_record(y: np.ndarray, normals: np.ndarray, groups: Clusters) -> dict:
    """How much brighter the lit side is than the shadow side, per colour group.

    Within one group (one material) ``log Y = a + g . n`` is fitted against the surface
    normal; ``exp(2 |g|)`` is the ratio between the side facing the painted light and
    the side facing away. Painted light is multiplicative, so the log makes the ratio
    independent of how dark the material is.
    """
    ratios, shares, vectors, per_group = [], [], [], []
    for index, share in enumerate(groups.shares):
        chosen = (groups.labels == index) & (y > 1e-3)
        if share < MIN_ZONE_SHARE or chosen.sum() < MIN_GROUP_SAMPLES:
            continue
        design = np.c_[np.ones(chosen.sum()), normals[chosen]]
        coef, *_ = np.linalg.lstsq(design, np.log(y[chosen]), rcond=None)
        gradient = coef[1:]
        ratio = float(np.exp(2 * np.linalg.norm(gradient)))
        ratios.append(ratio)
        shares.append(share)
        vectors.append(gradient)
        per_group.append({"group": index, "share": round(float(share), 4), "ratio": round(ratio, 3)})
    if not ratios:
        return check_record("baked_light_direction", "skipped", None, LIGHT_RATIO,
                            "no colour group is large enough to measure")
    value = float(np.average(ratios, weights=shares))
    direction = np.average(vectors, axis=0, weights=shares)
    direction = direction / (np.linalg.norm(direction) or 1.0)
    worst_group = max(per_group, key=lambda g: g["ratio"])
    status = _grade(value, LIGHT_RATIO)
    message = (f"lit side {value:.2f}x brighter than shadow side "
               f"(worst group {worst_group['group']}: {worst_group['ratio']:.2f}x)")
    return check_record("baked_light_direction", status, round(value, 3), LIGHT_RATIO, message,
                        {"direction": np.round(direction, 3), "groups": per_group})


def occlusion_record(y: np.ndarray, ao: np.ndarray | None, groups: Clusters) -> dict:
    """How strongly albedo darkens where the geometry occludes: ``Y ~ AO ** s``."""
    if ao is None:
        return check_record("baked_light_occlusion", "skipped", None, OCCLUSION_SLOPE,
                            "no ambient-occlusion bake given (scripts/blender_bake_ao.py)")
    if float(np.std(ao)) < 0.02:
        return check_record("baked_light_occlusion", "skipped", None, OCCLUSION_SLOPE,
                            "the ambient-occlusion bake is flat, nothing to compare against")
    slopes, shares = [], []
    for index, share in enumerate(groups.shares):
        chosen = (groups.labels == index) & (y > 1e-3)
        if share < MIN_ZONE_SHARE or chosen.sum() < MIN_GROUP_SAMPLES:
            continue
        log_ao = np.log(np.clip(ao[chosen], 0.02, 1.0))
        if np.std(log_ao) < 1e-3:
            continue
        design = np.c_[np.ones(chosen.sum()), log_ao]
        coef, *_ = np.linalg.lstsq(design, np.log(y[chosen]), rcond=None)
        slopes.append(float(coef[1]))
        shares.append(share)
    if not slopes:
        return check_record("baked_light_occlusion", "skipped", None, OCCLUSION_SLOPE,
                            "the bake does not vary inside any colour group")
    value = max(0.0, float(np.average(slopes, weights=shares)))
    status = _grade(value, OCCLUSION_SLOPE)
    return check_record("baked_light_occlusion", status, round(value, 3), OCCLUSION_SLOPE,
                        f"albedo follows the occlusion bake with exponent {value:.2f} (0 = clean)")


def highlight_record(points: np.ndarray, normals: np.ndarray, y: np.ndarray, lab: np.ndarray,
                     seed: int = 0) -> dict:
    """Share of the surface that is a painted highlight: bright, colourless, facing one way.

    A bright colourless spot on a darker neighbourhood is only a candidate: cream cloth
    beside dark purple is one too (3.4% of a generated girl's surface). What gives a
    painted highlight away is that one light put it there, so its normals point the
    same way, while a light material faces every way. Concentration is the share of
    candidates within 45 degrees of their mean normal: painted highlights measure about
    1.0, scattered light material 0.15-0.3. Below 0.4 nothing counts, above 0.7 every
    candidate in the cone does, with a straight ramp between.
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(points), size=min(5000, len(points)), replace=False)
    k = min(HIGHLIGHT_NEIGHBOURS, len(pick))
    _, neighbours = cKDTree(points[pick]).query(points, k=k)
    local = np.median(y[pick][neighbours], axis=1)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    spot = (y > HIGHLIGHT_GAIN * local) & (lab[:, 0] > HIGHLIGHT_MIN_L) & (chroma < HIGHLIGHT_MAX_CHROMA)
    details = {"candidates": round(float(spot.mean()), 5)}
    value = 0.0
    mean = normals[spot].mean(0) if spot.sum() >= 10 else np.zeros(3)
    if np.linalg.norm(mean) > 1e-6:
        direction = mean / np.linalg.norm(mean)
        cone = normals @ direction > np.cos(np.radians(HIGHLIGHT_CONE_DEGREES))
        concentration = float((spot & cone).sum() / spot.sum())
        low, high = HIGHLIGHT_CONCENTRATION
        value = float((spot & cone).mean()) * float(np.clip((concentration - low) / (high - low), 0, 1))
        details.update({"concentration": round(concentration, 3), "direction": np.round(direction, 3)})
    status = _grade(value, HIGHLIGHT_SHARE)
    return check_record("baked_light_highlights", status, round(value, 5), HIGHLIGHT_SHARE,
                        f"{value * 100:.2f}% of the surface looks like a painted highlight", details)


def tint_zones_record(zones: list[dict], source: str, distance: float | None) -> dict:
    count = len(zones)
    threshold = {"zones": 3, "min_share": MIN_ZONE_SHARE, "min_distance": MERGE_DISTANCE}
    if count >= 3:
        status, message = "pass", "3 recolourable zones"
    elif count == 2:
        status, message = "warn", "only 2 recolourable zones; channel B left empty"
    else:
        what = "material slot" if source == "materials" else "colour region"
        status, message = "fail", f"no separate {what}s to tint: the albedo needs zones"
    details = {"source": source, "shares": [z["area_share"] for z in zones]}
    if distance is not None:
        details["min_distance"] = round(distance, 2)
    return check_record("tint_zones", status, count, threshold, message, details)


# --------------------------------------------------------------------------- image maths

def uv_coverage(uv: np.ndarray, faces: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Texels covered by at least one UV triangle (glTF UVs: v = 0 is the top row)."""
    width, height = size
    tri = uv[faces] * np.array([width, height])
    coverage = np.zeros((height, width), np.uint8)
    try:
        import cv2
        cv2.fillPoly(coverage, np.round(tri * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    except ImportError:  # pragma: no cover - opencv ships with rembg in this repo
        from PIL import ImageDraw
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for t in tri:
            draw.polygon([tuple(p) for p in t], fill=1)
        coverage = np.asarray(canvas)
    # Sub-texel triangles fill nothing; one texel of growth catches them.
    return ndimage.binary_dilation(coverage.astype(bool), iterations=1)


def masked_blur(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores texels outside `mask` instead of averaging them in."""
    weight = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    if image.ndim == 3:
        return np.dstack([masked_blur(image[..., c], mask, sigma) for c in range(image.shape[2])])
    total = ndimage.gaussian_filter(np.where(mask, image, 0).astype(np.float32), sigma)
    return np.where(weight > 1e-6, total / np.maximum(weight, 1e-6), 0.0)


def fill_gutter(image: np.ndarray, coverage: np.ndarray, max_px: float | None = GUTTER_PX):
    """Copy each covered texel outward into the gutter, so mipmaps do not bleed black."""
    if coverage.all() or not coverage.any():
        return image
    distance, (rows, cols) = ndimage.distance_transform_edt(~coverage, return_indices=True)
    filled = image[rows, cols]
    if max_px is None:
        return filled
    reach = (distance <= max_px)
    if image.ndim == 3:
        reach = reach[..., None]
    return np.where(reach, filled, image)


def normal_from_height(height: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Tangent-space normal map, glTF convention (+Y up), same encoding as surface_detail."""
    if height.ndim != 2:
        raise ValueError("height must be a 2-D array")
    dy, dx = np.gradient(height.astype(np.float32))
    nx = -dx * strength
    ny = dy * strength
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    stacked = np.dstack([nx / length, ny / length, nz / length])
    return np.clip((stacked * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def height_from_luminance(lum: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Height as a weighted sum of luminance frequency bands (the Materialize idea).

    Each band is the difference between two masked blurs, so fine cracks and broad
    plates both contribute, and the gutter never does.
    """
    levels = [np.where(coverage, lum, 0.0)] + [masked_blur(lum, coverage, s) for s in HEIGHT_SIGMAS]
    height = np.full(lum.shape, 0.5, dtype=np.float64)
    for weight, fine, coarse in zip(HEIGHT_WEIGHTS, levels[:-1], levels[1:]):
        height += weight * (fine - coarse)
    return fill_gutter(np.where(coverage, height, 0.5), coverage, max_px=None)


def normal_from_albedo(lum: np.ndarray, coverage: np.ndarray, strength: float = 6.0) -> np.ndarray:
    """Normal map from an albedo's 0..1 luminance; `strength` is per 1024 texels."""
    height = height_from_luminance(np.asarray(lum, dtype=np.float64), coverage)
    return normal_from_height(height, strength * lum.shape[1] / 1024.0)


def pack_orm(occlusion: np.ndarray, roughness: np.ndarray, metallic: np.ndarray) -> np.ndarray:
    """glTF and Unreal ORM layout: R occlusion, G roughness, B metallic, linear 0..255."""
    stacked = np.dstack([occlusion, roughness, metallic]).astype(np.float64)
    return np.round(np.clip(stacked, 0.0, 1.0) * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- GLB

def set_view_bytes(document: dict, binary: bytes, view_index: int, data: bytes) -> bytes:
    """Replace one bufferView's bytes and lay every view out again, 4-byte aligned."""
    out = bytearray()
    for index, view in enumerate(document["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunk = data if index == view_index else binary[start:start + view["byteLength"]]
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"] = len(out)
        view["byteLength"] = len(chunk)
        out += chunk
    document["buffers"][0]["byteLength"] = len(out)
    return bytes(out)


def add_view(document: dict, binary: bytes, data: bytes) -> tuple[int, bytes]:
    binary = binary[:document["buffers"][0]["byteLength"]]
    binary += b"\0" * (-len(binary) % 4)
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
    binary += data
    document["buffers"][0]["byteLength"] = len(binary)
    return len(document["bufferViews"]) - 1, binary


def add_texture(document: dict, binary: bytes, data: bytes, name: str, sampler) -> tuple[int, bytes]:
    view, binary = add_view(document, binary, data)
    document.setdefault("images", []).append({"bufferView": view, "mimeType": "image/png", "name": name})
    texture = {"source": len(document["images"]) - 1}
    if sampler is not None:
        texture["sampler"] = sampler
    document.setdefault("textures", []).append(texture)
    return len(document["textures"]) - 1, binary


def _png(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG", optimize=False, compress_level=6)
    return out.getvalue()


def _texture_users(document: dict) -> dict[int, list[tuple[int, str]]]:
    """Which (material, slot) pairs reference each texture index."""
    users: dict[int, list[tuple[int, str]]] = {}

    def walk(node, material, key):
        if isinstance(node, dict):
            if "index" in node and isinstance(node["index"], int) and (
                    key.endswith("Texture") or key == "tintMask"):
                users.setdefault(node["index"], []).append((material, key))
            for k, v in node.items():
                walk(v, material, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, material, key)

    for index, material in enumerate(document.get("materials", [])):
        walk(material, index, "")
    return users


# --------------------------------------------------------------------------- analysis

@dataclass
class _Surface:
    material: int
    image: np.ndarray            # base colour, sRGB uint8 HxWx3
    factor: np.ndarray           # baseColorFactor rgb
    texcoord: int
    uv: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    positions: list = field(default_factory=list)


def _decode_image(glb: EditGLB, image_index: int, mode: str = "RGB") -> np.ndarray:
    image = glb.document["images"][image_index]
    if "bufferView" not in image:
        raise ValueError("only images embedded in the GLB are supported")
    view = glb.document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return np.asarray(Image.open(io.BytesIO(glb.binary[start:start + view["byteLength"]])).convert(mode))


def texture_source(texture: dict) -> int:
    """The image a texture reads: core ``source``, or one from an image-format extension."""
    if "source" in texture:
        return texture["source"]
    for extension in (texture.get("extensions") or {}).values():
        if isinstance(extension, dict) and isinstance(extension.get("source"), int):
            return extension["source"]
    raise ValueError("texture has no image source")


def _texture_image(glb: EditGLB, info: dict | None, mode: str = "RGB") -> np.ndarray | None:
    if not info:
        return None
    texture = glb.document["textures"][info["index"]]
    return _decode_image(glb, texture_source(texture), mode)


def _surfaces(glb: EditGLB) -> dict[int, _Surface]:
    document = glb.document
    globals_, _ = glb.node_globals()
    surfaces: dict[int, _Surface] = {}
    for node_index, node in enumerate(document.get("nodes", [])):
        if "mesh" not in node:
            continue
        matrix = globals_[node_index]
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            if primitive.get("mode", 4) != 4 or "material" not in primitive:
                continue
            material_index = primitive["material"]
            material = document["materials"][material_index]
            pbr = material.get("pbrMetallicRoughness", {})
            base = pbr.get("baseColorTexture")
            if base is None:
                continue
            texcoord = base.get("texCoord", 0)
            uv_key = f"TEXCOORD_{texcoord}"
            if uv_key not in primitive["attributes"]:
                continue
            if material_index not in surfaces:
                surfaces[material_index] = _Surface(
                    material_index, _texture_image(glb, base),
                    np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float64), texcoord)
            surface = surfaces[material_index]
            positions = glb.accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
            positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
            faces = (glb.accessor(primitive["indices"]).astype(np.int64).reshape(-1, 3)
                     if "indices" in primitive else np.arange(len(positions)).reshape(-1, 3))
            surface.positions.append(positions)
            surface.faces.append(faces)
            surface.uv.append(glb.accessor(primitive["attributes"][uv_key]).astype(np.float64))
    if not surfaces:
        raise ValueError("the GLB has no textured triangle primitives")
    return surfaces


def _lookup(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip((np.mod(uv[:, 0], 1.0) * width).astype(int), 0, width - 1)
    y = np.clip((np.mod(uv[:, 1], 1.0) * height).astype(int), 0, height - 1)
    return image[y, x]


@dataclass
class _Samples:
    points: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    material: np.ndarray
    srgb: np.ndarray       # 0..1
    linear: np.ndarray
    lab: np.ndarray
    y: np.ndarray


def _sample(surfaces: dict[int, _Surface], count: int, seed: int) -> _Samples:
    """Area-weighted points on every textured surface, with their albedo."""
    rng = np.random.default_rng(seed)
    triangles, uvs, owners = [], [], []
    for index, surface in surfaces.items():
        for positions, faces, uv in zip(surface.positions, surface.faces, surface.uv):
            triangles.append(positions[faces])
            uvs.append(uv[faces])
            owners.append(np.full(len(faces), index))
    tri, tri_uv, owner = np.concatenate(triangles), np.concatenate(uvs), np.concatenate(owners)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    if area.sum() <= 0:
        raise ValueError("the GLB has no surface area")
    chosen = rng.choice(len(tri), size=count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    bary = np.c_[1 - r1, r1 * (1 - r2), r1 * r2]
    points = (tri[chosen] * bary[..., None]).sum(1)
    uv = (tri_uv[chosen] * bary[..., None]).sum(1)
    normals = cross[chosen] / np.maximum(area[chosen], 1e-12)[:, None]
    material = owner[chosen]
    srgb = np.zeros((count, 3))
    linear = np.zeros((count, 3))
    for index, surface in surfaces.items():
        here = material == index
        texels = _lookup(surface.image, uv[here]) / 255.0
        srgb[here] = texels
        linear[here] = srgb_to_linear(texels) * surface.factor
    lab = color.rgb2lab(linear_to_srgb(linear)[None])[0]
    return _Samples(points, normals, uv, material, srgb, linear, lab, luminance(linear))


@dataclass
class _Analysis:
    surfaces: dict[int, _Surface]
    samples: _Samples
    zone_clusters: Clusters
    light_groups: Clusters
    zones: list[dict]
    zone_ids: list           # cluster index (colour) or material index (materials) per zone
    checks: list[dict]
    ao: dict[int, np.ndarray]


def _zones(samples: _Samples, clusters: Clusters, surfaces, zones_from: str):
    if zones_from == "materials":
        shares = {i: float((samples.material == i).mean()) for i in surfaces}
        ids = [i for i in sorted(shares, key=shares.get, reverse=True) if shares[i] >= MIN_ZONE_SHARE][:3]
        members = [samples.material == i for i in ids]
        distance = None
    elif zones_from == "colour":
        ids = [i for i, share in enumerate(clusters.shares) if share >= MIN_ZONE_SHARE][:3]
        members = [clusters.labels == i for i in ids]
        pairs = [np.linalg.norm(clusters.centres[a] - clusters.centres[b])
                 for a in ids for b in ids if a < b]
        distance = float(min(pairs)) if pairs else None
    else:
        raise ValueError("zones_from must be 'colour' or 'materials'")
    zones = []
    if len(ids) < 2:
        return [], [], distance
    for channel, chosen in zip(CHANNELS, members):
        mean_linear = samples.linear[chosen].mean(0)
        hex_colour = "#" + "".join(f"{round(v * 255):02x}" for v in linear_to_srgb(mean_linear))
        zones.append({"channel": channel, "source": zones_from,
                      "mean_linear": [round(float(v), 5) for v in mean_linear],
                      "mean_srgb_hex": hex_colour,
                      "lab": [round(float(v), 2) for v in samples.lab[chosen].mean(0)],
                      "area_share": round(float(chosen.mean()), 4)})
    return zones, ids, distance


def _load_ao(path: Path | None, surfaces) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    ao = Image.open(path).convert("L")
    return {index: np.asarray(ao.resize(s.image.shape[1::-1], Image.BILINEAR)) / 255.0
            for index, s in surfaces.items()}


def _analyse(glb: EditGLB, ao_path, zones_from: str, samples: int, seed: int) -> _Analysis:
    surfaces = _surfaces(glb)
    s = _sample(surfaces, samples, seed)
    zone_clusters = cluster_colours(zone_features(s.lab), seed=seed)
    light_groups = zone_clusters
    zones, ids, distance = _zones(s, zone_clusters, surfaces, zones_from)
    ao_maps = _load_ao(ao_path, surfaces)
    ao_samples = None
    if ao_maps:
        ao_samples = np.zeros(len(s.y))
        for index, ao in ao_maps.items():
            here = s.material == index
            ao_samples[here] = _lookup(ao, s.uv[here])
    checks = [
        tint_zones_record(zones, zones_from, distance),
        light_direction_record(s.y, s.normals, light_groups),
        occlusion_record(s.y, ao_samples, light_groups),
        highlight_record(s.points, s.normals, s.y, s.lab, seed),
    ]
    return _Analysis(surfaces, s, zone_clusters, light_groups, zones, ids, checks, ao_maps)


def run_checks(asset: Path, ao: Path | None = None, *, zones_from: str = "colour",
               samples: int = 60000, seed: int = 0) -> list[dict]:
    """The four material checks for one GLB, as shared-contract records."""
    glb = EditGLB(Path(asset).read_bytes())
    return _analyse(glb, ao, zones_from, samples, seed).checks


# --------------------------------------------------------------------------- maps

def _mask_image(surface: _Surface, analysis: _Analysis, coverage: np.ndarray,
                zones_from: str) -> np.ndarray:
    height, width = coverage.shape
    channels = np.zeros((height, width, 3), np.float32)
    if zones_from == "materials":
        for c, material in enumerate(analysis.zone_ids):
            if material == surface.material:
                channels[..., c] = coverage
        return channels
    srgb = surface.image.reshape(-1, 3) / 255.0
    linear = srgb_to_linear(srgb) * surface.factor
    lab = np.concatenate([color.rgb2lab(linear_to_srgb(linear[i:i + (1 << 20)])[None])[0]
                          for i in range(0, len(linear), 1 << 20)])
    labels = nearest_cluster(zone_features(lab), analysis.zone_clusters.centres).reshape(height, width)
    # Majority vote over 5x5 texels, so stray specks join their neighbourhood.
    best, best_label = np.full(labels.shape, -1.0, np.float32), labels.copy()
    for k in range(len(analysis.zone_clusters.centres)):
        votes = ndimage.uniform_filter(((labels == k) & coverage).astype(np.float32), size=5)
        better = votes > best
        best[better], best_label[better] = votes[better], k
    for c, cluster in enumerate(analysis.zone_ids):
        hard = ((best_label == cluster) & coverage).astype(np.float32)
        channels[..., c] = np.where(coverage, masked_blur(hard, coverage, 1.0), 0.0)
    return channels


def _zone_index(mask: np.ndarray) -> np.ndarray:
    """Per texel: 0..2 for the zone it belongs to, 3 for none."""
    index = mask.argmax(-1)
    return np.where(mask.max(-1) > 0.5, index, 3)


def _orm(surface: _Surface, material: dict, glb: EditGLB, mask: np.ndarray, coverage,
         ao: np.ndarray | None, roughness_by_zone: dict, metallic_by_zone: dict):
    height, width = coverage.shape
    pbr = material.get("pbrMetallicRoughness", {})
    rough_factor = float(pbr.get("roughnessFactor", 1.0))
    metal_factor = float(pbr.get("metallicFactor", 1.0))
    mr = _texture_image(glb, pbr.get("metallicRoughnessTexture"))
    if mr is not None and mr.shape[:2] != (height, width):
        mr = np.asarray(Image.fromarray(mr).resize((width, height), Image.BILINEAR))
    zone = _zone_index(mask)
    maps = {}

    if mr is not None and float(mr[..., 1][coverage].std()) >= 2.0:
        rough = mr[..., 1] / 255.0 * rough_factor
        maps["roughness"] = "kept"
    else:
        lum = luminance(srgb_to_linear(surface.image / 255.0))
        base = np.array([roughness_by_zone.get(c, DEFAULT_ROUGHNESS) for c in CHANNELS]
                        + [DEFAULT_ROUGHNESS])[zone]
        detail = np.zeros_like(lum)
        for z in range(4):
            here = (zone == z) & coverage
            if here.any():
                median = max(float(np.median(lum[here])), 0.02)
                detail[zone == z] = (median - lum[zone == z]) / median
        rough = base + ROUGHNESS_DETAIL * np.clip(detail, -1, 1)
        maps["roughness"] = "derived"

    if mr is not None:
        metal = mr[..., 2] / 255.0 * metal_factor
        maps["metallic"] = "kept"
    else:
        metal = np.full((height, width), metal_factor)
        maps["metallic"] = "factor"
    for c, value in metallic_by_zone.items():
        metal = np.where(zone == CHANNELS.index(c), value, metal)
        maps["metallic"] = "zone"

    occlusion = ao if ao is not None else np.ones((height, width))
    maps["occlusion"] = "baked" if ao is not None else "none"
    orm = pack_orm(occlusion, rough, metal)
    return fill_gutter(orm, coverage), maps


# --------------------------------------------------------------------------- entry point

@dataclass
class MaterialsResult:
    status: str
    checks: list[dict]
    zones: list[dict]
    glb_path: Path | None
    sidecar_path: Path
    textures: dict[str, Path]
    maps: dict[str, str]


def _tint_section(zones: list[dict]) -> dict:
    return {
        "space": "linear",
        "formula": "out = albedo + m * g * (t - c)",
        "g": "luminance(albedo) / luminance(c)",
        "luminance_weights": list(LUMINANCE_WEIGHTS),
        "clamp": [0.0, 1.0],
        "terms": {"albedo": "base colour, linear", "m": "tint mask channel, linear 0..1",
                  "c": "zone mean_linear", "t": "tint parameter, linear; t = c changes nothing"},
        "parameters": {z["channel"]: {"default": z["mean_linear"]} for z in zones},
    }


def make_game_materials(asset: Path, out_dir: Path, *, ao: Path | None = None,
                        zones_from: str = "colour", texture_prefix: str | None = None,
                        roughness_by_zone: dict | None = None, metallic_by_zone: dict | None = None,
                        normal_strength: float = 6.0, samples: int = 60000, seed: int = 0
                        ) -> MaterialsResult:
    """Check one GLB's albedo and, if it passes, write it back game-ready.

    Writes ``<out_dir>/<stem>.glb`` (normal, ORM and tint mask inside), the textures
    beside it and ``<stem>.materials.json``. On any failed check only the sidecar is
    written: the asset goes back with its reasons, nothing is half-made.
    """
    asset, out_dir = Path(asset), Path(out_dir)
    for overrides in (roughness_by_zone or {}, metallic_by_zone or {}):
        bad = set(overrides) - set(CHANNELS)
        if bad:
            raise ValueError(f"unknown zone channels: {sorted(bad)}")
    target = out_dir / f"{asset.stem}.glb"
    if target.exists() and target.samefile(asset):
        raise ValueError("out_dir is the asset's own folder: the result would overwrite the source")
    glb = EditGLB(asset.read_bytes())
    analysis = _analyse(glb, ao, zones_from, samples, seed)
    status = worst(r["status"] for r in analysis.checks)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = asset.stem
    prefix = texture_prefix or f"T_{stem}"
    sidecar_path = out_dir / f"{stem}.materials.json"
    sidecar = {"schema": SIDECAR_SCHEMA, "asset": asset.name, "status": status,
               "zones": analysis.zones, "tint": _tint_section(analysis.zones),
               "checks": analysis.checks}
    if status == "fail":
        # A batch reads <stem>.glb as "done"; an older pass must not outlive this verdict.
        (out_dir / f"{stem}.glb").unlink(missing_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return MaterialsResult(status, analysis.checks, analysis.zones, None, sidecar_path, {}, {})

    document = copy.deepcopy(glb.document)
    binary = glb.binary[:glb.buffer_size]
    users = _texture_users(document)
    per_material, first_textures, first_maps = [], {}, {}
    many = len(analysis.surfaces) > 1
    for surface in analysis.surfaces.values():
        material = document["materials"][surface.material]
        pbr = material.setdefault("pbrMetallicRoughness", {})
        base_info = pbr["baseColorTexture"]
        sampler = document["textures"][base_info["index"]].get("sampler")
        height, width = surface.image.shape[:2]
        coverage = uv_coverage(np.concatenate(surface.uv),
                               np.concatenate([f + sum(len(u) for u in surface.uv[:i])
                                               for i, f in enumerate(surface.faces)]),
                               (width, height))
        name = prefix + (f"_{material.get('name') or surface.material}" if many else "")
        written: dict[str, Path] = {}

        mask = _mask_image(surface, analysis, coverage, zones_from)
        mask_u8 = np.round(fill_gutter(mask, coverage) * 255).astype(np.uint8)
        written["tint_mask"] = out_dir / f"{name}_Tint.png"
        Image.fromarray(mask_u8).save(written["tint_mask"])
        mask_tex, binary = add_texture(document, binary, _png(mask_u8), f"{name}_Tint", sampler)

        orm, maps = _orm(surface, material, glb, mask, coverage, analysis.ao.get(surface.material),
                         roughness_by_zone or {}, metallic_by_zone or {})
        written["orm"] = out_dir / f"{name}_ORM.png"
        Image.fromarray(orm).save(written["orm"])
        old = pbr.get("metallicRoughnessTexture")
        exclusive = old is not None and all(
            m == surface.material and slot in ("metallicRoughnessTexture", "occlusionTexture")
            for m, slot in users.get(old["index"], []))
        old_texture = document["textures"][old["index"]] if old is not None else None
        if exclusive and sum(texture_source(t) == texture_source(old_texture)
                             for t in document["textures"]) == 1:
            # Reuse the old metallic-roughness slot rather than leave a dead 4K image behind.
            source = texture_source(old_texture)
            image = document["images"][source]
            binary = set_view_bytes(document, binary, image["bufferView"], _png(orm))
            image["mimeType"] = "image/png"
            old_texture["source"] = source
            old_texture.pop("extensions", None)     # a PNG is core glTF, not WebP
            orm_tex = old["index"]
        else:
            orm_tex, binary = add_texture(document, binary, _png(orm), f"{name}_ORM", sampler)
        texcoord = {"texCoord": surface.texcoord} if surface.texcoord else {}
        pbr["metallicRoughnessTexture"] = {"index": orm_tex, **texcoord}
        pbr["metallicFactor"] = 1.0
        pbr["roughnessFactor"] = 1.0
        if maps["occlusion"] == "baked":
            material["occlusionTexture"] = {"index": orm_tex, **texcoord}
        elif material.get("occlusionTexture", {}).get("index") == orm_tex:
            material.pop("occlusionTexture")

        if "normalTexture" in material:
            maps["normal"] = "kept"
        else:
            lum = (surface.image.astype(np.float64) / 255.0) @ np.array(LUMINANCE_WEIGHTS)
            normal = fill_gutter(normal_from_albedo(lum, coverage, normal_strength), coverage)
            written["normal"] = out_dir / f"{name}_N.png"
            Image.fromarray(normal).save(written["normal"])
            normal_tex, binary = add_texture(document, binary, _png(normal), f"{name}_N", sampler)
            material["normalTexture"] = {"index": normal_tex, **texcoord}
            maps["normal"] = "derived"

        extras = material.setdefault("extras", {})
        extras["assetfurnace"] = {
            "tintMask": {"index": mask_tex, **texcoord},
            "tintZones": [{"channel": z["channel"], "mean_linear": z["mean_linear"]}
                          for z in analysis.zones],
            "sidecar": sidecar_path.name,
        }
        per_material.append({"material": surface.material, "name": material.get("name"),
                             "textures": {k: p.name for k, p in written.items()}, "maps": maps})
        if not first_textures:
            first_textures, first_maps = written, maps

    glb_path = out_dir / f"{stem}.glb"
    glb_path.write_bytes(build_glb(document, binary))
    sidecar["glb"] = glb_path.name
    sidecar["textures"] = {**per_material[0]["textures"],
                           "orm_channels": {"R": "occlusion", "G": "roughness", "B": "metallic"},
                           "srgb": {"tint_mask": False, "orm": False, "normal": False}}
    sidecar["maps"] = first_maps
    sidecar["materials"] = per_material
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return MaterialsResult(status, analysis.checks, analysis.zones, glb_path, sidecar_path,
                           first_textures, first_maps)
