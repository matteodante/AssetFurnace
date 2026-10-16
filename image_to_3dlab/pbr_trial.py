"""The trial estimated roughness and metallic must pass (design section 6): Poly Haven assets with true maps.

Poly Haven's models are CC0 and come with their real roughness and metallic. Each one is
packed into a GLB, its metallic-roughness map taken out, estimated again by
``pbr_estimate``, and compared with the truth on the texels its triangles cover:

- roughness: RMSE <= 0.15, and at least 20% below a constant. The constant for an asset is
  the area-weighted mean roughness of the *other* assets, so it is never fitted on the asset
  it is scored on;
- metallic: F1 >= 0.8 (a texel is metal at 0.5 or more), wrong metal on under 5% of the area.
  The best F1 over every threshold, chosen in hindsight, is reported too: an optimistic bound
  that says whether any calibration could have saved the answer, never a pass;
- blind preference (preferred or equal in at least 60% of side-by-side renders) needs the
  user's eyes, so it stays open here and the step is not admitted on numbers alone.

Scores are pooled by area over the whole set; the per-asset table says where it fails. The
set is realistic photoscans, the domain Marigold was trained near: passing here says nothing
yet about stylised art.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb
from image_to_3dlab.materials import _same_uv_layout, _texture_image, surfaces

ROUGHNESS_RMSE = 0.15
BEAT_CONSTANT = 0.20
METAL_F1 = 0.80
WRONG_METAL = 0.05
METAL_AT = 0.5
BINS = 50                     # histogram of the metal answer, for the threshold chosen in hindsight
SKIP_CATEGORIES = {"nature", "plants", "trees", "rocks", "ground cover", "food"}
METAL_TAGS = {"metal", "metallic", "steel", "iron", "brass", "copper", "chrome", "aluminium", "aluminum"}
MAX_POLYCOUNT = 200_000


def select_assets(catalogue: dict, count: int = 30, seed: str = "assetfurnace-pbr-trial") -> list[str]:
    """`count` asset ids, half tagged as metal, the same ones every time for a given catalogue."""
    def eligible(entry):
        return (not SKIP_CATEGORIES & set(entry.get("categories", []))
                and 0 < entry.get("polycount", 0) <= MAX_POLYCOUNT)

    def order(name):
        return hashlib.sha1(f"{seed}:{name}".encode()).hexdigest()

    names = sorted((n for n, e in catalogue.items() if eligible(e)), key=order)
    metal = [n for n in names if METAL_TAGS & {t.lower() for t in catalogue[n].get("tags", [])}]
    other = [n for n in names if n not in set(metal)]
    half = count // 2
    chosen = metal[:half] + other[:count - min(half, len(metal))]
    return sorted(chosen[:count], key=order)


def pack_gltf(path: Path) -> bytes:
    """A .gltf with its .bin and textures beside it, as one GLB with everything embedded."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    binary = bytearray()
    offsets = []
    for buffer in document.get("buffers", []):
        uri = buffer.get("uri", "")
        if uri.startswith("data:"):
            raise ValueError("buffers embedded as data URIs are not supported; export with a .bin")
        offsets.append(len(binary))
        binary += (path.parent / uri).read_bytes()
        binary += b"\0" * (-len(binary) % 4)
    for view in document.get("bufferViews", []):
        view["byteOffset"] = view.get("byteOffset", 0) + offsets[view.get("buffer", 0)]
        view["buffer"] = 0
    for image in document.get("images", []):
        uri = image.pop("uri", None)
        if uri is None:
            continue
        if uri.startswith("data:"):
            data = base64.b64decode(uri.split(",", 1)[1])
        else:
            data = (path.parent / uri).read_bytes()
        image.setdefault("mimeType", "image/png" if uri.lower().endswith(".png") else "image/jpeg")
        document.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": len(binary),
                                                       "byteLength": len(data)})
        image["bufferView"] = len(document["bufferViews"]) - 1
        binary += data
        binary += b"\0" * (-len(binary) % 4)
    document["buffers"] = [{"byteLength": len(binary)}]
    return build_glb(document, bytes(binary))


def true_maps(data: bytes) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per textured material: (roughness, metallic) at its base colour's size, factors applied."""
    glb = EditGLB(data)
    out = {}
    for index, surface in surfaces(glb).items():
        pbr = glb.document["materials"][index].get("pbrMetallicRoughness", {})
        height, width = surface.image.shape[:2]
        info = pbr.get("metallicRoughnessTexture")
        if info is not None and not _same_uv_layout(glb.document, pbr["baseColorTexture"], info):
            continue                                   # scored only where texels line up
        image = _texture_image(glb, info)
        rough_factor = float(pbr.get("roughnessFactor", 1.0))
        metal_factor = float(pbr.get("metallicFactor", 1.0))
        if image is None:
            out[index] = (np.full((height, width), rough_factor), np.full((height, width), metal_factor))
            continue
        if image.shape[:2] != (height, width):
            image = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))
        out[index] = (image[..., 1] / 255.0 * rough_factor, image[..., 2] / 255.0 * metal_factor)
    return out


def strip_maps(data: bytes) -> bytes:
    """The GLB with every metallic-roughness texture unhooked, so the step has to estimate them."""
    glb = EditGLB(data)
    document = copy.deepcopy(glb.document)
    for material in document.get("materials", []):
        material.get("pbrMetallicRoughness", {}).pop("metallicRoughnessTexture", None)
    return build_glb(document, glb.binary[:glb.buffer_size])


def score(true_rough, true_metal, est_rough, est_metal, coverage) -> dict:
    """Sums over the covered texels, so assets of any size pool by area."""
    inside = np.asarray(coverage, bool)
    tr, tm = np.asarray(true_rough)[inside], np.asarray(true_metal)[inside] >= METAL_AT
    er, em = np.asarray(est_rough)[inside], np.asarray(est_metal)[inside] >= METAL_AT
    said = np.clip(np.asarray(est_metal)[inside], 0, 1)
    edges = np.linspace(0.0, 1.0, BINS + 1)
    return {"texels": int(inside.sum()), "rough_sq": float(((er - tr) ** 2).sum()),
            "metal_hist": np.histogram(said[tm], edges)[0].tolist(),
            "other_hist": np.histogram(said[~tm], edges)[0].tolist(),
            "rough_sum": float(tr.sum()), "rough_sq_sum": float((tr ** 2).sum()),
            "tp": int((em & tm).sum()), "fp": int((em & ~tm).sum()), "fn": int((~em & tm).sum())}


def summarise(rows: list[dict]) -> dict:
    total = sum(r["texels"] for r in rows)
    rough_sum = sum(r["rough_sum"] for r in rows)
    constant_sq, per_asset = 0.0, []
    for row in rows:
        others = total - row["texels"]
        c = (rough_sum - row["rough_sum"]) / others if others else 0.5
        # sum over the asset's texels of (c - truth)^2, from its sums
        sq = row["rough_sq_sum"] - 2 * c * row["rough_sum"] + c * c * row["texels"]
        constant_sq += sq
        n = max(row["texels"], 1)
        per_asset.append({"asset": row.get("asset"), "rmse": round(float(np.sqrt(row["rough_sq"] / n)), 4),
                          "constant_rmse": round(float(np.sqrt(sq / n)), 4),
                          "true_metal": row["tp"] + row["fn"], "said_metal": row["tp"] + row["fp"],
                          "wrong_metal": row["fp"] + row["fn"]})
    rmse = float(np.sqrt(sum(r["rough_sq"] for r in rows) / max(total, 1)))
    constant = float(np.sqrt(constant_sq / max(total, 1)))
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    wrong = (fp + fn) / max(total, 1)
    best_f1, best_at = None, None
    if all("metal_hist" in r for r in rows) and rows:
        metal_hist = np.sum([r["metal_hist"] for r in rows], axis=0)
        other_hist = np.sum([r["other_hist"] for r in rows], axis=0)
        for cut in range(1, BINS):                     # metal = the bins at and above `cut`
            t, f = metal_hist[cut:].sum(), other_hist[cut:].sum()
            n = metal_hist[:cut].sum()
            if 2 * t + f + n and (best_f1 is None or 2 * t / (2 * t + f + n) > best_f1):
                best_f1, best_at = float(2 * t / (2 * t + f + n)), cut / BINS
    verdict = {"roughness_rmse": rmse <= ROUGHNESS_RMSE,
               "roughness_beats_constant": constant > 0 and rmse <= (1 - BEAT_CONSTANT) * constant,
               "metallic_f1": f1 is not None and f1 >= METAL_F1,
               "metallic_wrong_area": wrong < WRONG_METAL,
               "blind_preference": None}
    verdict["numbers_pass"] = all(v for k, v in verdict.items() if k != "blind_preference")
    verdict["admitted"] = False                          # the blind comparison is the user's
    return {"assets": len(rows), "texels": total,
            "roughness": {"rmse": rmse, "constant_rmse": constant,
                          "improvement": (1 - rmse / constant) if constant else None},
            "metallic": {"f1": f1, "wrong_share": wrong, "true_share": (tp + fn) / max(total, 1),
                         "best_f1": best_f1, "best_threshold": best_at},
            "verdict": verdict, "per_asset": per_asset}
