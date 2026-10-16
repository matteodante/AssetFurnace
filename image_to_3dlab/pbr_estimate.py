"""Roughness and metallic where an asset has none, with Marigold-IID Appearance (prs-eth, v1.1).

Pixal3D writes its own metallic-roughness map, and the finish step bakes it; that map is kept.
For an asset without one, Marigold looks at each view of the orbit (``multiview.py``), shown
under a neutral studio light because it reads materials from shading (on unlit, flat albedo
it called a cork dartboard half metal), and says roughness and metallic per pixel; the answers are carried back to the atlas,
weighted by how squarely each view saw the surface, smoothed a little inside each UV island,
and written as a glTF metallic-roughness texture (G roughness, B metallic).

    result = estimate_pbr(asset, out_dir)          # only materials without their own map

Not admitted (design section 6) until it beats a constant on Poly Haven assets with true maps:
``image_to_3dlab.pbr_trial`` and ``scripts/pbr_trial.py``. The sidecar says so.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from image_to_3dlab import material_models as mm
from image_to_3dlab import multiview as mv
from image_to_3dlab.delight import run_intrinsics, smooth_within
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb
from image_to_3dlab.materials import (
    Surface,
    _material_names,
    _png,
    _publish,
    add_texture,
    drop_unused_extensions,
    fill_gutter,
    island_labels,
    source_provenance,
    surfaces,
)
from image_to_3dlab.provenance import sha256_file

MODEL = mm.MARIGOLD_APPEARANCE
SIDECAR_SCHEMA = "assetfurnace.pbr-estimate/1"
OPERATION = "Roughness and metallic estimated with Marigold-IID Appearance v1.1, where the asset had none"
SIGMA = 0.003                 # smoothing on the atlas, as a share of its longer side
TRIAL = ("not run: needs the Poly Haven trial of design section 6 (roughness RMSE <= 0.15 and 20% better "
         "than a constant; metallic F1 >= 0.8, wrong metal on < 5% of the area; blind preference)")


def roughness_metallic(material: np.ndarray, record: dict) -> tuple[np.ndarray, np.ndarray]:
    """(roughness, metallic) from Appearance's stacked answer, by the channel names it declares."""
    names = ((record.get("target_properties") or {}).get("material") or {}).get(
        "sub_target_names", ["roughness", "metallicity", None])
    material = np.asarray(material, dtype=np.float64)
    return (np.clip(material[..., names.index("roughness")], 0.0, 1.0),
            np.clip(material[..., names.index("metallicity")], 0.0, 1.0))


def surface_maps(views: list[mv.View], surface: Surface, answers: dict,
                 sigma: float = SIGMA) -> tuple[np.ndarray, np.ndarray, float]:
    """(roughness, metallic, share of covered texels some view saw) on `surface`'s atlas."""
    acc = mv.reproject(views, surface, lambda view: answers[view.index], channels=2)
    mean, seen = acc.mean()
    coverage = surface.coverage()
    islands = island_labels(coverage)
    support = acc.weights.reshape(acc.size)
    out = []
    for channel in range(2):
        smoothed, filled = smooth_within(mean[..., channel], support, islands, sigma * max(acc.size))
        out.append(mv.fill_unseen(smoothed[..., None], filled & coverage, coverage, islands)[..., 0])
    share = float((seen & coverage).sum() / max(coverage.sum(), 1))
    return np.clip(out[0], 0, 1), np.clip(out[1], 0, 1), share


def component(record: dict) -> dict:
    return {"component": f"{MODEL.repo}@{MODEL.revision}", "purpose": "roughness and metallic estimation",
            "license": MODEL.license, "license_url": mm.OPENRAIL_URL,
            "commercial_status": "commercial-conditional", "conditions": list(mm.OPENRAIL_CONDITIONS),
            "run": {k: record.get(k) for k in ("device", "steps", "resolution", "ensemble", "seconds")}}


def write_provenance(source: Path, source_sha256: str, licence: dict, output: Path, record: dict,
                     staged: Path | None = None, target: Path | None = None) -> Path:
    out = copy.deepcopy(licence)
    out["operation"] = OPERATION
    out["input"] = {"path": str(source), "sha256": source_sha256}
    out["output"] = {**(out.get("output") or {}), "path": str(output), "sha256": sha256_file(staged or output)}
    out["components"] = [*(out.get("components") or []), component(record)]
    path = target or output.with_suffix(".provenance.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


@dataclass
class PbrResult:
    glb_path: Path | None
    sidecar_path: Path | None
    estimated: list[int] = field(default_factory=list)


def estimate_pbr(asset: Path, out_dir: Path, *, force: bool = False, size: int = 768, around: int = 16,
                 render: Callable = mv.render_views, intrinsics: Callable = run_intrinsics,
                 work: Path | None = None) -> PbrResult:
    """Write the asset back with estimated roughness and metallic for materials that had none.

    A material with its own metallic-roughness texture keeps it unless `force`. When nothing
    needs estimating, nothing runs and nothing is written. Otherwise writes
    ``<out_dir>/<stem>.glb``, ``T_<name>_MR.png`` per estimated material,
    ``<stem>.pbr.json`` and ``<stem>.provenance.json``.
    """
    asset, out_dir = Path(asset), Path(out_dir)
    stem = asset.stem
    target = out_dir / f"{stem}.glb"
    if target.exists() and target.samefile(asset):
        raise ValueError("out_dir is the asset's own folder: the result would overwrite the source")
    data = asset.read_bytes()
    source_sha256, licence = hashlib.sha256(data).hexdigest(), source_provenance(asset)
    glb = EditGLB(data)
    found = surfaces(glb)
    todo = {index: s for index, s in found.items()
            if force or "metallicRoughnessTexture" not in glb.document["materials"][index]["pbrMetallicRoughness"]}
    if not todo:
        return PbrResult(None, None, [])

    with tempfile.TemporaryDirectory(prefix="af-pbr-") as scratch:
        folder = (Path(work) if work else Path(scratch)) / "views"
        # Appearance reads materials from shading: it gets the views under a neutral light.
        views = render(asset, folder, size=size, around=around, lit=True)
        colours = [folder / f"view_{v.index:02d}_lit.png" for v in views]
        outs = [folder / f"view_{v.index:02d}_appearance.npz" for v in views]
        record = intrinsics(colours, outs, MODEL)
        record.setdefault("model", {"repo": MODEL.repo, "revision": MODEL.revision, "license": MODEL.license})
        answers = {}
        for view, out in zip(views, outs):
            rough, metal = roughness_metallic(np.load(out)["material"], record)
            answers[view.index] = np.dstack([rough, metal])

    out_dir.mkdir(parents=True, exist_ok=True)
    names = _material_names(glb.document, found, f"T_{stem}")
    document = copy.deepcopy(glb.document)
    binary = glb.binary[:glb.buffer_size]
    images, per_material = [], []
    for surface in todo.values():
        rough, metal, seen = surface_maps(views, surface, answers)
        coverage = surface.coverage()
        mr = np.zeros((*rough.shape, 3), np.uint8)
        mr[..., 0] = 255
        mr[..., 1] = np.round(rough * 255).astype(np.uint8)
        mr[..., 2] = np.round(metal * 255).astype(np.uint8)
        mr = fill_gutter(mr, coverage)
        path = out_dir / f"{names[surface.material]}_MR.png"
        images.append((path, mr))
        material = document["materials"][surface.material]
        pbr = material["pbrMetallicRoughness"]
        base = pbr["baseColorTexture"]
        replaced = "metallicRoughnessTexture" in pbr
        sampler = document["textures"][base["index"]].get("sampler")
        index, binary = add_texture(document, binary, _png(mr), f"{names[surface.material]}_MR", sampler)
        info = {"index": index}
        if base.get("texCoord"):
            info["texCoord"] = base["texCoord"]
        if (base.get("extensions") or {}).get("KHR_texture_transform"):
            info["extensions"] = {"KHR_texture_transform": copy.deepcopy(base["extensions"]["KHR_texture_transform"])}
        pbr["metallicRoughnessTexture"] = info
        pbr["metallicFactor"] = pbr["roughnessFactor"] = 1.0
        inside = coverage
        per_material.append({
            "material": surface.material, "name": material.get("name"), "texture": path.name,
            "replaced_own_map": replaced, "seen_share": round(seen, 3),
            "roughness_median": round(float(np.median(rough[inside])), 3) if inside.any() else None,
            "metallic_share": round(float((metal[inside] >= 0.5).mean()), 3) if inside.any() else None})

    drop_unused_extensions(document)
    glb_path, sidecar_path = out_dir / f"{stem}.glb", out_dir / f"{stem}.pbr.json"
    sidecar = {"schema": SIDECAR_SCHEMA, "asset": asset.name, "source_sha256": source_sha256,
               "glb": glb_path.name, "admitted": False, "trial": TRIAL, "views": len(views),
               "model": record["model"], "run": component(record)["run"], "materials": per_material,
               "kept": sorted(set(found) - set(todo))}
    _publish(out_dir, images, build_glb(document, binary), glb_path, sidecar, sidecar_path,
             lambda staged, final: write_provenance(asset, source_sha256, licence, glb_path, record,
                                                    staged, final))
    return PbrResult(glb_path, sidecar_path, sorted(todo))
