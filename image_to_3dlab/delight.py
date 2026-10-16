"""Take the light painted into a texture back out, with Marigold-IID Lighting (prs-eth, v1.1).

A generated texture often has shadows and highlights drawn into its colour; the engine then
lights the model a second time. Marigold splits a picture into albedo, diffuse shading and a
residual. Its answer comes at 768 pixels, much coarser than a 2K atlas, so the texture is not
replaced by it: each pixel says how much brighter or darker its albedo is than its colour (a
gain). The gains are blurred in each view, where the surface is continuous, because Marigold
also reads fine painted detail (trims, engraving) as shading and only broad light should go;
then they are carried back to the atlas through the orbit of unlit views (``multiview.py``),
smoothed lightly inside each UV island, and the full-resolution texture is multiplied by them.
Only luminance moves; hue and painted detail stay.

The model's overall brightness is not trusted (it was trained on interiors and photos, not
stylised art): gains are taken relative to their median, so a texture that is evenly dark
stays as dark as its artist made it, and no texel moves by more than ``MAX_GAIN``.

    result = delight_asset(asset, out_dir)                 # a GLB with its base colour delit
    path = delight_image(concept, out_path)                # the concept, before the 3D

The step is off unless asked for, and is not *admitted* (design section 6) until it passes the
test against the user's labels; its sidecar says so.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from image_to_3dlab import gpu_lock
from image_to_3dlab import material_models as mm
from image_to_3dlab import multiview as mv
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb
from image_to_3dlab.materials import (
    Surface,
    _decode_image,
    _material_names,
    _png,
    _publish,
    _texture_users,
    drop_unused_extensions,
    fill_gutter,
    island_labels,
    linear_to_srgb,
    luminance,
    referenced_outside_materials,
    set_view_bytes,
    source_provenance,
    srgb_to_linear,
    surfaces,
    texture_source,
    view_shared,
)
from image_to_3dlab.materials import add_texture as _add_texture
from image_to_3dlab.provenance import sha256_file

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "scripts" / "materials_worker.py"
SIDECAR_SCHEMA = "assetfurnace.delight/1"
OPERATION = "Baked light removed with Marigold-IID Lighting v1.1 (luminance gain, per UV island)"
MAX_GAIN = 3.0               # no texel is made more than 3x brighter or darker
SIGMA = 0.004                # smoothing on the atlas, as a share of its longer side
VIEW_SIGMA = 0.02            # smoothing in each view, as a share of the view's side
DARK = 1e-3                  # linear luminance below which a pixel says nothing about light
TRIAL = ("not run: needs the labelled set of design section 6 (painted shadows under a side "
         "light, against the current output)")


# --------------------------------------------------------------------------- the model

def run_intrinsics(images: list[Path], outs: list[Path], model=mm.MARIGOLD_LIGHTING, *,
                   folder: Path | None = None, python: Path | None = None, steps: int | None = None,
                   ensemble: int = 1, resolution: int = 768, seed: int = 0,
                   run: Callable = subprocess.run, lock=gpu_lock.gpu, session: str = "materials") -> dict:
    """Marigold on each image, in the materials environment, with the GPU lock held only meanwhile."""
    folder = folder or mm.require(model)
    python = python or mm.require_env()
    work = Path(outs[0]).parent
    job_path, result_path = work / "intrinsics_job.json", work / "intrinsics_result.json"
    job = {"task": "intrinsics", "model": str(folder), "steps": steps, "ensemble": ensemble,
           "resolution": resolution, "seed": seed, "background": 0.5, "result": str(result_path),
           "images": [{"path": str(i), "out": str(o)} for i, o in zip(images, outs)]}
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    with lock(session):
        done = run([str(python), str(WORKER), str(job_path)], capture_output=True, text=True, check=False)
    if done.returncode:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"{model.label} failed: {' / '.join(tail) or f'exit code {done.returncode}'}")
    record = json.loads(result_path.read_text(encoding="utf-8"))
    record["model"] = {"repo": model.repo, "revision": model.revision, "license": model.license}
    return record


def albedo_linear(prediction: np.ndarray, record: dict) -> np.ndarray:
    space = (record.get("target_properties") or {}).get("albedo", {}).get("prediction_space", "linear")
    prediction = np.asarray(prediction, dtype=np.float64)
    return srgb_to_linear(prediction) if space == "srgb" else prediction


# --------------------------------------------------------------------------- pure parts

def pixel_gain(colour: np.ndarray, albedo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(log of albedo luminance over colour luminance, where that means something) per pixel."""
    shown = luminance(srgb_to_linear(np.asarray(colour)[..., :3] / 255.0))
    meant = luminance(np.asarray(albedo, dtype=np.float64)[..., :3])
    valid = (shown > DARK) & (meant > DARK)
    gain = np.zeros(shown.shape)
    gain[valid] = np.log(meant[valid]) - np.log(shown[valid])
    return gain, valid


def smooth_within(values: np.ndarray, weight: np.ndarray, islands: np.ndarray,
                  sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Gaussian smoothing that never crosses from one UV island into another.

    Returns the smoothed values and where they are defined (some weight nearby, same island).
    """
    out, filled = values.astype(np.float64).copy(), np.zeros(values.shape, bool)
    for number, box in enumerate(ndimage.find_objects(islands), start=1):
        if box is None:
            continue
        mine = islands[box] == number
        w = np.where(mine, weight[box], 0.0)
        num = ndimage.gaussian_filter(values[box] * w, sigma, mode="constant")
        den = ndimage.gaussian_filter(w, sigma, mode="constant")
        ok = mine & (den > 1e-9)
        part, done = out[box], filled[box]
        part[ok] = num[ok] / den[ok]
        done |= ok
    return out, filled


def texel_gain(views: list[mv.View], surface: Surface, gains: dict, sigma: float = SIGMA) -> np.ndarray:
    """The log gain of every texel of `surface`, from what each view said, relative to the median."""
    acc = mv.reproject(views, surface,
                       lambda view: np.dstack([gains[view.index][0] * gains[view.index][1],
                                               gains[view.index][1]]), channels=2)
    mean, _ = acc.mean()
    support = mean[..., 1] * acc.weights.reshape(acc.size)      # facing weight of valid pixels
    value = np.divide(mean[..., 0], mean[..., 1], out=np.zeros(acc.size), where=mean[..., 1] > 0)
    coverage = surface.coverage()
    islands = island_labels(coverage)
    smoothed, filled = smooth_within(value, support, islands, sigma * max(acc.size))
    gain = mv.fill_unseen(smoothed[..., None], filled & coverage, coverage, islands)[..., 0]
    inside = coverage & filled
    if inside.any():
        gain = gain - np.median(gain[inside])
    return np.clip(np.where(coverage, gain, 0.0), -np.log(MAX_GAIN), np.log(MAX_GAIN))


def apply_gain(image: np.ndarray, gain: np.ndarray, coverage: np.ndarray | None = None) -> np.ndarray:
    """The sRGB image with its linear colour multiplied by exp(gain); alpha untouched."""
    rgb = srgb_to_linear(image[..., :3] / 255.0) * np.exp(gain)[..., None]
    out = image.copy()
    out[..., :3] = np.round(linear_to_srgb(rgb) * 255.0).astype(np.uint8)
    if coverage is not None:
        out = fill_gutter(np.where(coverage[..., None], out, image), coverage)
    return out


# --------------------------------------------------------------------------- licence record

def component(record: dict) -> dict:
    model = record["model"]
    return {"component": f"{model['repo']}@{model['revision']}", "purpose": "baked light removal",
            "license": model["license"], "license_url": mm.OPENRAIL_URL,
            "commercial_status": "commercial-conditional", "conditions": list(mm.OPENRAIL_CONDITIONS),
            "run": {k: record.get(k) for k in ("device", "steps", "resolution", "ensemble", "seconds")}}


def write_provenance(source: Path, source_sha256: str, licence: dict, output: Path, model: dict,
                     staged: Path | None = None, target: Path | None = None) -> Path:
    """The source's licence record carried to the result, with Marigold added as a component."""
    record = copy.deepcopy(licence)
    record["operation"] = OPERATION
    record["input"] = {"path": str(source), "sha256": source_sha256}
    record["output"] = {**(record.get("output") or {}), "path": str(output),
                        "sha256": sha256_file(staged or output)}
    record["components"] = [*(record.get("components") or []), component(model)]
    path = target or output.with_suffix(".provenance.json")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- entry points

@dataclass
class DelightResult:
    glb_path: Path
    sidecar_path: Path
    textures: dict[int, Path]


def _gains(views: list[mv.View], folder: Path, intrinsics: Callable,
           view_sigma: float = VIEW_SIGMA) -> tuple[dict, dict]:
    """Each view's log gain, smoothed in the view where the surface is in one piece.

    The atlas is cut into islands, often tiny; on the picture the surface is continuous, so
    that is where broad light is told apart from fine painted detail: Marigold reads thin
    trims as shading, and only what survives a wide blur is light.
    """
    colours = [folder / f"view_{v.index:02d}_color.png" for v in views]
    outs = [folder / f"view_{v.index:02d}_intrinsics.npz" for v in views]
    record = intrinsics(colours, outs, mm.MARIGOLD_LIGHTING)
    record.setdefault("model", {"repo": mm.MARIGOLD_LIGHTING.repo, "revision": mm.MARIGOLD_LIGHTING.revision,
                                "license": mm.MARIGOLD_LIGHTING.license})
    gains = {}
    for view, out in zip(views, outs):
        albedo = albedo_linear(np.load(out)["albedo"], record)
        gain, valid = pixel_gain(view.color, albedo)
        valid = (valid & (view.material >= 0)).astype(np.float64)
        gain, filled = smooth_within(gain, valid, view.material + 1, view_sigma * max(gain.shape))
        gains[view.index] = (gain, valid * filled)
    return gains, record


def delight_asset(asset: Path, out_dir: Path, *, size: int = 768, around: int = 16,
                  render: Callable = mv.render_views, intrinsics: Callable = run_intrinsics,
                  work: Path | None = None) -> DelightResult:
    """Write the asset back with the light taken out of its base colour, plus sidecar and licence.

    Writes ``<out_dir>/<stem>.glb``, ``T_<name>_BaseColor.png`` per material,
    ``<stem>.delight.json`` and ``<stem>.provenance.json``; everything else in the GLB (rig,
    animations, other maps) is kept byte for byte.
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
    with tempfile.TemporaryDirectory(prefix="af-delight-") as scratch:
        folder = Path(work) if work else Path(scratch)
        views = render(asset, folder / "views", size=size, around=around)
        gains, record = _gains(views, folder / "views", intrinsics)

    out_dir.mkdir(parents=True, exist_ok=True)
    names = _material_names(glb.document, found, f"T_{stem}")
    document = copy.deepcopy(glb.document)
    binary = glb.binary[:glb.buffer_size]
    users = _texture_users(document)
    images, per_material, textures = [], [], {}
    for surface in found.values():
        coverage = surface.coverage()
        gain = texel_gain(views, surface, gains)
        info = document["materials"][surface.material]["pbrMetallicRoughness"]["baseColorTexture"]
        texture = document["textures"][info["index"]]
        source = texture_source(texture)
        original = _decode_image(glb, source, "RGBA")
        keep_alpha = bool((original[..., 3] < 255).any())
        delit = apply_gain(original if keep_alpha else original[..., :3], gain, coverage)
        path = out_dir / f"{names[surface.material]}_BaseColor.png"
        images.append((path, delit))
        textures[surface.material] = path
        exclusive = (users.get(info["index"], []) == [(surface.material, "baseColorTexture")]
                     and not referenced_outside_materials(document, info["index"])
                     and sum(texture_source(t) == source for t in document["textures"]) == 1
                     and not view_shared(document, source))
        if exclusive:
            image = document["images"][source]
            binary = set_view_bytes(document, binary, image["bufferView"], _png(delit))
            image["mimeType"] = "image/png"
            texture["source"] = source
            texture.pop("extensions", None)                 # a PNG is core glTF, not WebP
        else:
            index, binary = _add_texture(document, binary, _png(delit), f"{names[surface.material]}_BaseColor",
                                         texture.get("sampler"))
            info["index"] = index
        inside = coverage & (gain != 0)
        per_material.append({
            "material": surface.material, "name": document["materials"][surface.material].get("name"),
            "texture": path.name,
            "gain": {"p05": round(float(np.exp(np.percentile(gain[coverage], 5))), 3),
                     "p95": round(float(np.exp(np.percentile(gain[coverage], 95))), 3),
                     "changed_share": round(float(inside.sum() / max(coverage.sum(), 1)), 3)}})

    drop_unused_extensions(document)
    glb_path, sidecar_path = out_dir / f"{stem}.glb", out_dir / f"{stem}.delight.json"
    sidecar = {"schema": SIDECAR_SCHEMA, "asset": asset.name, "source_sha256": source_sha256,
               "glb": glb_path.name, "admitted": False, "trial": TRIAL, "views": len(views),
               "model": record["model"], "run": component(record)["run"], "materials": per_material}
    _publish(out_dir, images, build_glb(document, binary), glb_path, sidecar, sidecar_path,
             lambda staged, final: write_provenance(asset, source_sha256, licence, glb_path, record,
                                                    staged, final))
    return DelightResult(glb_path, sidecar_path, textures)


def delight_image(image: Path, out_path: Path, *, intrinsics: Callable = run_intrinsics,
                  sigma: float = SIGMA) -> Path:
    """The concept with its painted light taken out, before it goes to 3D; alpha kept."""
    image, out_path = Path(image), Path(out_path)
    if out_path.exists() and out_path.samefile(image):
        raise ValueError("the result would overwrite the concept")
    data = image.read_bytes()
    source_sha256, licence = hashlib.sha256(data).hexdigest(), source_provenance(image)
    picture = np.asarray(Image.open(image).convert("RGBA"))
    with tempfile.TemporaryDirectory(prefix="af-delight-") as scratch:
        npz = Path(scratch) / "concept_intrinsics.npz"
        record = intrinsics([image], [npz], mm.MARIGOLD_LIGHTING)
        record.setdefault("model", {"repo": mm.MARIGOLD_LIGHTING.repo, "revision": mm.MARIGOLD_LIGHTING.revision,
                                    "license": mm.MARIGOLD_LIGHTING.license})
        albedo = albedo_linear(np.load(npz)["albedo"], record)
    gain, valid = pixel_gain(picture[..., :3], albedo)
    shown = picture[..., 3] > 127
    weight = (valid & shown).astype(np.float64)
    smoothed, filled = smooth_within(gain, weight, shown.astype(np.int64), sigma * max(gain.shape))
    inside = shown & filled
    if inside.any():
        smoothed = smoothed - np.median(smoothed[inside])
    smoothed = np.clip(np.where(inside, smoothed, 0.0), -np.log(MAX_GAIN), np.log(MAX_GAIN))
    out = apply_gain(picture, smoothed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staged = out_path.with_name(f".{out_path.name}.staging")
    try:
        Image.fromarray(out).save(staged, format="PNG")
        write_provenance(image, source_sha256, licence, out_path, record, staged,
                         out_path.with_suffix(".provenance.json"))
        staged.replace(out_path)
    finally:
        staged.unlink(missing_ok=True)
    return out_path
