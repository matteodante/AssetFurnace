"""Tint zones by material (design section 2): name the materials, segment them in every view, vote on the UVs.

1. **Which materials.** Qwen3.5-4B looks at a few views of the asset and picks up to six from a
   closed list, the same words the user labels with in Studio (``VOCABULARY``); or the caller
   gives the list.
2. **Where they are.** A segmenter says, for every view of the orbit and every material, how
   likely each pixel is to be that material. Two stand behind one interface (``SEGMENTERS``):
   ``sam3``, which segments "every region of this concept" from a phrase (gated: its owner
   fills in Meta's form on Hugging Face), and ``sam2``, SAM 2.1's automatic regions each named
   by Qwen3.5, all Apache-2.0 and the default until SAM 3's access is granted.
3. **The vote.** The scores go back to the atlas (``multiview.reproject``), weighted by how
   squarely each view saw the surface, smoothed inside each UV island so seams and shadows do
   not split a zone; the asset's own metallic map counts for "metal". Each texel takes the
   material with the most support.
4. **The zones.** The three largest materials become R, G and B of the tint mask, their names go
   into the sidecar (``zones[i].material``). Three zones pass, two warn, one sends a creature or
   an equipment piece back.

Not admitted until it passes design section 6 against the user's points (``scripts/zones_trial.py``).
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_3dlab import gpu_lock
from image_to_3dlab import material_models as mm
from image_to_3dlab import multiview as mv
from image_to_3dlab.delight import WORKER, smooth_within
from image_to_3dlab.material_labels import MATERIALS
from image_to_3dlab.materials import CHANNELS, Surface, island_labels, srgb_to_linear
from image_to_3dlab.piece_materials import PBR

VOCABULARY = tuple(m for m in MATERIALS if m != "other")
MAX_MATERIALS = 6
SEGMENTERS = ("sam31", "sam3", "sam2")
ZONE_PROFILES = ("creature", "equipment_part")
SIGMA = 0.004
HINT_WEIGHT = 0.5
GREY = (214, 214, 214)
TRIAL = ("not run: needs the user's material points (design section 6, scripts/zones_trial.py human): "
         "at least 75% of the points in the zone of their material")

# Roughness and metallic per material (``piece_materials.PBR``). From zones they failed the Poly Haven
# trial on 2026-10-17 (SAM 2.1 zones: roughness RMSE 0.30, 32% worse than a constant; metal wrong on 42%
# of the area; design section 10), so zones write none of them into an asset; one material per piece does.
DEFAULT_PBR = PBR

SYNONYMS = {"fabric": "cloth", "textile": "cloth", "steel": "metal", "iron": "metal", "gold": "metal",
            "silver": "metal", "bronze": "metal", "brass": "metal", "copper": "metal", "chrome": "metal",
            "rock": "stone", "marble": "stone", "flesh": "skin", "feather": "feathers", "scale": "scales",
            "gems": "gem", "jewel": "gem", "crystal": "gem", "leaves": "plant", "leaf": "plant",
            "rubber": "plastic", "ceramic": "stone"}

VOCABULARY_QUESTION = ("Which materials is this object made of? Choose up to 6 from this list only: {words}. "
                       "Answer with the names separated by commas, largest area first, nothing else.")
REGION_QUESTION = ("The region inside the red outline is part of a 3D model. Which one material is it made of? "
                   "Choose one from this list only: {words}. Answer with the one name, nothing else.")


def parse_vocabulary(answer: str, allowed: tuple[str, ...] = VOCABULARY, limit: int = MAX_MATERIALS) -> list[str]:
    """The materials named in a model's answer, in order, only from `allowed`, at most `limit`."""
    found = []
    for word in re.split(r"[,\n;/]+|\band\b", answer.lower()):
        word = re.sub(r"[^a-z ]", "", word).strip()
        word = SYNONYMS.get(word, word)
        if word in allowed and word not in found:
            found.append(word)
    return found[:limit]


def merge_vocabularies(answers: list[list[str]], limit: int = MAX_MATERIALS) -> list[str]:
    """Several views' answers as one list: most often named first, earlier named breaking ties."""
    count, first = Counter(), {}
    for words in answers:
        for rank, word in enumerate(words):
            count[word] += 1
            first.setdefault(word, (len(first), rank))
    return sorted(count, key=lambda w: (-count[w], first[w]))[:limit]


# --------------------------------------------------------------------------- the vote

@dataclass
class ZoneMap:
    """One material's atlas, labelled by material, and what that means for the tint mask."""
    material: int
    phrases: list[str]
    labels: np.ndarray            # HxW index into phrases, -1 outside the atlas's triangles
    support: np.ndarray           # HxW share of the vote the winner got
    coverage: np.ndarray          # HxW

    def areas(self) -> dict[str, float]:
        inside = self.coverage & (self.labels >= 0)
        total = max(int(inside.sum()), 1)
        counts = np.bincount(self.labels[inside], minlength=len(self.phrases))
        return {p: float(c / total) for p, c in zip(self.phrases, counts) if c}

    def zones(self, count: int = 3) -> list[str]:
        """The `count` largest materials, largest first: R, G, B."""
        return [name for name, _ in sorted(self.areas().items(), key=lambda kv: -kv[1])][:count]

    def mask(self, count: int = 3) -> np.ndarray:
        """HxWx3 uint8: 255 in the channel of the texel's zone, 0 elsewhere."""
        out = np.zeros((*self.labels.shape, 3), np.uint8)
        for channel, name in enumerate(self.zones(count)):
            out[..., channel] = np.where(self.coverage & (self.labels == self.phrases.index(name)), 255, 0)
        return out

    def material_at(self, uv: tuple[float, float]) -> str | None:
        """The material voted for the texel at glTF `uv` (v downward), or None."""
        height, width = self.labels.shape
        row = min(int(uv[1] * height), height - 1)
        col = min(int(uv[0] * width), width - 1)
        index = int(self.labels[row, col])
        return self.phrases[index] if index >= 0 else None


def vote(views: list[mv.View], surface: Surface, scores: dict[int, np.ndarray], phrases: list[str],
         metal_hint: np.ndarray | None = None, hint_weight: float = HINT_WEIGHT,
         sigma: float = SIGMA) -> ZoneMap:
    """Each texel's material, from per-view scores (HxWxP each, P = len(phrases))."""
    acc = mv.reproject(views, surface, lambda view: scores[view.index], channels=len(phrases))
    mean, seen = acc.mean()
    if metal_hint is not None and "metal" in phrases:
        mean[..., phrases.index("metal")] += hint_weight * np.asarray(metal_hint, dtype=np.float64)
    coverage = surface.coverage()
    islands = island_labels(coverage)
    support = acc.weights.reshape(acc.size) * seen
    smoothed = np.zeros_like(mean)
    filled = np.zeros(acc.size, bool)
    for p in range(len(phrases)):
        smoothed[..., p], done = smooth_within(mean[..., p], support, islands, sigma * max(acc.size))
        filled |= done
    smoothed = mv.fill_unseen(smoothed, filled & coverage, coverage, islands)
    total = smoothed.sum(axis=-1)
    labels = np.where(coverage & (total > 0), smoothed.argmax(axis=-1), -1)
    share = np.divide(smoothed.max(axis=-1), total, out=np.zeros(acc.size), where=total > 0)
    return ZoneMap(surface.material, list(phrases), labels, share, coverage)


def status(count: int, profile: str | None) -> tuple[str, str]:
    """(status, message) for how many zones an asset has, by the rule kept from v1."""
    if profile not in ZONE_PROFILES or count >= 3:
        return "pass", f"{count} tint zones"
    if count == 2:
        return "warn", "only 2 tint zones: one colour slot will change nothing"
    return "fail", "fewer than 2 tint zones: recolouring would change the whole asset at once"


def zone_records(zone: ZoneMap, albedo_srgb: np.ndarray, count: int = 3, factor=(1.0, 1.0, 1.0)) -> list[dict]:
    """The sidecar's zones: channel, material, area and mean linear colour (the tint formula's c).

    The colour is the one the renderer shows: the texture times the material's baseColorFactor.
    """
    linear = srgb_to_linear(albedo_srgb[..., :3] / 255.0) * np.asarray(factor, dtype=np.float64)[:3]
    areas = zone.areas()
    out = []
    for channel, name in zip(CHANNELS, zone.zones(count)):
        inside = zone.coverage & (zone.labels == zone.phrases.index(name))
        out.append({"channel": channel, "material": name, "area": round(areas[name], 4),
                    "mean_linear": [round(float(v), 5) for v in linear[inside].mean(axis=0)]})
    return out


def default_maps(zone: ZoneMap) -> tuple[np.ndarray, np.ndarray]:
    """Roughness and metallic from each texel's material (``DEFAULT_PBR``); 0.5 / 0 where unknown."""
    rough = np.full(zone.labels.shape, 0.5)
    metal = np.zeros(zone.labels.shape)
    for index, name in enumerate(zone.phrases):
        r, mtl = DEFAULT_PBR.get(name, (0.5, 0.0))
        rough[zone.labels == index] = r
        metal[zone.labels == index] = mtl
    return rough, metal


# --------------------------------------------------------------------------- regions named one by one

def region_picture(colour: np.ndarray, material: np.ndarray, region: np.ndarray, pad: int = 12) -> Image.Image:
    """The view cropped around `region`, the object on grey and the region outlined in red."""
    from scipy import ndimage

    out = np.empty_like(colour)
    out[:] = GREY
    out[material >= 0] = colour[material >= 0]
    edge = region & ~ndimage.binary_erosion(region, iterations=2)
    out[edge] = (230, 30, 30)
    rows, cols = np.nonzero(region)
    top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad + 1, colour.shape[0])
    left, right = max(cols.min() - pad, 0), min(cols.max() + pad + 1, colour.shape[1])
    return Image.fromarray(out[top:bottom, left:right]).resize((224, 224), Image.BICUBIC)


def region_scores(regions: np.ndarray, names: list[str | None], phrases: list[str]) -> np.ndarray:
    """HxWxP scores from named regions (RxHxW): 1 for a region's material, smaller regions on top."""
    height, width = regions.shape[1:]
    out = np.zeros((height, width, len(phrases)))
    order = np.argsort([-r.sum() for r in regions])           # large first, small ones overwrite
    for index in order:
        name = names[index]
        if name not in phrases:
            continue
        region = regions[index]
        out[region] = 0.0
        out[region, phrases.index(name)] = 1.0
    return out


# --------------------------------------------------------------------------- the models

def _run(task: str, job: dict, work: Path, model: mm.HubModel, run=subprocess.run, lock=gpu_lock.gpu) -> dict:
    folder = mm.require_sam31() if model is mm.SAM31 else mm.require(model)
    python = mm.require_env()
    work.mkdir(parents=True, exist_ok=True)
    job_path, result = work / f"{task}_job.json", work / f"{task}_result.json"
    job_path.write_text(json.dumps({"task": task, "model": str(folder), "result": str(result), **job}, indent=2),
                        encoding="utf-8")
    with lock("materials"):
        done = run([str(python), str(WORKER), str(job_path)], capture_output=True, text=True, check=False)
    if done.returncode:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"{model.label} failed: {' / '.join(tail)}")
    return json.loads(result.read_text(encoding="utf-8"))


def ask_vocabulary(pictures: list[Path], work: Path, runner=_run) -> list[str]:
    """Up to six materials from the closed list, by Qwen3.5-4B looking at a few views."""
    answers = runner("name_materials", {"images": [{"path": str(p)} for p in pictures],
                                        "question": VOCABULARY_QUESTION.format(words=", ".join(VOCABULARY))},
                     work, mm.QWEN35)["answers"]
    return merge_vocabularies([parse_vocabulary(a) for a in answers])


def segment(views: list[mv.View], folder: Path, phrases: list[str], segmenter: str, work: Path,
            runner=_run) -> dict[int, np.ndarray]:
    """HxWxP scores per view index, from the chosen segmenter."""
    if segmenter not in SEGMENTERS:
        raise ValueError(f"unknown segmenter {segmenter!r}; choose from {', '.join(SEGMENTERS)}")
    colours = [folder / f"view_{v.index:02d}_color.png" for v in views]
    if segmenter in ("sam31", "sam3"):
        outs = [work / f"view_{v.index:02d}_{segmenter}.npz" for v in views]
        runner("segment_sam3", {"phrases": phrases,
                                "images": [{"path": str(c), "out": str(o)} for c, o in zip(colours, outs)]},
               work, mm.SAM31 if segmenter == "sam31" else mm.SAM3)
        return {v.index: np.moveaxis(np.load(o)["scores"].astype(np.float64), 0, -1) * (v.material >= 0)[..., None]
                for v, o in zip(views, outs)}
    outs = [work / f"view_{v.index:02d}_regions.npz" for v in views]
    runner("regions_sam2", {"images": [{"path": str(c), "out": str(o)} for c, o in zip(colours, outs)]},
           work, mm.SAM21)
    crops, owners = [], []
    for view, out in zip(views, outs):
        regions = np.load(out)["masks"].astype(bool) & (view.material >= 0)[None]
        for r, region in enumerate(regions):
            if region.sum() < 64:
                continue
            path = work / f"view_{view.index:02d}_region_{r:03d}.png"
            region_picture(view.color, view.material, region).save(path)
            crops.append(path)
            owners.append((view.index, r))
    question = REGION_QUESTION.format(words=", ".join(phrases))
    answers = runner("name_materials", {"images": [{"path": str(p)} for p in crops], "question": question},
                     work, mm.QWEN35)["answers"] if crops else []
    names: dict[int, dict[int, str | None]] = {}
    for (index, r), answer in zip(owners, answers):
        found = parse_vocabulary(answer, tuple(phrases), 1)
        names.setdefault(index, {})[r] = found[0] if found else None
    out = {}
    for view, path in zip(views, outs):
        regions = np.load(path)["masks"].astype(bool) & (view.material >= 0)[None]
        named = [names.get(view.index, {}).get(r) for r in range(len(regions))]
        out[view.index] = region_scores(regions, named, phrases)
    return out


def components(segmenter: str) -> list[dict]:
    """The models a zoning run used, for the provenance record."""
    used = [mm.QWEN35, {"sam31": mm.SAM31, "sam3": mm.SAM3}.get(segmenter, mm.SAM21)]
    return [{"component": f"{m.repo}@{m.revision}", "purpose": "material zones", "license": m.license,
             "license_url": m.license_url,
             "commercial_status": "commercial-conditional" if m in (mm.SAM3, mm.SAM31) else "commercial-clear"}
            for m in used]


def find_zones(asset: Path, work: Path, segmenter: str = "sam31", vocabulary: list[str] | None = None,
               size: int = 512, around: int = 16, render=mv.render_views, runner=_run,
               metal_hints: dict[int, np.ndarray] | None = None) -> tuple[dict[int, ZoneMap], dict]:
    """Zones for every textured material of `asset`, and a record of how they were found."""
    from image_to_3dlab.edit_glb import EditGLB
    from image_to_3dlab.materials import surfaces

    work = Path(work)
    folder = work / "views"
    views = render(asset, folder, size=size, around=around)
    if vocabulary is None:
        front = [folder / f"view_{v.index:02d}_color.png" for v in views[: around: max(around // 4, 1)]]
        vocabulary = ask_vocabulary(front, work / "vocabulary", runner)
    vocabulary = [w for w in vocabulary if w in VOCABULARY] or ["cloth", "metal", "leather"]
    scores = segment(views, folder, vocabulary, segmenter, work / segmenter, runner)
    found = surfaces(EditGLB(Path(asset).read_bytes()))
    zones = {index: vote(views, surface, scores, vocabulary, (metal_hints or {}).get(index))
             for index, surface in found.items()}
    record = {"segmenter": segmenter, "vocabulary": vocabulary, "views": len(views), "admitted": False,
              "trial": TRIAL, "components": components(segmenter)}
    return zones, record


# --------------------------------------------------------------------------- the test against the user's points

POINT_AGREEMENT = 0.75


def point_agreement(zones: dict[int, ZoneMap], points: list[dict]) -> dict:
    """How many of the user's points (``material_labels``) fall where the vote put their material.

    A point names a glTF material, a texel and a word; "other" and points on materials without
    zones are left out. ``in_zones`` counts only points whose word is one of the asset's three
    zones, the part a recolour touches.
    """
    hits = total = zone_hits = zone_total = 0
    for point in points:
        zone = zones.get(point.get("material"))
        if zone is None or point["label"] == "other":
            continue
        said = zone.material_at(point["uv"])
        total += 1
        hits += said == point["label"]
        if point["label"] in zone.zones():
            zone_total += 1
            zone_hits += said == point["label"]
    return {"points": total, "agree": hits, "agreement": hits / total if total else None,
            "zone_points": zone_total, "zone_agreement": zone_hits / zone_total if zone_total else None}
