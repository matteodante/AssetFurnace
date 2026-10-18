"""A registered base body: picture, rigged GLB, Pixal3D camera and bone roles, in one record.

`output/bodies/<name>/body.json` holds everything the slot steps need, so they never read
the rig again: the skeleton (glTF frame), the roles, and the camera that puts the bones on
the body picture. A concept drawn on the body keeps its framing, so the bones land on a
concept of any size by scaling from the body picture's size.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from image_to_3dlab.slot_parts import camera as camera_mod
from image_to_3dlab.slot_parts import skeleton

SCHEMA_VERSION = 1
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


# How a base body is dressed under its pieces: the user's choice when it is registered
# (17/10/2026), never assumed. Simple clothes are what 4Story and WoW show between armour pieces.
DRESSINGS = {
    "bodysuit": "bodysuit",
    "clothes": "simple clothes (a plain shirt, breeches and cloth shoes)",
    "underwear": "minimal plain underwear",
}


def dressing(kind, words: str = "") -> dict:
    """The body's dressing as body.json keeps it: one of DRESSINGS, or "other" with words."""
    words = str(words or "").strip()[:300]
    if kind not in DRESSINGS and not (kind == "other" and words):
        raise ValueError("say how the base body is dressed under its pieces: bodysuit, clothes, underwear, "
                         "or other with your own words")
    return {"kind": kind, "words": words}


def dressing_words(record: dict) -> str:
    """The words the outfit concept keeps the body's clothes with. A body.json written before the
    choice existed (16/10/2026) has no `dressing`: those bodies were all made in a bodysuit, so
    that is what it reads as. Kept while such bodies are registered."""
    chosen = record.get("dressing") or {"kind": "bodysuit", "words": ""}
    return ", ".join(part for part in (DRESSINGS.get(chosen.get("kind"), ""), chosen.get("words") or "") if part)


def record(name: str, *, image: Path, image_size: tuple[int, int], rigged: Path, views: Path,
           cam: camera_mod.Camera, parents: dict, heads: dict, found: dict, labels: dict | None = None,
           clips: dict | None = None, dressing: dict | None = None) -> dict:
    roles = skeleton.inherit_roles(skeleton.bone_roles(found), parents)  # refuses anything but a humanoid
    return {
        "schema_version": SCHEMA_VERSION, "name": name,
        "image": str(Path(image).resolve()), "image_size": [int(image_size[0]), int(image_size[1])],
        "rigged": str(Path(rigged).resolve()), "views": str(Path(views).resolve()),
        "camera": cam.to_dict(),
        "skeleton": {"parents": dict(parents), "heads": {b: np.asarray(h, float).round(6).tolist() for b, h in heads.items()}},
        "roles": {b: list(r) for b, r in roles.items()},
        "labels": dict(labels or {}),
        # the body's own animated clips (Kimodo, retargeted onto this rig): fit.clipping and
        # fit.holes pose the pieces with them
        "clips": {name: str(Path(path).resolve()) for name, path in (clips or {}).items()},
        **({"dressing": dict(dressing)} if dressing else {}),
    }


def register(name: str, image: Path, rigged: Path, views: Path | None = None, matte: Path | None = None,
             labels: dict | None = None, root: Path = Path("output/bodies"), clips: dict | None = None,
             dressing: dict | None = None) -> Path:
    """Read a rigged body and its Pixal3D run, check it is a humanoid, write body.json.

    `views` is the `<model>.svviews` folder of the Pixal3D run the body came from (asked for:
    the rig and finishing steps rename the GLB, so it cannot be guessed). `matte` defaults to
    Pixal3D's own cut-out, `<image stem>__matted.png` beside the picture, or the picture's own
    transparency when it arrived cut out.
    """
    from PIL import Image

    from image_to_3dlab.creature_family import classify
    from image_to_3dlab.glb_skin import SkinnedGLB

    image, rigged = Path(image), Path(rigged)
    if not dressing:
        raise ValueError("say how the base body is dressed under its pieces: bodysuit, clothes, underwear, "
                         "or other with your own words")
    if views is None:
        raise ValueError("give the Pixal3D camera folder (<model>.svviews) the body was made with")
    alpha = camera_mod.subject_alpha(image, matte)
    with Image.open(image) as picture:
        size = picture.size
    if alpha.shape[::-1] != size:
        raise ValueError(f"the cut-out is {alpha.shape[1]}x{alpha.shape[0]}, the picture {size[0]}x{size[1]}")
    rig = SkinnedGLB(rigged)
    parents, heads = rig.skeleton()
    found = classify(parents, heads, points=rig.mesh.vertices)
    skeleton.bone_roles(found)  # a humanoid, before anything else is read
    if not (Path(views) / "transforms.json").is_file():
        raise ValueError(f"{views} is not a Pixal3D camera folder (no transforms.json)")
    cam = camera_mod.from_views(views, alpha)
    return write(record(name, image=image, image_size=size, rigged=rigged, views=views, cam=cam,
                        parents=parents, heads=heads, found=found, labels=labels, clips=clips,
                        dressing=dressing), root)


def write(data: dict, root: Path) -> Path:
    name = str(data.get("name", ""))
    if not NAME.match(name):
        raise ValueError(f"body name {name!r}: letters, digits, '.', '_' or '-', as a folder name")
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "body.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    return path


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: body schema {data.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    return data


def segments_px(data: dict, size: tuple[int, int]) -> list[skeleton.Segment]:
    """The body's bone segments in pixels of a picture of `size` drawn on the body."""
    cam = camera_mod.Camera.from_dict(data["camera"])
    k = np.array([size[0] / data["image_size"][0], size[1] / data["image_size"][1]])
    parents = data["skeleton"]["parents"]
    heads = {b: np.asarray(h, float) for b, h in data["skeleton"]["heads"].items()}
    roles = {b: tuple(r) for b, r in data["roles"].items()}
    out = []
    for seg in skeleton.joint_segments(parents, heads, roles):
        head, end = cam.project(np.stack([seg.head, seg.end])) * k
        out.append(skeleton.Segment(seg.bone, seg.role, seg.side, head, end))
    return out
