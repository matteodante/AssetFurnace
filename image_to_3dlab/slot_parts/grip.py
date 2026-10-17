"""Where a weapon or shield is held: its grip, a point and an axis on its own 3D model.

A weapon or shield is a game asset of its own, not tied to a character (the user's decision,
17/10/2026). What a game needs to attach it is its grip: the static mesh's pivot sits there and
a `grip` socket names it (`scripts/export_unreal.py`). The grip is found on the canonical
picture the model was made from (upright, grip at the bottom, point or head at the top), by
the picture judge: Qwen3-VL's grounding gives the box of "the grip of the sword" on its 0-1000
grid (QwenLM/Qwen3-VL cookbooks/2d_grounding.ipynb, read 17/10/2026). For a hafted weapon it
gives the whole haft (asked several ways on a real axe, 17/10/2026), so where along that box
the hand closes is the class's: near the end for an axe, a hammer or a mace, the middle for a
staff. The Pixal3D camera brings that part of the box onto the model: the grip is the middle of
the model's surface seen inside it, and the axis is the model's long axis, toward the top of
the picture. A shield's grip is its
strap, on the back, which a front picture does not show: the centre of its back.

A person corrects it the same way on every route (the studio's marker, `slot_parts.py grip`,
MCP `pieces_grip`): a point on the picture, the class, the real size. The record is
`<piece>/grip.json` beside the picture.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from image_to_3dlab.slot_parts.camera import Camera


@dataclass(frozen=True)
class GripClass:
    phrase: str        # what the judge is asked to find; "" when the picture cannot show it
    size_cm: float     # a usual real length (the longest side), a person changes it
    words: str         # how the words name it (a regular expression)
    # where along the judge's box (top 0, bottom 1) the hand closes, and half the band it holds
    hold: tuple[float, float] = (0.5, 0.5)


CLASSES = {
    "dagger": GripClass("the grip of the dagger, the part a hand holds", 35, r"dagger|knife|dirk|stiletto"),
    "bow": GripClass("the grip in the middle of the bow, where a hand holds it", 140, r"bow\b|bows\b"),
    "axe": GripClass("the handle of the axe, where a hand holds it", 80, r"axe|hatchet", (0.8, 0.12)),
    "hammer": GripClass("the handle of the hammer, where a hand holds it", 80, r"hammer|maul", (0.8, 0.12)),
    "mace": GripClass("the handle of the mace, where a hand holds it", 70, r"mace|morning ?star|flail|club",
                      (0.75, 0.15)),
    "spear": GripClass("the shaft of the spear, where a hand holds it", 220,
                       r"spear|lance|pike|halberd|glaive|trident|polearm", (0.6, 0.12)),
    "staff": GripClass("the middle of the staff, where a hand holds it", 170, r"staff|wand|sceptre|scepter|rod\b",
                       (0.5, 0.12)),
    "sword": GripClass("the grip of the sword, the part a hand holds", 100,
                       r"sword|blade|sabre|saber|katana|scimitar|rapier|falchion|claymore"),
    "weapon": GripClass("the handle of the weapon, the part a hand holds", 100, r"$^"),
    "shield": GripClass("", 75, r"shield|buckler"),
}
QUESTION = "Locate {phrase}, output its bbox coordinates using JSON format."
WINDOW = 0.025       # a person's point: this share of the picture around it
SAMPLES = 20000      # points on the model's surface, to see what lies inside a box


def weapon_class(slot: str, words: str = "") -> str:
    if slot == "shield":
        return "shield"
    text = (words or "").lower()
    for name, entry in CLASSES.items():   # in order: a "dagger" is not a "sword", a "longbow" is a bow
        if name != "shield" and re.search(entry.words, text):
            return name
    return "weapon"


def parse_box(text: str, size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """The first `bbox_2d` the judge answered, in picture pixels (its grid is 0-1000)."""
    found = re.search(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]',
                      text or "")
    if not found:
        return None
    x0, y0, x1, y1 = (float(v) for v in found.groups())
    w, h = size
    return (min(x0, x1) / 1000 * w, min(y0, y1) / 1000 * h, max(x0, x1) / 1000 * w, max(y0, y1) / 1000 * h)


def long_axis(points: np.ndarray) -> np.ndarray:
    """The model's long axis, pointing up the picture (glTF +Y), toward the point or head."""
    p = np.asarray(points, float)
    _, _, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    return -axis if axis[1] < 0 else axis


def from_box(points: np.ndarray, cam: Camera, box, hold: tuple[float, float] = (0.5, 0.5)
             ) -> tuple[list[float], list[float]]:
    """The grip inside a picture box: the middle of the model's points seen in its `hold` band
    (where along the box the hand closes), front and back alike, and the long axis."""
    p = np.asarray(points, float)
    uv = cam.project(p)
    x0, y0, x1, y1 = box
    centre, half = hold
    y0, y1 = y0 + (centre - half) * (y1 - y0), y0 + (centre + half) * (y1 - y0)
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if not inside.any():   # a box beside the model: the points seen nearest its middle
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        nearest = np.argsort(np.linalg.norm(uv - centre, axis=1))[:max(len(p) // 200, 1)]
        inside = np.zeros(len(p), bool)
        inside[nearest] = True
    return [float(v) for v in p[inside].mean(0)], [float(v) for v in long_axis(p)]


def shield_grip(points: np.ndarray) -> tuple[list[float], list[float]]:
    """A shield is held by the strap on its back: the middle of its back (the side away from
    the camera, glTF -Z), upright."""
    p = np.asarray(points, float)
    lo, hi = p.min(0), p.max(0)
    return [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(lo[2])], [0.0, 1.0, 0.0]


def mesh_points(model: Path, count: int = SAMPLES) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(str(model), force="mesh", process=False)
    sampled, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
    return np.vstack([np.asarray(mesh.vertices, float), np.asarray(sampled, float)])


def _size(picture: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(picture) as image:
        return image.size


def find(model: Path, picture: Path, cam: Camera, *, slot: str, words: str = "",
         ask_box: Callable[[Path, str], str] | None, weapon_class_: str | None = None) -> dict:
    """The grip of a model made from `picture`: by the judge for a weapon (`ask_box` answers a
    grounding question with its text), by the shield's back for a shield. Without the judge a
    weapon has no grip yet, and it is said: the export then keeps the model's base, and a person
    can place it."""
    name = weapon_class_ or weapon_class(slot, words)
    record = {"slot": slot, "class": name, "size_cm": CLASSES[name].size_cm, "picture": str(picture),
              "model": str(model), "camera": cam.to_dict(), "box": None, "point": None, "axis": None, "by": "none"}
    if name == "shield":
        point, axis = shield_grip(mesh_points(model))
        return {**record, "point": point, "axis": axis, "by": "rule"}
    if ask_box is None:
        print(f"{slot}: no grip found, the picture judge is not installed (Setup & Status installs it); "
              "place it with `slot_parts.py grip` or the studio's marker", flush=True)
        return record
    box = parse_box(ask_box(Path(picture), QUESTION.format(phrase=CLASSES[name].phrase)), _size(picture))
    if box is None:
        print(f"{slot}: the picture judge found no grip; place it with `slot_parts.py grip` or the studio's marker",
              flush=True)
        return record
    point, axis = from_box(mesh_points(model), cam, box, CLASSES[name].hold)
    return {**record, "box": [round(v, 1) for v in box], "point": point, "axis": axis, "by": "judge"}


def correct(record: dict, *, at: tuple[float, float] | None = None, weapon_class: str | None = None,
            size_cm: float | None = None) -> dict:
    """A person's correction: `at` a point on the picture (fractions of its width and height),
    `weapon_class` one of CLASSES, `size_cm` the real length. What is not given stays."""
    out = dict(record)
    if weapon_class is not None:
        if weapon_class not in CLASSES:
            raise ValueError(f"unknown class {weapon_class!r}: choose one of {', '.join(CLASSES)}")
        if weapon_class != record.get("class") and size_cm is None:
            out["size_cm"] = CLASSES[weapon_class].size_cm
        out["class"] = weapon_class
    if size_cm is not None:
        if not 1 <= float(size_cm) <= 1000:
            raise ValueError("the size is the real length in centimetres, 1 to 1000")
        out["size_cm"] = float(size_cm)
    if at is not None:
        u, v = (float(a) for a in at)
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError("the grip is a point on the picture: two fractions between 0 and 1")
        w, h = _size(Path(record["picture"]))
        half = WINDOW * max(w, h)
        box = (u * w - half, v * h - half, u * w + half, v * h + half)
        points = mesh_points(Path(record["model"]))
        point, axis = from_box(points, Camera.from_dict(record["camera"]), box)
        if out.get("class") == "shield":   # still held on its back, upright: only where on it moves
            point, axis = [point[0], point[1], float(points[:, 2].min())], [0.0, 1.0, 0.0]
        out.update(box=[round(c, 1) for c in box], point=point, axis=axis)
    if at is not None or weapon_class is not None or size_cm is not None:
        out["by"] = "person"
    return out


def path_for(picture: Path) -> Path:
    return Path(picture).parent / "grip.json"


def write(record: dict, path: Path) -> Path:
    Path(path).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return Path(path)


def load(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def correct_in(folder: Path, slot: str, *, at: tuple[float, float] | None = None, weapon_class: str | None = None,
               size_cm: float | None = None) -> dict:
    """A person's correction of a set's weapon or shield (`folder` is output/sets/<set>/<body>),
    saved: the studio's marker, `slot_parts.py grip` and MCP `pieces_grip` all come here."""
    from image_to_3dlab.slot_parts import slots

    if slot not in slots.STANDALONE:
        raise ValueError(f"{slot}: only a weapon or shield has a grip (slot {', '.join(slots.STANDALONE)})")
    path = Path(folder) / "pieces" / slot / "grip.json"
    for parts in sorted(Path(folder).glob("*.parts.json")):   # a carried set names the one it was made with
        try:
            named = (json.loads(parts.read_text(encoding="utf-8")).get("pieces") or {}).get(slot) or {}
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(named, dict) and named.get("grip"):
            path = Path(named["grip"])
    record = load(path)
    if record is None:
        raise ValueError(f"no grip record for the {slot} in {folder}: make the set first")
    corrected = correct(record, at=at, weapon_class=weapon_class, size_cm=size_cm)
    write(corrected, path)
    return corrected


def picture_point(record: dict) -> list[float] | None:
    """Where the grip is on its picture, as fractions of its width and height (the studio's
    marker): the middle of its box, else its point seen through the camera."""
    try:
        w, h = _size(Path(record["picture"]))
    except (OSError, KeyError, ValueError):
        return None
    if record.get("point") and record.get("camera"):
        u, v = Camera.from_dict(record["camera"]).project(np.asarray([record["point"]], float))[0]
        return [round(min(max(u / w, 0.0), 1.0), 4), round(min(max(v / h, 0.0), 1.0), 4)]
    if record.get("box"):
        x0, y0, x1, y1 = record["box"]
        return [round((x0 + x1) / 2 / w, 4), round((y0 + y1) / 2 / h, 4)]
    return None
