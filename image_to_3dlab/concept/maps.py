"""The pictures a concept is drawn from: renders and depth maps of a GLB, a pose map of a rig, lineart.

- `render` runs Blender headless on the CPU (`scripts/blender_concept_views.py`): a lit
  colour picture and a depth map per view, and the bones in pixels.
- `flatten` puts a render on plain grey, the background the prompts ask for.
- `depth_on_black` stretches a depth render over the subject's own range, so the shape
  reads at full contrast whatever the model's size.
- `lineart` draws a picture's lines, for a reference that is a picture and not a model.
- `pose_points` and `draw_pose` turn a rig's bones into an OpenPose (COCO-18) map.

These are inputs for the model, not checks: nothing here judges a result.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_to_3dlab.blender import find_blender, missing_help

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "blender_concept_views.py"
VIEWS = ("front", "side", "back")
GREY = (217, 217, 217)
DEPTH_FAR, DEPTH_NEAR = 40, 255  # the farthest part stays apart from the black background

# OpenPose COCO-18: the order, the limbs and the colours the edit model knows.
KEYPOINTS = ("nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
             "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear")
LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17))
COLOURS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
           (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
           (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85))

# Joint -> the bone whose head sits on it, per naming: the UE mannequin (what the Unreal
# profile exports) and SOMA (what the retargeting reads). "l"/"r" are the character's own
# sides, as in OpenPose.
_UE = {"neck": "neck_01", "head": "head"}
_SOMA = {"neck": "Neck1", "head": "Head"}
for _side, _soma in (("l", "Left"), ("r", "Right")):
    for _joint, _ue_bone, _soma_bone in (("shoulder", "upperarm", "Arm"), ("elbow", "lowerarm", "ForeArm"),
                                         ("wrist", "hand", "Hand"), ("hip", "thigh", "Leg"),
                                         ("knee", "calf", "Shin"), ("ankle", "foot", "Foot")):
        _UE[f"{_side}_{_joint}"] = f"{_ue_bone}_{_side}"
        _SOMA[f"{_side}_{_joint}"] = f"{_soma}{_soma_bone}"
NAMINGS = (_UE, _SOMA)


def render_command(blender: str | Path, glb: Path, folder: Path, views=("front",), size: int = 1024) -> list[str]:
    unknown = [v for v in views if v not in VIEWS]
    if unknown or not views:
        raise ValueError(f"each view is one of {', '.join(VIEWS)}, got {list(views)!r}")
    return [str(blender), "-b", "--factory-startup", "-P", str(SCRIPT), "--", str(glb), str(folder),
            ",".join(views), str(size)]


def render(glb: Path, folder: Path, views=("front",), size: int = 1024, blender: Path | None = None,
           find: Callable[[], Path | None] = find_blender, run: Callable = subprocess.run) -> dict:
    """{view: {"colour": path, "depth": path}, "bones": path or None} for `glb`, rendered into `folder`."""
    blender = blender or find()
    if blender is None:
        raise RuntimeError(missing_help())
    folder = Path(folder)
    result = run(render_command(blender, glb, folder, views, size), capture_output=True, text=True,
                 errors="replace", check=False)
    files = {v: {"colour": folder / f"{v}_colour.png", "depth": folder / f"{v}_depth.png"} for v in views}
    missing = [p.name for v in views for p in files[v].values() if not p.is_file()]
    if result.returncode != 0 or missing:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"Blender did not render {Path(glb).name}: " + " | ".join(tail or missing))
    bones = folder / "bones.json"
    files["bones"] = bones if bones.is_file() else None
    return files


def flatten(rgba: Path, out: Path, background: tuple[int, int, int] = GREY) -> Path:
    image = Image.open(rgba).convert("RGBA")
    canvas = Image.new("RGBA", image.size, background + (255,))
    canvas.alpha_composite(image)
    canvas.convert("RGB").save(out)
    return Path(out)


def depth_on_black(rgba: Path, out: Path) -> Path:
    """Near white, far `DEPTH_FAR` grey, over the subject's 1st-99th percentile; background black."""
    image = np.asarray(Image.open(rgba).convert("RGBA")).astype(np.float32)
    mask = image[..., 3] > 127
    if not mask.any():
        raise ValueError(f"{Path(rgba).name} shows nothing: no opaque pixel to take a depth from")
    value = image[..., 0]
    low, high = np.percentile(value[mask], 1), np.percentile(value[mask], 99)
    stretched = np.clip((value - low) / max(high - low, 1.0), 0, 1) * (DEPTH_NEAR - DEPTH_FAR) + DEPTH_FAR
    grey = np.where(mask, stretched, 0).astype(np.uint8)
    Image.fromarray(np.dstack([grey] * 3)).save(out)
    return Path(out)


def lineart(picture: Path, out: Path, size: int = 1024) -> Path:
    """Black lines on white: a soft difference of Gaussians on the light, plus edges of colour.

    The colour edges catch what light alone misses, such as orange on light grey. The
    picture is scaled to fit `size` without changing its proportions (a round shield stays
    round) and the lines are centred on white paper; padding added before the lines would
    draw its own border.
    """
    colour = cv2.cvtColor(np.asarray(Image.open(picture).convert("RGB")), cv2.COLOR_RGB2BGR)
    scale = size / max(colour.shape[:2])
    fitted = (max(1, round(colour.shape[1] * scale)), max(1, round(colour.shape[0] * scale)))
    colour = cv2.resize(colour, fitted, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    grey = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    difference = cv2.GaussianBlur(grey, (0, 0), 1.0) - 0.98 * cv2.GaussianBlur(grey, (0, 0), 1.6)
    lines = np.where(difference >= 0.0, 1.0, 1.0 + np.tanh(40.0 * difference))
    median = float(np.median(colour))
    edges = cv2.Canny(cv2.GaussianBlur(colour, (0, 0), 1.2), 0.25 * median, 0.5 * median)
    lines = np.where(cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0, 0.0, lines)
    paper = np.full((size, size), 255, np.uint8)
    top, left = (size - lines.shape[0]) // 2, (size - lines.shape[1]) // 2
    paper[top:top + lines.shape[0], left:left + lines.shape[1]] = (np.clip(lines, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(paper).save(out)
    return Path(out)


def pose_points(bones: dict[str, dict], naming: dict[str, str] | None = None) -> dict[str, tuple[float, float]]:
    """The 18 OpenPose joints of one view of a rig.

    The bones are found by name, like the UE mannequin or SOMA, unless `naming` maps each
    joint (`NAMINGS`' keys) to a bone: a rig with numbered bones, such as one from
    SkinTokens, needs its roles found some other way. The face points are placed from the
    head bone, which runs from the neck up: the nose a little over halfway, eyes and ears
    to either side.
    """
    if naming is not None:
        absent = sorted(set(_UE) - set(naming)) + sorted(b for b in naming.values() if b not in bones)
        if absent:
            raise ValueError("the joint naming does not cover the rig; missing: " + ", ".join(absent))
    else:
        naming = next((n for n in NAMINGS if all(b in bones for b in n.values())), None)
    if naming is None:
        best = min(NAMINGS, key=lambda n: sum(b not in bones for b in n.values()))
        missing = sorted(b for b in best.values() if b not in bones)
        raise ValueError("the rig's bones are not named like the UE mannequin or SOMA; missing: "
                         + ", ".join(missing))
    head = {joint: np.array(bones[bone]["head"], float) for joint, bone in naming.items()}
    top = np.array(bones[naming["head"]]["tail"], float)
    base = head.pop("head")
    span = max(float(np.linalg.norm(top - base)), 1e-6)
    nose = base + (top - base) * 0.55
    up = (top - base) / span * span * 0.12
    # The character's left is on the viewer's right in a front view, and its eyes follow its shoulders.
    across = head["l_shoulder"] - head["r_shoulder"]
    side = across / max(float(np.linalg.norm(across)), 1e-6) * span * 0.18
    points = {**head, "nose": nose, "r_eye": nose - side + up, "l_eye": nose + side + up,
              "r_ear": nose - side * 2.2, "l_ear": nose + side * 2.2}
    return {name: (float(points[name][0]), float(points[name][1])) for name in KEYPOINTS}


def draw_pose(points: dict[str, tuple[float, float]], size: int, out: Path) -> Path:
    """The OpenPose drawing: dimmed coloured limbs as ellipses, bright joints, on black."""
    canvas = np.zeros((size, size, 3), np.uint8)
    stick = max(4, size // 128)
    for i, (p, q) in enumerate(LIMBS):
        a, b = np.array(points[KEYPOINTS[p]]), np.array(points[KEYPOINTS[q]])
        centre = tuple(int(v) for v in (a + b) / 2)
        length = int(np.linalg.norm(a - b) / 2)
        angle = int(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
        polygon = cv2.ellipse2Poly(centre, (length, stick), angle, 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, tuple(int(c * 0.6) for c in COLOURS[i]))
    for i, name in enumerate(KEYPOINTS):
        cv2.circle(canvas, tuple(int(v) for v in points[name]), stick + 2, COLOURS[i], -1)
    Image.fromarray(canvas).save(out)
    return Path(out)
