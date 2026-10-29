"""A toy rig for the two-camera tests: a small quadruped with a tail, built the way
AnyMove's `build_rig.py` builds bones (local X kept horizontal and across the bone).

Its limb names are deliberately not the wolf's (`leg_front_L`, not `upperarm_L`), and its
layout is not the default, so a test passes only if the code reads both from the rig.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

LAYOUT = {"H": 720, "W_FRONT": 640, "W_SIDE": 640, "M_PER_PX": 0.004, "GROUND_V": 630}

JOINTS = {
    "hips": (0, 0.30, 0.80), "chest": (0, -0.30, 0.85), "neck": (0, -0.50, 1.05), "head": (0, -0.75, 1.10),
    "tail1": (0, 0.55, 0.80), "tail2": (0, 0.85, 0.70),
    "shoulder_L": (0.15, -0.30, 0.75), "elbow_L": (0.16, -0.32, 0.42), "paw_L": (0.16, -0.34, 0.05),
    "toe_L": (0.16, -0.44, 0.02),
    "hip_L": (0.14, 0.30, 0.75), "knee_L": (0.15, 0.22, 0.42), "foot_L": (0.15, 0.32, 0.05),
    "heel_L": (0.15, 0.22, 0.02),
}
# name, head joint, tail joint, parent
BONES = [
    ("pelvis", "hips", "chest", None), ("neck", "neck", "head", "pelvis"),
    ("tail_a", "tail1", "tail2", "pelvis"),
    ("leg_front_L", "shoulder_L", "elbow_L", "pelvis"), ("shin_front_L", "elbow_L", "paw_L", "leg_front_L"),
    ("paw_front_L", "paw_L", "toe_L", "shin_front_L"),
    ("leg_hind_L", "hip_L", "knee_L", "pelvis"), ("shin_hind_L", "knee_L", "foot_L", "leg_hind_L"),
    ("paw_hind_L", "foot_L", "heel_L", "shin_hind_L"),
]
LIMITS = {"pelvis": [[-30, 30], [-20, 20], [-30, 30]], "neck": [[-40, 40], [-20, 20], [-40, 40]],
          "tail_a": [[-40, 40], [-20, 20], [-40, 40]], "leg": [[-70, 50], [-10, 10], [-15, 15]],
          "shin": [[-10, 110], [-5, 5], [-5, 5]], "paw": [[-60, 60], [-5, 5], [-5, 5]]}


def mirror(name: str | None) -> str | None:
    return name[:-2] + "_R" if name and name.endswith("_L") else name


def bone_matrix(head, tail) -> list[list[float]]:
    head, tail = np.asarray(head, float), np.asarray(tail, float)
    y = (tail - head) / np.linalg.norm(tail - head)
    z = np.cross([1.0, 0, 0], y)
    if np.linalg.norm(z) < 0.5:
        xl = np.cross([0, 0, 1.0], y)
        z = np.cross(xl / np.linalg.norm(xl), y) if np.linalg.norm(xl) > 1e-4 else np.array([0, 0, 1.0])
    z = z - np.dot(z, y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, head
    return m.tolist()


def toy_rig() -> dict:
    joints = dict(JOINTS)
    for k, v in JOINTS.items():
        if k.endswith("_L"):
            joints[mirror(k)] = (-v[0], v[1], v[2])
    bones = []
    for b in BONES:
        bones.append(b)
        if b[0].endswith("_L"):
            bones.append(tuple(mirror(x) for x in b))
    order = [b[0] for b in bones]
    info = {"bones": [], "keypoints": [], "layout": dict(LAYOUT)}
    for name, h, t, parent in bones:
        stem = name if name in LIMITS else name.split("_")[0]
        info["bones"].append({
            "name": name, "parent": order.index(parent) if parent else -1,
            "matrix_local": bone_matrix(joints[h], joints[t]),
            "length": float(np.linalg.norm(np.subtract(joints[t], joints[h]))),
            "head_joint": h, "tail_joint": t, "limits_deg": copy.deepcopy(LIMITS[stem])})
    for j, p in joints.items():
        src = next(((i, "head") for i, b in enumerate(bones) if b[1] == j), None) or \
            next(((i, "tail") for i, b in enumerate(bones) if b[2] == j), None)
        info["keypoints"].append({"name": j, "bone": src[0], "end": src[1], "rest": list(p)})
    return info


def write_rig(folder: Path, info: dict | None = None) -> Path:
    path = Path(folder) / "rig_info.json"
    path.write_text(json.dumps(info or toy_rig()), encoding="utf-8")
    return path
