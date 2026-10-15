"""A tiny rigged GLB for tests: a strip two units tall on two bones, and optional clips.

The lower bone sits at the origin and the upper one at y = 1; vertices below y = 1 follow
the lower bone, the rest the upper one. A clip is a list of (time, quaternion xyzw) keys
for the upper bone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from image_to_3dlab.glb_turn import build_glb

# a 1 x 2 strip, three rows of two vertices, in the XY plane
VERTICES = np.array([[0, 0, 0], [0.5, 0, 0], [0, 1, 0], [0.5, 1, 0], [0, 2, 0], [0.5, 2, 0]], np.float32)
FACES = np.array([[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]], np.uint16)


def turn_z(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2
    return [0.0, 0.0, math.sin(half), math.cos(half)]


class _Buffer:
    def __init__(self):
        self.data, self.views, self.accessors = b"", [], []

    def add(self, array: np.ndarray, kind: str, component: int, **extra) -> int:
        self.data += b"\0" * (-len(self.data) % 4)
        self.views.append({"buffer": 0, "byteOffset": len(self.data), "byteLength": array.nbytes})
        self.data += array.tobytes()
        accessor = {"bufferView": len(self.views) - 1, "componentType": component, "count": len(array),
                    "type": kind, **extra}
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def write(path: Path, clips: dict[str, list[tuple[float, list[float]]]] | None = None,
          weights: np.ndarray | None = None) -> Path:
    buf = _Buffer()
    pos = buf.add(VERTICES, "VEC3", 5126, min=VERTICES.min(0).tolist(), max=VERTICES.max(0).tolist())
    idx = buf.add(FACES.reshape(-1), "SCALAR", 5123)
    upper = (VERTICES[:, 1] >= 1).astype(np.uint8)
    joints = np.stack([upper, np.zeros(6, np.uint8), np.zeros(6, np.uint8), np.zeros(6, np.uint8)], 1)
    if weights is None:
        weights = np.tile(np.array([1, 0, 0, 0], np.float32), (6, 1))
    jnt = buf.add(joints.astype(np.uint8), "VEC4", 5121)
    wgt = buf.add(weights.astype(np.float32), "VEC4", 5126)
    ibm = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    ibm[1][1, 3] = -1.0  # upper bone rests at y = 1
    ibms = buf.add(ibm.transpose(0, 2, 1).reshape(2, 16), "MAT4", 5126)
    animations = []
    for name, keys in (clips or {}).items():
        times = np.array([t for t, _ in keys], np.float32)
        quats = np.array([q for _, q in keys], np.float32)
        t_acc = buf.add(times, "SCALAR", 5126, min=[float(times.min())], max=[float(times.max())])
        q_acc = buf.add(quats, "VEC4", 5126)
        animations.append({"name": name, "samplers": [{"input": t_acc, "output": q_acc}],
                           "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}]})
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "strip", "mesh": 0, "skin": 0},
                  {"name": "bone_0", "children": [2]},
                  {"name": "bone_1", "translation": [0, 1, 0]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pos, "JOINTS_0": jnt, "WEIGHTS_0": wgt},
                                    "indices": idx}]}],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": ibms}],
        "accessors": buf.accessors, "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
    }
    if animations:
        doc["animations"] = animations
    data = buf.data + b"\0" * (-len(buf.data) % 4)
    doc["buffers"][0]["byteLength"] = len(data)
    Path(path).write_bytes(build_glb(json.loads(json.dumps(doc)), data))
    return Path(path)

