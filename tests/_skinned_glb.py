"""A tiny rigged GLB for tests: a strip two units tall on two bones, and optional clips.

The lower bone sits at the origin and the upper one at y = 1; vertices below y = 1 follow
the lower bone, the rest the upper one. A clip is a list of (time, quaternion xyzw) keys
for one bone: a rotation (xyzw) by default, or any `channel` and `interpolation`; a
CUBICSPLINE key's value is an (in-tangent, value, out-tangent) triple.
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


def turn(axis: str, degrees: float) -> list[float]:
    half = math.radians(degrees) / 2
    q = [0.0, 0.0, 0.0, math.cos(half)]
    q["xyz".index(axis)] = math.sin(half)
    return q


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
          weights: np.ndarray | None = None, node: int = 2, channel: str = "rotation",
          interpolation: str = "LINEAR", rest_turn: list[float] | None = None, second_skin: bool = False,
          rest_scale: list[float] | None = None, second_weights: bool = False,
          second_primitive: bool = False, duplicate_clip: bool = False) -> Path:
    """`node` is the bone the clips move: 2 the upper one (a bend), 1 the lower (the whole strip).
    `rest_turn` turns the lower bone at rest without changing the bind matrices, so the
    rest pose and the bind pose differ, as they do in many exported rigs; `rest_scale` mirrors
    it the same way. `second_weights` moves every weight to a JOINTS_1/WEIGHTS_1 set on the
    upper bone, leaving the first set empty. `second_primitive` adds a copy of the strip with
    only the first set (four influences beside eight). `duplicate_clip` repeats every clip
    with its keys reset to rest, under the same name."""
    buf = _Buffer()
    pos = buf.add(VERTICES, "VEC3", 5126, min=VERTICES.min(0).tolist(), max=VERTICES.max(0).tolist())
    idx = buf.add(FACES.reshape(-1), "SCALAR", 5123)
    upper = (VERTICES[:, 1] >= 1).astype(np.uint8)
    joints = np.stack([upper, np.zeros(6, np.uint8), np.zeros(6, np.uint8), np.zeros(6, np.uint8)], 1)
    if weights is None:
        weights = np.tile(np.array([1, 0, 0, 0], np.float32), (6, 1))
    jnt = buf.add(joints.astype(np.uint8), "VEC4", 5121)
    wgt = buf.add((weights * (0 if second_weights else 1)).astype(np.float32), "VEC4", 5126)
    extra = {}
    if second_weights:
        extra["JOINTS_1"] = buf.add(np.tile(np.array([1, 0, 0, 0], np.uint8), (6, 1)), "VEC4", 5121)
        extra["WEIGHTS_1"] = buf.add(np.tile(np.array([1, 0, 0, 0], np.float32), (6, 1)), "VEC4", 5126)
    ibm = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    ibm[1][1, 3] = -1.0  # upper bone rests at y = 1
    ibms = buf.add(ibm.transpose(0, 2, 1).reshape(2, 16), "MAT4", 5126)
    animations = []
    for name, keys in (clips or {}).items():
        times = np.array([t for t, _ in keys], np.float32)
        values = np.array([v for _, v in keys], np.float32).reshape(-1, 4 if channel == "rotation" else 3)
        t_acc = buf.add(times, "SCALAR", 5126, min=[float(times.min())], max=[float(times.max())])
        q_acc = buf.add(values, "VEC4" if channel == "rotation" else "VEC3", 5126)
        animations.append({"name": name, "samplers": [{"input": t_acc, "output": q_acc,
                                                       "interpolation": interpolation}],
                           "channels": [{"sampler": 0, "target": {"node": node, "path": channel}}]})
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "strip", "mesh": 0, "skin": 0},
                  {"name": "bone_0", "children": [2], **({"rotation": rest_turn} if rest_turn else {}),
                   **({"scale": rest_scale} if rest_scale else {})},
                  {"name": "bone_1", "translation": [0, 1, 0]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": pos, "JOINTS_0": jnt, "WEIGHTS_0": wgt, **extra},
                                    "indices": idx}]}],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": ibms}],
        "accessors": buf.accessors, "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
    }
    if duplicate_clip:
        rest_keys = buf.add(np.tile(np.array([0, 0, 0, 1], np.float32), (2, 1)), "VEC4", 5126)
        animations += [{"name": a["name"], "samplers": [{"input": a["samplers"][0]["input"], "output": rest_keys}],
                        "channels": a["channels"]} for a in list(animations)]
    if animations:
        doc["animations"] = animations
    if second_primitive:
        doc["meshes"][0]["primitives"].append(
            {"attributes": {"POSITION": pos, "JOINTS_0": jnt, "WEIGHTS_0": buf.add(
                np.tile(np.array([1, 0, 0, 0], np.float32), (6, 1)), "VEC4", 5126)}, "indices": idx})
        doc["buffers"][0]["byteLength"] = len(buf.data)
    if second_skin:
        doc["skins"].append({"joints": [2]})
        doc["nodes"].append({"name": "strip_2", "mesh": 0, "skin": 1})
        doc["scenes"][0]["nodes"].append(3)
    data = buf.data + b"\0" * (-len(buf.data) % 4)
    doc["buffers"][0]["byteLength"] = len(data)
    Path(path).write_bytes(build_glb(json.loads(json.dumps(doc)), data))
    return Path(path)

