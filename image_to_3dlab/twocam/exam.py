"""The exam: a made-up motion with known joints, read back through the fit and scored in centimetres.

AnyMove's `mocap/exam.py` films the made-up motion in Blender and reads it with the trained
reader, which tests everything at once. This keeps its motion maker and adds the pure half:
project the true joints into both views, add reader-like noise, fit, and score. It needs no
Blender, no video and no weights, so it says in seconds whether the fit itself is sound for
a rig, and it is what the unit tests run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .layout import Layout
from .motion import quaternions_to_matrices
from .solve import Rig, limb_pairs, validate_rig_info


def make_motion(info: Mapping[str, Any], seed: int = 0, frames: int = 120, fps: float = 24,
                amplitude: float = 1.0) -> dict[str, Any]:
    """A random smooth motion inside every bone's limits, in `motion.json`'s schema."""
    validate_rig_info(info)
    rng = np.random.default_rng(seed)
    bones = [b["name"] for b in info["bones"]]
    t = np.arange(frames) / fps
    eul = np.zeros((frames, len(bones), 3))

    def wave(amp):
        f = rng.uniform(0.4, 1.8, 3)
        ph = rng.uniform(0, 2 * math.pi, 3)
        w = rng.dirichlet(np.ones(3))
        return amp * (w[:, None] * np.sin(2 * math.pi * f[:, None] * t + ph[:, None])).sum(0)

    for i, b in enumerate(info["bones"]):
        if i == 0:
            continue
        lim = np.radians(np.array(b["limits_deg"], float))
        for k in range(3):
            lo, hi = lim[k]
            c = np.clip(0.0, lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo))
            amp = amplitude * (0.45 if k == 0 else 0.25) * (hi - lo) / 2
            eul[:, i, k] = np.clip(c + wave(amp), lo, hi)
    yaw = wave(math.radians(20) * amplitude)
    pitch = wave(math.radians(8) * amplitude)
    roll = wave(math.radians(5) * amplitude)
    quat = []
    for y, p, r in zip(yaw, pitch, roll):
        # pelvis basis: local X = pitch, local Y (along the bone) = roll, local Z = yaw
        qx = np.array([math.cos(p / 2), math.sin(p / 2), 0, 0])
        qy = np.array([math.cos(r / 2), 0, math.sin(r / 2), 0])
        qz = np.array([math.cos(y / 2), 0, 0, math.sin(y / 2)])
        quat.append(qmul(qz, qmul(qy, qx)).tolist())
    trans = np.stack([wave(0.05), wave(0.12), 0.03 * np.abs(np.sin(2 * math.pi * 1.2 * t))], 1)
    return {"fps": float(fps), "bones": bones, "trans": trans.round(5).tolist(), "root_quat": quat,
            "euler": eul.round(5).tolist()}


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def motion_joints(info: Mapping[str, Any], motion: Mapping[str, Any]) -> np.ndarray:
    """World joints (T, J, 3) of a motion on this rig, by the same kinematics the fit uses."""
    rig = Rig(info)
    f64 = {"dtype": torch.float64}
    with torch.no_grad():
        joints, _ = rig.forward(torch.tensor(motion["trans"], **f64),
                                torch.tensor(quaternions_to_matrices(motion["root_quat"]), **f64),
                                torch.tensor(motion["euler"], **f64))
    return joints.numpy()


def synthetic_detections(joints: np.ndarray, layout: Layout, noise_px: float = 0.0,
                         seed: int = 0) -> dict[str, np.ndarray]:
    """What a perfect reader would see, plus Gaussian noise of `noise_px` pixels."""
    rng = np.random.default_rng(seed)
    front, side = layout.project(joints)
    shape = joints.shape[:2]
    return {"front_uv": front + rng.normal(0, noise_px, front.shape) if noise_px else front,
            "side_uv": side + rng.normal(0, noise_px, side.shape) if noise_px else side,
            "front_conf": np.ones(shape), "side_conf": np.ones(shape)}


def score(info: Mapping[str, Any], truth: np.ndarray, solved: np.ndarray) -> dict[str, Any]:
    """Per-joint 3D error in centimetres: overall, per limb (from the rig's own `_L`/`_R`
    limbs, not the wolf's names as in AnyMove) and the worst five joints."""
    names = [k["name"] for k in info["keypoints"]]
    frames = min(len(truth), len(solved))
    e = np.linalg.norm(np.asarray(truth)[:frames] - np.asarray(solved)[:frames], axis=-1) * 100
    out: dict[str, Any] = {"mean_cm": float(e.mean()), "median_cm": float(np.median(e)),
                           "p95_cm": float(np.percentile(e, 95))}
    in_limb = set()
    for limb, pairs in limb_pairs(info).items():
        idx = sorted({i for pair in pairs for i in pair})
        in_limb.update(idx)
        out[f"{limb}_cm"] = float(e[:, idx].mean())
    body = [i for i in range(len(names)) if i not in in_limb]
    if body:
        out["body_cm"] = float(e[:, body].mean())
    worst = np.argsort(-e.mean(0))[:5]
    out["worst_joints"] = {names[i]: round(float(e[:, i].mean()), 2) for i in worst}
    return out
