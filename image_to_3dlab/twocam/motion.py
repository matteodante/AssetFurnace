"""The solved motion on disk (`motion.json`), and the rotation maths both sides share.

numpy only: Blender's own Python imports this module to key the motion onto the rig.

`motion.json` keeps AnyMove's schema, so its `apply_motion.py` can read ours:
`fps`, `bones` (root first), `trans` (T x 3, metres, world offset of the root's head),
`root_quat` (T x 4, w x y z, the root bone's pose basis), `euler` (T x B x 3, Blender
XYZ Euler per bone; the root's row is unused), plus `joints`, `joint_names`,
`reproj_px`, `sync` and `swaps` from the fit. We add `layout` and `provenance`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

MOTION_KEYS = ("fps", "bones", "trans", "root_quat", "euler")


class MotionError(ValueError):
    pass


def matrices_to_quaternions(m: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotations to unit quaternions (w, x, y, z) with w >= 0.

    Shepperd's method: build the quaternion from whichever of w, x, y, z is largest.
    The usual trick (all four from the diagonal, signs from the off-diagonal) picks
    wrong signs near half a turn, where the off-diagonal differences are all ~0.
    """
    m = np.asarray(m, np.float64)
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    tr = np.trace(m, axis1=1, axis2=2)
    d = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], 1)
    pick = d.argmax(1)
    for n, (r, k) in enumerate(zip(m, pick)):
        if k == 0:
            s = np.sqrt(1 + tr[n]) * 2
            q[n] = (s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
        elif k == 1:
            s = np.sqrt(max(1 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2
            q[n] = ((r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
        elif k == 2:
            s = np.sqrt(max(1 - r[0, 0] + r[1, 1] - r[2, 2], 1e-12)) * 2
            q[n] = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s)
        else:
            s = np.sqrt(max(1 - r[0, 0] - r[1, 1] + r[2, 2], 1e-12)) * 2
            q[n] = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q.reshape(*shape, 4)


def quaternions_to_matrices(q: np.ndarray) -> np.ndarray:
    """[..., 4] quaternions (w, x, y, z), not necessarily unit, to [..., 3, 3]."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def continuous_quaternions(q: np.ndarray) -> np.ndarray:
    """(T, 4) quaternions with each one's sign flipped to sit closest to the one before.

    q and -q are the same turn; without this, a creature turning past half a turn
    jumps between the two and anything that averages over time breaks.
    """
    q = np.array(q, np.float64)
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q


def quaternions_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """[..., 4] quaternions to axis * angle vectors. Angles can exceed pi when w < 0,
    which keeps a continuous quaternion track continuous."""
    q = np.asarray(q, np.float64)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, q[..., :1])
    return np.where(s > 1e-12, v / np.maximum(s, 1e-12) * angle, 2 * v)


def euler_xyz_matrix(e: np.ndarray) -> np.ndarray:
    """Blender Euler XYZ (X applied first, then Y, then Z) to a 3x3: Rz @ Ry @ Rx."""
    x, y, z = np.asarray(e, np.float64)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_euler_xyz(r: np.ndarray) -> np.ndarray:
    """Inverse of `euler_xyz_matrix` for one 3x3 (Y kept within +-90 degrees)."""
    r = np.asarray(r, np.float64)
    y = -np.arcsin(np.clip(r[2, 0], -1, 1))
    return np.array([np.arctan2(r[2, 1], r[2, 2]), y, np.arctan2(r[1, 0], r[0, 0])])


def root_location(rest_matrix: Sequence[Sequence[float]], trans: Sequence[float]) -> np.ndarray:
    """The root pose bone's `location` that moves its head by `trans` in armature space.

    A pose bone's location is in its own rest frame, so the world offset is turned by
    the transpose of the bone's rest rotation (`matrix_local`'s 3x3).
    """
    r = np.asarray(rest_matrix, np.float64)[:3, :3]
    return r.T @ np.asarray(trans, np.float64)


def validate_motion(motion: Mapping[str, Any], rig_bones: Sequence[str] | None = None) -> int:
    """Check a motion's shape before it is keyed; returns its frame count."""
    missing = [k for k in MOTION_KEYS if k not in motion]
    if missing:
        raise MotionError(f"motion is missing {', '.join(missing)}")
    fps = float(motion["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise MotionError(f"motion fps must be positive, got {motion['fps']!r}")
    bones = list(motion["bones"])
    if not bones:
        raise MotionError("motion has no bones")
    trans = np.asarray(motion["trans"], np.float64)
    quat = np.asarray(motion["root_quat"], np.float64)
    euler = np.asarray(motion["euler"], np.float64)
    frames = len(euler)
    if frames == 0:
        raise MotionError("motion has no frames")
    if trans.shape != (frames, 3) or quat.shape != (frames, 4) or euler.shape != (frames, len(bones), 3):
        raise MotionError(
            f"motion arrays disagree: trans {trans.shape}, root_quat {quat.shape}, "
            f"euler {euler.shape} for {len(bones)} bones"
        )
    for name, a in (("trans", trans), ("root_quat", quat), ("euler", euler)):
        if not np.isfinite(a).all():
            raise MotionError(f"motion {name} holds NaN or infinite values")
    if rig_bones is not None:
        absent = sorted(set(bones) - set(rig_bones))
        if absent:
            raise MotionError(f"the rig has no bones named {', '.join(absent[:8])}")
    return frames


def write_motion(motion: Mapping[str, Any], path: Path) -> None:
    validate_motion(motion)
    Path(path).write_text(json.dumps(motion), encoding="utf-8")


def load_motion(path: Path) -> dict[str, Any]:
    try:
        motion = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MotionError(f"cannot read motion {path}: {exc}") from exc
    validate_motion(motion)
    return motion
