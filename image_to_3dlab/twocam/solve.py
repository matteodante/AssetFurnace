"""Fit the rig to the joints read from both views: one pose per frame, smooth, inside the joint limits.

Ported from AnyMove's `mocap/solve.py` (Patch, Apache-2.0). Before fitting, the clip is
scored: a joint has the same height in both halves, so the height gap between the views
says how well the video model kept them in step, and dynamic time warping re-times the
front view onto the side view when that closes the gap. The fit then redoes Blender's
forward kinematics in torch (pose = parent pose x rest offset x Euler XYZ), with the
root free and every other bone squashed into its limits by a sigmoid, and minimises a
robust reprojection error in both views plus smoothness, a pull to rest and a floor.

Inputs are a detections dict (`front_uv`, `side_uv`: T x J x 2 layout px; `front_conf`,
`side_conf`: T x J) and a `rig_info` dict as AnyMove's `build_rig.py` writes it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .layout import Layout
from .motion import (
    continuous_quaternions,
    euler_xyz_matrix,
    matrices_to_quaternions,
    matrix_to_euler_xyz,
    quaternions_to_axis_angle,
)

DETECTION_KEYS = ("front_uv", "side_uv", "front_conf", "side_conf")


class RigInfoError(ValueError):
    pass


class DetectionError(ValueError):
    pass


# --- the rig ---------------------------------------------------------------------------


def validate_rig_info(info: Mapping[str, Any]) -> None:
    """Fail early, with a reason, on a rig the fit could not use."""
    bones = info.get("bones")
    keypoints = info.get("keypoints")
    if not bones or not keypoints:
        raise RigInfoError("rig_info needs non-empty 'bones' and 'keypoints'")
    for i, b in enumerate(bones):
        for key in ("name", "parent", "matrix_local", "length", "limits_deg", "head_joint", "tail_joint"):
            if key not in b:
                raise RigInfoError(f"bone {i} has no {key!r}")
        parent = b["parent"]
        if (i == 0) != (parent == -1):
            raise RigInfoError(f"bone {b['name']!r}: only the first bone may be the root (parent -1)")
        if i and not 0 <= parent < i:
            raise RigInfoError(f"bone {b['name']!r}: parents must come before their children")
        if np.asarray(b["matrix_local"], float).shape != (4, 4):
            raise RigInfoError(f"bone {b['name']!r}: matrix_local must be 4x4")
        limits = np.asarray(b["limits_deg"], float)
        if limits.shape != (3, 2) or (limits[:, 0] > limits[:, 1]).any():
            raise RigInfoError(f"bone {b['name']!r}: limits_deg must be three [low, high] pairs")
    for k in keypoints:
        if not 0 <= k.get("bone", -1) < len(bones) or k.get("end") not in ("head", "tail"):
            raise RigInfoError(f"keypoint {k.get('name')!r} must name a bone index and 'head' or 'tail'")
        if len(k.get("rest", ())) != 3:
            raise RigInfoError(f"keypoint {k.get('name')!r} needs a rest position")
    names = [k["name"] for k in keypoints]
    if len(set(names)) != len(names):
        raise RigInfoError("keypoint names must be unique")
    if bones[0]["head_joint"] not in names:
        raise RigInfoError("the root bone's head must be a keypoint")


def mirror_name(name: str) -> str:
    if name.endswith("_L"):
        return name[:-2] + "_R"
    if name.endswith("_R"):
        return name[:-2] + "_L"
    return name


def limb_pairs(info: Mapping[str, Any]) -> dict[str, list[tuple[int, int]]]:
    """(left, right) keypoint index pairs, grouped by limb.

    A limb is every `_L` bone hanging, directly or not, from an `_L` bone whose parent
    is not `_L` (a leg off the pelvis, a wing off the chest). Its joints are the heads
    and tails of those bones that have a `_R` twin among the keypoints. AnyMove knew
    only the wolf's leg names here, and crashed on any other creature.
    """
    names = [k["name"] for k in info["keypoints"]]
    index = {n: i for i, n in enumerate(names)}
    bones = info["bones"]
    roots = [i for i, b in enumerate(bones)
             if b["name"].endswith("_L") and (b["parent"] < 0 or not bones[b["parent"]]["name"].endswith("_L"))]
    out: dict[str, list[tuple[int, int]]] = {}
    for r in roots:
        members = set()
        for i in range(len(bones)):
            j = i
            while j >= 0 and j != r:
                j = bones[j]["parent"]
            if j == r:
                members.add(i)
        pairs = []
        for i in sorted(members):
            for joint in (bones[i]["head_joint"], bones[i]["tail_joint"]):
                twin = mirror_name(joint)
                if joint.endswith("_L") and joint in index and twin in index:
                    pair = (index[joint], index[twin])
                    if pair not in pairs:
                        pairs.append(pair)
        if pairs:
            out[bones[r]["name"][:-2]] = pairs
    return out


def ground_joints(info: Mapping[str, Any]) -> list[str]:
    """The feet: in each limb that reaches the floor, its lowest left/right pair. A limb
    reaches the floor when its lowest pair sits within 5% of the creature's height (at
    least 3 cm) of the lowest joint. What the feet-agreement score watches by default.

    Taking whole pairs keeps both sides even on a rig that is not quite symmetric at rest
    (a height band alone took 5 of the spider's 8 feet). A rig with no `_L`/`_R` limbs
    falls back to every joint in the band."""
    keypoints = info["keypoints"]
    z = np.array([k["rest"][2] for k in keypoints], float)
    band = max(0.03, 0.05 * float(z.max() - z.min()))
    floor = z.min() + band
    feet: list[str] = []
    for pairs in limb_pairs(info).values():
        low = min(pairs, key=lambda pair: z[list(pair)].mean())
        if z[list(low)].mean() <= floor:
            feet += [keypoints[i]["name"] for i in low]
    return feet or [k["name"] for k, h in zip(keypoints, z) if h <= floor]


class Rig:
    """Blender's forward kinematics for one armature, in torch (float64)."""

    def __init__(self, info: Mapping[str, Any], device: str = "cpu"):
        validate_rig_info(info)
        bones = info["bones"]
        self.names = [b["name"] for b in bones]
        self.parent = [b["parent"] for b in bones]
        B = torch.tensor([b["matrix_local"] for b in bones], dtype=torch.float64, device=device)
        self.B = B
        self.length = torch.tensor([b["length"] for b in bones], dtype=torch.float64, device=device)
        # rest offset of each bone in its parent's frame
        self.offset = torch.stack(
            [B[i] if p < 0 else torch.linalg.inv(B[p]) @ B[i] for i, p in enumerate(self.parent)])
        limits = np.radians(np.array([b["limits_deg"] for b in bones], float))
        self.lo = torch.tensor(limits[..., 0], device=device)
        self.hi = torch.tensor(limits[..., 1], device=device)
        self.keypoints = [(k["bone"], k["end"]) for k in info["keypoints"]]
        self.kp_names = [k["name"] for k in info["keypoints"]]
        self.rest_kp = torch.tensor([k["rest"] for k in info["keypoints"]], dtype=torch.float64, device=device)

    def forward(self, trans: torch.Tensor, root_rot: torch.Tensor, euler: torch.Tensor):
        """trans (T, 3) world offset of the root, root_rot (T, 3, 3) its pose basis,
        euler (T, B, 3) Blender XYZ Euler per bone (the root's row is ignored).
        Returns joints (T, J, 3) and bone pose matrices (T, B, 4, 4)."""
        frames, nb = euler.shape[:2]
        rot = euler_xyz(euler)
        poses: list[torch.Tensor] = [None] * nb  # type: ignore[list-item]
        for i, p in enumerate(self.parent):
            basis = torch.zeros(frames, 4, 4, dtype=euler.dtype, device=euler.device)
            basis[:, :3, :3] = root_rot if p < 0 else rot[:, i]
            basis[:, 3, 3] = 1
            if p < 0:
                m = (self.offset[i].expand(frames, 4, 4) @ basis).clone()
                m[:, :3, 3] += trans
            else:
                m = poses[p] @ self.offset[i] @ basis
            poses[i] = m
        pose = torch.stack(poses, 1)
        joints = []
        for b, end in self.keypoints:
            if end == "head":
                joints.append(pose[:, b, :3, 3])
            else:
                joints.append(pose[:, b, :3, 1] * self.length[b] + pose[:, b, :3, 3])
        return torch.stack(joints, 1), pose


def euler_xyz(a: torch.Tensor) -> torch.Tensor:
    """Blender Euler XYZ (applied X, then Y, then Z) to rotation matrices: Rz @ Ry @ Rx."""
    x, y, z = a.unbind(-1)
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    o, one = torch.zeros_like(x), torch.ones_like(x)
    rx = torch.stack([one, o, o, o, cx, -sx, o, sx, cx], -1).view(*x.shape, 3, 3)
    ry = torch.stack([cy, o, sy, o, one, o, -sy, o, cy], -1).view(*x.shape, 3, 3)
    rz = torch.stack([cz, -sz, o, sz, cz, o, o, o, one], -1).view(*x.shape, 3, 3)
    return rz @ ry @ rx


def axis_angle(v: torch.Tensor) -> torch.Tensor:
    """Axis * angle vectors (..., 3) to rotation matrices (Rodrigues)."""
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, dtype=v.dtype, device=v.device) + th.sin() * K + (1 - th.cos()) * K @ K


def project(joints: torch.Tensor, layout: Layout) -> tuple[torch.Tensor, torch.Tensor]:
    v = layout.h / 2 - (joints[..., 2] - layout.z_center) / layout.m_per_px
    front = torch.stack([layout.w_front / 2 + joints[..., 0] / layout.m_per_px, v], -1)
    side = torch.stack([layout.w_side / 2 - joints[..., 1] / layout.m_per_px, v], -1)
    return front, side


# --- the clip: checks before the fit ------------------------------------------------


def validate_detections(d: Mapping[str, np.ndarray], n_joints: int) -> int:
    missing = [k for k in DETECTION_KEYS if k not in d]
    if missing:
        raise DetectionError(f"detections are missing {', '.join(missing)}")
    frames = len(d["front_uv"])
    if frames < 2:
        raise DetectionError("detections need at least two frames")
    for k, shape in (("front_uv", (frames, n_joints, 2)), ("side_uv", (frames, n_joints, 2)),
                     ("front_conf", (frames, n_joints)), ("side_conf", (frames, n_joints))):
        a = np.asarray(d[k])
        if a.shape != shape:
            raise DetectionError(f"{k} is {a.shape}, the rig needs {shape}: was it read with this rig's reader?")
        if not np.isfinite(a).all():
            raise DetectionError(f"{k} holds NaN or infinite values")
    return frames


def height_features(uv, conf, names):
    """Per-frame heights that do not care about left/right labels: middle joints as they
    are, each left/right pair as its mean height and its height difference."""
    v = uv[..., 1]
    index = {n: i for i, n in enumerate(names)}
    feats, w = [], []
    for i, n in enumerate(names):
        twin = index.get(mirror_name(n))
        if twin is None or twin == i:
            feats.append(v[:, i])
            w.append(conf[:, i])
        elif n.endswith("_L"):
            c = np.minimum(conf[:, i], conf[:, twin])
            feats += [(v[:, i] + v[:, twin]) / 2, np.abs(v[:, i] - v[:, twin])]
            w += [c, c]
    return np.stack(feats, 1), np.stack(w, 1).clip(0, 1)


def align_views(d, names, band=48, step_penalty=1.0):
    """Re-time the front view onto the side view by dynamic time warping on joint heights,
    which both views share (video models often play one half ahead of the other).
    Changes `d` in place; returns how far each side frame's match moved, in frames."""
    F, Fw = height_features(d["front_uv"], d["front_conf"], names)
    S, Sw = height_features(d["side_uv"], d["side_conf"], names)
    T = len(F)
    C = np.full((T, T), np.inf)
    for i in range(T):
        lo, hi = max(0, i - band), min(T, i + band + 1)
        w = np.minimum(Fw[i][None], Sw[lo:hi])
        C[i, lo:hi] = (w * np.abs(F[i][None] - S[lo:hi])).sum(1) / w.sum(1).clip(1e-6)
    D = np.full((T, T), np.inf)
    D[0, 0] = C[0, 0]
    for i in range(T):
        for j in range(max(0, i - band), min(T, i + band + 1)):
            if i == j == 0:
                continue
            best = min(D[i - 1, j - 1] if i and j else np.inf,
                       D[i - 1, j] + step_penalty if i else np.inf,
                       D[i, j - 1] + step_penalty if j else np.inf)
            D[i, j] = C[i, j] + best
    i, j, path = T - 1, T - 1, [(T - 1, T - 1)]
    while i or j:
        cands = [(D[i - 1, j - 1], i - 1, j - 1) if i and j else (np.inf, 0, 0),
                 (D[i - 1, j], i - 1, j) if i else (np.inf, 0, 0),
                 (D[i, j - 1], i, j - 1) if j else (np.inf, 0, 0)]
        _, i, j = min(cands)
        path.append((i, j))
    path = np.array(path[::-1])
    match = np.array([path[path[:, 1] == j, 0].mean() for j in range(T)])  # front frame per side frame
    match = np.convolve(np.pad(match, 3, mode="edge"), np.ones(7) / 7, mode="valid")  # no stair steps
    t = np.arange(T)
    for k in ("front_uv", "front_conf"):
        a = d[k].reshape(T, -1)
        d[k] = np.stack([np.interp(match, t, a[:, c]) for c in range(a.shape[1])], 1).reshape(d[k].shape)
    return match - t


def fix_swaps(d, pairs: Mapping[str, Sequence[tuple[int, int]]], switch_px: float = 8.0, bias_px: float = 1.0):
    """Swap near and far limbs in the side view where the heights say the reader mixed them up.

    For each limb, keeping and swapping the side view's labels each give a height gap to
    the front view. Unlike AnyMove's frame-by-frame threshold, the choice is made over the
    whole clip (a two-state Viterbi path): every change of mind costs `switch_px`, so one
    unclear frame cannot swap a leg. Changes `d` in place; returns frames swapped per limb.
    """
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    swapped: dict[str, int] = {}
    for limb, limb_pairs_ in pairs.items():
        if not limb_pairs_:
            continue
        L, R = (np.array(x) for x in zip(*limb_pairs_))
        keep = (c[:, L] * np.abs(sv[:, L] - fv[:, L]) + c[:, R] * np.abs(sv[:, R] - fv[:, R])).sum(1) / len(L)
        swap = (c[:, L] * np.abs(sv[:, R] - fv[:, L]) + c[:, R] * np.abs(sv[:, L] - fv[:, R])).sum(1) / len(L)
        do = viterbi_two_states(keep, swap + bias_px, switch_px)
        for t in np.where(do)[0]:
            for k in ("side_uv", "side_conf"):
                d[k][t, L], d[k][t, R] = d[k][t, R].copy(), d[k][t, L].copy()
        swapped[limb] = int(do.sum())
    return swapped


def viterbi_two_states(cost0: np.ndarray, cost1: np.ndarray, switch: float) -> np.ndarray:
    """The cheapest 0/1 path through per-frame costs, paying `switch` per change; starts in 0."""
    T = len(cost0)
    acc = np.array([cost0[0], cost1[0] + switch])
    back = np.zeros((T, 2), int)
    for t in range(1, T):
        stay, move = acc, acc[::-1] + switch
        back[t] = np.where(stay <= move, [0, 1], [1, 0])
        acc = np.minimum(stay, move) + np.array([cost0[t], cost1[t]])
    state = int(acc.argmin())
    path = np.empty(T, bool)
    for t in range(T - 1, -1, -1):
        path[t] = bool(state)
        state = back[t, state]
    return path


def paw_agreement(d, names, feet, still_px: float = 0.5):
    """Correlation of each foot's height between the two views over the clip: +1 when both
    halves show the same footfalls, near 0 or negative when they animate differently.

    A foot that hardly moves in either view (its height wanders less than `still_px`) has
    no footfalls to compare: its correlation would be the reader's noise, so it is left
    out. AnyMove used 0.5 px; a dragon's planted hind feet scored -0.75 at that, while its
    front feet agreed at 0.97, so `assess` scales it with the creature's height."""
    out: dict[str, float | None] = {}
    for k in feet:
        if k in names:
            j = names.index(k)
            a, b = d["front_uv"][:, j, 1], d["side_uv"][:, j, 1]
            moving = a.std() > still_px and b.std() > still_px
            out[k] = float(np.corrcoef(a, b)[0, 1]) if moving else None
    vals = [v for v in out.values() if v is not None]
    out["score"] = float(np.mean(vals)) if vals else None
    return out


def sync_report(d, max_lag: int = 12):
    """Confidence-weighted height gap between the halves (layout px), and the constant lag
    (frames) that minimises it."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    err = (c * np.abs(fv - sv)).sum(1) / c.sum(1).clip(1e-6)
    per_joint = (c * np.abs(fv - sv)).sum(0) / c.sum(0).clip(1e-6)
    T = len(fv)
    best = None
    for lag in range(-max_lag, max_lag + 1):
        sl_f = slice(max(0, lag), T + min(0, lag))
        sl_s = slice(max(0, -lag), T + min(0, -lag))
        if sl_f.stop - sl_f.start < 2:
            continue
        w = np.minimum(c[sl_f], c[sl_s])
        e = float((w * np.abs(fv[sl_f] - sv[sl_s])).sum() / max(w.sum(), 1e-6))
        if best is None or e < best[1]:
            best = (lag, e)
    return {"mean_height_gap_px": float(err.mean()), "median_joint_gap_px": float(np.median(per_joint)),
            "worst_frame_gap_px": float(err.max()),
            "per_frame_gap_px": err.round(2).tolist(), "best_lag_frames": best[0], "gap_at_best_lag_px": best[1]}


def creature_height_px(info: Mapping[str, Any], layout: Layout) -> float:
    z = np.array([k["rest"][2] for k in info["keypoints"]], float)
    return float(z.max() - z.min()) / layout.m_per_px


def verdict(sync: Mapping[str, Any], paws: Mapping[str, Any], height_px: float) -> dict[str, Any]:
    """Whether a clip is worth fitting. Two halves showing different motions cannot make
    one true motion: AnyMove's eight-legged walks (feet agreement 0.36-0.49) never fit,
    its good captures scored 0.73 and up.

    The height gap is the typical joint's (the median over joints), not the mean: a leg
    hidden behind another in one view is guessed there, and one such leg pulled a clip
    that fitted well to 4.1% on the mean (MiniMax H3 run locally on AnyMove's dragon:
    median 2.8%; the same roar from fal's H3 Max: median 1.2%, mean 1.6%)."""
    gap_px = sync.get("median_joint_gap_px", sync["mean_height_gap_px"])
    gap = gap_px / max(height_px, 1e-6)
    score = paws.get("score")
    reasons = []
    if gap > 0.04:
        reasons.append(f"the views disagree by {gap:.1%} of the creature's height for a typical joint")
    if score is not None and score < 0.5:
        reasons.append(f"the feet move differently in the two views (agreement {score:.2f})")
    level = "bad" if reasons else "doubtful" if gap > 0.025 or (score is not None and score < 0.7) else "good"
    return {"level": level, "height_gap_fraction": round(gap, 4), "feet_agreement": score, "reasons": reasons}


def follow_report(per_joint_px: Mapping[str, float], height_px: float) -> dict[str, Any]:
    """How closely the fitted rig follows the clip: each joint's distance from its readings,
    as a share of the creature's height, for the typical joint and the worst tenth.

    The two views can agree and still show a body no skeleton can make: MiniMax H3 drew a
    rearing wolf's hips-to-head 1.7 to 1.9 times its length in every take. Measured on the
    typical joint: AnyMove's dragon roar from fal 1.3%, the same roar filmed here 2.3%, the
    stretched wolf 3.9%."""
    v = np.array(list(per_joint_px.values()), float)
    if not len(v):
        return {"level": "good", "typical": 0.0, "worst_tenth": 0.0, "reasons": []}
    typical = float(np.median(v)) / max(height_px, 1e-6)
    worst = float(np.percentile(v, 90)) / max(height_px, 1e-6)
    level = "good" if typical <= 0.025 else "doubtful" if typical <= 0.045 else "bad"
    loose = (f"the rig follows it loosely: a typical joint stays {typical:.1%} of its height away "
             "(the video may stretch or squash the body)")
    reasons = [] if level == "good" else [loose]
    return {"level": level, "typical": round(typical, 4), "worst_tenth": round(worst, 4), "reasons": reasons}


def combined_verdict(check: Mapping[str, Any], follow: Mapping[str, Any]) -> dict[str, Any]:
    """One verdict for a take: the worse of the two views' agreement and the rig's follow."""
    rank = {"good": 0, "doubtful": 1, "bad": 2}
    out = dict(check)
    if rank[follow["level"]] > rank[check["level"]]:
        out["level"] = follow["level"]
    out["reasons"] = [*check.get("reasons", []), *follow.get("reasons", [])]
    out["follow"] = follow["typical"]
    return out


# --- the fit ------------------------------------------------------------------------------


def triangulate(d, layout: Layout, front_height_weight: float = 0.25) -> np.ndarray:
    """Joints in metres: x from the front view, y from the side view, height mostly from the side."""
    fx = (d["front_uv"][..., 0] - layout.w_front / 2) * layout.m_per_px
    sy = -(d["side_uv"][..., 0] - layout.w_side / 2) * layout.m_per_px
    fz = layout.z_center + (layout.h / 2 - d["front_uv"][..., 1]) * layout.m_per_px
    sz = layout.z_center + (layout.h / 2 - d["side_uv"][..., 1]) * layout.m_per_px
    w1 = front_height_weight * np.asarray(d["front_conf"]).clip(1e-3)
    w2 = np.asarray(d["side_conf"]).clip(1e-3)
    return np.stack([fx, sy, (fz * w1 + sz * w2) / (w1 + w2)], -1)


def root_from_triangulation(tri, conf, rig: Rig, info) -> np.ndarray:
    """Per-frame root rotation (axis * angle of the root bone's pose basis) from the points
    that ride rigidly on it, by weighted Kabsch against the rest pose, smoothed over time.

    Converted through quaternions kept on one hemisphere, so a creature turning past half
    a turn keeps a continuous track (AnyMove's axis extraction collapsed there)."""
    names = rig.kp_names
    bones = info["bones"]
    ride = {bones[0]["head_joint"], bones[0]["tail_joint"]} | {b["head_joint"] for b in bones if b["parent"] == 0}
    body = sorted(names.index(n) for n in ride if n in names)
    frames = len(tri)
    if len(body) < 3:
        return np.zeros((frames, 3))  # not enough points to turn the body: start at rest
    h = names.index(bones[0]["head_joint"])
    rest = rig.rest_kp.cpu().numpy()
    a = rest[body] - rest[h]
    brot = rig.B[0, :3, :3].cpu().numpy()
    rots = []
    for t in range(frames):
        b = tri[t, body] - tri[t, h]
        w = conf[t, body]
        U, _, Vt = np.linalg.svd((a * w[:, None]).T @ b)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        rots.append(brot.T @ (Vt.T @ D @ U.T) @ brot)
    aa = quaternions_to_axis_angle(continuous_quaternions(matrices_to_quaternions(np.array(rots))))
    k = np.ones(5) / 5
    return np.stack([np.convolve(np.pad(aa[:, i], 2, mode="edge"), k, mode="valid") for i in range(3)], 1)


def rot_between(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def ik_init(tri, root_aa, info, rig: Rig) -> np.ndarray:
    """Point every bone at its triangulated child joint, down the hierarchy, untwisted and
    inside its limits: a start close to the answer, so the fit need not find a rear from a stand."""
    names = rig.kp_names
    index = {n: i for i, n in enumerate(names)}
    bones = info["bones"]
    off = rig.offset.cpu().numpy()
    brot = rig.B[0, :3, :3].cpu().numpy()
    lo, hi = rig.lo.cpu().numpy(), rig.hi.cpu().numpy()
    frames, nb = len(tri), len(bones)
    eul = np.zeros((frames, nb, 3))
    y = np.array([0.0, 1.0, 0.0])
    root_rot = axis_angle(torch.tensor(root_aa, dtype=torch.float64)).numpy()
    for t in range(frames):
        world = [None] * nb
        world[0] = brot @ root_rot[t]
        for i in range(1, nb):
            base = world[bones[i]["parent"]] @ off[i][:3, :3]
            head, tail = index.get(bones[i]["head_joint"]), index.get(bones[i]["tail_joint"])
            if head is None or tail is None:
                world[i] = base  # e.g. fur bones, whose root is not a tracked joint: start at rest
                continue
            d = tri[t, tail] - tri[t, head]
            if np.linalg.norm(d) < 1e-6:
                world[i] = base
                continue
            e = np.clip(matrix_to_euler_xyz(rot_between(y, base.T @ (d / np.linalg.norm(d)))), lo[i], hi[i])
            eul[t, i] = e
            world[i] = base @ euler_xyz_matrix(e)
    return eul


def pick_device(name: str = "cpu") -> str:
    """The CPU unless CUDA is asked for. The fit's tensors are small, so the CPU is faster
    (18 s against 50 s for the dragon's roar on an RTX 5070 Ti) and gives the same motion
    on every machine. `auto` takes CUDA when present. Apple's MPS has no float64."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was asked for but torch sees no CUDA device")
    return name


def fit(d, info, layout: Layout, iters: int = 1500, device: str = "cpu",
        front_height_weight: float = 0.25, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    rig = Rig(info, device)
    names = rig.kp_names
    validate_detections(d, len(names))
    dt = {"dtype": torch.float64, "device": device}
    f_obs = torch.tensor(np.asarray(d["front_uv"]), **dt)
    s_obs = torch.tensor(np.asarray(d["side_uv"]), **dt)
    f_c = torch.tensor(np.asarray(d["front_conf"]), **dt).clamp(0, 1) ** 2
    s_c = torch.tensor(np.asarray(d["side_conf"]), **dt).clamp(0, 1) ** 2
    tri_np = triangulate(d, layout, front_height_weight)
    tri = torch.tensor(tri_np, **dt)
    hips = names.index(info["bones"][0]["head_joint"])
    trans = (tri[:, hips] - rig.rest_kp[hips]).clone().requires_grad_(True)
    conf = np.minimum(d["front_conf"], d["side_conf"]).clip(1e-3, 1)
    root0 = root_from_triangulation(tri_np, conf, rig, info) + 1e-4
    root = torch.tensor(root0, **dt).requires_grad_(True)
    # Euler = lo + (hi - lo) * sigmoid(phi), starting from the analytic IK pose. A locked
    # axis (lo == hi) has nothing to squash: AnyMove divided by its zero span and every
    # angle of every bone became NaN.
    span = rig.hi - rig.lo
    e0 = torch.tensor(ik_init(tri_np, root0, info, rig), **dt)
    z = torch.where(span > 1e-9, (e0 - rig.lo) / span.clamp(min=1e-9), torch.full_like(e0, 0.5))
    phi = torch.logit(z.clamp(1e-3, 1 - 1e-3)).clone().requires_grad_(True)
    sigma = 6.0  # px, robust scale (annealed from 40 px so far-off joints still pull early on)
    # The side view sees legs swing and the hips clearly; the front view foreshortens leg
    # lifts and video models animate it less faithfully. So heights come mostly from the
    # side view and the front view mainly supplies left-right position.
    f_axis = torch.tensor([1.0, front_height_weight], **dt)
    px2 = layout.m_per_px ** 2
    floor = torch.clamp(rig.rest_kp[:, 2], max=0.0)

    def loss_fn():
        eul = rig.lo + span * torch.sigmoid(phi)
        J, _ = rig.forward(trans, axis_angle(root), eul)
        pf, ps = project(J, layout)
        rf = ((pf - f_obs) ** 2 * f_axis).sum(-1)
        rs = ((ps - s_obs) ** 2).sum(-1)
        data = (f_c * torch.log1p(rf / sigma ** 2)).sum() + (s_c * torch.log1p(rs / sigma ** 2)).sum()
        acc = ((J[2:] - 2 * J[1:-1] + J[:-2]) ** 2).sum() / px2 * 0.02
        vel = ((eul[1:] - eul[:-1]) ** 2).sum() * 2.0 + ((root[1:] - root[:-1]) ** 2).sum() * 2.0
        prior = (eul ** 2).sum() * 0.05
        # nothing goes under the floor (AnyMove applied this to the wolf's paw names only); a
        # joint that is already under it at rest (a SkinTokens toe tip pokes out of the boot)
        # keeps its own depth as its floor, or the fit would bend the foot to lift it
        ground = (torch.relu(floor - J[..., 2]) ** 2).sum() / px2 * 0.5
        total = data + acc + vel + prior + ground
        parts = {"data": data.item(), "acc": acc.item(), "vel": vel.item(), "prior": prior.item(), "ground": ground.item()}
        return total, parts, J, pf, ps, eul

    opt = torch.optim.Adam([{"params": [trans], "lr": 0.01}, {"params": [root], "lr": 0.03},
                            {"params": [phi], "lr": 0.05}])
    for it in range(iters):
        sigma = 6.0 + 34.0 * max(0.0, 1 - it / (0.6 * iters))
        if it == int(iters * 0.7):
            for g in opt.param_groups:
                g["lr"] *= 0.3
        total, parts, *_ = loss_fn()
        opt.zero_grad()
        total.backward()
        opt.step()
        if log and it % 250 == 0:
            log(f"iter {it} loss {total.item():.1f} " + str({k: round(v, 1) for k, v in parts.items()}))
    sigma = 6.0
    with torch.no_grad():
        total, parts, J, pf, ps, eul = loss_fn()
        err_f = (pf - f_obs).norm(dim=-1)
        err_s = (ps - s_obs).norm(dim=-1)
        reproj = {"front": float(err_f.mean()), "front_x_only": float((pf - f_obs)[..., 0].abs().mean()),
                  "side": float(err_s.mean())}
        per_joint = ((err_f + err_s) / 2).mean(0).cpu().numpy()
    if not all(np.isfinite(v) for v in reproj.values()):
        raise FloatingPointError("the fit diverged (NaN): check the rig's limits and the detections")
    return {"trans": trans.detach().cpu().numpy(), "root_rotmat": axis_angle(root).detach().cpu().numpy(),
            "euler": eul.detach().cpu().numpy(), "joints": J.cpu().numpy(), "reproj_px": reproj,
            "reproj_px_per_joint": {n: round(float(e), 2) for n, e in zip(names, per_joint)},
            "bones": rig.names, "joint_names": names, "loss": parts}


def assess(d, info, *, feet: Sequence[str] | None = None, align: bool = True, swaps: bool = False,
           log: Callable[[str], None] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Get a read clip ready for the fit and say whether it is worth fitting: re-time the
    front view when that closes the height gap, fix side-view swaps if asked, then judge
    the clip as it will be fitted. Returns the prepared detections and the report.

    The verdict is on the re-timed clip: judged as generated, two halves merely played out
    of step looked hopeless, though re-timing is exactly what fixes them. Both are kept."""
    layout = Layout.from_rig_info(info)
    validate_rig_info(info)
    names = [k["name"] for k in info["keypoints"]]
    validate_detections(d, len(names))
    d = {k: np.array(d[k], np.float64) for k in DETECTION_KEYS}
    feet = list(feet) if feet else ground_joints(info)
    height = creature_height_px(info, layout)
    say = log or (lambda _msg: None)
    before = sync_report(d)
    paws_before = paw_agreement(d, names, feet, still_px=max(0.5, 0.015 * height))
    say(f"sync as generated: mean height gap {before['mean_height_gap_px']:.1f} px, "
        f"worst {before['worst_frame_gap_px']:.1f} px, best constant lag {before['best_lag_frames']} frames")
    shift = None
    if align:
        trial = {k: v.copy() for k, v in d.items()}
        s = align_views(trial, names)
        gain = 1 - sync_report(trial)["mean_height_gap_px"] / max(before["mean_height_gap_px"], 1e-6)
        if gain > 0.15:  # only re-time when the halves really were out of step
            d.update(trial)
            shift = s
            say(f"re-timed the front view onto the side view: mean shift {np.abs(s).mean():.1f} frames "
                f"(height gap down {gain:.0%})")
        else:
            say(f"views already in step (re-timing would cut the height gap by {gain:.0%}); left as generated")
    # painted limbs make the reader's left/right labels trustworthy, so swaps are only fixed on request
    swapped = fix_swaps(d, limb_pairs(info)) if swaps else {}
    sync = sync_report(d)
    paws = paw_agreement(d, names, feet, still_px=max(0.5, 0.015 * height))
    before = {k: v for k, v in before.items() if k != "per_frame_gap_px"}
    sync.update(before=before, paw_agreement=paws, paw_agreement_before=paws_before,
                front_shift_frames=None if shift is None else shift.round(2).tolist(), swaps=swapped,
                feet=feet, verdict=verdict(sync, paws, height),
                verdict_as_generated=verdict(before, paws_before, height))
    return d, sync


def capture(d, info, fps: float, *, iters: int = 1500, device: str = "cpu", align: bool = True,
            swaps: bool = False, feet: Sequence[str] | None = None,
            log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The whole fit stage: `assess` the clip, then fit the rig to it.
    Returns the motion as `motion.json` holds it (see `motion.py`)."""
    if not np.isfinite(fps) or fps <= 0:
        raise DetectionError(f"the clip's frame rate must be positive, got {fps!r}")
    d, sync = assess(d, info, feet=feet, align=align, swaps=swaps, log=log)
    layout = Layout.from_rig_info(info)
    res = fit(d, info, layout, iters=iters, device=pick_device(device), log=log)
    # one hemisphere from frame to frame: a sign flip at half a turn made Blender's
    # interpolation swing through the rest pose between two keys
    root_quat = continuous_quaternions(matrices_to_quaternions(res["root_rotmat"]))
    return {"fps": float(fps), "bones": res["bones"], "trans": res["trans"].round(5).tolist(),
            "root_quat": root_quat.round(6).tolist(),
            "euler": res["euler"].round(5).tolist(), "joints": res["joints"].round(4).tolist(),
            "joint_names": res["joint_names"], "reproj_px": res["reproj_px"],
            "reproj_px_per_joint": res["reproj_px_per_joint"], "sync": sync, "swaps": sync["swaps"],
            "follow": follow_report(res["reproj_px_per_joint"], creature_height_px(info, layout)),
            "feet": sync["feet"], "layout": layout.to_dict()}
