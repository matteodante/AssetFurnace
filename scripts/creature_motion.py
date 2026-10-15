#!/usr/bin/env python3
"""Test clips for a creature family, as pure functions: a walk or trot and an attack.

They exist to prove a rig, not to ship as game animation: does each leg bend the right
way, does the body ride over planted feet, does the skin hold up. So they are procedural
and work on any skeleton `image_to_3dlab.creature_family` has given roles to, with every
distance a fraction of the creature's own leg length.

* **Legs** are IK: each foot gets a target on the ground (planted during stance, lifted
  and carried forward during swing, from `quadruped_gait.foot_offset`) and the chain is
  solved with FABRIK, started from the rest pose every frame so the knees keep bending
  the way the rig was built. The foot segment (ankle or hock to toe) keeps its rest angle,
  which is what makes a digitigrade leg stay on its toes.
* **Body** motion is FK: small turns at the hips, spine, neck, arms and tail, phased
  against the feet (`quadruped_gait.follow_through`), keyed through the attack timing of
  `attack_pose`.

A frame is `(turn, root)`: a world-space rotation per bone, applied at that bone's joint
on top of its parent's, plus where the root's head goes. `frame_rotations` turns that into
the absolute rotations `kimodo_retarget.pose_basis` keys in Blender.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from attack_pose import _blend  # the anticipate / rear / strike / settle timing
from kimodo_retarget import rotation_between
from quadruped_gait import GAITS, follow_through, foot_offset

FPS = 30
CLIPS = {"digitigrade": ("walk", "attack"), "humanoid": ("walk", "attack"),
         "quadruped": ("trot", "attack"), "multileg": ("walk", "attack"),
         "serpent": ("slither", "attack")}
# humanoids already have the preset moves; they get test clips only when asked for
DEFAULT_CLIPS = {"digitigrade", "quadruped", "multileg", "serpent"}

# Biped walk, as fractions of leg length (hip to toe, vertically).
WALK = {"frames": 32, "duty": 0.6, "stride": 0.42, "lift": 0.12, "bob": 0.018, "sway": 0.012,
        "crouch": {"humanoid": 0.03, "digitigrade": 0.07}, "hip_yaw": 5.0, "arm_swing": 16.0,
        "arm_drop": 50.0}
TROT = {"frames": GAITS["trot"]["frames"], "stride": 0.32, "lift_per_stride": 0.3, "bob": 0.02,
        "crouch": 0.03, "head_nod": 4.0, "tail_swing": 6.0, "tail_lag": 0.09}
ATTACK_FRAMES = 30
# Many legs: two alternating groups (a tripod for six, a tetrapod for eight), as fractions
# of each leg's own length along its bones -- a spider's legs reach out further than down.
CRAWL = {"frames": 24, "duty": 0.6, "stride": 0.16, "lift_per_stride": 0.6, "ripple": 0.04, "bob": 0.01}
# Serpents: a wave travelling from head to tail, growing towards the tail. Degrees per joint.
SLITHER = {"frames": 40, "bend": 9.0, "waves": 1.2}
STRIKE = {"neck": 0.35, "coil": 14.0, "rear": 12.0}


@dataclass
class Rig:
    """The skeleton in armature space: parents, parent-first order and rest heads."""
    parents: dict[str, str | None]
    order: list[str]
    heads: dict[str, np.ndarray]

    @property
    def kids(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {b: [] for b in self.parents}
        for b, p in self.parents.items():
            if p is not None:
                out[p].append(b)
        return out


def axis_angle(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    th = math.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k


def accumulate(rig: Rig, turn: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each bone's total world-space turn: its parent's, then its own on top."""
    total = {}
    for b in rig.order:
        p = rig.parents[b]
        own = turn.get(b, np.eye(3))
        total[b] = own if p is None else own @ total[p]
    return total


def posed_heads(rig: Rig, total: dict[str, np.ndarray], root: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for b in rig.order:
        p = rig.parents[b]
        out[b] = np.asarray(root, float) if p is None else out[p] + total[p] @ (rig.heads[b] - rig.heads[p])
    return out


def frame_rotations(rig: Rig, rest_rot: dict[str, np.ndarray], turn, root):
    """(absolute world rotation per bone, root head) for `kimodo_retarget.pose_basis`."""
    total = accumulate(rig, turn)
    return {b: total[b] @ rest_rot[b] for b in rig.order}, np.asarray(root, float)


def fabrik(points: list[np.ndarray], target: np.ndarray, iterations: int = 80,
           bend: np.ndarray | None = None) -> list[np.ndarray]:
    """Joint positions reaching `target` with the first joint fixed and lengths kept.

    A dead straight chain gives FABRIK no side to fold to, so its middle joints are first
    nudged towards `bend` (a knee's forward), or to any side when none is given.
    """
    pts = [np.asarray(p, float).copy() for p in points]
    lengths = [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])]
    base = pts[0].copy()
    line = _unit(pts[-1] - base)
    if len(pts) > 2 and max(float(np.linalg.norm(np.cross(p - base, line))) for p in pts[1:-1]) < 1e-4 * sum(lengths):
        side = np.asarray(bend, float) if bend is not None else np.cross(line, [1.0, 0.0, 0.0])
        if np.linalg.norm(np.cross(side, line)) < 1e-9:
            side = np.cross(line, [0.0, 1.0, 0.0])
        side = _unit(side - (side @ line) * line)
        for i in range(1, len(pts) - 1):
            pts[i] = pts[i] + side * 0.05 * sum(lengths)
    if float(np.linalg.norm(target - base)) >= sum(lengths):
        direction = (target - base) / max(float(np.linalg.norm(target - base)), 1e-12)
        for i, length in enumerate(lengths):
            pts[i + 1] = pts[i] + direction * length
        return pts
    for _ in range(iterations):
        pts[-1] = np.asarray(target, float).copy()
        for i in range(len(pts) - 2, -1, -1):
            pts[i] = pts[i + 1] + _unit(pts[i] - pts[i + 1]) * lengths[i]
        pts[0] = base.copy()
        for i in range(len(pts) - 1):
            pts[i + 1] = pts[i] + _unit(pts[i + 1] - pts[i]) * lengths[i]
        if float(np.linalg.norm(pts[-1] - target)) < 1e-8:
            break
    return pts


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def solve_leg(rig: Rig, leg: dict, turn: dict[str, np.ndarray], root: np.ndarray,
              toe: np.ndarray, foot_turn: np.ndarray | None = None, bend: np.ndarray | None = None) -> None:
    """Add the turns that put this leg's last joint on `toe`, in place.

    The body's turns must already be in `turn`: the hip moves with them. Up to the ankle
    the chain is solved by FABRIK; the foot (ankle to toe) keeps its rest direction in the
    world, whatever the hips do, turned by `foot_turn` if given.
    """
    bones = leg["bones"]
    total = accumulate(rig, turn)
    heads = posed_heads(rig, total, root)
    parent = rig.parents[bones[0]]
    inherited = total[parent] if parent is not None else np.eye(3)
    start = heads[bones[0]]
    rest_points = [start + inherited @ (rig.heads[b] - rig.heads[bones[0]]) for b in bones]
    ankle = leg.get("ankle")
    k = bones.index(ankle) if ankle in bones else len(bones) - 1
    pitch = foot_turn if foot_turn is not None else np.eye(3)
    foot = [pitch @ (rig.heads[b] - rig.heads[bones[k]]) for b in bones[k + 1:]]
    goal = toe - foot[-1] if foot else np.asarray(toe, float)
    solved = fabrik(rest_points[:k + 1], goal, bend=bend) + [goal + v for v in foot]
    current = inherited
    for i, b in enumerate(bones[:-1]):
        want = solved[i + 1] - solved[i]
        have = current @ (rig.heads[bones[i + 1]] - rig.heads[b])
        own = rotation_between(have, want) if np.linalg.norm(have) > 1e-9 else np.eye(3)
        turn[b] = own  # legs are wholly IK: whatever was there is replaced
        current = own @ current


def reach_drop(rig: Rig, legs: list[dict], turn: dict[str, np.ndarray], root: np.ndarray,
               toes: list[np.ndarray], up: np.ndarray, slack: float = 0.97) -> float:
    """How far to lower the body so every foot can reach its target without locking the
    knee straight. A straight leg cannot keep a foot planted, it drags it."""
    total = accumulate(rig, turn)
    heads = posed_heads(rig, total, root)
    drop = 0.0
    for leg, toe in zip(legs, toes):
        bones = leg["bones"]
        ankle = leg.get("ankle")
        k = bones.index(ankle) if ankle in bones else len(bones) - 1
        goal = toe - (rig.heads[bones[-1]] - rig.heads[bones[k]])
        reach = slack * sum(float(np.linalg.norm(rig.heads[b] - rig.heads[a]))
                            for a, b in zip(bones[:k], bones[1:k + 1]))
        gap = heads[bones[0]] - goal
        vertical = float(gap @ up)
        across = float(np.linalg.norm(gap - vertical * up))
        if across >= reach:
            continue  # out of reach sideways: lowering cannot help
        drop = max(drop, vertical - math.sqrt(reach * reach - across * across))
    return drop


def plant(rig: Rig, legs: list[dict], turn: dict[str, np.ndarray], root: np.ndarray, toes: list[np.ndarray],
          up: np.ndarray, pitches: list[np.ndarray | None] | None = None,
          bend: np.ndarray | None = None) -> np.ndarray:
    """Lower the body as far as the feet need, then solve every leg (knees folding towards
    `bend` if a leg is dead straight). Returns the root."""
    root = root - up * reach_drop(rig, legs, turn, root, toes, up)
    for leg, toe, pitch in zip(legs, toes, pitches or [None] * len(legs)):
        solve_leg(rig, leg, turn, root, toe, pitch, bend)
    return root


def leg_length(rig: Rig, leg: dict, up: np.ndarray) -> float:
    return float((rig.heads[leg["bones"][0]] - rig.heads[leg["bones"][-1]]) @ up)


def _frame(family: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = family["frame"]
    return np.asarray(f["up"], float), np.asarray(f["forward"], float), np.asarray(f["left"], float)


def _arm_upper(arm: dict) -> str:
    """The upper arm: past a collar bone when the chain is long enough to have one."""
    return arm["bones"][1] if len(arm["bones"]) >= 4 else arm["bones"][0]


def _lowered_arms(rig: Rig, family: dict, degrees: float) -> dict[str, np.ndarray]:
    """Turns that bring arms held out (a T or A pose) down by up to `degrees`."""
    up, forward, left = _frame(family)
    out = {}
    for arm in family.get("arms", []):
        upper = _arm_upper(arm)
        i = arm["bones"].index(upper)
        tip = arm["bones"][min(i + 2, len(arm["bones"]) - 1)]
        direction = _unit(rig.heads[tip] - rig.heads[upper])
        below = math.degrees(math.asin(max(-1.0, min(1.0, -float(direction @ up)))))
        drop = max(0.0, min(degrees, 75.0 - below))
        side = 1.0 if arm["side"] == "L" else -1.0
        out[upper] = axis_angle(forward, -side * drop)
    return out


def biped_walk(rig: Rig, family: dict, frames: int | None = None) -> list[tuple[dict, np.ndarray]]:
    """A walk in place: feet alternate, hips bob and turn, arms swing against the legs."""
    legs = sorted(family["legs"], key=lambda leg: leg["side"])
    if len(legs) != 2:
        raise ValueError(f"a biped walk needs two legs, found {len(legs)}")
    up, forward, left = _frame(family)
    frames = frames or WALK["frames"]
    length = float(np.mean([leg_length(rig, leg, up) for leg in legs]))
    stride, lift = WALK["stride"] * length, WALK["lift"] * length
    crouch = WALK["crouch"].get(family["family"], WALK["crouch"]["humanoid"]) * length
    root = rig.order[0]
    arms = _lowered_arms(rig, family, WALK["arm_drop"])
    spine = [b for b in family.get("spine", []) if b != root]
    out = []
    for index in range(frames + 1):
        u = (index % frames) / frames
        # lowest just after each foot lands (u = 0 and 0.5), highest over the standing leg
        bob = -WALK["bob"] * length * math.cos(4 * math.pi * (u - 0.05))
        sway = WALK["sway"] * length * math.sin(2 * math.pi * u)  # towards the standing leg
        root_pos = rig.heads[root] + up * (bob - crouch) + left * sway
        # the left foot is furthest forward at u = 0, and the left hip with it
        yaw = -WALK["hip_yaw"] * math.cos(2 * math.pi * u)
        turn = {root: axis_angle(up, yaw)}
        if spine:  # the chest turns back against the hips
            turn[spine[-1]] = axis_angle(up, -yaw * 1.6)
        for bone, lower in arms.items():
            arm = next(a for a in family["arms"] if _arm_upper(a) == bone)
            sign = 1.0 if arm["side"] == "L" else -1.0
            swing = WALK["arm_swing"] * sign * math.cos(2 * math.pi * u)  # + swings it back
            turn[bone] = axis_angle(left, swing) @ lower
        toes, pitches = [], []
        for leg, phase in zip(legs, (0.0, 0.5)):
            fore_aft, rise = foot_offset(u, phase, WALK["duty"], stride, lift)
            toes.append(rig.heads[leg["bones"][-1]] + forward * fore_aft + up * rise)
            pitches.append(None if rise == 0.0 else axis_angle(left, 18.0 * math.sin(math.pi * min(1.0, rise / lift))))
        root_pos = plant(rig, legs, turn, root_pos, toes, up, pitches, bend=forward)
        out.append((turn, root_pos))
    return out


def biped_attack(rig: Rig, family: dict, frames: int = ATTACK_FRAMES) -> list[tuple[dict, np.ndarray]]:
    """A two-handed swipe: lean back and raise the arms, slash down and forward, recover."""
    legs = family["legs"]
    up, forward, left = _frame(family)
    length = float(np.mean([leg_length(rig, leg, up) for leg in legs])) if legs else 1.0
    root = rig.order[0]
    spine = [b for b in family.get("spine", []) if b != root]
    arms = _lowered_arms(rig, family, WALK["arm_drop"])
    out = []
    for index in range(frames + 1):
        t = index / frames
        lean = _blend(t, 0.0, -6.0, -14.0, 18.0, 4.0)        # degrees, + leans forward
        raise_ = _blend(t, 0.0, 25.0, 130.0, 20.0, 5.0)      # arm lift, + raises forward/up
        sink = _blend(t, 0.0, 0.03, 0.05, 0.08, 0.02) * length
        root_pos = rig.heads[root] - up * sink
        turn = {}
        for bone in spine[-2:]:
            turn[bone] = axis_angle(left, lean / 2)
        for bone, lower in arms.items():
            turn[bone] = axis_angle(left, -raise_) @ lower
        root_pos = plant(rig, legs, turn, root_pos, [rig.heads[leg["bones"][-1]] for leg in legs], up, bend=forward)
        out.append((turn, root_pos))
    return out


def _trot_slot(leg: dict) -> str:
    return ("F" if leg["end"] == "front" else "B") + leg["side"]


def quadruped_trot(rig: Rig, family: dict, frames: int | None = None) -> list[tuple[dict, np.ndarray]]:
    """A diagonal trot in place: feet from `quadruped_gait`, a level head, a trailing tail."""
    legs = family["legs"]
    slots = {_trot_slot(leg): leg for leg in legs}
    if sorted(slots) != ["BL", "BR", "FL", "FR"]:
        raise ValueError(f"a trot needs four legs front and back, left and right; found {sorted(slots)}")
    up, forward, left = _frame(family)
    frames = frames or TROT["frames"]
    gait = GAITS["trot"]
    root = rig.order[0]
    neck, tail = family.get("neck", []), family.get("tail", [])
    lengths = {k: leg_length(rig, leg, up) for k, leg in slots.items()}
    mean = float(np.mean(list(lengths.values())))
    out = []
    for index in range(frames + 1):
        u = (index % frames) / frames
        bob = TROT["bob"] * mean * 0.5 * math.sin(2 * math.pi * gait["bob_cycles"] * u)
        root_pos = rig.heads[root] + up * (bob - TROT["crouch"] * mean)
        turn = {}
        for i, bone in enumerate(neck[:2]):
            turn[bone] = axis_angle(left, follow_through(u, gait["bob_cycles"], TROT["head_nod"] / 2, 0.05 * (i + 1)))
        for i, bone in enumerate(tail):
            swing = follow_through(u, 1, TROT["tail_swing"] * 0.7 ** i, TROT["tail_lag"] * i)
            turn[bone] = axis_angle(up, swing)
        toes = []
        for slot, leg in slots.items():
            stride = TROT["stride"] * lengths[slot]
            fore_aft, rise = foot_offset(u, gait["phases"][slot], gait["duty"], stride,
                                         stride * TROT["lift_per_stride"])
            toes.append(rig.heads[leg["bones"][-1]] + forward * fore_aft + up * rise)
        root_pos = plant(rig, list(slots.values()), turn, root_pos, toes, up, bend=forward)
        out.append((turn, root_pos))
    return out


def quadruped_attack(rig: Rig, family: dict, frames: int = ATTACK_FRAMES) -> list[tuple[dict, np.ndarray]]:
    """Rear and slam: the chest lifts the forefeet off the ground, then drives them down."""
    legs = family["legs"]
    up, forward, left = _frame(family)
    root = rig.order[0]
    spine, neck = [b for b in family.get("spine", []) if b != root], family.get("neck", [])
    mean = float(np.mean([leg_length(rig, leg, up) for leg in legs]))
    out = []
    for index in range(frames + 1):
        t = index / frames
        rear = _blend(t, 0.0, -4.0, -16.0, 6.0, 2.0)          # degrees, - lifts the chest
        lift = _blend(t, 0.0, 0.02, 0.35, 0.0, 0.0) * mean     # forefeet off the ground
        reach = _blend(t, 0.0, -0.02, 0.05, 0.18, 0.05) * mean
        nod = _blend(t, 0.0, 4.0, -12.0, 16.0, 4.0)
        sink = _blend(t, 0.0, 0.05, 0.08, 0.02, 0.01) * mean
        root_pos = rig.heads[root] - up * sink
        turn = {}
        for bone in spine:
            turn[bone] = axis_angle(left, rear / len(spine))
        for bone in neck[:2]:
            turn[bone] = axis_angle(left, nod / 2)
        toes = [rig.heads[leg["bones"][-1]] + ((up * lift + forward * reach) if leg["end"] == "front" else 0)
                for leg in legs]
        root_pos = plant(rig, [leg for leg in legs if leg["end"] != "front"], turn, root_pos,
                         [t for t, leg in zip(toes, legs) if leg["end"] != "front"], up, bend=forward)
        for leg, toe in zip(legs, toes):
            if leg["end"] == "front":
                solve_leg(rig, leg, turn, root_pos, toe)
        out.append((turn, root_pos))
    return out


def leg_rows(family: dict) -> dict[str, list[int]]:
    """Indices of the legs on each side, front first, by where each foot stands at rest
    (`foot`, added by `_with_feet`), else in the order they were found."""
    forward = np.asarray(family["frame"]["forward"], float)
    rows: dict[str, list[int]] = {"L": [], "R": []}
    for i, leg in enumerate(family["legs"]):
        if leg["side"] in rows:
            rows[leg["side"]].append(i)

    def ahead(i: int) -> float:
        foot = family["legs"][i].get("foot")
        return -float(np.asarray(foot, float) @ forward) if foot is not None else float(i)

    return {side: sorted(row, key=ahead) for side, row in rows.items()}


def _chain_length(rig: Rig, bones: list[str]) -> float:
    return float(sum(np.linalg.norm(rig.heads[b] - rig.heads[a]) for a, b in zip(bones, bones[1:])))


def _with_feet(rig: Rig, family: dict) -> dict:
    """`family` with each leg's rest foot position, which `leg_rows` sorts by."""
    legs = [{**leg, "foot": rig.heads[leg["bones"][-1]].tolist()} for leg in family["legs"]]
    return {**family, "legs": legs}


def twins(rig: Rig, legs: list[dict], near: float = 0.1) -> dict[int, int]:
    """{leg: the leg it doubles}: chains whose feet stand within `near` of a leg length of
    an earlier leg's. The auto-rig sometimes puts two chains inside one mesh leg; stepping
    them apart tears the skin between them, so they must move as one."""
    out: dict[int, int] = {}
    for i, leg in enumerate(legs):
        for j in range(i):
            if j in out:
                continue
            gap = float(np.linalg.norm(rig.heads[leg["bones"][-1]] - rig.heads[legs[j]["bones"][-1]]))
            if gap < near * _chain_length(rig, legs[j]["bones"]):
                out[i] = j
                break
    return out


def multileg_walk(rig: Rig, family: dict, frames: int | None = None) -> list[tuple[dict, np.ndarray]]:
    """Two alternating groups: each leg steps opposite its neighbours and its partner."""
    family = _with_feet(rig, family)
    double = twins(rig, family["legs"])
    rows = {side: [i for i in row if i not in double] for side, row in leg_rows(family).items()}
    if len(rows["L"]) < 2 or len(rows["R"]) < 2:
        raise ValueError(f"a crawl needs pairs of legs, found {len(rows['L'])} left and {len(rows['R'])} right")
    up, forward, _ = _frame(family)
    frames = frames or CRAWL["frames"]
    phase = {}
    for side, offset in (("L", 0), ("R", 1)):
        for rank, i in enumerate(rows[side]):
            phase[i] = 0.5 * ((rank + offset) % 2) + CRAWL["ripple"] * rank
    legs = family["legs"]
    lead = {i: double.get(i, i) for i in range(len(legs))}  # a twin steps with the leg it doubles
    for i in range(len(legs)):
        phase.setdefault(i, phase.get(lead[i], 0.25))  # a leg straight under the body: off-beat
    root = rig.order[0]
    bob = CRAWL["bob"] * float(np.mean([_chain_length(rig, leg["bones"]) for leg in legs]))
    out = []
    for index in range(frames + 1):
        u = (index % frames) / frames
        root_pos = rig.heads[root] + up * bob * math.sin(4 * math.pi * u)
        turn: dict[str, np.ndarray] = {}
        toes = []
        for i, leg in enumerate(legs):
            stride = CRAWL["stride"] * _chain_length(rig, legs[lead[i]]["bones"])
            fore_aft, rise = foot_offset(u, phase[lead[i]], CRAWL["duty"], stride, stride * CRAWL["lift_per_stride"])
            toes.append(rig.heads[leg["bones"][-1]] + forward * fore_aft + up * rise)
        root_pos = plant(rig, legs, turn, root_pos, toes, up, bend=forward)
        out.append((turn, root_pos))
    return out


def multileg_attack(rig: Rig, family: dict, frames: int = ATTACK_FRAMES) -> list[tuple[dict, np.ndarray]]:
    """Rear up on the back legs and lash out with the front pair."""
    family = _with_feet(rig, family)
    double = twins(rig, family["legs"])
    rows = {side: [i for i in row if i not in double] for side, row in leg_rows(family).items()}
    front = {rows["L"][0], rows["R"][0]} if rows["L"] and rows["R"] else set()
    front |= {i for i, j in double.items() if j in front}
    up, forward, left = _frame(family)
    root = rig.order[0]
    legs = family["legs"]
    reach = float(np.mean([_chain_length(rig, legs[i]["bones"]) for i in front])) if front else 0.0
    standing = [i for i in range(len(legs)) if i not in front]
    out = []
    for index in range(frames + 1):
        t = index / frames
        rear = _blend(t, 0.0, -2.0, -7.0, 3.0, 1.0)
        lift = _blend(t, 0.0, 0.04, 0.32, 0.04, 0.0) * reach
        lunge = _blend(t, 0.0, -0.02, 0.05, 0.3, 0.05) * reach
        turn = {root: axis_angle(left, rear)}
        root_pos = plant(rig, [legs[i] for i in standing], turn, rig.heads[root] + up * 0.3 * lift,
                         [rig.heads[legs[i]["bones"][-1]] for i in standing], up)
        for i in front:
            solve_leg(rig, legs[i], turn, root_pos, rig.heads[legs[i]["bones"][-1]] + up * lift + forward * lunge)
        out.append((turn, root_pos))
    return out


def pose_path(rig: Rig, path: list[str], desired: dict[str, np.ndarray], turn: dict[str, np.ndarray]) -> np.ndarray:
    """Turn the bones along `path` so its joints land on `desired`; returns the root's head.

    The path may run with or against the tree (a snake can be rooted at its head, its tail
    or its middle): each bone is turned towards its own child on the path. Bones off the
    path follow their parents.
    """
    on_path = set(path)
    root = rig.order[0]
    if root not in on_path:
        raise ValueError("the skeleton's root is not on the body chain")
    kids = rig.kids
    total: dict[str, np.ndarray] = {}
    actual = {root: np.asarray(desired[root], float)}
    for b in rig.order:
        parent = rig.parents[b]
        inherited = total[parent] if parent is not None else np.eye(3)
        if parent is not None:
            actual[b] = actual[parent] + total[parent] @ (rig.heads[b] - rig.heads[parent])
        children = [c for c in kids[b] if c in on_path] if b in on_path else []
        if children:  # aim from where this joint really is, so no error carries down the chain
            have = [inherited @ (rig.heads[c] - rig.heads[b]) for c in children]
            want = [desired[c] - actual[b] for c in children]
            turn[b] = rotation_between(have[0], want[0]) if len(children) == 1 else _fit(have, want)
        total[b] = turn.get(b, np.eye(3)) @ inherited
    return actual[root]


def _fit(have: list[np.ndarray], want: list[np.ndarray]) -> np.ndarray:
    """The rotation turning the `have` directions closest onto `want` (Kabsch)."""
    u, _, vt = np.linalg.svd(sum(np.outer(w, h) for h, w in zip(have, want)))
    fix = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)) or 1.0)])
    return u @ fix @ vt


def _rigid_at_root(rig: Rig, path: list[str], order: list[str], bends: list[np.ndarray]) -> list[np.ndarray]:
    """No bend where the root sits in the middle of the chain: both of its segments hang
    from the one root bone, so the angle between them cannot change."""
    root = rig.order[0]
    if root in order[1:-1]:
        bends = list(bends)
        bends[order.index(root)] = np.eye(3)
    return bends


def _bent(rig: Rig, path: list[str], anchor: str, bends: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Joint positions along `path`, walked out from the `anchor` end, each segment turned by
    the accumulated `bends` (one rotation per joint). Segment lengths are kept."""
    order = path if anchor == path[0] else path[::-1]
    out = {order[0]: rig.heads[order[0]].copy()}
    total = np.eye(3)
    for bend, a, b in zip(bends, order, order[1:]):
        total = bend @ total
        out[b] = out[a] + total @ (rig.heads[b] - rig.heads[a])
    return out


def _arc(rig: Rig, path: list[str]) -> np.ndarray:
    """Distance along the path from its first joint, as a fraction of its length."""
    steps = [0.0] + [float(np.linalg.norm(rig.heads[b] - rig.heads[a])) for a, b in zip(path, path[1:])]
    s = np.cumsum(steps)
    return s / max(float(s[-1]), 1e-12)


def serpent_slither(rig: Rig, family: dict, frames: int | None = None) -> list[tuple[dict, np.ndarray]]:
    """A side-to-side wave from head to tail, in place, the head held steady."""
    path = family.get("body") or []
    if len(path) < 4:
        raise ValueError("a slither needs a body chain of at least four joints")
    up = np.asarray(family["frame"]["up"], float)
    frames = frames or SLITHER["frames"]
    s = _arc(rig, path)
    out = []
    for index in range(frames + 1):
        u = (index % frames) / frames
        bends = [axis_angle(up, SLITHER["bend"] * s[i] * math.sin(2 * math.pi * (u - SLITHER["waves"] * s[i])))
                 for i in range(len(path) - 1)]
        bends = _rigid_at_root(rig, path, path, bends)
        turn: dict[str, np.ndarray] = {}
        root = pose_path(rig, path, _bent(rig, path, path[0], bends), turn)
        out.append((turn, root))
    return out


def serpent_strike(rig: Rig, family: dict, frames: int = ATTACK_FRAMES) -> list[tuple[dict, np.ndarray]]:
    """Coil into an S and raise the head, then straighten and drive it forward. The tail stays."""
    path = family.get("body") or []
    if len(path) < 4:
        raise ValueError("a strike needs a body chain of at least four joints")
    up, forward, left = _frame(family)
    s = _arc(rig, path)
    out = []
    for index in range(frames + 1):
        t = index / frames
        coil = _blend(t, 0.0, 0.5, 1.0, -0.15, 0.0)
        rear = _blend(t, 0.0, 0.4, 1.0, 0.0, 0.0)  # never below rest: the head must not dive into the ground
        bends = []
        for k in range(len(path) - 1, 0, -1):  # the joint each segment starts from, tail to head
            neck = max(0.0, (STRIKE["neck"] - s[k]) / STRIKE["neck"])  # 1 at the head, 0 past the neck
            lateral = STRIKE["coil"] * coil * math.sin(2 * math.pi * (1.0 - s[k])) * (1.0 - neck)
            bends.append(axis_angle(up, lateral) @ axis_angle(left, -STRIKE["rear"] * rear * neck))
        bends = _rigid_at_root(rig, path, path[::-1], bends)
        turn: dict[str, np.ndarray] = {}
        root = pose_path(rig, path, _bent(rig, path, path[-1], bends), turn)
        out.append((turn, root))
    return out


MOTIONS = {
    ("humanoid", "walk"): biped_walk, ("digitigrade", "walk"): biped_walk,
    ("humanoid", "attack"): biped_attack, ("digitigrade", "attack"): biped_attack,
    ("quadruped", "trot"): quadruped_trot, ("quadruped", "attack"): quadruped_attack,
    ("multileg", "walk"): multileg_walk, ("multileg", "attack"): multileg_attack,
    ("serpent", "slither"): serpent_slither, ("serpent", "attack"): serpent_strike,
}


def clips_for(family: str) -> tuple[str, ...]:
    """The test clips a family gets, or () when there are none yet."""
    return CLIPS.get(family, ())


def clip_name(family: str, motion: str) -> str:
    """File-safe clip name, e.g. `digitigrade-walk`: hyphenated so no preset move id
    (letters, digits, underscores) can ever collide with it."""
    return f"{family}-{motion}"


def make(motion: str, rig: Rig, family: dict) -> list[tuple[dict, np.ndarray]]:
    key = (family["family"], motion)
    if key not in MOTIONS:
        raise ValueError(f"no {motion!r} clip for a {family['family']}")
    return MOTIONS[key](rig, family)

