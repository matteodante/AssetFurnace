"""Joint limits for the two-camera fit, read from the bone names the creature rig gives.

The creature rig names every bone after what it is (`Shin.L`, `Forearm.R`, `HindPaw.L`,
`Neck2`...), so the fit no longer has to guess what a joint can do from its place alone:

* a knee (`Shin`, `Calf`, `LowerLeg`) folds back and an elbow (`Forearm`, `LowerArm`) forward,
  one way only, from where the bone lines up with its parent to 155 degrees, so a creature
  standing on bent legs can still straighten them; a joint already bent more than 12 degrees
  at rest folds further that way whatever its name says (a bird's knee folds forward, and the
  foot below such a knee gets a paw's range). Its twist is held to 26 degrees (a knee) or 80
  (an elbow turning the palm over) and it does not bend sideways (11 degrees);
* a paw or hoof folds its own way, back below an elbow (a wrist) and forward below a knee (a
  hock), and only 20 degrees the other way;
* feet, toes, hands, the spine, the neck, the head, shoulders and thighs get a range each,
  in degrees from the rest pose: bend, twist about the bone, sideways;
* an upper arm that is not a leg the creature stands on swings back at most 75 degrees;
* a jaw only opens, by up to 40 degrees.

A bone whose name says none of this keeps the limits its role gave it (`rigging.plan_bones`).
Everything is in the prepared frame (Z up, facing -Y) and in each bone's own frame as the
two-camera prepare sets it: Y along the bone, Z its roll (`rigging.roll_vector`), X = Y x Z;
limits are XYZ Euler angles in degrees, counted from the rest pose.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from image_to_3dlab.twocam.rigging import roll_vector

KNEE = re.compile(r"(shin|calf|knee|lower_?leg)", re.IGNORECASE)
ELBOW = re.compile(r"(forearm|fore_arm|lower_?arm|elbow)", re.IGNORECASE)
NOT_KNEE = re.compile(r"(thigh|up_?leg)", re.IGNORECASE)
PAW = re.compile(r"(paw|hoof)", re.IGNORECASE)
FOOT = re.compile(r"(foot|ankle)", re.IGNORECASE)
UPPER_ARM = re.compile(r"(^|[^a-z]|left|right|upper)arm([._]|$)", re.IGNORECASE)
JAW = re.compile(r"^jaw", re.IGNORECASE)
# the middle of the body bends up and down: a neck meeting the head a little to one side must
# not turn that small sideways angle into its bend axis
MIDDLE = re.compile(r"^(root|hips|spine|chest|neck|head|tail|jaw)", re.IGNORECASE)
# bend, twist about the bone, sideways: degrees either way from the rest pose
RANGES = [
    (FOOT, (45, 20, 25)),
    (re.compile(r"toe", re.IGNORECASE), (90, 10, 10)),        # toes along the sole curl right under a lifted paw
    (re.compile(r"(hand|wrist)", re.IGNORECASE), (70, 40, 35)),
    (PAW, (100, 20, 25)),                            # a galloping hock turns 90 degrees from rest
    (re.compile(r"(spine|chest|torso|abdomen)", re.IGNORECASE), (35, 30, 30)),
    (re.compile(r"neck", re.IGNORECASE), (45, 50, 35)),
    (re.compile(r"(^|[^a-z])head$", re.IGNORECASE), (60, 50, 35)),   # a howl lifts the head past 45
    (re.compile(r"(shoulder|clavicle|collar)", re.IGNORECASE), (40, 20, 40)),
    (re.compile(r"(up_?leg|thigh)", re.IGNORECASE), (130, 45, 60)),
]
HINGE_FOLD = 155.0        # a knee or an elbow folds this far, degrees from rest
HINGE_PAST = 3.0          # and straightens this far past straight
HINGE_TWIST = {"knee": 26.0, "elbow": 80.0}
HINGE_SIDE = 11.5
REST_BENT = 12.0          # bent more than this at rest: it folds further that way
PLANE = 8.0               # a bone and its parent at more than this define the plane it bends in
PAW_AGAINST = 20.0        # a paw folds this far against its own way
ARM_BACK = 75.0           # an upper arm swings back at most this far
JAW_OPEN = (3.0, 40.0)    # a jaw opens this much, from rest
JAW_OTHER = 4.6
FLOOR_REACH = 0.1         # of the height: a limb whose lowest point is this close to the floor stands on it
FACING = np.array([0.0, -1.0, 0.0])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _frame(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Columns X, Y, Z of a bone's own frame, as the two-camera prepare rolls it."""
    y = _unit(tail - head)
    z = roll_vector(head, tail)
    z = _unit(z - (z @ y) * y)
    return np.column_stack([np.cross(y, z), y, z])


def _turn(axis: np.ndarray, v: np.ndarray, angle: float) -> np.ndarray:
    """`v` turned by `angle` (radians) about the unit `axis`."""
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def bend_axis(frame: np.ndarray, own: np.ndarray, parent: np.ndarray | None) -> int:
    """0 (X) or 2 (Z): the axis a joint bends about. The one nearest the normal of the plane
    the bone and its parent make, when they meet at an angle; in line, X (the sideways axis
    the roll gives a bone in the body's plane)."""
    if parent is not None:
        n = np.cross(parent, own)
        if np.linalg.norm(n) > math.sin(math.radians(PLANE)):
            n = _unit(n)
            return max((0, 2), key=lambda a: abs(float(frame[:, a] @ n)))
    return 0


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


def _tip_shift(frame: np.ndarray, own: np.ndarray, axis: int, angle: float = 0.3) -> np.ndarray:
    return _turn(frame[:, axis], own, angle) - own


def straight_at(frame: np.ndarray, own: np.ndarray, parent: np.ndarray, axis: int) -> float:
    """The turn (degrees) about `axis` at which the bone lines up best with its parent."""
    a = frame[:, axis]
    moves = np.cross(a, _unit(own))                  # where the bone's direction goes for a + turn
    return math.degrees(math.atan2(float(parent @ moves), float(parent @ _unit(own))))


def _fold_sign(frame, own, parent, axis, way: str) -> float:
    """+1 or -1: the turn about `axis` that folds the joint `way` ("back" or "front")."""
    d = float(_tip_shift(frame, own, axis) @ FACING)
    forward = d > 0
    return 1.0 if forward == (way == "front") else -1.0


def _rest_fold(own, parent, frame, axis) -> float | None:
    """+1 or -1 when the rest pose already bends the joint more than REST_BENT: the turn that
    folds it further; None for a joint at rest nearly straight."""
    if _angle(parent, own) < REST_BENT:
        return None
    turned = _turn(frame[:, axis], own, 0.15)
    return 1.0 if _angle(parent, turned) > _angle(parent, own) else -1.0


def swing_back(frame: np.ndarray, own: np.ndarray) -> tuple[int, float]:
    """(channel, sign): the turn, about X or Z, that swings the bone's tip back the most (an
    arm held out in an A-pose swings back about a different axis than it bends)."""
    shifts = {a: float(_tip_shift(frame, own, a) @ -FACING) for a in (0, 2)}
    channel = max(shifts, key=lambda a: abs(shifts[a]))
    return channel, 1.0 if shifts[channel] > 0 else -1.0


def _channels(axis: int) -> tuple[int, int, int]:
    """(bend, twist, side) channels for a bone bending about `axis`."""
    return axis, 1, 2 - axis


def standing_bones(names: Sequence[str], parents: Sequence[int], heads: np.ndarray, tails: np.ndarray) -> set[int]:
    """Bones of the limbs the creature stands on: everything from a limb's top bone down,
    when the lowest point below it is near the floor."""
    pts = np.concatenate([heads, tails])
    floor, height = float(pts[:, 2].min()), float(np.ptp(pts[:, 2])) + 1e-12
    children: list[list[int]] = [[] for _ in names]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    def below(i: int) -> list[int]:
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(children[j])
        return out
    sided = [n.endswith((".L", ".R", "_L", "_R")) for n in names]
    out: set[int] = set()
    for i, p in enumerate(parents):
        if sided[i] and (p < 0 or not sided[p]):          # a limb's top bone
            sub = below(i)
            if min(float(heads[sub, 2].min()), float(tails[sub, 2].min())) <= floor + FLOOR_REACH * height:
                out.update(sub)
    return out


def named_limits(names: Sequence[str], parents: Sequence[int], heads: np.ndarray, tails: np.ndarray,
                 defaults: Sequence[Sequence[Sequence[float]]]) -> list[list[list[float]]]:
    """Each bone's XYZ Euler limits (degrees from rest): from its name where the name says
    what it is, else `defaults[i]` (the limits its role gave it)."""
    heads, tails = np.asarray(heads, float), np.asarray(tails, float)
    out = [[list(map(float, a)) for a in lim] for lim in defaults]
    legs = standing_bones(names, parents, heads, tails)
    kind: dict[int, str] = {}
    for i, name in enumerate(names):
        if KNEE.search(name) and not NOT_KNEE.search(name):
            kind[i] = "knee"
        elif ELBOW.search(name):
            kind[i] = "elbow"
    reversed_knees: set[int] = set()
    for i, name in enumerate(names):
        p = parents[i]
        own = tails[i] - heads[i]
        if np.linalg.norm(own) < 1e-9:
            continue
        frame = _frame(heads[i], tails[i])
        parent = _unit(tails[p] - heads[p]) if p >= 0 and np.linalg.norm(tails[p] - heads[p]) > 1e-9 else None
        axis = 0 if MIDDLE.search(name) else bend_axis(frame, _unit(own), parent)
        bend, twist, side = _channels(axis)
        lim = [[-160.0, 160.0], [-69.0, 69.0], [-160.0, 160.0]]
        if i in kind and parent is not None:
            sign = _fold_sign(frame, own, parent, axis, "back" if kind[i] == "knee" else "front")
            rest = _rest_fold(own, parent, frame, axis)
            if rest is not None:
                if rest != sign and kind[i] == "knee":
                    reversed_knees.add(i)
                sign = rest
            st = straight_at(frame, own, parent, axis)
            lim[bend] = [min(0.0, st) - HINGE_PAST, HINGE_FOLD] if sign > 0 else [-HINGE_FOLD, max(0.0, st) + HINGE_PAST]
            lim[twist] = [-HINGE_TWIST[kind[i]], HINGE_TWIST[kind[i]]]
            lim[side] = [-HINGE_SIDE, HINGE_SIDE]
            out[i] = lim
            continue
        if JAW.search(name):
            sign = 1.0 if float(_tip_shift(frame, own, axis)[2]) < 0 else -1.0   # opening takes the chin down
            lo, hi = JAW_OPEN
            lim[bend] = [lo, hi] if sign > 0 else [-hi, -lo]
            lim[twist] = lim[side] = [-JAW_OTHER, JAW_OTHER]
            out[i] = lim
            continue
        rng = next((r for rx, r in RANGES if rx.search(name)), None)
        if rng is not None and FOOT.search(name) and p in reversed_knees:
            rng = next(r for rx, r in RANGES if rx is PAW)    # a bird's long hock, not a foot
        if rng is None and not (UPPER_ARM.search(name) and i not in legs):
            continue                                          # its role's limits stand
        if rng is not None:
            for channel, degrees in zip((bend, twist, side), rng):
                lim[channel] = [-float(degrees), float(degrees)]
        else:
            lim = [list(map(float, a)) for a in defaults[i]]
        if PAW.search(name) and p >= 0 and parent is not None:
            way = "back" if kind.get(p) == "elbow" else "front" if kind.get(p) == "knee" else None
            if way is not None:
                fold = _fold_sign(frame, own, parent, axis, way)
                if fold > 0:
                    lim[bend][0] = max(lim[bend][0], -PAW_AGAINST)
                else:
                    lim[bend][1] = min(lim[bend][1], PAW_AGAINST)
        if UPPER_ARM.search(name) and i not in legs:
            channel, back = swing_back(frame, own)
            if back > 0:
                lim[channel][1] = min(lim[channel][1], ARM_BACK)
            else:
                lim[channel][0] = max(lim[channel][0], -ARM_BACK)
        out[i] = [[round(a, 2), round(b, 2)] for a, b in lim]
    return out
