"""Make any rigged model ready for two-camera capture: turned, sized, sides named, painted.

AnyMove's rigs are placed by hand, with bones named `upperarm_L`. AssetFurnace's come from
SkinTokens, which names bones `bone_0..N` for any model, and from a picture that may show
the subject at three-quarters. Everything here works from the geometry instead of names,
so it serves a wolf, a knight or a dragon alike:

* the left/right symmetry plane of the mesh gives the turn; the feet, which point forward
  on two legs and on four, say which way is front;
* a bone off the middle is paired with its mirror image, and the pair gets `_L`/`_R`;
* limbs are numbered from the front, painted one colour each (so the video model and the
  reader can tell left from right), and given joint limits by role;
* the two-camera layout is sized to the creature and to room for it to rear up.

numpy only: Blender's Python runs it inside `scripts/blender_twocam_prepare.py`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# AnyMove's limb colours first (front-left pink, front-right blue, ...), then four more.
PALETTE = [("pink", "#F53CD1"), ("blue", "#2F6BFF"), ("yellow", "#FFD21F"), ("cyan", "#19D3E0"),
           ("orange", "#FF7A1A"), ("green", "#2BD94A"), ("purple", "#9B4DFF"), ("red", "#FF2B2B"),
           ("white", "#F2F2F2"), ("black", "#1A1A1A")]
ROOT_LIMITS = [[-30, 30], [-20, 20], [-30, 30]]
SPINE_LIMITS = [[-30, 30], [-15, 15], [-30, 30]]
NECK_LIMITS = [[-60, 60], [-25, 25], [-50, 50]]
TAIL_LIMITS = [[-60, 60], [-20, 20], [-60, 60]]
LIMB_ROOT_LIMITS = [[-80, 80], [-20, 20], [-45, 45]]
LIMB_MIDDLE_LIMITS = [[-110, 110], [-10, 10], [-20, 20]]
LIMB_END_LIMITS = [[-50, 50], [-10, 10], [-20, 20]]


class RiggingError(ValueError):
    pass


@dataclass
class Skeleton:
    """Bones in world space: parents[i] < 0 for a root."""

    names: list[str]
    parents: list[int]
    heads: np.ndarray
    tails: np.ndarray
    children: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.heads = np.asarray(self.heads, float)
        self.tails = np.asarray(self.tails, float)
        n = len(self.names)
        if len(self.parents) != n or self.heads.shape != (n, 3) or self.tails.shape != (n, 3):
            raise RiggingError("names, parents, heads and tails must describe the same bones")
        roots = [i for i, p in enumerate(self.parents) if p < 0]
        if len(roots) != 1:
            raise RiggingError(f"the armature needs exactly one root bone, it has {len(roots)}")
        self.children = [[] for _ in range(n)]
        for i, p in enumerate(self.parents):
            if p >= 0:
                self.children[p].append(i)

    @property
    def root(self) -> int:
        return next(i for i, p in enumerate(self.parents) if p < 0)

    def order(self) -> list[int]:
        """Parents before children (depth first from the root)."""
        out, stack = [], [self.root]
        while stack:
            i = stack.pop()
            out.append(i)
            stack.extend(reversed(self.children[i]))
        return out

    def subtree(self, i: int) -> list[int]:
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(self.children[j])
        return out

    def transformed(self, matrix: np.ndarray) -> Skeleton:
        m = np.asarray(matrix, float)
        apply = lambda p: p @ m[:3, :3].T + m[:3, 3]
        return Skeleton(list(self.names), list(self.parents), apply(self.heads), apply(self.tails))


# --- turn, size and place ------------------------------------------------------------------


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _mirror_error(v: np.ndarray, angle: float) -> float:
    r = v @ _rot_z(angle).T  # centred: a symmetric body's centroid is on its plane
    m = r * [-1, 1, 1]
    return float(np.sqrt(((r[:, None, :] - m[None, :, :]) ** 2).sum(-1).min(1)).mean())


def symmetry_yaw(verts: np.ndarray, samples: int = 900, seed: int = 0) -> float:
    """The turn about Z (radians, in [0, pi)) that puts the mesh's left/right mirror plane
    on X = 0: the angle at which the mesh, mirrored in X, lands closest to itself. Searched
    every 5 degrees, then every half degree around the best."""
    v = np.asarray(verts, float)
    if len(v) > samples:
        v = v[np.random.default_rng(seed).choice(len(v), samples, replace=False)]
    v = v - v.mean(0)
    coarse = min(np.arange(0.0, 180.0, 5.0), key=lambda d: _mirror_error(v, math.radians(d)))
    fine = min(np.arange(coarse - 5.0, coarse + 5.25, 0.5), key=lambda d: _mirror_error(v, math.radians(d)))
    return math.radians(fine % 180.0)


def forward_sign(skeleton: Skeleton) -> float:
    """+1 when the creature faces -Y, -1 when it faces +Y, after its mirror plane is on X = 0.

    Feet point forward on two legs and on four: the lowest leaf bones' tails say which way.
    With no clear feet, the end the head is on: the leaf farthest from the root along Y
    that is also the highest. A creature facing neither way keeps -Y (+1)."""
    z = skeleton.heads[:, 2]
    leaves = [i for i, c in enumerate(skeleton.children) if not c]
    # feet stand off the middle line: a tail tip resting on the floor (a lizard, a T. rex)
    # points backwards and once turned such creatures round
    width = float(np.ptp(np.concatenate([skeleton.heads[:, 0], skeleton.tails[:, 0]]))) + 1e-9
    off_middle = [i for i in leaves if abs(skeleton.heads[i, 0]) > 0.05 * width]
    low = [i for i in off_middle if z[i] <= z.min() + 0.1 * (skeleton.heads[:, 2].max() - z.min() + 1e-9)]
    toes = sum(float(skeleton.tails[i, 1] - skeleton.heads[i, 1]) for i in low)
    span = float(np.ptp(skeleton.heads[:, 1])) + 1e-9
    if abs(toes) > 0.05 * span:
        return 1.0 if toes < 0 else -1.0
    top = max(leaves, key=lambda i: skeleton.tails[i, 2])
    lean = float(skeleton.tails[top, 1] - skeleton.heads[skeleton.root, 1])
    if abs(lean) > 0.1 * span:
        return 1.0 if lean < 0 else -1.0
    return 1.0


def normalising_matrix(verts: np.ndarray, skeleton: Skeleton, length_m: float,
                       yaw: float | None = None, axis: str = "y") -> tuple[np.ndarray, float]:
    """The 4x4 that turns the creature to face -Y, scales it so its size along `axis` is
    `length_m`, centres it on X and Y and stands it on Z = 0. Returns it and the turn used."""
    if length_m <= 0:
        raise RiggingError("the creature's length must be positive")
    v = np.asarray(verts, float)
    if yaw is None:
        yaw = symmetry_yaw(v)
        turned = skeleton.transformed(_affine(_rot_z(yaw)))
        if forward_sign(turned) < 0:
            yaw += math.pi
    rot = _rot_z(yaw)
    r = v @ rot.T
    k = "xyz".index(axis)
    size = float(np.ptp(r[:, k]))
    if size <= 1e-9:
        raise RiggingError(f"the mesh has no size along {axis}")
    scale = length_m / size
    lo, hi = r.min(0), r.max(0)
    offset = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]) * scale
    m = np.eye(4)
    m[:3, :3] = rot * scale
    m[:3, 3] = offset
    return m, float(yaw % (2 * math.pi))


def _affine(r: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    return m


# --- sides and names --------------------------------------------------------------------------


def find_mirrors(skeleton: Skeleton, tolerance: float) -> dict[int, int]:
    """Each off-middle bone and its mirror image (both ways), for a skeleton facing -Y with
    its mirror plane on X = 0.

    Off a middle bone, a child is a limb when its whole subtree sits more than `tolerance`
    to one side (judged on the subtree, not the first joint: SkinTokens may start both legs
    at the middle of the hips); its mirror is the sibling subtree, of the same size or one
    bone more or less (SkinTokens sometimes gives one leg an extra bone), whose mirrored
    centre lands nearest. Below a mirrored pair, children pair up by structure: each with
    the child of the mirror whose head lands nearest its own, mirrored, the same size
    preferred. A bone left over in one limb keeps its limb's side in `plan_bones`."""
    mirrors: dict[int, int] = {}
    heads = skeleton.heads
    size = {i: len(skeleton.subtree(i)) for i in range(len(heads))}
    centre = {i: heads[skeleton.subtree(i)].mean(0) for i in range(len(heads))}
    for i in skeleton.order():
        p = skeleton.parents[i]
        if i in mirrors or p < 0:
            continue
        if p in mirrors:
            free = [c for c in skeleton.children[mirrors[p]] if c not in mirrors]
            pool = [c for c in free if size[c] == size[i]] or [c for c in free if abs(size[c] - size[i]) <= 1]
            if pool:
                target = heads[i] * [-1, 1, 1]
                j = min(pool, key=lambda c: np.linalg.norm(heads[c] - target))
                mirrors[i], mirrors[j] = j, i
            continue
        if abs(centre[i][0]) <= tolerance:
            continue
        pool = [c for c in skeleton.children[p] if c != i and c not in mirrors and abs(size[c] - size[i]) <= 1
                and centre[c][0] * centre[i][0] < 0 and abs(centre[c][0]) > tolerance]
        if not pool:
            continue
        target = centre[i] * [-1, 1, 1]
        j = min(pool, key=lambda c: np.linalg.norm(centre[c] - target))
        if np.linalg.norm(centre[j] - target) <= max(4 * tolerance, 0.35 * abs(2 * centre[i][0])):
            mirrors[i], mirrors[j] = j, i
    return mirrors


@dataclass
class Plan:
    """What `scripts/blender_twocam_prepare.py` does to the armature."""

    names: list[str]                 # new name per bone
    sides: list[str]                 # "L", "R" or "" per bone
    limbs: list[int]                 # limb number per bone (1 = frontmost), 0 in the middle
    tails: np.ndarray                # new tail per bone (child joints joined up)
    limits: list[list[list[float]]]  # Euler limits per bone, degrees
    paint: dict[str, dict]           # group -> {"color", "colour_name", "bones", "label"}
    unpaired: list[str] = field(default_factory=list)  # bones off the middle with no mirror


def plan_bones(skeleton: Skeleton, tolerance: float) -> Plan:
    """Names, sides, joined-up tails, limits and limb paint for a turned, sized skeleton."""
    mirrors = find_mirrors(skeleton, tolerance)
    n = len(skeleton.names)
    sides = [""] * n
    for i, j in mirrors.items():  # facing -Y, left is +X; judged on the whole subtree, since a
        # pair of legs may start at one point
        xi = skeleton.heads[skeleton.subtree(i), 0].mean()
        xj = skeleton.heads[skeleton.subtree(j), 0].mean()
        sides[i] = "L" if (xi, skeleton.heads[i, 0]) > (xj, skeleton.heads[j, 0]) else "R"
    # A bone left over inside a limb (one leg a bone longer) keeps its limb's side.
    for i in skeleton.order():
        p = skeleton.parents[i]
        if not sides[i] and p >= 0 and sides[p]:
            sides[i] = sides[p]
    pair_roots = [i for i in range(n) if sides[i] == "L" and i in mirrors and sides[skeleton.parents[i]] == ""]
    # A limb reaches: an ear or a horn pair is named left and right but not painted or counted,
    # or a wolf with ears became "six limbs" and its legs lost their colours.
    points = np.concatenate([skeleton.heads, skeleton.tails])
    size = float(np.ptp(points, 0).max()) + 1e-9

    def reach(root: int) -> float:
        sub = skeleton.subtree(root)
        ends = np.concatenate([skeleton.heads[sub], skeleton.tails[sub]])
        return float(np.linalg.norm(ends - skeleton.heads[root], axis=1).max())

    limb_roots = [i for i in pair_roots if reach(i) >= 0.2 * size]
    part_roots = [i for i in pair_roots if i not in limb_roots]
    # clearly to one side, like a lone arm: a tail curling a little off the middle (the
    # studio's wolf) is not worth a warning
    lone = {i for i in range(n)
            if skeleton.parents[i] >= 0 and not sides[skeleton.parents[i]] and i not in mirrors
            and abs(skeleton.heads[skeleton.subtree(i), 0].mean()) > 3 * tolerance}
    unpaired = [skeleton.names[i] for i in sorted(lone) if skeleton.parents[i] not in lone]  # where each starts
    at = skeleton.heads[limb_roots] if limb_roots else np.zeros((0, 3))
    # limbs that sit above one another (arms over legs) count from the top, others from the front
    upright = len(limb_roots) > 1 and np.ptp(at[:, 2]) > np.ptp(at[:, 1])
    limb_roots.sort(key=lambda i: (-skeleton.heads[i, 2], skeleton.heads[i, 1]) if upright
                    else (skeleton.heads[i, 1], -skeleton.heads[i, 2]))
    limbs = [0] * n
    group = [0] * n  # limbs 1.., then the other pairs (ears, horns) as negative numbers
    names = list(skeleton.names)
    for k, root in enumerate(limb_roots, 1):
        for start in (root, mirrors[root]):
            for i in skeleton.subtree(start):
                limbs[i] = group[i] = k
    for k, root in enumerate(part_roots, 1):
        for start in (root, mirrors[root]):
            for i in skeleton.subtree(start):
                group[i] = -k
    branches: dict[int, str] = {}  # the letters of every fork on the way down a limb (fingers)
    for i in skeleton.order():
        if sides[i] != "L" or not group[i] or i not in mirrors:
            continue
        k = group[i]
        depth = 1
        p = skeleton.parents[i]
        while p >= 0 and sides[p]:
            depth += 1
            p = skeleton.parents[p]
        parent = skeleton.parents[i]
        siblings = [c for c in skeleton.children[parent] if sides[c] == "L" and group[c] == k and c in mirrors]
        letter = "" if len(siblings) <= 1 or depth == 1 else "abcdefghijklmnopqrstuvwxyz"[siblings.index(i) % 26]
        branches[i] = (branches.get(parent, "") if depth > 1 else "") + letter
        stem = f"{'limb' if k > 0 else 'part'}{abs(k)}_{depth}{branches[i]}"
        names[i] = f"{stem}_L"
        names[mirrors[i]] = f"{stem}_R"
    for i in skeleton.order():  # a leftover bone in one limb: its parent's name, marked x
        if sides[i] and i not in mirrors:
            parent_name = names[skeleton.parents[i]]
            names[i] = f"{parent_name[:-2]}x{parent_name[-2:]}" if parent_name.endswith(("_L", "_R")) \
                else f"{skeleton.names[i]}_{sides[i]}"
    if len(set(names)) != n:
        raise RiggingError("two bones would share a name after naming sides")

    tails = skeleton.tails.copy()
    for i in range(n):
        kids = skeleton.children[i]
        same = [c for c in kids if (sides[c] == "") == (sides[i] == "") and limbs[c] == limbs[i]]
        if not same:
            continue
        main = max(same, key=lambda c: len(skeleton.subtree(c)))
        if np.linalg.norm(skeleton.heads[main] - skeleton.heads[i]) > 1e-4:
            tails[i] = skeleton.heads[main]

    role = middle_roles(skeleton, sides, limbs)
    limits = []
    for i in range(n):
        if skeleton.parents[i] < 0:
            limits.append(ROOT_LIMITS)
        elif not sides[i]:
            limits.append({"spine": SPINE_LIMITS, "neck": NECK_LIMITS, "tail": TAIL_LIMITS}[role[i]])
        elif not limbs[i]:  # an ear, a horn: it swings like a neck
            limits.append(NECK_LIMITS)
        elif not sides[skeleton.parents[i]]:
            limits.append(LIMB_ROOT_LIMITS)
        elif not skeleton.children[i]:
            limits.append(LIMB_END_LIMITS)
        else:
            limits.append(LIMB_MIDDLE_LIMITS)

    paint: dict[str, dict] = {}
    count = len(limb_roots)
    for k, root in enumerate(limb_roots, 1):
        for side in ("L", "R"):
            bones = [names[i] for i in skeleton.subtree(root if side == "L" else mirrors[root])]
            colour_name, colour = PALETTE[(2 * (k - 1) + (side == "R")) % len(PALETTE)]
            paint[f"{k}{side}"] = {"color": colour, "colour_name": colour_name, "bones": bones,
                                   "label": limb_label(k, count, side, upright)}
    return Plan(names, sides, limbs, tails, [[list(map(float, a)) for a in lim] for lim in limits], paint,
                unpaired)


def middle_roles(skeleton: Skeleton, sides: Sequence[str], limbs: Sequence[int] | None = None) -> dict[int, str]:
    """"spine", "neck" or "tail" for each bone in the middle, so a neck can lift a head to
    howl while the back stays a back.

    The head is the middle chain's end farthest forward and highest. On the way to it from
    the root, the bones up to the last one a limb hangs from are the spine; past it, and
    whatever hangs off them (a jaw, ears), the neck and head. Any other middle chain (a
    tail, a crest) bends freely. A howling wolf pinned its head at a single middle limit."""
    middle = [i for i in range(len(skeleton.names)) if not sides[i]]
    if not middle:
        return {}
    ends = [i for i in middle if not any(not sides[c] for c in skeleton.children[i])]
    head = max(ends, key=lambda i: skeleton.tails[i, 2] - skeleton.tails[i, 1])
    path = [head]
    while skeleton.parents[path[-1]] >= 0:
        path.append(skeleton.parents[path[-1]])
    path.reverse()
    limbs = limbs if limbs is not None else [1 if x else 0 for x in sides]
    hangs = [k for k, i in enumerate(path) if any(limbs[c] for c in skeleton.children[i])]
    split = hangs[-1] if hangs else 0
    roles = {i: "spine" for i in path[1:split + 1]}
    for i in path[split + 1:]:
        for j in skeleton.subtree(i):
            if not sides[j]:
                roles.setdefault(j, "neck")
    for i in middle:
        if skeleton.parents[i] >= 0:
            roles.setdefault(i, "tail")
    return roles


def limb_label(k: int, count: int, side: str, upright: bool = False) -> str:
    """How the video prompt names limb `k` of `count`: "left arm" and "left leg" when two
    pairs sit one above the other, "front-left leg" and "hind-left leg" when one is ahead
    of the other, "second left limb" past two pairs."""
    s = "left" if side == "L" else "right"
    if count == 1:
        return f"{s} limb"
    if count == 2:
        if upright:
            return f"{s} {'arm' if k == 1 else 'leg'}"
        return f"{'front' if k == 1 else 'hind'}-{s} leg"
    ordinal = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
    return f"{ordinal[k - 1] if k <= len(ordinal) else f'{k}th'} {s} limb"


def colour_code(paint: dict[str, dict]) -> str:
    """The limb colours in words: "front-left limb pink, front-right limb blue, ..."."""
    return ", ".join(f"{g['label']} {g['colour_name']}" for g in paint.values())


def roll_vector(head: Sequence[float], tail: Sequence[float]) -> np.ndarray:
    """The direction a bone's local Z should point (AnyMove's `build_rig.py`): across the
    bone and horizontal-ish, so local X is world X for bones in the body's plane and
    bending a knee or a neck is a turn about X."""
    d = np.asarray(tail, float) - np.asarray(head, float)
    d /= max(np.linalg.norm(d), 1e-12)
    z = np.cross([1.0, 0, 0], d)
    if np.linalg.norm(z) < 0.5:  # runs mostly along X
        xl = np.cross([0, 0, 1.0], d)
        z = np.cross(xl / np.linalg.norm(xl), d) if np.linalg.norm(xl) > 1e-4 else np.array([0, 0, 1.0])
    return z / np.linalg.norm(z)


# --- keypoints and layout ------------------------------------------------------------------


def keypoints(names: Sequence[str], parents: Sequence[int], heads: np.ndarray, tails: np.ndarray,
              children: Sequence[Sequence[int]]) -> tuple[list[dict], list[tuple[str, str]]]:
    """The joints the reader looks for: every bone's head (named after the bone) and the tip
    of every bone with no children. Returns the keypoints and each bone's (head, tail) joint."""
    kps, ends = [], []
    for i, name in enumerate(names):
        kps.append({"name": name, "bone": i, "end": "head", "rest": [round(float(x), 5) for x in heads[i]]})
    for i, name in enumerate(names):
        joined = [c for c in children[i] if np.linalg.norm(heads[c] - tails[i]) < 1e-4]
        if joined:
            ends.append((name, names[joined[0]]))
            continue
        side = name[-2:] if name.endswith(("_L", "_R")) else ""
        tip = f"{name[:-2] if side else name}_tip{side}"
        kps.append({"name": tip, "bone": i, "end": "tail", "rest": [round(float(x), 5) for x in tails[i]]})
        ends.append((name, tip))
    return kps, ends


def layout_for(verts: np.ndarray, height_room: float = 1.75, side_room: float = 1.45) -> dict[str, float]:
    """A two-camera frame (1280 x 720, ground on row 630) sized to the creature: room above
    it to rear up, room along it to lunge, and the front view wide enough for a 30 degree
    turn. The margins are guesses in the spirit of AnyMove's hand-picked frames (its wolf:
    420 + 860 px at 2.6 mm a pixel); widen `height_room` for creatures that leap."""
    v = np.asarray(verts, float)
    width, length, height = (float(x) for x in np.ptp(v, 0))
    need_front = max(1.6 * width, width + 0.4 * length)
    need_side = side_room * length
    need_up = height_room * height
    w_front = int(round(1280 * need_front / (need_front + need_side) / 2) * 2)
    w_front = min(max(w_front, 320), 960)
    w_side = 1280 - w_front
    m = max(need_up / 630, need_front / w_front, need_side / w_side)
    return {"H": 720, "W_FRONT": w_front, "W_SIDE": w_side, "M_PER_PX": m, "GROUND_V": 630}
