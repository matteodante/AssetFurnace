"""What a skeleton's ends are, and the names its bones get from that body plan.

The ends of the joint tree say what the creature is: ends on the floor are feet, the
middle end furthest forward and up is the head, a middle end far behind is a tail, and the
other ends come in mirrored pairs (arms, wings). The spine runs from the tail (or the
pelvis) to the head; every limb hangs from it where its own path meets it. Two pairs of
legs make a quadruped, one a biped; an upper pair is a wing when the body is flat across
it, an arm when it is round.

Joints keep their drawn places; a name goes to the joint nearest to where a body of that
kind has it (a hind knee about a third of the way down the leg).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .shell import FORWARD, REACH_HIGH, Shell
from .skeleton import (
    LIMB_KINDS,
    Bone,
    Bones,
    Tree,
    arc,
    centre_line,
    place,
    resample,
    side_of,
    smooth,
    stem_of,
)

MIDDLE_DRAWN = 0.12     # an end this close to the middle plane (x height) is on it, for drawn trees
MIDDLE_CENTRE = 0.06    # the same for centre lines, which sit exactly in the middle
FOOT = 0.08             # ends this close to the floor are feet
TAIL_ON_FLOOR = 0.25    # a middle "foot" this far behind the feet is a tail lying on the floor
TAIL_SIDE = 0.20        # a tail end lies this close to the middle ...
TAIL_BEHIND = 0.15      # ... and this far behind the mean end
NEAR_HEAD = 0.15        # ends this close to the head are ears and horns
HEAD_PART = 0.25        # so is a pair this short (x height) ...
HEAD_REACH = 0.35       # ... leaving the spine this close to the head end (a T-rex's arms leave it further back)
GROUP = 0.15            # ends this close together are one hand or foot
MERGE_OUT = 0.12        # limbs sharing the node this far out from the spine are one limb
PAIR = 0.35             # twin ends lie this close, ignoring left and right
PROP_SIDE = 0.10        # an unpaired end this far off the middle is something held: no bones
WING = 5.0              # flatter than this across: a wing (a wing measures about 10, an arm 1-2.5)
SKIP = 0.25             # cost of a planned joint the drawn limb has no joint for
MIN_BONE = 0.01         # no bone shorter than this (x height): one cell

# kind: (optional top bone, bones along the limb, where its next joints sit as shares of
# its length from its root joint)
PLANS = {
    "hind": (None, ("Thigh", "Shin", "HindPaw", "HindToes"), (0.32, 0.66, 0.88)),
    "front": ("Scapula", ("UpperArm", "Forearm", "FrontPaw", "FrontToes"), (0.30, 0.66, 0.88)),
    "biped": (None, ("Thigh", "Shin", "Foot", "Toes"), (0.45, 0.82, 0.92)),
    "arm": ("Shoulder", ("Arm", "LowerArm", "Hand"), (0.42, 0.80)),
    "wing": ("WingShoulder", ("WingUpper", "WingForearm", "WingHand"), (0.30, 0.62)),
}
FOLDS = ("hind", "front", "biped", "arm")      # fingers and toes past the first fork fold into the end
TRUNK_KINDS = ("root", "spine", "neck", "head", "tail", "jaw")


@dataclass
class Limb:
    end: int
    path: list[int]          # the spine node first, out to the end
    at: int                  # where it meets the spine: an index into Body.spine
    side: int                # +1 left (+X), -1 right
    foot: bool
    length: float


@dataclass
class Body:
    spine: list[int]
    head: int
    tail: int | None
    feet: list[int]
    leg_pairs: list[tuple[Limb, Limb]]        # back to front
    upper_pairs: list[tuple[Limb, Limb]]      # longest first
    upper_kinds: list[str]                    # "wing" or "arm", one per upper pair
    flatness: list[float]
    singles: list[Limb]
    props: list[Limb]
    kind: str                                 # quadruped, biped, centaur or winged
    notes: list[str] = field(default_factory=list)


def _forward(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, float) @ FORWARD


# --- the wing test -----------------------------------------------------------------------


def flatness(shell: Shell, limb: np.ndarray, trunk: np.ndarray, cuts=(0.30, 0.45, 0.60, 0.75)) -> float:
    """How flat the body is across a limb: the cross-section's width over its thickness
    (sqrt of the two largest spreads), the median of four cuts along the limb, counting
    only what lies nearer the limb than the trunk. Shape, not pose: a man in a T-pose
    spreads his arms as wide as a bat its wings, but they are round."""
    limb = np.asarray(limb, float)
    s = arc(limb)
    if len(limb) < 2 or s[-1] <= 0:
        return 1.0
    kd_limb, kd_trunk = cKDTree(limb), cKDTree(np.asarray(trunk, float))
    out = []
    for t in cuts:
        i = min(int(np.searchsorted(s, t * s[-1])), len(limb) - 1)
        p = limb[i]
        axis = limb[min(i + 3, len(limb) - 1)] - limb[max(i - 3, 0)]
        norm = np.linalg.norm(axis)
        if norm <= 0:
            continue
        axis = axis / norm
        v = shell.cells[np.asarray(shell.kd.query_ball_point(p, 0.35 * shell.height), np.int64)]
        v = v[np.abs((v - p) @ axis) < 1.5 * shell.size]
        if len(v):
            v = v[kd_limb.query(v)[0] < kd_trunk.query(v)[0]]
        if len(v) < 5:
            continue
        q = (v - p) - np.outer((v - p) @ axis, axis)
        w = np.sort(np.linalg.eigvalsh(np.cov(q.T)))[::-1]
        out.append(float(np.sqrt(w[0] / max(w[1], 1e-12))))
    return float(np.median(out)) if out else 1.0


def _dense(points: np.ndarray, per: int = 20) -> np.ndarray:
    p = np.asarray(points, float)
    if len(p) < 2:
        return p
    t = np.linspace(0, 1, per, endpoint=False)[None, :, None]
    return np.concatenate([(p[:-1, None] + (p[1:] - p[:-1])[:, None] * t).reshape(-1, 3), p[-1:]])


# --- the analysis ------------------------------------------------------------------------


def _spine(tree: Tree, shell: Shell, head: int, tail: int | None, feet: list[int]) -> list[int]:
    """Tail to head. Without a tail, from the pelvis: where the paths of the rearmost foot
    and its twin meet. A rearmost foot without a twin (its leg not drawn) would meet the
    front legs at the chest and lose the back half, so then the spine starts where that
    foot's path first reaches the middle."""
    p = tree.points
    if tail is not None:
        return tree.path(tail, head)
    h = shell.height
    if feet:
        rear = min(feet, key=lambda e: _forward(p[e]))
        twins = [e for e in feet if e != rear and np.sign(shell.lateral(p[e])) != np.sign(shell.lateral(p[rear]))
                 and np.linalg.norm(p[e, 1:] - p[rear, 1:]) < PAIR * h]
        path = tree.path(rear, head)
        if twins:
            other = set(tree.path(min(twins, key=lambda e: np.linalg.norm(p[e, 1:] - p[rear, 1:])), head))
            common = [n for n in path if n in other]
            if len(common) > 2:
                return common
        for k, n in enumerate(path):
            if abs(shell.lateral(p[n])) <= MIDDLE_CENTRE * h:
                return path[k:]
        return path
    middle = [n for n in range(tree.n) if n != head and abs(shell.lateral(p[n])) <= MIDDLE_CENTRE * h]
    if not middle:
        return [head]
    return tree.path(min(middle, key=lambda n: p[n, 2]), head)


def _tail_on_floor(tree: Tree, shell: Shell, end: int, feet: list[int]) -> bool:
    """A foot far behind the others that is a tail lying on the floor: it has no twin among
    the feet (narrow hind legs stand 0.1 h off the middle, but in pairs), and it lies on the
    middle or its branch runs along the floor more than it rises to the body (a dragon's
    tail curled 0.11 h to one side; a leg climbs to its hip)."""
    h, p = shell.height, tree.points
    if any(np.linalg.norm(p[o] - shell.mirror(p[end])) < GROUP * h for o in feet if o != end):
        return False
    if abs(shell.lateral(p[end])) < 0.1 * h:
        return True
    others = [o for o in feet if o != end]
    path = tree.path(end, others[0]) if others else [end]
    joint = next((n for n in path if abs(shell.lateral(p[n])) <= MIDDLE_CENTRE * h), path[-1])
    run = p[joint] - p[end]
    return float(np.linalg.norm(run[:2])) > float(run[2])


def analyse(tree: Tree, shell: Shell, middle: float = MIDDLE_DRAWN, wing_test: bool = True) -> Body:
    """The body plan of a joint tree on its shell."""
    h = shell.height
    p = tree.points
    lat = shell.lateral
    ends = tree.ends()
    if len(ends) < 2:
        raise ValueError("a skeleton needs at least two ends")
    # feet from the lowest end (legs drawn only to the ankles still stand), but never from
    # higher than a bone needs to reach a floor contact: a hovering robed figure's hands are no feet
    floor = min(min(p[e, 2] for e in ends), shell.floor + REACH_HIGH * h)
    feet = [e for e in ends if p[e, 2] - floor < FOOT * h]
    if len(feet) > 2:
        back = float(np.mean([_forward(p[e]) for e in feet]))
        feet = [e for e in feet if not (_forward(p[e]) < back - TAIL_ON_FLOOR * h and _tail_on_floor(tree, shell, e, feet))]
    rest = [e for e in ends if e not in feet]
    on_middle = [e for e in rest if abs(lat(p[e])) < middle * h]
    if on_middle:
        head = max(on_middle, key=lambda e: _forward(p[e]) + 0.25 * p[e, 2])
    else:   # a head sunk between the shoulders: the highest middle point of the tree
        mids = [n for n in range(tree.n) if abs(lat(p[n])) < MIDDLE_CENTRE * h and n not in feet] or rest or ends
        head = max(mids, key=lambda n: p[n, 2] + 0.25 * _forward(p[n]))
    mean_end = float(np.mean([_forward(p[e]) for e in ends]))
    tails = [e for e in rest if e != head and abs(lat(p[e])) < TAIL_SIDE * h
             and _forward(p[e]) < mean_end - TAIL_BEHIND * h]
    tail = min(tails, key=lambda e: _forward(p[e])) if tails else None
    others = [e for e in rest if e not in (head, tail) and np.linalg.norm(p[e] - p[head]) > NEAR_HEAD * h]
    centre = p.mean(0)
    groups: list[list[int]] = []
    for e in sorted(others, key=lambda e: -np.linalg.norm(p[e] - centre)):
        for g in groups:   # one side only: two horns 0.14 h apart are no hand
            if np.linalg.norm(p[e] - p[g[0]]) < GROUP * h and np.sign(lat(p[e])) == np.sign(lat(p[g[0]])):
                g.append(e)
                break
        else:
            groups.append([e])
    others = [g[0] for g in groups]
    spine = _spine(tree, shell, head, tail, feet)
    on_spine = {n: i for i, n in enumerate(spine)}

    limbs = []
    for e in feet + others:
        if e in on_spine:
            continue
        path = tree.path(e, spine[0])
        k = next(k for k, n in enumerate(path) if n in on_spine)
        out = path[:k + 1][::-1]
        if len(out) < 3:
            continue
        limbs.append(Limb(e, out, on_spine[path[k]], 1 if lat(p[e]) > 0 else -1, e in feet, tree.path_length(out)))

    # one limb per stretch of path leaving the trunk: the toes of one foot, the claws of one hand
    merged: dict[int, Limb] = {}
    for limb in limbs:
        key, run = limb.path[-1], 0.0
        for a, b in zip(limb.path, limb.path[1:]):
            run += float(np.linalg.norm(p[b] - p[a]))
            if run >= MERGE_OUT * h:
                key = b
                break
        kept = merged.get(key)
        if kept is None or limb.length > kept.length:
            if kept is not None:
                limb.foot = limb.foot or kept.foot
            merged[key] = limb
        else:
            kept.foot = kept.foot or limb.foot
    limbs = list(merged.values())

    # pairs: opposite sides, same kind, ends close once left and right are ignored; the
    # closest pairs first, so a far match cannot take a near one's twin
    flat2 = lambda q: np.asarray(q, float)[1:]
    candidates = sorted((float(np.linalg.norm(flat2(p[a.end]) - flat2(p[b.end]))), i, j)
                        for i, a in enumerate(limbs) for j, b in enumerate(limbs)
                        if i < j and a.side != b.side and a.foot == b.foot)
    used, pairs = set(), []
    for d, i, j in candidates:
        if d < PAIR * h and i not in used and j not in used:
            used |= {i, j}
            pairs.append(tuple(sorted((limbs[i], limbs[j]), key=lambda limb: -limb.side)))
    singles = [limb for k, limb in enumerate(limbs) if k not in used]
    props = [limb for limb in singles if not limb.foot and abs(lat(p[limb.end])) > PROP_SIDE * h]
    singles = [limb for limb in singles if limb not in props]
    leg_pairs = sorted([q for q in pairs if q[0].foot], key=lambda q: q[0].at + q[1].at)
    upper_pairs = sorted([q for q in pairs if not q[0].foot], key=lambda q: -(q[0].length + q[1].length))
    # horns and ears: a short pair leaving the spine at the head is no pair of arms (a horned
    # dragon's centre lines made it a centaur)
    s_spine = arc(p[spine])
    at_head = lambda limb: s_spine[-1] - s_spine[limb.at] < HEAD_REACH * h and limb.length < HEAD_PART * h
    head_parts = [q for q in upper_pairs if at_head(q[0]) and at_head(q[1])]
    upper_pairs = [q for q in upper_pairs if q not in head_parts]

    trunk = _dense(p[spine])
    flat, kinds = [], []
    for pair in upper_pairs:
        f = float(np.mean([flatness(shell, _dense(p[limb.path[1:]]), trunk) for limb in pair])) if wing_test else 1.0
        flat.append(f)
        kinds.append("wing" if f > WING else "arm")
    if len(leg_pairs) >= 2:
        kind = "centaur" if "arm" in kinds else "quadruped"
    else:
        kind = "winged" if "wing" in kinds else "biped"
    notes = []
    if props:
        notes.append(f"{len(props)} unpaired end(s) off the middle: held things, no bones")
    return Body(spine, head, tail, feet, leg_pairs, upper_pairs, kinds, flat, singles, props, kind, notes)


# --- names -------------------------------------------------------------------------------


def choose(found: list[float], planned: list[float], skip: float = SKIP) -> dict[int, int]:
    """In-order match of planned joint shares to drawn ones, the total gap smallest; a
    planned joint left out costs `skip`, a drawn joint left out nothing.
    Returns {planned index: drawn index}."""
    m, k = len(found), len(planned)
    cost = np.zeros((k + 1, m + 1))
    cost[1:, 0] = skip * np.arange(1, k + 1)
    how = np.zeros((k + 1, m + 1), np.int8)       # 0 skip a plan, 1 take, 2 pass a drawn joint
    for i in range(1, k + 1):
        for j in range(1, m + 1):
            options = (cost[i - 1, j] + skip, cost[i - 1, j - 1] + abs(found[j - 1] - planned[i - 1]), cost[i, j - 1])
            how[i, j] = int(np.argmin(options))
            cost[i, j] = options[how[i, j]]
    out, i, j = {}, k, m
    while i > 0:
        if j == 0 or how[i, j] == 0:
            i -= 1
        elif how[i, j] == 1:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return out


def limb_plan(body: Body) -> list[tuple[str, tuple[Limb, ...], int | None]]:
    """(kind, its limbs, its number or None) for every limb that gets bones. Several pairs
    of one kind are numbered from the front."""
    legs = body.leg_pairs
    leg_kinds = ["hind"] + ["front"] * (len(legs) - 1) if len(legs) >= 2 else ["biped"] * len(legs)
    entries = list(zip(leg_kinds, legs)) + list(zip(body.upper_kinds, body.upper_pairs))
    out = []
    for kind in dict.fromkeys(k for k, _ in entries):
        same = [pair for k, pair in entries if k == kind]
        same.sort(key=lambda pair: -(pair[0].at + pair[1].at))
        for n, pair in enumerate(same):
            out.append((kind, pair, n + 1 if len(same) > 1 else None))
    out += [("chain", (limb,), k + 1) for k, limb in enumerate(body.singles)]
    return out


def _name(stem: str, number: int | None, side: str) -> str:
    return f"{stem}{number or ''}{side}"


def name_bones(tree: Tree, body: Body, shell: Shell) -> tuple[Bones, dict[str, int]]:
    """Named bones on the joints of an analysed tree, and the joint each name is on."""
    h = shell.height
    spine = body.spine
    hips = spine[int(round((body.leg_pairs[0][0].at + body.leg_pairs[0][1].at) / 2))] if body.leg_pairs else spine[0]
    if tree.tip[hips]:
        hips = spine[1] if hips == spine[0] and len(spine) > 1 else spine[-2] if len(spine) > 1 else hips
    t = tree.rerooted(hips)
    par, pos, tip = t.parent, t.points, t.tip
    kids = [list(k) for k in t.children()]
    name: dict[int, str] = {}
    kind: dict[int, str] = {}
    tail: dict[int, np.ndarray] = {}
    base: dict[int, str] = {}
    further: dict[int, list[np.ndarray]] = {}    # where a too-short bone may reach on to
    gone: set[int] = set()

    def live(n: int) -> list[int]:
        return [c for c in kids[n] if c not in gone]

    def descendants(n: int) -> set[int]:
        out, stack = set(), list(kids[n])
        while stack:
            j = stack.pop()
            out.add(j)
            stack.extend(kids[j])
        return out

    # the spine: Hips where the hind legs meet it, Tail behind, Spine, Neck and Head in front
    hk = spine.index(hips)
    front = [n for n in spine[hk + 1:] if not tip[n]]
    back = [n for n in spine[:hk][::-1] if not tip[n]]
    attach = [limb.at for pair in body.leg_pairs + body.upper_pairs[:1] for limb in pair]
    last = max([a for a in attach if a > hk], default=hk)
    name[hips], kind[hips] = "Hips", "spine"
    if back:
        tail[hips] = pos[back[0]]
        further[hips] = [pos[m] for m in spine[hk + 1:]]     # the tail starts on the hips: point along the trunk
    elif hk + 1 < len(spine):
        tail[hips] = pos[spine[hk + 1]]
        further[hips] = [pos[m] for m in spine[hk + 1:]]
    for k, n in enumerate(back):
        name[n], kind[n] = f"Tail{k + 1}", "tail"
        i = spine.index(n)
        tail[n] = pos[spine[i - 1]]
        further[n] = [pos[m] for m in spine[:i][::-1]]
    n_spine = n_neck = 0
    for n in front:
        i = spine.index(n)
        tail[n] = pos[spine[i + 1]] if i + 1 < len(spine) else pos[n]
        further[n] = [pos[m] for m in spine[i + 1:]]
        if n == front[-1]:
            name[n], kind[n] = "Head", "head"
        elif i <= last:
            n_spine += 1
            name[n], kind[n] = f"Spine{n_spine}", "spine"
        else:
            n_neck += 1
            name[n], kind[n] = f"Neck{n_neck}", "neck"
    for n, nm in name.items():
        base[n] = stem_of(nm)

    # held things: their branch goes, back to where it leaves something else
    for limb in body.props:
        n = limb.end
        while par[n] >= 0 and len(kids[par[n]]) == 1 and par[n] not in name:
            n = par[n]
        gone |= {n} | descendants(n)

    for limb_kind, limbs, number in limb_plan(body):
        for limb in limbs:
            side = (".L" if limb.side > 0 else ".R") if limb_kind != "chain" or \
                abs(shell.lateral(pos[limb.end])) > MIDDLE_CENTRE * h else ""
            path = [n for n in limb.path[1:] if n not in gone]
            if not path:
                continue
            if limb_kind == "arm":
                _drop_spurs(path, kids, tip, pos, gone, descendants)
            end = None
            cut = next((k for k, n in enumerate(path) if not tip[n] and len(live(n)) > 1), None)
            if cut is not None:
                path = path[:cut + 1]
                fork = path[-1]
                if limb_kind in FOLDS:
                    far = []
                    for c in live(fork):
                        sub = [c, *descendants(c)]
                        far.append(pos[max(sub, key=lambda m: np.linalg.norm(pos[m] - pos[fork]))])
                    end = np.mean(far, axis=0)
                    gone |= descendants(fork)
            joints = [n for n in path if not tip[n]]
            if not joints:
                continue
            if end is None:
                if tip[path[-1]]:
                    end = pos[path[-1]]
                else:
                    ahead = live(joints[-1])
                    end = np.mean(pos[ahead], axis=0) if ahead else pos[joints[-1]]
            title = limb_kind.capitalize()
            if limb_kind in PLANS:
                top, names, shares = PLANS[limb_kind]
                use_top = top is not None and len(joints) >= len(names) + 1
                chain = joints[1:] if use_top else joints
                s = arc(np.vstack([pos[chain], end]))
                found = list(s[1:-1] / max(s[-1], 1e-12))
                pick = {k: k for k in range(len(shares))} if len(chain) == len(names) else choose(found, list(shares))
                starts = {0: names[0], **{m + 1: names[k + 1] for k, m in pick.items()}}
                if use_top:
                    name[joints[0]] = _name(top, number, side)
                    base[joints[0]] = top
                for m, n in enumerate(chain):
                    stem = starts.get(m, f"{title}{number or ''}Seg{m}")
                    name[n] = _name(stem, number if m in starts else None, side)
                    base[n] = stem
            else:
                for m, n in enumerate(joints):
                    stem = f"Chain{number}Seg{m + 1}"
                    name[n], base[n] = stem + side, stem
            for k, n in enumerate(joints):
                kind[n] = limb_kind
                tail[n] = pos[joints[k + 1]] if k + 1 < len(joints) else end
                further[n] = [pos[m] for m in joints[k + 1:]] + [end]

    # everything else hangs from a named bone and is numbered after it: HeadX1, WingHandX2.L
    order = [n for n in t.order() if n not in gone]
    count: dict[tuple[str, str], int] = {}
    for n in order:
        if tip[n] or n in name:
            continue
        a = par[n]
        while a not in name:
            a = par[a]
        sfx = name[a][-2:] if side_of(name[a]) else ""
        stem = base[a]
        count[stem, sfx] = count.get((stem, sfx), 0) + 1
        name[n], base[n] = f"{stem}X{count[stem, sfx]}{sfx}", stem
        kind[n] = kind[a] if kind[a] in LIMB_KINDS else "head" if kind[a] in ("head", "jaw") else "part"
    for n in order:
        if tip[n] or n in tail:
            continue
        ahead = [c for c in kids[n] if c not in gone]
        if ahead:
            tail[n] = np.mean(pos[ahead], axis=0)
        else:
            before = pos[par[n]] if par[n] >= 0 else pos[n] - [0.0, 0.0, 0.05 * h]
            tail[n] = pos[n] + (pos[n] - before) * 0.3
    # no bone shorter than a cell: drawn joints often sit on each other (a pelvis on the
    # root, the first tail joint on the hips), and Blender drops a bone of no length
    for n in order:
        if tip[n] or np.linalg.norm(tail[n] - pos[n]) >= MIN_BONE * h:
            continue
        on = next((q for q in further.get(n, []) if np.linalg.norm(q - pos[n]) >= MIN_BONE * h), None)
        if on is None:
            way = pos[n] - pos[par[n]] if par[n] >= 0 else np.array([0.0, 0.0, 1.0])
            way = way / np.linalg.norm(way) if np.linalg.norm(way) > 1e-9 else np.array([0.0, 0.0, 1.0])
            on = pos[n] + way * 0.05 * h
        tail[n] = on

    floor = shell.floor
    root_head = np.array([pos[hips, 0], pos[hips, 1], floor])
    bones = [Bone("Root", None, root_head, root_head + [0.0, 0.0, 0.1 * h], "root", "", False)]
    joint_of: dict[str, int] = {}
    taken = {"Root"}
    final: dict[int, str] = {}
    for n in order:
        if tip[n]:
            continue
        nm = name[n]
        if nm in taken:
            k = 2
            while f"{stem_of(nm)}_{k}{nm[-2:] if side_of(nm) else ''}" in taken:
                k += 1
            nm = f"{stem_of(nm)}_{k}{nm[-2:] if side_of(nm) else ''}"
        taken.add(nm)
        final[n] = nm
        parent = "Root" if n == hips else final[par[n]]
        bones.append(Bone(nm, parent, pos[n], tail[n], kind[n], side_of(nm)))
        joint_of[nm] = n
    return bones, joint_of


def _drop_spurs(path, kids, tip, pos, gone, descendants) -> None:
    """A forearm drawn as two bones side by side (radius and ulna): at a fork where the arm
    runs on to a joint of its own, single bones that end nearer than the arm's end go."""
    reach_end = pos[path[-1]]
    for k, n in enumerate(path[:-1]):
        ahead = [c for c in kids[n] if c not in gone]
        if tip[n] or len(ahead) < 2 or path[k + 1] not in ahead:
            continue
        spurs = [c for c in ahead if c != path[k + 1]]
        if tip[path[k + 1]]:
            break
        reach = float(np.linalg.norm(reach_end - pos[n]))

        def lone(c, fork=n, reach=reach):
            sub = [c, *descendants(c)]
            return sum(1 for m in sub if not tip[m]) <= 1 and \
                max(float(np.linalg.norm(pos[m] - pos[fork])) for m in sub) <= 0.85 * reach
        if not all(lone(c) for c in spurs):
            break
        for c in spurs:
            gone |= {c} | descendants(c)


# --- the centre-line skeleton --------------------------------------------------------------


def joints_on_centre_line(dense: Tree, body: Body, shell: Shell) -> Tree:
    """A joint tree on the centre lines: spine joints about every 12% of the height between
    the limbs, three tail joints, two neck joints and the head; each limb's joints at the
    strongest bends near where its plan has them. Ears, spurs and held things drop out."""
    h = shell.height
    step = 0.01 * h
    window = max(2, int(round(0.035 * h / step)))
    spine = body.spine
    sp = smooth(dense.points[spine], 2.0)
    s = arc(dense.points[spine])
    radius = dense.radius if dense.radius is not None else np.full(dense.n, shell.size)
    legs, ups = body.leg_pairs, body.upper_pairs
    hips_i = int(round((legs[0][0].at + legs[0][1].at) / 2)) if legs else 0
    hips_s = s[hips_i]
    near = lambda x: int(np.argmin(np.abs(s - x)))
    chosen = {hips_i}
    if body.tail is not None and hips_s > 0.08 * h:
        r = radius[spine]
        base_i = next((k for k in range(hips_i, -1, -1) if r[k] < 0.5 * r[hips_i]), 0)
        base_s = min(s[base_i], hips_s - 0.02 * h)
        seg = 0.16 * h
        long_back = hips_s - base_s > 1.0 * h    # a seal or a slug: the whole back body is tail
        if long_back:
            base_s = hips_s - 1.5 * seg
        n_tail = int(np.clip(round(base_s / seg), 3, 10)) if long_back else 3
        chosen |= {near(x) for x in np.linspace(base_s, 0.0, n_tail + 1)[:-1]}
    marks = sorted({float(s[limb.at]) for pair in legs + ups for limb in pair}
                   | {float(s[limb.at]) for limb in body.singles})
    s0 = hips_s
    for m in [m for m in marks if m > hips_s + 0.04 * h]:
        n_seg = max(1, int(round((m - s0) / (0.12 * h))))
        chosen |= {near(x) for x in np.linspace(s0, m, n_seg + 1)[1:]}
        s0 = m
    rest = s[-1] - s0
    chosen |= {near(s0 + rest * f) for f in (0.55 / 3, 1.1 / 3, 0.55)}
    chosen.discard(len(spine) - 1)
    if body.tail is not None:
        chosen.discard(0)
    else:
        chosen = {i for i in chosen if i >= hips_i}
    order = sorted(chosen)

    points, parent, tip = [], [], []

    def add(q, up, is_tip=False):
        points.append(np.asarray(q, float))
        parent.append(up)
        tip.append(is_tip)
        return len(points) - 1

    node_at = {hips_i: add(sp[hips_i], -1)}
    prev = node_at[hips_i]
    for i in [i for i in order if i > hips_i]:
        prev = node_at[i] = add(sp[i], prev)
    add(dense.points[spine[-1]], prev, True)
    prev = node_at[hips_i]
    for i in [i for i in order if i < hips_i][::-1]:
        prev = node_at[i] = add(sp[i], prev)
    if body.tail is not None:
        add(dense.points[spine[0]], prev, True)
    spine_nodes = sorted(node_at)

    for kind_, limbs, _ in limb_plan(body):
        for limb in limbs:
            path = dense.points[limb.path]
            r0 = float(radius[limb.path[0]])
            off = np.min(np.linalg.norm(path[:, None] - dense.points[spine][None, ::2], axis=2), axis=1)
            out = np.nonzero(off > 0.75 * r0)[0]
            e = int(out[0]) if len(out) else 0
            if len(path) - e < 4:
                e = 0
            line = smooth(resample(path[e:], step), 1.5)
            host = node_at[min(spine_nodes, key=lambda i: abs(s[i] - s[limb.at]))]
            joints = []
            if kind_ in PLANS:
                top, _, shares = PLANS[kind_]
                if top is not None:
                    lift = np.array([0.0, 0.25 * r0, 0.7 * r0]) if kind_ == "front" else np.zeros(3)
                    joints.append(path[0] + lift)
                joints.append(line[0])
                joints += [line[i] for i in place(line, shares, window)]
            else:
                n = max(2, int(round(arc(line)[-1] / (0.12 * h))))
                joints += [line[i] for i in np.linspace(0, len(line) - 1, n + 1).astype(int)[:-1]]
            up = host
            for q in joints:
                up = add(q, up)
            add(line[-1], up, True)
    return Tree(np.array(points), np.array(parent), np.array(tip))


def centre_line_bones(shell: Shell) -> tuple[Bones, Body]:
    """Named bones from the shell's centre lines alone."""
    dense = centre_line(shell)
    first = analyse(dense, shell, middle=MIDDLE_CENTRE)
    tree = joints_on_centre_line(dense, first, shell)
    body = analyse(tree, shell, middle=MIDDLE_CENTRE)
    bones, _ = name_bones(tree, body, shell)
    body.notes = first.notes + [n for n in body.notes if n not in first.notes]
    return bones, body


def is_trunk(bone: Bone) -> bool:
    return bone.kind in TRUNK_KINDS
