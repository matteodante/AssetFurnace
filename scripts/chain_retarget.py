#!/usr/bin/env python3
"""Carry a clip made for one creature onto another skeleton, chain by chain.

The source is a hand-made clip from the CC0 library (`image_to_3dlab/clip_library.py`)
with its chains named by hand; the target is a SkinTokens rig whose chains
`image_to_3dlab.creature_family` found. Chains are matched by role: front left leg to
front left leg, wing to wing, tail to tail. Nothing here needs Blender: the result is the
same list of frames `(turn, root)` as `creature_motion`, which `creature_clips.py` keys.

* **Body chains** (spine, neck, tail, a serpent's body) take the source's turn from rest,
  bone by bone at the same fraction along the chain, so the creature keeps its own
  posture and only moves the way the source moves.
* **Limbs** (legs, arms, wings) turn the same way, so they keep their own rest shape: forcing
  the source's shape onto them stretched the skin before the clip even moved (1-5% of the
  edges on a werewolf and a spider). A leg is then solved so its foot stands at the
  source's height above the ground, scaled like the body, wherever its turned leg puts it
  along the ground.
* **The root** moves like the source's matching point, scaled by leg length (or body
  length when there are no legs). Bones on no chain follow their parents.

Directions are converted between the two creatures' own frames (left, up, forward), so a
source facing +Z drives a rig facing anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from creature_motion import FPS, Rig, fabrik, twins
from kimodo_retarget import rotation_between

BODY_CHAINS = ("spine", "neck", "tail", "body")
# A source foot this far above its rest height, as a fraction of its leg, is on the ground
# (first) or lifted (second); in between the two rules blend.
CONTACT = (0.02, 0.08)
# Each frame a lifted foot keeps this much of the offset that held it in place.
RELEASE = 0.8
# and the offset moves by at most this fraction of the leg's length per frame.
STEP = 0.03


@dataclass
class Source:
    """A sampled source clip: bone world matrices at rest and per frame, plus its roles
    (the same shape `creature_family.classify` returns: legs, arms, wings, spine, ...)."""
    parents: dict[str, str | None]
    rest: dict[str, np.ndarray]
    poses: list[dict[str, np.ndarray]]
    roles: dict


# -- small rotation helpers ----------------------------------------------------------------

def _rot(m: np.ndarray) -> np.ndarray:
    """The rotation part of a world matrix, scale removed."""
    r = np.asarray(m, float)[:3, :3]
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    return out if np.linalg.det(out) > 0 else u @ np.diag([1, 1, -1]) @ vt


def _quat(r: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    qa, qb = _quat(a), _quat(b)
    if qa @ qb < 0:
        qb = -qb
    dot = min(1.0, float(qa @ qb))
    angle = math.acos(dot)
    if angle < 1e-6:
        q = qa + (qb - qa) * u
    else:
        q = (math.sin((1 - u) * angle) * qa + math.sin(u * angle) * qb) / math.sin(angle)
    return _matrix(q / np.linalg.norm(q))


def _basis(frame: dict) -> np.ndarray:
    """Columns: left, up, forward."""
    return np.column_stack([np.asarray(frame[k], float) for k in ("left", "up", "forward")])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# -- matching chains -----------------------------------------------------------------------

def pick_rows(n_target: int, n_source: int) -> list[int]:
    """Which source row (front to back) drives each target row. A single target leg per
    side is a hind leg: a biped stands on the legs a quadruped walks behind on."""
    if n_target == 1:
        return [n_source - 1]
    return [math.floor(i * (n_source - 1) / (n_target - 1) + 0.5) for i in range(n_target)]


def _ordered(legs: list[dict], heads: dict[str, np.ndarray], forward: np.ndarray, side: str) -> list[dict]:
    """One side's legs, front first, by where their feet are: a spider's legs may all start
    from one hub in the middle of the body."""
    mine = [leg for leg in legs if leg["side"] == side]
    return sorted(mine, key=lambda leg: -float(heads[leg["bones"][-1]] @ forward))


def match_chains(rig: Rig, roles: dict, source: Source) -> list[tuple[str, list[str], list[str]]]:
    """(kind, target bones, source bones) for every chain both creatures have. `kind` is
    "leg", "limb" (arms and wings) or "body"; or "twin", a second chain inside one mesh leg,
    paired with that leg's target bones instead."""
    src_heads = {b: m[:3, 3] for b, m in source.rest.items()}
    t_fwd = np.asarray(roles["frame"]["forward"], float)
    s_fwd = np.asarray(source.roles["frame"]["forward"], float)
    pairs = []
    for side in ("L", "R"):
        legs = _ordered(roles.get("legs", []), rig.heads, t_fwd, side)
        double = twins(rig, legs)  # two chains in one mesh leg move as one
        mine = [leg for i, leg in enumerate(legs) if i not in double]
        theirs = _ordered(source.roles.get("legs", []), src_heads, s_fwd, side)
        if mine and theirs:
            row_of = dict(zip((id(leg) for leg in mine), pick_rows(len(mine), len(theirs))))
            for i, leg in enumerate(legs):
                if i in double:  # it copies its leg's turns, so the two stay parallel
                    pairs.append(("twin", leg["bones"], legs[double[i]]["bones"]))
                else:
                    pairs.append(("leg", leg["bones"], theirs[row_of[id(leg)]]["bones"]))
        for kind in ("arms", "wings"):
            mine = [c for c in roles.get(kind, []) if c["side"] == side]
            theirs = [c for c in source.roles.get(kind, []) if c["side"] == side]
            if mine and theirs:
                pairs.append(("limb", mine[0]["bones"], theirs[0]["bones"]))
    serpent = roles.get("family") == "serpent"
    for kind in BODY_CHAINS:
        if serpent != (kind == "body"):
            continue  # a serpent's spine and neck lie along its body: one chain, not three
        if roles.get(kind) and source.roles.get(kind):
            pairs.append(("body", _joined(roles, kind, rig.parents), _joined(source.roles, kind, source.parents)))
    return pairs


def _joined(roles: dict, kind: str, parents: dict) -> list[str]:
    """A body chain, starting one joint earlier when it grows from the last bone of another
    (a neck on the end of the spine): that bone's segment belongs to no chain otherwise."""
    bones = list(roles[kind])
    start = parents.get(bones[0])
    ends = {roles[k][-1] for k in BODY_CHAINS if k != kind and roles.get(k)}
    return [start, *bones] if start in ends else bones


# -- along a chain -------------------------------------------------------------------------

def _fractions(points: list[np.ndarray]) -> np.ndarray:
    """Each joint's distance along the chain, as a fraction of its length."""
    steps = [float(np.linalg.norm(b - a)) for a, b in pairwise(points)]
    total = sum(steps) or 1.0
    return np.concatenate([[0.0], np.cumsum(steps) / total])


def _at(points: list[np.ndarray], fractions: np.ndarray, f: float) -> np.ndarray:
    """The point a fraction `f` along a polyline whose joints sit at `fractions`."""
    i = int(np.clip(np.searchsorted(fractions, f, side="right") - 1, 0, len(points) - 2))
    span = fractions[i + 1] - fractions[i]
    u = 0.0 if span <= 1e-12 else float(np.clip((f - fractions[i]) / span, 0.0, 1.0))
    return points[i] + (points[i + 1] - points[i]) * u


def _owner(parents: dict, a: str, b: str) -> str:
    """The bone that turns the segment between joints a and b: the parent of the two."""
    return b if parents.get(a) == b else a


def _chain_length(heads: dict[str, np.ndarray], bones: list[str]) -> float:
    return sum(float(np.linalg.norm(heads[b] - heads[a])) for a, b in pairwise(bones))


# -- the retarget --------------------------------------------------------------------------

def retarget(rig: Rig, roles: dict, source: Source) -> list[tuple[dict, np.ndarray]]:
    """The source clip on `rig`, as `creature_motion` frames."""
    convert = _basis(roles["frame"]) @ _basis(source.roles["frame"]).T
    s_up = np.asarray(source.roles["frame"]["up"], float)
    src_rest = {b: m[:3, 3] for b, m in source.rest.items()}
    src_rest_rot = {b: _rot(m) for b, m in source.rest.items()}
    pairs = match_chains(rig, roles, source)
    scale = _scale(rig, src_rest, pairs)
    anchor = _anchor(rig, pairs, src_rest)

    frames = []
    locks: dict[str, tuple] = {}  # per leg: its foot's offset, and where it touched down
    for pose in source.poses:
        delta = {b: _rot(m) @ src_rest_rot[b].T for b, m in pose.items() if b in src_rest_rot}
        posed = {b: m[:3, 3] for b, m in pose.items()}
        total: dict[str, np.ndarray] = {}
        for kind, mine, theirs in pairs:
            if kind == "body":
                total.update(_body(rig, mine, source, theirs, src_rest, delta, convert))
        root_name = rig.order[0]
        if anchor is not None:
            theirs, f = anchor
            fr = _fractions([src_rest[b] for b in theirs])
            moved = _at([posed[b] for b in theirs], fr, f) - _at([src_rest[b] for b in theirs], fr, f)
            root = rig.heads[root_name] + scale * (convert @ moved)
        else:
            root = rig.heads[root_name].copy()
        limbs = {mine[0]: (kind, mine, theirs) for kind, mine, theirs in pairs if kind in ("leg", "limb")}
        heads: dict[str, np.ndarray] = {}
        inherits = set()
        for b in rig.order:
            p = rig.parents[b]
            inherited = total[p] if p is not None else np.eye(3)
            heads[b] = root if p is None else heads[p] + total[p] @ (rig.heads[b] - rig.heads[p])
            if b in limbs:
                kind, mine, theirs = limbs[b]
                # the body's middle runs through its hips, not a helper root that stays still
                # while the body rolls over (a dying fox)
                hub = next((h for h in (roles.get("spine") or [])[:1] if h in heads), rig.order[0])
                rest_left = np.asarray(roles["frame"]["left"], float)
                midline = (hub, heads[hub], total[hub] @ rest_left, rest_left)
                total.update(_limb(rig, kind, mine, theirs, heads[b], inherited, source, src_rest, posed, delta,
                                   convert, scale, s_up, locks, midline))
            if b not in total:
                total[b] = inherited
                inherits.add(b)
        twinned = [(mine, theirs) for kind, mine, theirs in pairs if kind == "twin"]
        for mine, leg in twinned:
            t_fr, l_fr = _fractions([rig.heads[b] for b in mine]), _fractions([rig.heads[b] for b in leg])
            for b, f in zip(mine, t_fr):
                total[b] = total[leg[int(np.abs(l_fr - f).argmin())]]
                inherits.discard(b)
        if twinned:  # what hangs off a twin follows it again
            for b in rig.order:
                if b in inherits and rig.parents[b] is not None:
                    total[b] = total[rig.parents[b]]
        turn = {b: total[b] @ (total[rig.parents[b]].T if rig.parents[b] is not None else np.eye(3))
                for b in rig.order}
        frames.append((turn, root))
    return frames


def _scale(rig, src_rest, pairs) -> float:
    """Target size over source size: leg length when both stand on legs, else the length of
    the body chains. Not hip height: a flat spider's hips are low and its steps long, and a
    squat rig scaled by its hips would take steps its legs cannot reach."""
    legs = [(_chain_length(rig.heads, mine), _chain_length(src_rest, theirs))
            for kind, mine, theirs in pairs if kind == "leg"]
    legs = [t / s for t, s in legs if t > 1e-9 and s > 1e-9]
    if legs:
        return float(np.mean(legs))
    body = [p for p in pairs if p[0] == "body"]
    t_len = sum(_chain_length(rig.heads, mine) for _, mine, _ in body)
    s_len = sum(_chain_length(src_rest, theirs) for _, _, theirs in body)
    return t_len / s_len if t_len > 1e-9 and s_len > 1e-9 else 1.0


def _anchor(rig: Rig, pairs, src_rest) -> tuple[list[str], float] | None:
    """The source chain and fraction along it that sit where the target's root does."""
    root = rig.order[0]
    for kind, mine, theirs in pairs:
        if kind == "body" and root in mine:
            f = _fractions([rig.heads[b] for b in mine])[mine.index(root)]
            return theirs, float(f)
    for kind, mine, theirs in pairs:  # a root off every chain moves like the spine's start
        if kind == "body":
            return theirs, 0.0
    return None


def _body(rig, mine, source, theirs, src_rest, delta, convert, heading: bool = False) -> dict[str, np.ndarray]:
    """Each target segment turns like the source's at the same fraction along the chain,
    blended between source segments so a long chain bends smoothly. With `heading` the
    turns are taken about the chain's own heading: a leg reaching out sideways lifts the
    way a source leg reaching forward lifts, instead of sweeping."""
    if heading:
        convert = _heading(convert @ (src_rest[theirs[-1]] - src_rest[theirs[0]]),
                           rig.heads[mine[-1]] - rig.heads[mine[0]], convert @ _up(source)) @ convert
    s_fr = _fractions([src_rest[b] for b in theirs])
    s_mid = (s_fr[:-1] + s_fr[1:]) / 2
    s_rot = [delta.get(_owner(source.parents, a, b), np.eye(3)) for a, b in pairwise(theirs)]
    t_fr = _fractions([rig.heads[b] for b in mine])
    out = {}
    for j, (a, b) in enumerate(pairwise(mine)):
        m = (t_fr[j] + t_fr[j + 1]) / 2
        if m <= s_mid[0] or len(s_rot) == 1:
            r = s_rot[0]
        elif m >= s_mid[-1]:
            r = s_rot[-1]
        else:
            i = int(np.searchsorted(s_mid, m, side="right") - 1)
            r = _slerp(s_rot[i], s_rot[i + 1], float((m - s_mid[i]) / (s_mid[i + 1] - s_mid[i])))
        out[_owner(rig.parents, a, b)] = convert @ r @ convert.T
    return out


def _up(source: Source) -> np.ndarray:
    return np.asarray(source.roles["frame"]["up"], float)


def _heading(src: np.ndarray, tgt: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The turn about `up` that brings the source chain's heading onto the target's. No turn
    (the identity) when either chain runs (nearly) straight up or down: it has no heading."""
    up = _unit(up)
    a, b = src - (src @ up) * up, tgt - (tgt @ up) * up
    if np.linalg.norm(a) < 0.2 * np.linalg.norm(src) or np.linalg.norm(b) < 0.2 * np.linalg.norm(tgt):
        return np.eye(3)
    angle = math.atan2(float(np.cross(a, b) @ up), float(a @ b))
    k = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def above_ground(foot: np.ndarray, rest_height: float, lift: float, up: np.ndarray) -> np.ndarray:
    """`foot`, lifted to the ground if it went through it. A source foot below its rest height
    (a body lying down, a creature on a slope) lets it go as low."""
    up = _unit(np.asarray(up, float))
    own = float(foot @ up)
    return foot + max(0.0, rest_height + min(lift, 0.0) - own) * up


def approach(current: np.ndarray, wanted: np.ndarray, step: float) -> np.ndarray:
    """`current` moved towards `wanted` by at most `step`."""
    gap = wanted - current
    n = float(np.linalg.norm(gap))
    return wanted if n <= step else current + gap * (step / n)


def own_side(foot: np.ndarray, rest_offset: np.ndarray, centre: np.ndarray, left: np.ndarray,
             rest_left: np.ndarray, kept: float = 1 / 3) -> np.ndarray:
    """`foot` kept on its own side of the body, at least `kept` as far out as at rest. The
    same tuck that keeps a drake's front feet apart crossed them on a rig whose feet stand
    closer together. `kept` is how far out the source's own foot is (`kept_out`): a foot
    the source puts across the middle (a spider's, reaching under its head) is not held."""
    out = float(rest_offset @ rest_left)
    if abs(out) < 1e-9 or kept <= 0:
        return foot
    margin = np.sign(out) * abs(out) * min(kept, 1 / 3)
    lateral = float((foot - centre) @ left)
    if np.sign(out) * (lateral - margin) >= 0:
        return foot
    return foot + (margin - lateral) * left


def kept_out(source: Source, tip: str, posed, delta, src_rest) -> float:
    """How far out from the body's middle the source's foot is now, as a fraction of how far
    it is at rest: 1 where it stands, 0 on the middle, below 0 across it."""
    spine = source.roles.get("spine") or []
    hub = spine[0] if spine else next((b for b, p in source.parents.items() if p is None and b in src_rest), None)
    if hub is None:
        return 1.0
    left = np.asarray(source.roles["frame"]["left"], float)
    rest = float((src_rest[tip] - src_rest[hub]) @ left)
    if abs(rest) < 1e-9:
        return 1.0
    now = float((posed[tip] - posed[hub]) @ (delta.get(hub, np.eye(3)) @ left))
    return now / rest


def contact(lift: float, length: float) -> float:
    """How much a source foot `lift` above its rest height is on the ground: 1 near it, 0
    clear of it, above or below (a flying creature's feet hang below their standing height)."""
    lifted = abs(lift) / max(length, 1e-9)
    return float(np.clip((CONTACT[1] - lifted) / (CONTACT[1] - CONTACT[0]), 0.0, 1.0))


def _limb(rig, kind, mine, theirs, start, inherited, source, src_rest, posed, delta, convert, scale, s_up,
          locks, midline):
    """A limb turns like a body chain, so it keeps its own shape. A leg is then solved so its
    foot stands at the source's height; while the source's foot is on the ground it is held
    where it touched down (a turned leg of other proportions would let it drift), and as it
    lifts off it goes back to where the turned leg puts it.

    Only the limb's own motion, relative to the body it hangs from, is turned to its
    heading; the body's turn (`inherited`) is then put back on top unchanged. Turning the
    whole of it would make a plain body roll bend a limb that points another way."""
    on_body = {x for k in BODY_CHAINS for x in source.roles.get(k) or []}
    hub = source.parents.get(theirs[0])
    while hub is not None and hub not in on_body:  # a limb's anchor swings with the limb, not the body
        hub = source.parents.get(hub)
    # The limb's motion in the body's own frame (L = body^T limb), turned to the rig's
    # heading, then put under the rig's body: total = body @ H L H^T. The other order (the
    # world-frame motion turned, the body put on afterwards) bent an arm 22.7 degrees wrong
    # under a 45-degree body roll.
    base = delta.get(hub, np.eye(3)).T
    own = {b: base @ d for b, d in delta.items()}
    out = {b: inherited @ r for b, r in _body(rig, mine, source, theirs, src_rest, own, convert, True).items()}
    if kind != "leg":
        return out
    pts = [np.asarray(start, float)]
    for a, b in pairwise(mine):
        pts.append(pts[-1] + out[_owner(rig.parents, a, b)] @ (rig.heads[b] - rig.heads[a]))
    # A foot the source keeps on the ground stands at the source's height, scaled like the
    # body; a lifted one goes where its turned leg puts it, but never lower.
    lift = float((posed[theirs[-1]] - src_rest[theirs[-1]]) @ s_up)
    w = contact(lift, _chain_length(src_rest, theirs))
    t_up = _unit(convert @ s_up)
    # The foot sits at an offset from where the turned leg puts it. On the ground the offset
    # aims to hold it where it touched down (moving only as the source's foot moves since)
    # at the source's scaled height; lifted, it dies away. Either way it moves by at most a
    # little each frame: snapping between the two made feet leap as they lifted off or
    # touched down (by 12-17% of a werewolf's height in one frame).
    turned = pts[-1]
    offset, lock = locks.get(mine[0], (np.zeros(3), None))
    height = float(rig.heads[mine[-1]] @ t_up)
    if w > 0:
        if lock is None:
            lock = (turned + offset, posed[theirs[-1]].copy())
        held, then = lock
        target = held + scale * (convert @ (posed[theirs[-1]] - then))
        target = target + (height + scale * lift - float(target @ t_up)) * t_up
        wanted = w * (target - turned) + (1 - w) * offset * RELEASE
    else:
        lock, wanted = None, offset * RELEASE
    offset = approach(offset, wanted, STEP * _chain_length(rig.heads, mine))
    locks[mine[0]] = (offset, lock)
    hub, *line = midline
    tip = own_side(turned + offset, rig.heads[mine[-1]] - rig.heads[hub], *line,
                   kept=kept_out(source, theirs[-1], posed, delta, src_rest))
    tip = above_ground(tip, height, scale * lift, t_up)
    solved = fabrik(pts, tip)
    for j, (a, b) in enumerate(pairwise(mine)):
        have, want = pts[j + 1] - pts[j], solved[j + 1] - solved[j]
        if np.linalg.norm(have) > 1e-12 and np.linalg.norm(want) > 1e-12:
            owner = _owner(rig.parents, a, b)
            out[owner] = rotation_between(have, want) @ out[owner]
    return out


def longer(source: Source, factor: float, seed: int, loop: bool) -> Source:
    """A take of the source clip `factor` times as long, made of its own pieces in a new
    order (`image_to_3dlab.motion_variation`, after GenMM)."""
    from image_to_3dlab.motion_variation import vary

    poses = vary(source.parents, source.rest, source.poses, _up(source),
                 length=round(len(source.poses) * factor), seed=seed, loop=loop)
    return Source(parents=source.parents, rest=source.rest, poses=poses, roles=source.roles)


def closed_loop(frames: list, loop: bool) -> list:
    """A loop's frames with the first again at the end, as the procedural clips have them: its
    last key is then its first pose, and an engine playing it round blends back over one frame
    instead of jumping from the last pose to the first in no time. `load_source` drops the
    source's own closing frame, so it is put back here, after the retarget."""
    return [*frames, frames[0]] if loop and frames else list(frames)


# -- reading a library clip ----------------------------------------------------------------

def load_source(glb: Path, roles: dict, clip: str, fps: int = FPS, loop: bool = False) -> Source:
    """Sample one animation of a library GLB at `fps`. A looping clip leaves out its last
    frame, which repeats the first."""
    from image_to_3dlab.glb_skin import SkinnedGLB

    skin = SkinnedGLB(glb)
    names = [n for n, _ in skin.animations()]
    if clip not in names:
        raise LookupError(f"{Path(glb).name} has no clip {clip!r} (it has {', '.join(names)})")
    index, duration = names.index(clip), skin.animations()[names.index(clip)][1]
    parents, _ = skin.skeleton()
    by_name = {skin.name(j): j for j in skin.joint_nodes}
    rest_world = skin.world(skin.rest_locals())
    rest = {n: rest_world[j] for n, j in by_name.items()}
    count = max(1, round(duration * fps)) + (0 if loop else 1)
    poses = []
    for k in range(count):
        world = skin.pose(index, min(k / fps, duration))
        poses.append({n: world[j] for n, j in by_name.items()})
    return Source(parents=parents, rest=rest, poses=poses, roles=roles)


# -- the command line ----------------------------------------------------------------------

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Carry one CC0 library clip onto a rigged creature, chain by chain.")
    parser.add_argument("--list", action="store_true", help="the library's creatures and clips, then stop")
    parser.add_argument("--rig", type=Path, help="the rigged GLB (SkinTokens output)")
    parser.add_argument("--source", help="the library creature the clip comes from (see --list)")
    parser.add_argument("--clip", help="the clip, as --list names it")
    parser.add_argument("--family", default="auto", help="the rig's family (default: recognised)")
    parser.add_argument("--longer", action="store_true", help="a longer take, varied (GenMM), of a looping clip")
    parser.add_argument("--factor", type=float, default=None, help="how much longer (default the library's)")
    parser.add_argument("--seed", type=int, default=None, help="the variation's seed (default the library's)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """`python scripts/chain_retarget.py --rig RIGGED.glb --source kaiju --clip walk [--longer]`:
    one library clip on a rig, written beside it as `<name>_<family>-<clip>.glb` with its
    provenance, as `creature_clips.py` writes the whole set. Keying the clip needs Blender's
    `bpy`, so run it with SkinTokens' Python (`vendor/SkinTokens/.venv`); `--list` does not."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_to_3dlab import clip_library as cl

    args = build_parser().parse_args(argv)
    if args.list:
        for name, source in cl.SOURCES.items():
            print(f"{name:8} {', '.join(source['clips'])}")
        return 0
    if not (args.rig and args.source and args.clip):
        print("chain_retarget: give --rig, --source and --clip (or --list)", file=sys.stderr)
        return 1
    source = cl.SOURCES.get(args.source)
    if source is None or args.clip not in source["clips"]:
        known = ", ".join(source["clips"]) if source else ", ".join(cl.SOURCES)
        print(f"chain_retarget: {args.source} has no clip {args.clip!r} ({known})", file=sys.stderr)
        return 1
    if args.longer and not source["clips"][args.clip][1]:
        print(f"chain_retarget: {args.clip} does not loop, so it has no longer take", file=sys.stderr)
        return 1
    try:
        import bpy  # noqa: F401 - keying the clip needs it
    except ImportError:
        print("chain_retarget: keying a clip needs Blender's bpy: run it with SkinTokens' Python "
              "(vendor/SkinTokens/.venv)", file=sys.stderr)
        return 1
    import creature_clips  # pragma: no cover - needs bpy

    factor = cl.LONGER_FACTOR if args.factor is None else args.factor
    seed = cl.LONGER_SEED if args.seed is None else args.seed
    chosen = None if args.family == "auto" else args.family
    found = creature_clips.recognise(args.rig, chosen)
    if found["family"] is None:
        print(f"chain_retarget: {found['clips_error']}", file=sys.stderr)
        return 1
    motion = args.clip + (cl.LONGER if args.longer else "")
    name = creature_clips._write_clip(args.rig, found["family"], motion, None, args.source, factor=factor, seed=seed)
    clip = creature_clips.clip_path(args.rig, name)
    creature_clips.write_provenance(args.rig, clip, name, found["family"], cl.motion_record(args.source, motion, factor, seed))
    print(clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
