"""Which family a rigged creature belongs to, read from its skeleton alone.

The auto-rig (SkinTokens) gives any creature a skeleton, with bones named `bone_0..N`.
Animating it needs to know which chains are legs, arms, wings, a tail or a neck, and what
kind of body they make. That is all decided here from the tree's shape and where its
joints sit, never from bone names or one creature's measurements:

* a **leg** is a branch that hangs from the body and ends on or near the ground, touching
  it at one spot (a tail or tentacle lying along the ground is not one). Once three legs are
  found, long chains that reach out sideways and down count too: the auto-rig stops a
  generated spider's leg bones at different heights, not all on the ground;
* a leg is **digitigrade** when it stands on its toes: the mesh touches the ground only in
  front of the last joint (the hock), where a human foot puts its heel down behind the
  ankle. The joints alone cannot tell: SkinTokens puts a werewolf's hock and a booted
  knight's ankle at nearly the same height (18% and 16% of the leg), while the soles differ
  plainly (no heel against a heel 0.6 of a foot behind the ankle);
* the creature **faces** across the line between its hips (two legs) or along the line its
  legs attach on (more): feet can point outwards, and taking them as the front turned one
  werewolf 17 degrees off its own body;
* the **head** is the trunk tip furthest forward and up; a **tail** is the one furthest
  back; branches from the trunk that reach out sideways are **arms** or **wings**.

The families: humanoid, digitigrade (a two-legged creature on digitigrade legs), quadruped,
multileg (spiders, insects), winged, serpent and plant. A person can choose the family
instead; the roles found here stay the same either way.

Pure numpy, so it runs in the studio, in the checks and in the clip maker alike. Positions
may be in any frame: pass which way is `up`, and `front` for a hint of which way the model
faces when its feet do not say (a generated glTF faces +Z).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FAMILIES = ("humanoid", "digitigrade", "quadruped", "multileg", "winged", "serpent", "plant")
LABELS = {
    "humanoid": "Humanoid", "digitigrade": "Two legs, digitigrade (werewolf, demon)",
    "quadruped": "Four legs", "multileg": "Many legs (spider, insect)", "winged": "Winged (dragon, bird)",
    "serpent": "Serpent", "plant": "Plant",
}
SCHEMA_VERSION = 1
SUFFIX = ".family.json"

# Fractions of the creature's height unless said otherwise. Provisional, like every
# threshold the pilot has not calibrated yet.
GROUND = 0.08            # a joint this close to the lowest one is on the ground
NEAR_GROUND = 0.3        # of its attachment's height: a leg ends at least this low
SPLAYED = 0.3            # of body size: a chain this long reaching out and down is a leg too
LEG_ATTACH = 0.25        # a leg hangs from a body at least this far up (roots fan out lower)
TAIL_REACH = 1.5         # of its drop: a chain reaching out further than this along the ground is a tail
CONTACT = 0.06           # of leg length: joints this low are the foot's contact
DIGITIGRADE_JOINTS = 0.25  # of leg length: without a mesh, only a hock this high says digitigrade
SOLE = 0.03              # of the mesh height: vertices this low are a sole on the ground
HEEL = -0.25             # of the foot's length: a sole reaching further behind the ankle has a heel
DRAG_SPREAD = 0.3        # of the drop: ground contact spread wider than this is dragging
DESCENT = 0.1            # of its length along the bones: a leg drops at least this much
LATERAL = 0.15           # of body size: a branch reaching this far sideways is an arm or wing


def _kids(parents: dict[str, str | None]) -> dict[str, list[str]]:
    kids: dict[str, list[str]] = {b: [] for b in parents}
    for bone, parent in parents.items():
        if parent is not None:
            kids[parent].append(bone)
    return kids


def _subtree(bone: str, kids) -> list[str]:
    out, stack = [], [bone]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(kids[b])
    return out


def _path_up(bone: str, parents) -> list[str]:
    """Root first, ending at `bone`."""
    out = [bone]
    while parents[out[-1]] is not None:
        out.append(parents[out[-1]])
    return out[::-1]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class _Body:
    """The skeleton plus the measurements every rule below shares."""

    def __init__(self, parents, heads, up):
        self.parents = dict(parents)
        self.heads = {b: np.asarray(p, float) for b, p in heads.items()}
        self.up = _unit(np.asarray(up, float))
        self.kids = _kids(self.parents)
        roots = [b for b, p in self.parents.items() if p is None]
        if len(roots) != 1:
            raise ValueError(f"expected one root bone, found {len(roots)}")
        self.root = roots[0]
        self.h = {b: float(p @ self.up) for b, p in self.heads.items()}
        self.ground = min(self.h.values())
        self.height = max(max(self.h.values()) - self.ground, 1e-9)
        flat = np.array([self.flat(p) for p in self.heads.values()])
        self.width = float(np.max(np.ptp(flat, axis=0))) if len(flat) > 1 else 0.0
        self.size = max(self.height, self.width)
        self.low = {}
        for b in reversed(self.order()):
            self.low[b] = min([self.h[b]] + [self.low[c] for c in self.kids[b]])

    def order(self) -> list[str]:
        out = [self.root]
        for b in out:
            out.extend(self.kids[b])
        return out

    def flat(self, p: np.ndarray) -> np.ndarray:
        return p - (p @ self.up) * self.up

    def above(self, b: str) -> float:
        return self.h[b] - self.ground

    def lowest_path(self, start: str) -> list[str]:
        path = [start]
        while self.kids[path[-1]]:
            path.append(min(self.kids[path[-1]], key=self.low.__getitem__))
        return path


def _reaches_down(body: _Body, start: str, fork: str) -> bool:
    """The branch from `fork` through `start` ends on or near the ground."""
    return body.low[start] - body.ground <= max(GROUND * body.height, NEAR_GROUND * body.above(fork))


def _legs(body: _Body) -> list[dict]:
    """Every branch that hangs from the body to one spot on the ground."""
    found = []
    for fork in body.order():
        if len(body.kids[fork]) < 2:  # a leg hangs beside something: a lone chain is the body itself
            continue
        if body.above(fork) < LEG_ATTACH * body.height:
            continue
        for start in body.kids[fork]:
            if not _reaches_down(body, start, fork):
                continue
            path = body.lowest_path(start)
            if body.h[fork] - body.low[start] < DESCENT * _length(body, [fork, *path]):
                continue  # runs along the ground rather than down to it: a body, a tail
            top = body.above(start)
            # a trunk carrying two legs further on (a spine to the forelegs) is not a leg
            if any(body.above(x) > 0.25 * top and sum(_reaches_down(body, c, x) for c in body.kids[x]) >= 2
                   for x in path[:-1]):
                continue
            # contact is what lies as low as the branch goes, all of it: a tail hanging just
            # above the ground drags along it, a plant's roots fan out over it, while a
            # toe-walker's hock stands clear of its toes
            contact = [body.flat(body.heads[x]) for x in _subtree(start, body.kids)
                       if body.h[x] - body.low[start] <= GROUND * body.height]
            spread = max((float(np.linalg.norm(a - b)) for a in contact for b in contact), default=0.0)
            if len(contact) >= 2 and spread > DRAG_SPREAD * body.above(fork):
                continue  # lies along the ground: a tail or tentacle
            found.append({"fork": fork, "bones": path})
    starts = {leg["bones"][0] for leg in found}
    # a knee pad or a toe that also reaches the ground sits inside a leg already found
    return [leg for leg in found
            if not any(a in starts for a in _path_up(leg["bones"][0], body.parents)[:-1])]


def _splayed(body: _Body, legs: list[dict], left: np.ndarray) -> list[dict]:
    """Long chains that reach out sideways and down from where legs attach, but stop short
    of the ground: the rest of a many-legged creature's legs."""
    taken = {b for leg in legs for b in _subtree(leg["bones"][0], body.kids)}
    starts = {leg["bones"][0] for leg in legs}
    found, used = [], set()
    for fork in body.order():
        if fork in taken or (len(body.kids[fork]) < 2 and fork != body.root):
            continue
        if body.above(fork) < LEG_ATTACH * body.height:
            continue
        for start in body.kids[fork]:
            branch = _subtree(start, body.kids)
            if start in taken or starts & set(branch) or used & set(branch):
                continue
            # a hub carrying several legs is not one leg: its own legs come up as the loop
            # reaches it
            if any(sum(len(_subtree(c, body.kids)) >= 3 for c in body.kids[x]) >= 2 for x in branch):
                continue
            tip = max(branch, key=lambda b: float(np.linalg.norm(body.heads[b] - body.heads[fork])))
            path = [b for b in _path_up(tip, body.parents) if b in set(branch)]
            reach = body.heads[tip] - body.heads[fork]
            flat = body.flat(reach)
            if (len(path) < 4 or _length(body, [fork, *path]) < SPLAYED * body.size
                    or abs(float(reach @ left)) < 0.5 * float(np.linalg.norm(flat))
                    or body.h[fork] - body.h[tip] < 0.15 * body.above(fork)):
                continue
            found.append({"fork": fork, "bones": path})
            used |= set(branch)
    return found


def _tail_like(body: _Body, leg: dict) -> bool:
    """A chain that reaches out along the ground much further than it drops, and never
    rises above where it starts: a tail resting on the ground, not a leg standing on it."""
    fork, bones = leg["fork"], leg["bones"]
    tip = bones[-1]
    drop = body.h[fork] - body.h[tip]
    reach = float(np.linalg.norm(body.flat(body.heads[tip] - body.heads[fork])))
    rises = max(body.h[b] for b in bones) > body.h[fork] + 0.02 * body.height
    return drop > 0 and reach > TAIL_REACH * drop and not rises


def _lone_tail(body: _Body, legs: list[dict]) -> dict | None:
    """The one tail-like chain among at least three candidates, or None when there is none
    or several (then they are all legs)."""
    tails = [leg for leg in legs if _tail_like(body, leg)]
    return tails[0] if len(tails) == 1 and len(legs) >= 3 else None


def _stem(body: _Body) -> list[str]:
    """A plant's stem, base first: from the highest joint down through the tree to the
    lowest, stopping where it fans out into roots (a fork near the ground; a leaf high on
    the stem is no reason to stop)."""
    top = max(body.parents, key=body.h.__getitem__)
    low = min(body.parents, key=body.h.__getitem__)
    up_top, up_low = _path_up(top, body.parents), _path_up(low, body.parents)
    shared = [a for a, b in zip(up_top, up_low) if a == b]
    path = list(reversed(up_low[len(shared) - 1:])) + up_top[len(shared):]  # low ... meeting point ... top
    forks = [i for i, b in enumerate(path) if len(body.kids[b]) >= 2 and body.above(b) <= 0.25 * body.height]
    return path[forks[0]:] if forks else path


def _digitigrade(body: _Body, bones: list[str], forward: np.ndarray,
                 soles: np.ndarray | None) -> tuple[bool, str | None]:
    """(stands on its toes, the ankle or hock joint). `soles` are mesh vertices on the ground."""
    end = bones[-1]
    length = body.h[bones[0]] - body.h[end]
    if length <= 1e-9:
        return False, None
    ankle = next((b for b in reversed(bones) if body.h[b] - body.h[end] > CONTACT * length), None)
    if ankle is None:
        return False, None
    foot = float(np.linalg.norm(body.flat(body.heads[end] - body.heads[ankle])))
    if soles is not None and len(soles) and foot > 1e-9:
        near = soles[np.linalg.norm([body.flat(p - body.heads[end]) for p in soles], axis=1) < 0.5 * length]
        if len(near) >= 3:
            heel = min(float(body.flat(p - body.heads[ankle]) @ forward) for p in near) / foot
            return bool(heel > HEEL), ankle
    return bool((body.h[ankle] - body.h[end]) / length >= DIGITIGRADE_JOINTS), ankle


def _forward(body: _Body, legs: list[dict], front) -> np.ndarray:
    """Which way the creature faces: across its hips, or along where its legs attach; the
    feet (or, failing them, the `front` hint) only say which of the two ways."""
    hint = _unit(body.flat(np.asarray(front, float)))
    feet = [body.flat(body.heads[leg["bones"][-1]] - body.heads[leg["bones"][-2]])
            for leg in legs if len(leg["bones"]) >= 2]
    toes = np.mean(feet, axis=0) if feet else np.zeros(3)
    typical = float(np.mean([np.linalg.norm(f) for f in feet])) if feet else 0.0
    if len(legs) >= 5:
        # Many legs radiate from one hub: neither where they attach nor where their feet
        # point says where the head is. A generated model faces the camera; trust that.
        return hint
    starts = np.array([body.flat(body.heads[leg["bones"][0]]) for leg in legs])
    axis = None
    if len(legs) == 2:
        axis = _unit(np.cross(starts[0] - starts[1], body.up))
    elif len(legs) >= 3:
        values, vectors = np.linalg.eigh(np.cov((starts - starts.mean(axis=0)).T))
        if values[-1] > 1.3 * values[-2]:
            axis = _unit(vectors[:, -1])
    if axis is None or np.linalg.norm(axis) < 0.5:
        return _unit(toes) if np.linalg.norm(toes) > 0.4 * typical > 0 else hint
    lean = float(toes @ axis)
    if abs(lean) > 0.25 * typical > 0:
        return axis if lean > 0 else -axis
    return axis if float(hint @ axis) >= 0 else -axis


def _trunk_tip(body: _Body, trunk: set[str], score) -> str:
    leaves = [b for b in trunk if not any(c in trunk for c in body.kids[b])]
    return max(leaves, key=score)


def _side(value: float, scale: float) -> str:
    if value > 0.02 * scale:
        return "L"
    if value < -0.02 * scale:
        return "R"
    return "C"


def _length(body: _Body, bones: list[str]) -> float:
    return float(sum(np.linalg.norm(body.heads[b] - body.heads[a]) for a, b in zip(bones, bones[1:])))


def classify(parents: dict[str, str | None], heads: dict, up=(0.0, 1.0, 0.0), front=(0.0, 0.0, 1.0),
             family: str | None = None, points=None) -> dict:
    """The family and the role of every chain. `family` forces the answer (a person's choice);
    `points` are the mesh's rest vertices, in the same frame, to see how the feet stand.

    Returns a JSON-ready dict: family, guessed, chosen, confidence, legs, arms, wings, spine,
    neck, tail, body (the main chain of a serpent or plant), frame (up/forward/left), detail.
    """
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose one of {FAMILIES}")
    body = _Body(parents, heads, up)
    legs = _legs(body)
    if len(legs) == 1:  # nothing stands on one leg: a lone chain to the ground is the body (a stem)
        legs = []
    lone = _lone_tail(body, legs)
    if lone is not None:
        legs.remove(lone)
    forward = _forward(body, legs, front)
    left = _unit(np.cross(body.up, forward))
    if len(legs) >= 3:
        # the tail rule again, over every candidate: one chain lying out along the ground is a
        # tail, several are a sprawling creature's legs
        known = legs + ([lone] if lone is not None else [])
        found = known + _splayed(body, known, left)  # the tail set aside is not found again
        lone = _lone_tail(body, found)
        legs = [leg for leg in found if leg is not lone]
        forward = _forward(body, legs, front)  # with every leg: five or more change the rule
        left = _unit(np.cross(body.up, forward))
    soles = None
    if points is not None and len(points):
        points = np.asarray(points, float)
        h = points @ body.up
        soles = points[h < h.min() + SOLE * max(float(np.ptp(h)), 1e-9)]
    in_legs = {b for leg in legs for b in _subtree(leg["bones"][0], body.kids)}
    trunk = {b for b in body.parents if b not in in_legs}
    centre = np.mean([body.heads[leg["fork"]] for leg in legs], axis=0) if legs else body.heads[body.root]

    # front or back from where each leg starts; legs that all start on one hub joint
    # (an auto-rig sometimes builds them so) are told apart by where their feet stand
    anchors = [body.heads[leg["bones"][0]] for leg in legs]
    if legs and np.ptp([float(a @ forward) for a in anchors]) < 0.1 * body.size:
        anchors = [body.heads[leg["bones"][-1]] for leg in legs]
    middle = np.mean(anchors, axis=0) if legs else centre
    for leg, anchor in zip(legs, anchors):
        foot = body.heads[leg["bones"][-1]]
        leg["side"] = _side(float((foot - centre) @ left), body.size)
        along = float((anchor - middle) @ forward)
        leg["end"] = "front" if along > 0.05 * body.size else "back" if along < -0.05 * body.size else "mid"
        leg["digitigrade"], leg["ankle"] = _digitigrade(body, leg["bones"], forward, soles)

    # the head is forward and up, and on the middle line: a wing swept forward is not a head
    head = _trunk_tip(body, trunk, lambda b: float((body.heads[b] - centre) @ (forward + body.up))
                      - 2 * abs(float((body.heads[b] - centre) @ left)))
    head_path = _path_up(head, body.parents)
    on_head = set(head_path)
    lateral = []
    for joint in head_path:
        for start in body.kids[joint]:
            if start in on_head or start in in_legs:
                continue
            branch = _subtree(start, body.kids)
            reach = max(float((body.heads[b] - body.heads[joint]) @ left) for b in branch)
            reach_r = min(float((body.heads[b] - body.heads[joint]) @ left) for b in branch)
            side_reach = reach if abs(reach) >= abs(reach_r) else reach_r
            if abs(side_reach) < LATERAL * body.size:
                continue
            tip = max(branch, key=lambda b: abs(float((body.heads[b] - body.heads[joint]) @ left)))
            bones = [b for b in _path_up(tip, body.parents) if b not in on_head]
            lateral.append({"fork": joint, "bones": bones, "side": "L" if side_reach > 0 else "R",
                            "length": _length(body, [joint, *bones])})

    in_lateral = {b for x in lateral for b in _subtree(x["bones"][0], body.kids)}
    rest = trunk - on_head - in_lateral
    tail = []
    if rest:
        tip = _trunk_tip(body, rest, lambda b: -float((body.heads[b] - centre) @ forward))
        back = float((centre - body.heads[tip]) @ forward)
        if back > 0.15 * body.size and abs(float((body.heads[tip] - centre) @ left)) < 0.5 * back:
            tail = [b for b in _path_up(tip, body.parents) if b not in on_head]

    arms, wings = [], []
    if 1 <= len(legs) <= 2:  # a biped's side branches are arms, unless a pair is wings too
        for side in ("L", "R"):
            mine = sorted((x for x in lateral if x["side"] == side), key=lambda x: x["length"])
            if len(mine) >= 2 or (len(legs) == 2 and mine and mine[0]["length"] > 0.9 * body.height):
                wings.append(mine.pop())
            arms += mine
    else:
        # A wing is long and spreads out level or upwards; feelers and fangs reach down.
        trunk_length = _length(body, head_path)
        wings = [x for x in lateral if x["length"] >= max(0.5 * trunk_length, SPLAYED * body.size)
                 and body.h[x["fork"]] - body.h[x["bones"][-1]] < 0.15 * body.above(x["fork"])]

    attach = [x["fork"] for x in arms + wings] + [leg["fork"] for leg in legs if leg["fork"] in on_head]
    last = max((head_path.index(f) for f in attach), default=0)
    spine, neck = head_path[:last + 1], head_path[last + 1:]

    guess, main, confident = _guess(body, legs, arms, wings, head_path)
    chosen = family or guess
    if chosen == "plant" and guess != "plant":  # chosen by a person: the stem, not the guess's chain
        main = _stem(body)
    if chosen == "serpent":  # a serpent faces along its body, towards its head
        main = main or _diameter(body)
        towards = body.flat(body.heads[main[0]] - body.heads[main[min(2, len(main) - 1)]])
        if np.linalg.norm(towards) > 1e-9:
            forward = _unit(towards)
            left = _unit(np.cross(body.up, forward))
    detail = f"{len(legs)} legs, {len(arms)} arms, {len(wings)} wings, tail of {len(tail)} bones"
    return {
        "schema_version": SCHEMA_VERSION,
        "family": chosen, "guessed": guess, "chosen": family is not None,
        "confidence": "high" if confident else "low",
        "legs": [{k: leg[k] for k in ("bones", "side", "end", "digitigrade", "ankle")} for leg in legs],
        "arms": [{"bones": a["bones"], "side": a["side"]} for a in arms],
        "wings": [{"bones": w["bones"], "side": w["side"]} for w in wings],
        "spine": spine, "neck": neck, "tail": tail, "body": main,
        "frame": {"up": body.up.round(6).tolist(), "forward": forward.round(6).tolist(),
                  "left": left.round(6).tolist()},
        "ground": body.ground, "height": body.height, "detail": detail,
    }


def _guess(body: _Body, legs, arms, wings, head_path) -> tuple[str, list[str], bool]:
    sides = sorted(leg["side"] for leg in legs)
    balanced = sides.count("L") == sides.count("R")
    n = len(legs)
    if n == 0:
        if len(wings) >= 2:
            return "winged", [], False
        if body.height >= 0.75 * body.width:
            stem = _stem(body)
            return "plant", stem, body.above(stem[0]) < 0.25 * body.height  # the stem stands on or near the ground
        return "serpent", _diameter(body), True
    if n <= 2:
        if len(wings) >= 2 and not arms:
            return "winged", [], n == 2 and balanced
        digi = all(leg["digitigrade"] for leg in legs)
        return ("digitigrade" if digi else "humanoid"), [], n == 2 and balanced
    if n <= 4:
        return ("winged" if len(wings) >= 2 else "quadruped"), [], n == 4 and balanced
    return "multileg", [], n % 2 == 0 and balanced


def _diameter(body: _Body) -> list[str]:
    """The longest path through the tree, head end (the higher one) first."""
    def far(start):
        dist, prev, todo = {start: 0.0}, {start: None}, [start]
        while todo:
            b = todo.pop()
            for n in body.kids[b] + ([body.parents[b]] if body.parents[b] else []):
                if n not in dist:
                    dist[n] = dist[b] + float(np.linalg.norm(body.heads[n] - body.heads[b]))
                    prev[n] = b
                    todo.append(n)
        end = max(dist, key=dist.__getitem__)
        path = [end]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        return path

    one = far(body.root)[0]
    path = far(one)[::-1]
    return path if body.h[path[0]] >= body.h[path[-1]] else path[::-1]


def classify_glb(glb: Path, family: str | None = None) -> dict:
    """`classify` on a rigged GLB: its rest skeleton and mesh, glTF's Y up, facing +Z."""
    from image_to_3dlab.glb_skin import SkinnedGLB

    rig = SkinnedGLB(glb)
    parents, heads = rig.skeleton()
    return classify(parents, heads, family=family, points=rig.mesh.vertices)


# ---------------------------------------------------------------------- on disk

PROCEDURAL_LICENSE = {
    "name": "MIT",
    "note": "Procedural test clip made by AssetFurnace (scripts/creature_motion.py); no motion data or model.",
}


def clip_record(rig_record: dict | None, clip: str, family: str | None, tool: str,
                output: Path | None = None) -> dict:
    """Provenance for a test clip: the rig's record (its licences), plus how the motion was
    made, and `output` (the clip file) in place of the rig's own file and hash. Without a
    rig record the licence stays unknown rather than invented."""
    import time

    record = dict(rig_record or {"license": None})
    if output is not None:
        from image_to_3dlab.provenance import sha256_file

        before = record.get("output") if isinstance(record.get("output"), dict) else {}
        record["output"] = {**before, "path": str(output), "sha256": sha256_file(output)}
    record.update({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": tool,
        "motion": {"clip": clip, "family": family, "made_with": "procedural test clip",
                   "license": PROCEDURAL_LICENSE},
    })
    return record


def family_path(glb: Path) -> Path:
    return Path(glb).with_suffix(SUFFIX)


def write_family(glb: Path, found: dict) -> Path:
    path = family_path(glb)
    path.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    return path


def read_family(glb: Path) -> dict | None:
    try:
        found = json.loads(family_path(glb).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) and found.get("family") in FAMILIES else None
