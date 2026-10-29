"""Clean a drawn skeleton, score it on the mesh, keep the best of several.

The learned rigger draws a different skeleton each run, and none is right everywhere. So
each draw is first cleaned of what is plainly wrong (a pelvis stub above the root,
fingers too small to read in a video, a branch lying outside the body, ends poking out of
the skin), then scored on what the mesh shows for sure: every place the creature stands
has a bone reaching it, the bones mirror each other where the body does, and no large part
of the body is far from every bone. Lowest score wins; on a tie the earlier draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .shell import Shell
from .skeleton import Tree

DIGIT = 0.22           # every branch at a joint ending this close (x height, along the bones): fold them
OUTSIDE = 2.5          # a branch point this far off the skin (in cells) is outside ...
OUTSIDE_SHARE = 0.5    # ... and a branch with more than this share of them goes
SKIN = 1.5             # an end this close to the skin (in cells) is in
PULL = np.linspace(1.0, 0.3, 15)   # where along its bone an end may be pulled back to
GROW = 0.3             # a joint left without an end gets one this much further along its parent bone
FAR = 0.15             # body this far from every bone (x the longest side) ...
FAR_SHARE = 0.33       # ... over this share of it: the draw left part of the body without bones
MISS, GAP, LOOSE = 100.0, 100.0, 100.0   # score per missed contact, per height of mirror gap, for a far share


def read_draw(path: Path) -> Tree:
    """A draw file (`names`, `parents`, `heads`, `tails`, model frame) as a joint tree."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Tree.from_bones(d["parents"], d["heads"], d["tails"])


# --- cleaning ----------------------------------------------------------------------------


def drop_root_stubs(tree: Tree) -> Tree:
    """A root with a single child is no end of the body (a pelvis stub): it goes, again
    until the root forks."""
    while True:
        kids = tree.children()[tree.root]
        if tree.tip[tree.root] or len(kids) != 1 or tree.tip[kids[0]]:
            return tree
        child = kids[0]
        tree = tree.rerooted(child).remove([tree.root])


def fold_digits(tree: Tree, height: float) -> tuple[Tree, int]:
    """Where every branch leaving a joint ends within DIGIT x height of it (fingers at a
    wrist, toes at an ankle, ears and a jaw at a head), the branches go and the joint gets
    one end at the mean of their farthest points. Returns (tree, branches folded)."""
    folded = 0
    while True:
        kids, order, length = tree.children(), tree.order(), tree.lengths()
        reach = np.zeros(tree.n)
        far = np.arange(tree.n)
        for i in reversed(order):
            for c in kids[i]:
                if length[c] + reach[c] > reach[i] or far[i] == i:
                    reach[i], far[i] = length[c] + reach[c], far[c]
        short = np.array([not tree.tip[i] and i != tree.root and len(kids[i]) >= 2
                          and all(length[c] + reach[c] < DIGIT * height for c in kids[i]) for i in range(tree.n)])
        if not short.any():
            return tree, folded
        forks, blocked = [], np.zeros(tree.n, bool)
        for i in order:                    # the joint nearest the root folds; those under it go with it
            if tree.parent[i] >= 0 and blocked[tree.parent[i]]:
                blocked[i] = True
            elif short[i]:
                forks.append(i)
                blocked[i] = True
        keep = np.ones(tree.n, bool)
        ends = {}
        for f in forks:
            ends[f] = tree.points[[far[c] for c in kids[f]]].mean(0)
            folded += len(kids[f])
            for c in kids[f]:
                keep[tree.subtree(c)] = False
        new_index = np.cumsum(keep) - 1
        tree = tree.keep(keep)
        for f, end in ends.items():
            tree, _ = tree.add(end, int(new_index[f]), tip=True)


def _branches(tree: Tree) -> list[list[int]]:
    """Every end's branch: the end, up to the first joint where it forks (or the root)."""
    kids = tree.children()
    out = []
    for t in np.nonzero(tree.tip)[0]:
        path, n = [int(t)], int(t)
        while tree.parent[n] >= 0:
            n = int(tree.parent[n])
            path.append(n)
            if len(kids[n]) != 1 or n == tree.root:
                break
        out.append(path)
    return out


def drop_outside(tree: Tree, shell: Shell) -> tuple[Tree, int]:
    """Branches lying mostly outside the skin go (a stray bone out of the back). Each is
    judged as drawn, so two bad branches at one fork both go."""
    branches = [b for b in _branches(tree) if len(b) >= 2]
    if not branches:
        return tree, 0
    t = np.linspace(0, 1, 6)[None, :, None]
    samples = []
    for b in branches:
        p = tree.points[b]
        samples.append((p[:-1, None] + (p[1:] - p[:-1])[:, None] * t).reshape(-1, 3))
    off = shell.off_skin(np.concatenate(samples))
    bad, start = [], 0
    for b, s in zip(branches, samples):
        if (off[start:start + len(s)] > OUTSIDE * shell.size).mean() > OUTSIDE_SHARE:
            bad.append(b)
        start += len(s)
    if not bad or len(bad) == len(branches):
        return tree, 0
    gone = {n for b in bad for n in b[:-1]}
    return tree.remove(gone), len(bad)


def add_missing_ends(tree: Tree) -> Tree:
    """A joint left without an end (its branches dropped) gets one GROW further along its
    parent bone's direction."""
    kids = tree.children()
    bare = [i for i in range(tree.n) if not tree.tip[i] and not kids[i] and tree.parent[i] >= 0]
    for i in bare:
        a, b = tree.points[tree.parent[i]], tree.points[i]
        tree, _ = tree.add(b + (b - a) * GROW, i, tip=True)
    return tree


def pull_in(tree: Tree, shell: Shell) -> Tree:
    """Ends slid back along their bone to the first point within SKIN cells of the body
    (at most to 30% of the bone), never below the floor."""
    tree = tree.copy()
    tips = np.nonzero(tree.tip & (tree.parent >= 0))[0]
    if not len(tips):
        return tree
    a = tree.points[tree.parent[tips]]
    b = tree.points[tips]
    tries = a[:, None] + (b - a)[:, None] * PULL[None, :, None]
    ok = (shell.off_skin(tries.reshape(-1, 3)) <= SKIN * shell.size).reshape(len(tips), len(PULL))
    first = np.where(ok.any(1), ok.argmax(1), len(PULL) - 1)
    new = tries[np.arange(len(tips)), first]
    new[:, 2] = np.maximum(new[:, 2], shell.floor)
    tree.points[tips] = new
    return tree


def clean(tree: Tree, shell: Shell) -> tuple[Tree, dict]:
    """All the cleaning, in order. Returns (tree, {"folded": n, "dropped": n})."""
    tree = drop_root_stubs(tree)
    tree, folded = fold_digits(tree, shell.height)
    tree, dropped = drop_outside(tree, shell)
    tree = add_missing_ends(tree)
    return pull_in(tree, shell), {"folded": folded, "dropped": dropped}


# --- scoring -----------------------------------------------------------------------------


def mirror_gap(tree: Tree, shell: Shell) -> float:
    """Every joint and end mirrored across the middle plane, its distance to the nearest
    joint: the 90th percentile, as a share of the height."""
    d = cKDTree(tree.points).query(shell.mirror(tree.points))[0]
    return float(np.percentile(d, 90)) / shell.height


def far_share(tree: Tree, shell: Shell) -> float:
    """Share of the body (every 4th cell) farther than FAR x its longest side from every
    bone. Of a robed figure's four draws one had no bone below the waist: 45%, the others
    8-10%."""
    d = cKDTree(tree.samples(11)).query(shell.cells[::4])[0]
    return float((d > FAR * shell.longest).mean())


def score(tree: Tree, shell: Shell) -> dict:
    """The score of a cleaned draw and what it is made of."""
    missed = shell.unreached(tree.samples(11))
    gap = mirror_gap(tree, shell)
    far = far_share(tree, shell)
    total = MISS * len(missed) + GAP * gap + (LOOSE if far > FAR_SHARE else 0.0)
    return {"score": round(total, 4), "missed_contacts": len(missed), "contacts": len(shell.contacts),
            "mirror_gap": round(gap, 4), "far_share": round(far, 4), "joints": tree.joints(), "missed": missed}


def best(scores: list[dict | None]) -> int | None:
    """The draw to keep: the lowest score, the earlier on a tie; None when none is usable."""
    usable = [(s["score"], k) for k, s in enumerate(scores) if s is not None and s.get("score") is not None]
    return min(usable)[1] if usable else None
