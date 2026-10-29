"""Skeletons as joint trees, named bones, and the centre lines of a voxel body.

A drawn skeleton is read as a tree of joints: one node per bone head, and one end node
(a tip) at the tail of every bone without children. Cleaning, scoring and the body
analysis all walk that tree; only the last step turns joints into named `Bone`s.

The centre-line skeleton is the same kind of tree, from the voxels thinned to their
middle lines: the fallback when no learned skeleton is available, and where a leg the
learned one missed comes from.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

SPUR = 0.05          # side spurs of the centre lines shorter than this (x height) are cut
TIP_BRANCH = 0.15    # short end branches at one fork (fingers, toes): only the longest stays
LIMB_KINDS = ("hind", "front", "biped", "arm", "wing", "chain")
SIDES = (".L", ".R")


# --- the joint tree ------------------------------------------------------------------------


@dataclass
class Tree:
    """Joints (n, 3) and each one's parent (-1 for the root). `tip[i]` marks a bone's end
    rather than a joint; `radius` is the body's thickness at a centre-line node."""

    points: np.ndarray
    parent: np.ndarray
    tip: np.ndarray
    radius: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, float).reshape(-1, 3)
        self.parent = np.asarray(self.parent, np.int64).reshape(-1)
        self.tip = np.asarray(self.tip, bool).reshape(-1)
        n = len(self.points)
        if len(self.parent) != n or len(self.tip) != n:
            raise ValueError("points, parents and tips must describe the same joints")
        if (self.parent < 0).sum() != 1:
            raise ValueError(f"a joint tree needs exactly one root, it has {(self.parent < 0).sum()}")

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def root(self) -> int:
        return int(np.nonzero(self.parent < 0)[0][0])

    def _cached(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def children(self) -> list[list[int]]:
        def make():
            kids: list[list[int]] = [[] for _ in range(self.n)]
            for i, p in enumerate(self.parent):
                if p >= 0:
                    kids[p].append(i)
            return kids
        return self._cached("children", make)

    def order(self) -> list[int]:
        """Parents before children (breadth first from the root)."""
        def make():
            kids, out, queue = self.children(), [], deque([self.root])
            while queue:
                i = queue.popleft()
                out.append(i)
                queue.extend(kids[i])
            return out
        return self._cached("order", make)

    def depth(self) -> np.ndarray:
        def make():
            d = np.zeros(self.n, np.int64)
            for i in self.order()[1:]:
                d[i] = d[self.parent[i]] + 1
            return d
        return self._cached("depth", make)

    def degree(self) -> np.ndarray:
        kids = self.children()
        return np.array([len(k) + (p >= 0) for k, p in zip(kids, self.parent)])

    def ends(self) -> list[int]:
        """Nodes with one neighbour: the ends of the body."""
        return [int(i) for i in np.nonzero(self.degree() == 1)[0]]

    def lengths(self) -> np.ndarray:
        """Each node's distance to its parent (0 for the root)."""
        p = np.where(self.parent >= 0, self.parent, np.arange(self.n))
        return np.linalg.norm(self.points - self.points[p], axis=1)

    def path(self, a: int, b: int) -> list[int]:
        """The nodes from a to b."""
        d = self.depth()
        up, down = [a], [b]
        while up[-1] != down[-1]:
            if d[up[-1]] >= d[down[-1]]:
                up.append(int(self.parent[up[-1]]))
            else:
                down.append(int(self.parent[down[-1]]))
        return up + down[-2::-1]

    def subtree(self, i: int) -> list[int]:
        kids, out, stack = self.children(), [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(kids[j])
        return out

    def path_length(self, nodes: list[int]) -> float:
        p = self.points[nodes]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0

    def joints(self) -> int:
        return int((~self.tip).sum())

    def copy(self) -> Tree:
        return Tree(self.points.copy(), self.parent.copy(), self.tip.copy(),
                    None if self.radius is None else self.radius.copy())

    def keep(self, mask: np.ndarray) -> Tree:
        """The tree without the nodes `mask` leaves out; a kept node hangs from its nearest
        kept ancestor. The root must stay."""
        mask = np.asarray(mask, bool)
        if not mask[self.root]:
            raise ValueError("the root cannot be removed")
        new = -np.ones(self.n, np.int64)
        new[mask] = np.arange(mask.sum())
        anchor = np.arange(self.n)
        for i in self.order()[1:]:
            if not mask[i]:
                anchor[i] = anchor[self.parent[i]]
        parent = np.array([new[anchor[self.parent[i]]] if self.parent[i] >= 0 else -1
                           for i in np.nonzero(mask)[0]], np.int64)
        return Tree(self.points[mask], parent, self.tip[mask], None if self.radius is None else self.radius[mask])

    def remove(self, nodes) -> Tree:
        mask = np.ones(self.n, bool)
        mask[list(nodes)] = False
        return self.keep(mask)

    def add(self, point, parent: int, tip: bool = True) -> tuple[Tree, int]:
        radius = None if self.radius is None else np.append(self.radius, 0.0)
        tree = Tree(np.vstack([self.points, np.asarray(point, float)]), np.append(self.parent, parent),
                    np.append(self.tip, tip), radius)
        return tree, self.n

    def rerooted(self, r: int) -> Tree:
        """The same joints hung from r."""
        kids = self.children()
        adj = [list(k) + ([int(p)] if p >= 0 else []) for k, p in zip(kids, self.parent)]
        parent = -np.ones(self.n, np.int64)
        seen = np.zeros(self.n, bool)
        seen[r] = True
        queue = deque([r])
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                if not seen[j]:
                    seen[j] = True
                    parent[j] = i
                    queue.append(j)
        return Tree(self.points.copy(), parent, self.tip.copy(), None if self.radius is None else self.radius.copy())

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """(starts, ends) of every bone: parent joint to child node."""
        child = np.nonzero(self.parent >= 0)[0]
        return self.points[self.parent[child]], self.points[child]

    def samples(self, per_bone: int = 11) -> np.ndarray:
        """Points spread along every bone, its two ends included."""
        a, b = self.segments()
        if not len(a):
            return self.points.copy()
        t = np.linspace(0, 1, per_bone)[None, :, None]
        return (a[:, None] + (b - a)[:, None] * t).reshape(-1, 3)

    @classmethod
    def from_bones(cls, parents, heads, tails) -> Tree:
        """A tree from bones: one joint per bone head, and a tip at the tail of every bone
        without children. Several roots: the one with most bones below it is kept."""
        parents = [int(p) for p in parents]
        heads = np.asarray(heads, float).reshape(-1, 3)
        tails = np.asarray(tails, float).reshape(-1, 3)
        n = len(parents)
        if n == 0 or len(heads) != n or len(tails) != n:
            raise ValueError("a draw needs as many heads, tails and parents as bones")
        parents = [p if 0 <= p < n and p != i else -1 for i, p in enumerate(parents)]
        kids: list[list[int]] = [[] for _ in range(n)]
        for i, p in enumerate(parents):
            if p >= 0:
                kids[p].append(i)

        def size(r):
            count, stack = 0, [r]
            while stack:
                j = stack.pop()
                count += 1
                stack.extend(kids[j])
            return count
        roots = [i for i, p in enumerate(parents) if p < 0]
        if not roots:
            raise ValueError("a draw needs a root bone")
        root = max(roots, key=size)
        keep, stack = [], [root]
        while stack:
            j = stack.pop()
            keep.append(j)
            stack.extend(kids[j])
        keep.sort()
        index = {b: k for k, b in enumerate(keep)}
        points = [heads[b] for b in keep]
        parent = [index[parents[b]] if parents[b] >= 0 else -1 for b in keep]
        tip = [False] * len(keep)
        for b in keep:
            if not kids[b]:
                points.append(tails[b])
                parent.append(index[b])
                tip.append(True)
        return cls(np.array(points), np.array(parent), np.array(tip))


# --- named bones ---------------------------------------------------------------------------


@dataclass
class Bone:
    """One bone of the result: `kind` is its role (hind, wing, spine, ...), `side` L, R or ''."""

    name: str
    parent: str | None
    head: np.ndarray
    tail: np.ndarray
    kind: str
    side: str = ""
    deform: bool = True

    def __post_init__(self) -> None:
        self.head = np.asarray(self.head, float).copy()
        self.tail = np.asarray(self.tail, float).copy()

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.tail - self.head))

    def samples(self, count: int) -> np.ndarray:
        return self.head + (self.tail - self.head) * np.linspace(0, 1, count)[:, None]

    def to_json(self) -> dict:
        return {"name": self.name, "parent": self.parent, "head": [round(float(v), 6) for v in self.head],
                "tail": [round(float(v), 6) for v in self.tail], "kind": self.kind, "side": self.side,
                "deform": bool(self.deform)}


Bones = list[Bone]


def side_of(name: str) -> str:
    return name[-1] if name[-2:] in SIDES else ""


def stem_of(name: str) -> str:
    return name[:-2] if name[-2:] in SIDES else name


def children_of(bones: Bones) -> dict[str | None, list[Bone]]:
    out: dict[str | None, list[Bone]] = {}
    for b in bones:
        out.setdefault(b.parent, []).append(b)
    return out


def below(bones: Bones, name: str) -> list[Bone]:
    """The bone called `name` and every bone under it, parents first."""
    kids = children_of(bones)
    by = {b.name: b for b in bones}
    out, queue = [], deque([by[name]])
    while queue:
        b = queue.popleft()
        out.append(b)
        queue.extend(kids.get(b.name, []))
    return out


def in_order(bones: Bones) -> Bones:
    """Parents before children, the original order otherwise (Blender builds them so)."""
    names = {b.name for b in bones}
    kids = children_of(bones)
    out, queue = [], deque(b for b in bones if b.parent is None or b.parent not in names)
    while queue:
        b = queue.popleft()
        out.append(b)
        queue.extend(kids.get(b.name, []))
    return out


def all_samples(bones: Bones, per_bone: int = 11) -> np.ndarray:
    if not bones:
        return np.zeros((0, 3))
    return np.concatenate([b.samples(per_bone) for b in bones])


# --- paths ---------------------------------------------------------------------------------


def arc(points: np.ndarray) -> np.ndarray:
    """Distance along a polyline at each of its points."""
    p = np.asarray(points, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def resample(points: np.ndarray, step: float) -> np.ndarray:
    """A polyline with its points `step` apart (both ends kept)."""
    p = np.asarray(points, float)
    s = arc(p)
    if len(p) < 2 or s[-1] < 1e-12:
        return p[:1].copy()
    t = np.linspace(0.0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], 1)


def smooth(points: np.ndarray, sigma: float) -> np.ndarray:
    """A polyline smoothed along itself, its two ends where they were."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return p.copy()
    out = ndimage.gaussian_filter1d(p, sigma, axis=0, mode="nearest")
    out[0], out[-1] = p[0], p[-1]
    return out


def bends(points: np.ndarray, window: int) -> np.ndarray:
    """The turn (degrees) at every point of an evenly spaced polyline, over +-window points."""
    p = np.asarray(points, float)
    out = np.zeros(len(p))
    if len(p) <= 2 * window:
        return out
    u = p[window:-window] - p[:-2 * window]
    v = p[2 * window:] - p[window:-window]
    c = (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
    out[window:-window] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def place(points: np.ndarray, fractions, window: int, reach: float = 0.12, min_bend: float = 12.0) -> list[int]:
    """Indices of joints along an evenly spaced polyline near the given fractions of its
    length: the strongest bend within +-reach of each, or the fraction itself where the
    path runs straight (no turn over `min_bend` degrees)."""
    p = np.asarray(points, float)
    s = arc(p)
    total = max(s[-1], 1e-12)
    turn = bends(p, window)
    used: list[int] = []
    for f in fractions:
        near = [i for i in np.nonzero(np.abs(s / total - f) <= reach)[0] if all(abs(i - u) > window for u in used)]
        if near and turn[near].max() > min_bend:
            i = int(near[int(np.argmax(turn[near]))])
        else:
            i = int(np.argmin(np.abs(s - f * total)))
        used.append(i)
    return used


# --- centre lines of the voxels ------------------------------------------------------------

_FORWARD_STEPS = [np.array(d) for d in np.argwhere(np.ones((3, 3, 3))) - 1 if tuple(d) > (0, 0, 0)]


def centre_graph(solid: np.ndarray):
    """(cells of the thinned body (k, 3), neighbour lists, edge lengths in cells): the
    largest connected piece of the voxels thinned to lines."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import (
        breadth_first_order,
        connected_components,
        minimum_spanning_tree,
    )
    from skimage.morphology import skeletonize

    thin = skeletonize(solid)
    idx = np.argwhere(thin)
    if len(idx) < 2:
        return idx, -np.ones(len(idx), np.int64)
    lookup = -np.ones(solid.shape, np.int64)
    lookup[tuple(idx.T)] = np.arange(len(idx))
    rows, cols, w = [], [], []
    shape = np.array(solid.shape)
    for d in _FORWARD_STEPS:
        j = idx + d
        ok = np.all((j >= 0) & (j < shape), axis=1)
        a = np.nonzero(ok)[0]
        b = lookup[tuple(j[ok].T)]
        hit = b >= 0
        rows.append(a[hit])
        cols.append(b[hit])
        w.append(np.full(hit.sum(), float(np.linalg.norm(d))))
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    graph = coo_matrix((w, (rows, cols)), shape=(len(idx), len(idx))).tocsr()
    _, label = connected_components(graph, directed=False)
    big = np.argmax(np.bincount(label))
    keep = np.nonzero(label == big)[0]
    sub = graph[keep][:, keep]
    tree = minimum_spanning_tree(sub + sub.T)
    _, pred = breadth_first_order(tree, 0, directed=False, return_predecessors=True)
    parent = np.where(pred >= 0, pred, -1).astype(np.int64)
    return idx[keep], parent


def _adjacency(parent: np.ndarray) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(len(parent))]
    for i, p in enumerate(parent):
        if p >= 0:
            adj[i].add(int(p))
            adj[int(p)].add(i)
    return adj


def _end_branches(points: np.ndarray, adj: list[set[int]], alive: np.ndarray):
    """Every end's branch: (end, nodes from the end up to before the fork, fork or -1,
    length to the fork)."""
    out = []
    for leaf in np.nonzero(alive)[0]:
        if len(adj[leaf]) != 1:
            continue
        path, prev, n, length = [int(leaf)], -1, int(leaf), 0.0
        while True:
            nxt = [m for m in adj[n] if m != prev]
            if len(nxt) != 1:
                break
            length += float(np.linalg.norm(points[nxt[0]] - points[n]))
            prev, n = n, nxt[0]
            if len(adj[n]) != 2:
                break
            path.append(n)
        out.append((int(leaf), path, n if len(adj[n]) >= 3 else -1, length))
    return out


def _cut(adj: list[set[int]], alive: np.ndarray, nodes) -> None:
    for n in nodes:
        for m in adj[n]:
            adj[m].discard(n)
        adj[n] = set()
        alive[n] = False


def prune(points: np.ndarray, adj: list[set[int]], alive: np.ndarray, min_len: float) -> None:
    """Side spurs shorter than `min_len` cut, again until none is left (cutting one can
    make another)."""
    changed = True
    while changed:
        changed = False
        for _, path, fork, length in _end_branches(points, adj, alive):
            if fork >= 0 and length < min_len and alive[path[0]] and len(adj[fork]) >= 3:
                _cut(adj, alive, path)
                changed = True


def collapse_tips(points: np.ndarray, adj: list[set[int]], alive: np.ndarray, max_len: float) -> None:
    """Where two or more short end branches leave one fork that has at most one other way
    on (a hand, a foot), only the longest stays: a hand is one limb end, not four limbs."""
    changed = True
    while changed:
        changed = False
        by_fork: dict[int, list] = {}
        for _, path, fork, length in _end_branches(points, adj, alive):
            if fork >= 0 and length < max_len:
                by_fork.setdefault(fork, []).append((length, path))
        for fork, short in by_fork.items():
            if len(short) >= 2 and len(adj[fork]) - len(short) <= 1:
                short.sort(key=lambda b: -b[0])
                for _, path in short[1:]:
                    _cut(adj, alive, path)
                changed = True


def centre_line(shell) -> Tree:
    """The centre-line tree of the shell's voxels: thinned, side spurs under 5% of the
    height cut, a fork's short end branches (under 15%) down to the longest, spurs cut
    again. Rooted at its deepest node; `radius` is the body's thickness there."""
    idx, parent = centre_graph(shell.solid)
    if len(idx) < 2:
        raise ValueError("the body is too small to thin to centre lines")
    points = shell.lo + (idx + 0.5) * shell.size
    radius = shell.depth[tuple(idx.T)].astype(float) * shell.size
    adj = _adjacency(parent)
    alive = np.ones(len(idx), bool)
    h = shell.height
    prune(points, adj, alive, SPUR * h)
    collapse_tips(points, adj, alive, TIP_BRANCH * h)
    prune(points, adj, alive, SPUR * h)
    return _tree_from(points, radius, adj, alive)


def _tree_from(points: np.ndarray, radius: np.ndarray, adj: list[set[int]], alive: np.ndarray) -> Tree:
    live = np.nonzero(alive)[0]
    root = int(live[np.argmax(radius[live])])
    new = -np.ones(len(points), np.int64)
    order, parent_of = [root], {root: -1}
    queue = deque([root])
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if j not in parent_of:
                parent_of[j] = i
                order.append(j)
                queue.append(j)
    new[order] = np.arange(len(order))
    parent = np.array([new[parent_of[i]] if parent_of[i] >= 0 else -1 for i in order], np.int64)
    tree = Tree(points[order], parent, np.zeros(len(order), bool), radius[order])
    tree.tip = tree.degree() == 1
    tree.tip[tree.root] = False
    return tree
