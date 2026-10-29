"""Six or more legs (a spider, a crab, an insect): traced from the body, not drawn.

A learned rigger draws a spider's legs only to the knees, and the centre lines tangle
where thin legs touch. So the body is taken to be its thick core, grown by its own depth
(that takes in the crowded roots of the legs); what is left, in connected pieces, is legs.
Each leg is the cheapest path from where its piece touches the body to its far end, a step
dearer near the skin so the path keeps to the middle of the tube. Joints go at the
strongest bends near a quarter, a half and three quarters of the leg: Coxa, Femur, Tibia,
Tarsus, numbered front to back, no knee or elbow names (each joint folds the way the
model already bends it).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .shell import FORWARD, LEFT, Shell, voxelize
from .skeleton import Bone, Bones, arc, place, resample, smooth

CORE = 0.5            # cells at least this share of the deepest depth are the body's core
GROW = 1.2            # the core grown by this x the deepest depth (in cells) is the body
PIECE_CELLS = 20      # quick test: a leg piece has more cells than this,
PIECE_LOW = 0.06      # comes this close to the floor (x height)
PIECE_SPAN = 0.25     # and spans more than this
QUICK = 5             # pieces like that needed before tracing
PER_PIECE = 4         # legs traced in one piece at most (legs touching each other)
APART = 0.12          # a next leg in a piece ends this far from the paths already taken
LEG_MIN = 0.30        # a leg is at least this long
THIN = 2.5            # legs thinner than this across (cells): trace again on half-size cells
ON_FLOOR = 0.08       # a leg ending this close to the floor stands on it
TRIGGER = 6           # legs on the floor that make a many-legged body
BENDS = (0.22, 0.5, 0.78)
SEGMENTS = ("Coxa", "Femur", "Tibia", "Tarsus")
_STEPS = [np.array(d) for d in np.argwhere(np.ones((3, 3, 3))) - 1 if tuple(d) > (0, 0, 0)]


def split_body(solid: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(body cells, leg pieces labelled 1..n, n). The growing is measured only in the
    core's box plus the growth: nothing outside it can be that close to the core, and a
    spider's legs spread its grid five times wider than its body."""
    deepest = float(depth.max())
    core = depth >= CORE * deepest
    reach = GROW * deepest
    idx = np.argwhere(core)
    lo = np.maximum(idx.min(0) - int(np.ceil(reach)) - 1, 0)
    hi = np.minimum(idx.max(0) + int(np.ceil(reach)) + 2, solid.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    body = np.zeros_like(solid)
    body[box] = (ndimage.distance_transform_edt(~core[box]) <= reach) & solid[box]
    labels, n = ndimage.label(solid & ~body, structure=np.ones((3, 3, 3)))
    return body, labels, n


def _touching(body: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Which cells have a body cell among their 26 neighbours."""
    out = np.zeros(len(cells), bool)
    shape = np.array(body.shape)
    for d in np.argwhere(np.ones((3, 3, 3))) - 1:
        j = cells + d
        ok = np.all((j >= 0) & (j < shape), axis=1)
        out[ok] |= body[tuple(j[ok].T)]
    return out


def leg_pieces(shell: Shell) -> int:
    """Quick test: pieces left after the body that are big, reach near the floor and are
    long. Five or more and the legs are worth tracing."""
    _, labels, _ = split_body(shell.solid, shell.depth)
    h = shell.height
    count = 0
    for k, sl in enumerate(ndimage.find_objects(labels)):
        if sl is None:
            continue
        idx = np.argwhere(labels[sl] == k + 1) + [s.start for s in sl]
        if len(idx) <= PIECE_CELLS:
            continue
        p = shell.lo + (idx + 0.5) * shell.size
        if p[:, 2].min() - shell.floor < PIECE_LOW * h and np.ptp(p, axis=0).max() > PIECE_SPAN * h:
            count += 1
    return count


def trace_piece(cells: np.ndarray, depth: np.ndarray, touches: np.ndarray, lo: np.ndarray, size: float,
                height: float) -> list[np.ndarray]:
    """Legs in one piece (its cells, their depth, which touch the body), each the points
    from the body out to the tip."""
    if len(cells) < 8 or not touches.any():
        return []
    origin = cells.min(0)
    local = cells - origin
    lookup = -np.ones(local.max(0) + 1, np.int64)
    lookup[tuple(local.T)] = np.arange(len(cells))
    rows, cols, w = [], [], []
    for d in _STEPS:
        j = local + d
        ok = np.all((j >= 0) & (j < np.array(lookup.shape)), axis=1)
        a = np.nonzero(ok)[0]
        b = lookup[tuple(j[ok].T)]
        a, b = a[b >= 0], b[b >= 0]
        rows.append(a)
        cols.append(b)
        w.append(float(np.linalg.norm(d)) * (1 + 3.0 / (0.5 * (depth[a] + depth[b]) + 0.5)))
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    graph = coo_matrix((np.concatenate([w, w]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                       shape=(len(cells), len(cells))).tocsr()
    dist, pred, _ = dijkstra(graph, indices=np.nonzero(touches)[0], min_only=True, return_predecessors=True)
    pos = lo + (cells + 0.5) * size
    legs, taken = [], np.zeros(len(cells), bool)
    for _ in range(PER_PIECE):
        cand = np.where(np.isfinite(dist) & ~taken, dist, -1.0)
        tip = int(np.argmax(cand))
        if cand[tip] <= 0:
            break
        path = [tip]
        while pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        line = pos[path[::-1]]
        if arc(line)[-1] < LEG_MIN * height:
            break
        legs.append(line)
        near = np.min(np.linalg.norm(pos[:, None] - line[None, ::4], axis=2), axis=1)
        taken |= near < APART * height
    return legs


def trace(solid: np.ndarray, lo: np.ndarray, size: float, height: float,
          depth: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray, float]:
    """(legs, body cell centres, the legs' thickness across in cells) on one grid. The
    thickness is measured along the traced centre lines (twice the depth there), not over
    every leg cell: most cells of a tube lie near its skin, so that median called a leg
    eight cells across thin and sent every spider to the eight-times-bigger grid."""
    if depth is None:
        depth = ndimage.distance_transform_edt(solid).astype(np.float32)
    body, labels, _ = split_body(solid, depth)
    legs = []
    for k, sl in enumerate(ndimage.find_objects(labels)):
        if sl is None:
            continue
        idx = np.argwhere(labels[sl] == k + 1) + [s.start for s in sl]
        legs += trace_piece(idx, depth[tuple(idx.T)], _touching(body, idx), lo, size, height)
    across = 9.0
    if legs:
        on_line = np.concatenate([np.floor((leg - lo) / size).astype(np.int64) for leg in legs])
        across = 2.0 * float(np.median(depth[tuple(on_line.T)]))
    return legs, lo + (np.argwhere(body) + 0.5) * size, across


def rig(shell: Shell) -> tuple[Bones, list[str], str] | None:
    """(bones, notes, facing) for a creature on six or more legs, else None."""
    h = shell.height
    if leg_pieces(shell) < QUICK:
        return None
    legs, body_pts, across = trace(shell.solid, shell.lo, shell.size, h, shell.depth)
    size, notes = shell.size, []
    stands = lambda found: [leg for leg in found if leg[-1, 2] - shell.floor < ON_FLOOR * h]
    if across < THIN or len(stands(legs)) < TRIGGER:   # legs 1-2 cells across merge and break apart
        solid, lo = voxelize(shell.meshes, shell.size / 2)
        legs, body_pts, across = trace(solid, lo, shell.size / 2, h)
        size = shell.size / 2
        notes.append("legs too thin for the grid: traced again on half-size cells")
    standing = stands(legs)
    if len(standing) < TRIGGER or not len(body_pts):
        return None
    fwd, left, facing = FORWARD.copy(), LEFT.copy(), "-Y"
    ahead = (np.mean([leg[0] for leg in legs], axis=0) - body_pts.mean(0)) @ fwd
    if abs(ahead) > 0.1 * np.ptp(body_pts @ fwd) and ahead < 0:
        fwd, left, facing = -fwd, -left, "+Y"
        notes.append("the legs leave the body behind its middle: it faces +Y")
    notes.insert(0, f"{len(standing)} legs on the floor ({len(legs)} traced): rigged from the body, "
                    "the drawn skeletons not used")

    s = body_pts @ fwd
    a, b = np.percentile(s, 1), np.percentile(s, 99)

    def slab(t: float) -> np.ndarray:
        sel = np.abs(s - (a + (b - a) * t)) < max(0.04 * (b - a), 2 * size)
        return body_pts[sel].mean(0) if sel.any() else body_pts.mean(0)
    p0, p1, p2, p3 = slab(0.03), slab(1 / 3), slab(2 / 3), slab(0.97)
    root = np.array([p1[0], p1[1], shell.floor])
    bones = [Bone("Root", None, root, root + [0.0, 0.0, 0.1 * h], "root", "", False),
             Bone("Hips", "Root", p1, p2, "spine"), Bone("Tail1", "Hips", p1, p0, "tail"),
             Bone("Head", "Hips", p2, p3, "head")]
    trunk = bones[1:]

    lines = [smooth(resample(leg, size), 1.5) for leg in legs]
    side = np.array([1 if (line[0] - [shell.mid, 0, 0]) @ left > 0 else -1 for line in lines])
    lefts, rights = list(np.nonzero(side > 0)[0]), list(np.nonzero(side < 0)[0])
    pairs: list[tuple[int | None, int | None]] = []
    if lefts and rights:     # twins by place, mirrored: one side may have a leg more
        cost = np.array([[np.linalg.norm(lines[i][0] - shell.mirror(lines[j][0]))
                          + np.linalg.norm(lines[i][-1] - shell.mirror(lines[j][-1])) for j in rights] for i in lefts])
        li, ri = linear_sum_assignment(cost)
        pairs = [(lefts[i], rights[j]) for i, j in zip(li, ri)]
    paired = {k for pair in pairs for k in pair}
    pairs += [(k, None) if side[k] > 0 else (None, k) for k in range(len(lines)) if k not in paired]
    pairs.sort(key=lambda pair: -np.mean([lines[k][0] @ fwd for k in pair if k is not None]))   # front first

    window = max(2, int(round(0.035 * h / size)))

    def shares(line: np.ndarray) -> np.ndarray:
        s_ = arc(line)
        return s_[place(line, BENDS, window)] / max(s_[-1], 1e-12)

    def at(line: np.ndarray, f: float) -> np.ndarray:
        s_ = arc(line)
        return np.array([np.interp(f * s_[-1], s_, line[:, c]) for c in range(3)])

    def nearest_trunk(q: np.ndarray) -> str:
        def d(bone: Bone) -> float:
            ab = bone.tail - bone.head
            t = np.clip((q - bone.head) @ ab / max(ab @ ab, 1e-12), 0, 1)
            return float(np.linalg.norm(q - (bone.head + t * ab)))
        return min(trunk, key=d).name

    for number, pair in enumerate(pairs, start=1):
        present = [k for k in pair if k is not None]
        fr = np.mean([shares(lines[k]) for k in present], axis=0)
        for k in present:
            line, sfx = lines[k], ".L" if side[k] > 0 else ".R"
            joints = [line[0]] + [at(line, f) for f in fr] + [line[-1]]
            parent = nearest_trunk(line[0])
            for seg, j0, j1 in zip(SEGMENTS, joints, joints[1:]):
                bones.append(Bone(f"{seg}{number}{sfx}", parent, j0, j1, "chain", sfx[1]))
                parent = f"{seg}{number}{sfx}"
    return bones, notes, facing
