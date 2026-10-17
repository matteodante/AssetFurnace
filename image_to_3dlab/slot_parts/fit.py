"""The pure half of fitting a piece onto the base body; `scripts/blender_fit_parts.py` does the rest.

Everything here works in Blender's frame (Z up, the camera on the -Y side looking +Y),
because that is where the fitting happens; glTF's frame (Y up, camera on +Z) maps to it by
blender = (x, -z, y).

* **Placement**: a piece's box on the concept, carried into the body picture and back
  through the body's camera at the depth of the bones it sits on, gives its size and
  place. A front view says nothing about depth, so a piece made on its own is stretched in
  depth to cover the body under it, plus a margin.
* **Paired pieces fit per side**: the left one is placed from the right one's box mirrored
  about the body's midline, then each is centred on its own limb. Mirroring the fitted
  right piece was the worst in the trial (a shoulder at 6% clipping).
* **Limbs**: a gauntlet's long axis turns onto the forearm and hand bones.
* **Spikes**: a cage fit can fling a few vertices far out (a thin spike from the torso past
  a shoulder, in the trial and here). A vertex that moved several times further than its
  neighbours, or sits on an edge the fit stretched to many times the usual length, goes
  back to the mean of its sound neighbours.
"""

from __future__ import annotations

import numpy as np

from image_to_3dlab.slot_parts.camera import Camera

GLTF_TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def to_blender(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, float) @ GLTF_TO_BLENDER.T


def unproject(cam: Camera, u: float, v: float, depth: float) -> tuple[float, float]:
    """Body-picture pixel (u, v) to Blender (x, z) at Blender depth y."""
    U = (u - cam.offset[0]) / cam.scale
    V = (v - cam.offset[1]) / cam.scale
    f = (cam.size / 2) / np.tan(cam.fov / 2)
    d = cam.dist + depth  # glTF z = -y; the camera sits at glTF z = dist
    return float((U - cam.size / 2) * d / f), float(-(V - cam.size / 2) * d / f)


def midline_px(cam: Camera, x: float) -> float:
    """Image x, in the body picture, of the vertical line through Blender x at depth 0."""
    f = (cam.size / 2) / np.tan(cam.fov / 2)
    return float((cam.size / 2 + f * x / cam.dist) * cam.scale + cam.offset[0])


def mirror_box(box, centre_px: float) -> list[float]:
    """A box mirrored about the vertical line x = centre_px."""
    x0, y0, x1, y1 = box
    return [2 * centre_px - x1, y0, 2 * centre_px - x0, y1]


def concept_to_body(box, concept_size, body_size) -> list[float]:
    """A box in concept pixels, in body-picture pixels (the concept keeps the body's framing)."""
    kx, ky = body_size[0] / concept_size[0], body_size[1] / concept_size[1]
    return [box[0] * kx, box[1] * ky, box[2] * kx, box[3] * ky]


def placement(points: np.ndarray, box_px, cam: Camera, depth: float, body_depth: float | None = None,
              frame=None, margin: float = 0.02) -> tuple[np.ndarray, dict]:
    """The 4x4 that puts a piece (Blender points) into `box_px` of the body picture at `depth`.

    `frame` = (x0, z0, x1, z1, y_centre) places a piece cut from a whole-character model by
    that model's extent instead of its own (route B0). `body_depth` is the front-to-back
    size of the body under the piece; given (and no frame), a thinner piece is stretched in
    depth to cover it plus `margin`.
    """
    p = np.asarray(points, float)
    x0, y0, x1, y1 = box_px
    wx0, wz1 = unproject(cam, x0, y0, depth)
    wx1, wz0 = unproject(cam, x1, y1, depth)
    if frame is not None:
        mx0, mz0, mx1, mz1, yc = frame
    else:
        mx0, mz0, mx1, mz1 = p[:, 0].min(), p[:, 2].min(), p[:, 0].max(), p[:, 2].max()
        yc = (p[:, 1].min() + p[:, 1].max()) / 2
    sx, sz = (wx1 - wx0) / max(mx1 - mx0, 1e-9), (wz1 - wz0) / max(mz1 - mz0, 1e-9)
    k = (sx + sz) / 2
    ky = k
    if frame is None and body_depth is not None:
        ky = max(k, (body_depth + margin) / max(p[:, 1].max() - p[:, 1].min(), 1e-9))
    m = np.diag([k, ky, k, 1.0])
    m[:3, 3] = [(wx0 + wx1) / 2 - k * (mx0 + mx1) / 2, depth - ky * yc, (wz0 + wz1) / 2 - k * (mz0 + mz1) / 2]
    return m, {"scale": float(k), "depth_scale": float(ky), "aspect_mismatch": float(sx / max(sz, 1e-9))}


def cage_sizes(base: float) -> list[float]:
    """Voxel sizes to try for the cage, the base first: Surface Deform's bind depends on small
    details of the cage's topology, so a near size often binds where one did not."""
    return [round(base * k, 6) for k in (1.0, 1.1, 0.9, 1.3, 0.75, 1.6)]


# How big a piece made on its own is, as a share of the body's height, and where it is held:
# a weapon by its grip (a fifth of its height up from the bottom), a shield by its middle.
HELD = {"weapon": (0.42, 0.2), "shield": (0.30, 0.5)}


def held_placement(points: np.ndarray, slot: str, bone_head: np.ndarray, body_height: float) -> tuple[np.ndarray, dict]:
    """4x4 putting a weapon or shield (upright, as generated) in the hand or on the forearm."""
    share, grip_at = HELD[slot]
    p = np.asarray(points, float)
    lo, hi = p.min(0), p.max(0)
    k = share * body_height / max(float((hi - lo).max()), 1e-9)
    grip = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + grip_at * (hi[2] - lo[2])])
    m = np.diag([k, k, k, 1.0])
    m[:3, 3] = np.asarray(bone_head, float) - k * grip
    return m, {"scale": float(k), "held_at": [round(float(v), 4) for v in bone_head]}


def enclose(points: np.ndarray, region: np.ndarray, margin: float = 0.15) -> tuple[np.ndarray, float]:
    """4x4 growing a rigid piece (a helmet) until it holds the body region under it with a margin.

    A front picture gives a helmet's outline, not the room inside it: placed to the outline,
    a solid helmet came out the size of the head and the face showed through its visor. The
    piece is scaled up uniformly, never down, and centred on the region.
    """
    p, r = np.asarray(points, float), np.asarray(region, float)
    need = (r.max(0) - r.min(0)) * (1 + margin)
    have = np.maximum(p.max(0) - p.min(0), 1e-9)
    k = max(1.0, float((need / have).max()))
    m = np.diag([k, k, k, 1.0])
    m[:3, 3] = (r.min(0) + r.max(0)) / 2 - k * (p.min(0) + p.max(0)) / 2
    return m, k


def region_transfer(source: tuple, target: tuple) -> np.ndarray:
    """4x4 moving a piece fitted on one body onto another: the source region's box (lo, hi) onto
    the target region's, centre onto centre and each axis scaled by the ratio of the sizes.

    The regions are the body vertices the slot's bones carry, so a longer forearm or broader
    shoulders stretch the piece along with them; the cage fit that follows does the rest.
    """
    (slo, shi), (tlo, thi) = (np.asarray(source[0], float), np.asarray(source[1], float)), \
        (np.asarray(target[0], float), np.asarray(target[1], float))
    scale = (thi - tlo) / np.maximum(shi - slo, 1e-9)
    m = np.diag([*scale, 1.0])
    m[:3, 3] = (tlo + thi) / 2 - scale * (slo + shi) / 2
    return m


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The 3x3 rotation turning unit vector a onto unit vector b (Rodrigues)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        other = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, other)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def turn_onto(points: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float]:
    """4x4 turning the points' long axis (sign towards `axis`) onto `axis`, about their centre."""
    p = np.asarray(points, float)
    c = p.mean(0)
    main = np.linalg.svd(p - c, full_matrices=False)[2][0]
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    if main @ axis < 0:
        main = -main
    r = rotation_between(main, axis)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = c - r @ c
    return m, float(np.degrees(np.arccos(np.clip(main @ axis, -1, 1))))


def centre_across(centre: np.ndarray, target: np.ndarray, axis: np.ndarray | None = None) -> np.ndarray:
    """The shift moving `centre` onto `target`, leaving out any part along `axis`."""
    d = np.asarray(target, float) - np.asarray(centre, float)
    if axis is not None:
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        d = d - (d @ a) * a
    return d


def neighbours(edges: np.ndarray, count: int) -> list[np.ndarray]:
    links: list[list[int]] = [[] for _ in range(count)]
    for a, b in np.asarray(edges, int):
        links[a].append(b)
        links[b].append(a)
    return [np.array(x, int) for x in links]


def spikes(rest: np.ndarray, fitted: np.ndarray, edges: np.ndarray, factor: float = 4.0,
           floor: float = 0.01) -> np.ndarray:
    """Vertices that moved more than `factor` times their neighbours' median move, and more than `floor`."""
    moved = np.linalg.norm(np.asarray(fitted, float) - np.asarray(rest, float), axis=1)
    links = neighbours(edges, len(moved))
    near = np.array([np.median(moved[n]) if len(n) else m for n, m in zip(links, moved)])
    return (moved > factor * np.maximum(near, 1e-9)) & (moved > floor)


def stretched(rest: np.ndarray, fitted: np.ndarray, edges: np.ndarray, stretch: float = 4.0,
              long: float = 5.0) -> np.ndarray:
    """Vertices on edges the fit stretched `stretch` times and that end up `long` times the median
    edge: a spike drawn out as a strip of vertices, which `spikes` misses because its
    neighbours moved as far as it did."""
    e = np.asarray(edges, int)
    if not len(e):
        return np.zeros(len(fitted), bool)
    before = np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=1)
    after = np.linalg.norm(fitted[e[:, 0]] - fitted[e[:, 1]], axis=1)
    bad = (after > stretch * np.maximum(before, 1e-9)) & (after > long * np.median(after))
    out = np.zeros(len(fitted), bool)
    out[e[bad].ravel()] = True
    return out


def smooth_spikes(fitted: np.ndarray, edges: np.ndarray, bad: np.ndarray, rounds: int = 20) -> np.ndarray:
    """Spike vertices moved to the mean of their sound neighbours, a few rounds out from the edge."""
    out = np.asarray(fitted, float).copy()
    bad = np.asarray(bad, bool).copy()
    links = neighbours(edges, len(out))
    for _ in range(rounds):
        if not bad.any():
            break
        fixed = []
        for i in np.nonzero(bad)[0]:
            good = links[i][~bad[links[i]]] if len(links[i]) else links[i]
            if len(good):
                out[i] = out[good].mean(0)
                fixed.append(i)
        bad[fixed] = False
    return out


def smooth_push(points: np.ndarray, edges: np.ndarray, need: np.ndarray, rounds: int = 80,
                keep: float = 0.96) -> np.ndarray:
    """The piece moved off the skin as one smooth surface: every vertex with a `need` (how far,
    and which way, it must go to clear the skin) goes at least that far, and its neighbours
    follow, less and less with distance, so a dent becomes a gentle bulge instead of a
    crumple. A cage shrinkwrapped onto the body tore a breastplate into shards: each cage
    vertex jumped to its own nearest skin point.
    """
    points = np.asarray(points, float)
    need = np.asarray(need, float)
    edges = np.asarray(edges, int).reshape(-1, 2)
    length = np.linalg.norm(need, axis=1)
    hard = length > 0
    if not hard.any() or not len(edges):
        return points + need
    unit = np.zeros_like(need)
    unit[hard] = need[hard] / length[hard, None]
    degree = np.maximum(np.bincount(edges.ravel(), minlength=len(points)), 1)[:, None]
    shift = need.copy()
    for _ in range(rounds):
        total = np.zeros_like(shift)
        np.add.at(total, edges[:, 0], shift[edges[:, 1]])
        np.add.at(total, edges[:, 1], shift[edges[:, 0]])
        spread = keep * total / degree
        along = np.einsum("ij,ij->i", spread, unit)
        spread[hard] += np.maximum(0.0, length[hard] - along[hard])[:, None] * unit[hard]
        shift = spread
    return points + shift


def inward_islands(vertices: np.ndarray, triangles: np.ndarray) -> list[np.ndarray]:
    """The triangles of each closed island whose signed volume is negative: it faces inward.

    A cage fit can fold a small closed part of a piece (a strap, a rivet cut out of a whole
    model) through itself, and a refit can do it on one body and not the other. An island
    with open edges says nothing by its volume, so it is left alone.
    """
    triangles = np.asarray(triangles, int)
    if not len(triangles):
        return []
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(triangles)), 3)
    _, edge_id, uses = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_id = edge_id.reshape(-1)
    # union-find over triangles sharing an edge
    parent = np.arange(len(triangles))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(edge_id, kind="stable")
    for a, b in zip(order[:-1], order[1:]):
        if edge_id[a] == edge_id[b]:
            ra, rb = find(owner[a]), find(owner[b])
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(len(triangles))])
    open_edge = uses[edge_id] == 1  # as mesh.flipped_faces reads it: an edge with three faces is not open
    vertices = np.asarray(vertices, float)
    v = vertices[triangles]
    volume = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])) / 6.0
    out = []
    for root in np.unique(roots):
        faces = np.nonzero(roots == root)[0]
        if open_edge[np.isin(owner, faces)].any() or volume[faces].sum() >= 0:
            continue
        # an inward shell inside the rest of the piece is a cavity's wall, which is right
        others = np.setdiff1d(np.arange(len(triangles)), faces)
        if len(others) and np.median(winding_number(vertices[triangles[faces[:3], 0]], vertices[triangles[others]])) > 0.5:
            continue
        out.append(faces)
    return out


def winding_number(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """How many times the triangles (n, 3, 3) wrap each point: about 1 inside a closed outward surface."""
    total = np.zeros(len(points))
    for i, p in enumerate(np.asarray(points, float)):
        a, b, c = (triangles[:, k] - p for k in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        num = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la             + np.einsum("ij,ij->i", c, a) * lb
        total[i] = np.arctan2(num, den).sum() / (2 * np.pi)
    return total


def pose_frames(start: int, end: int, count: int = 8) -> list[int]:
    """`count` frames spread evenly over a clip, its first and last included."""
    if end <= start or count <= 1:
        return [start]
    return sorted({round(f) for f in np.linspace(start, end, count)})


def coverage_report(per_frame: list[dict]) -> dict:
    """Worst and mean over frames of {piece: share} measures, and the worst frame's index."""
    names = sorted({n for f in per_frame for n in f["clipping"]})
    clip = {n: {"max": max(f["clipping"].get(n, 0.0) for f in per_frame),
                "mean": float(np.mean([f["clipping"].get(n, 0.0) for f in per_frame]))} for n in names}
    holes = [f["holes"] for f in per_frame]
    worst = int(np.argmax([f["holes"] + sum(f["clipping"].values()) for f in per_frame]))
    return {"clipping": clip, "holes": {"max": max(holes), "mean": float(np.mean(holes))}, "worst_frame": worst}
