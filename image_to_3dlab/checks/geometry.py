"""Mesh measurements the checks share, in plain numpy.

**Weld by position first.** glTF splits a vertex at every UV and normal seam, so counting
edges on the mesh as stored measures UV islands, not geometry. This is the same welding as
`scripts/mesh_health.py::weld_by_position` (quantised at 1e-6 of the asset's size), and
`tests/test_checks_geometry.py` holds the two to the same numbers so they cannot drift.
The package does not import `scripts/` (see `scripts/README.md`), hence the copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Welded:
    vertices: np.ndarray
    faces: np.ndarray


def weld(vertices: np.ndarray, faces: np.ndarray) -> Welded:
    """Merge coincident vertices by position and drop the faces that collapse."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return Welded(vertices.reshape(-1, 3), faces)
    scale = float(np.ptp(vertices, axis=0).max()) or 1.0
    quantised = np.round(vertices / (scale * 1e-6)).astype(np.int64)
    _, index, inverse = np.unique(quantised, axis=0, return_index=True, return_inverse=True)
    welded = inverse.reshape(-1)[faces]
    keep = (welded[:, 0] != welded[:, 1]) & (welded[:, 1] != welded[:, 2]) & (welded[:, 0] != welded[:, 2])
    return Welded(vertices[index], welded[keep])


@dataclass(frozen=True)
class Edges:
    """Every face corner's edge, mapped to a unique undirected edge."""

    unique: np.ndarray       # (E, 2) sorted vertex pairs
    counts: np.ndarray       # (E,) faces using each edge
    of_face: np.ndarray      # (F, 3) unique-edge index of each face's three edges
    forward: np.ndarray      # (F, 3) True where the face walks the edge low -> high


def edges(faces: np.ndarray) -> Edges:
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)  # (F,3,2)
    forward = directed[..., 0] < directed[..., 1]
    flat = np.sort(directed.reshape(-1, 2), axis=1)
    unique, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    return Edges(unique, counts, inverse.reshape(-1, 3), forward)


def boundary_shares(faces: np.ndarray) -> tuple[float, float]:
    """(percent of edges that are open, percent of faces touching an open edge).

    The second number is `scripts/ribbon_metric.py`'s tear measure: 0% is closed, 1-3% is
    a few tears, tens of percent is a mesh of ribbons that no repair will save.
    """
    if len(faces) == 0:
        return 0.0, 0.0
    table = edges(faces)
    open_edge = table.counts == 1
    touching = open_edge[table.of_face].any(axis=1)
    return 100.0 * float(open_edge.mean()), 100.0 * float(touching.mean())


def nonmanifold_share(faces: np.ndarray) -> tuple[int, float]:
    """(edges shared by three or more faces, as a count and as a percent of all edges)."""
    if len(faces) == 0:
        return 0, 0.0
    counts = edges(faces).counts
    bad = int((counts > 2).sum())
    return bad, 100.0 * bad / len(counts)


def _face_graph(table: Edges, n_faces: int, which: np.ndarray):
    """Sparse graph joining the two faces of every two-face edge selected by `which`."""
    slots = np.argsort(table.of_face.reshape(-1), kind="stable")
    edge_of_slot = table.of_face.reshape(-1)[slots]
    face_of_slot = slots // 3
    forward_of_slot = table.forward.reshape(-1)[slots]
    starts = np.searchsorted(edge_of_slot, np.arange(len(table.counts)))
    pair = np.flatnonzero((table.counts == 2) & which)
    a, b = starts[pair], starts[pair] + 1
    same_direction = forward_of_slot[a] == forward_of_slot[b]
    return face_of_slot[a], face_of_slot[b], same_direction


def components(faces: np.ndarray) -> np.ndarray:
    """Component label of every face; faces join when they share an edge."""
    n = len(faces)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    table = edges(faces)
    a, b, _ = _face_graph(table, n, np.ones(len(table.counts), dtype=bool))
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Each face's signed contribution to the enclosed volume (sum it for the total)."""
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    return np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0


def orientation(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float]:
    """(mask of faces wound against their neighbours, volume once they are turned back).

    Two faces sharing an edge agree when they walk it in opposite directions. Faces that
    agree form regions; regions are told apart across edges where they disagree. In each
    connected piece the smaller orientation is the flipped one. Counting faces, rather
    than disagreeing edges, is what makes a large flipped patch count as large: its border
    is short, its area is not.

    The volume is computed with the flipped faces turned back. If it is still negative,
    the whole surface points inward.
    """
    n = len(faces)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    table = edges(faces)
    a, b, same = _face_graph(table, n, np.ones(len(table.counts), dtype=bool))
    agree = coo_matrix((np.ones(int((~same).sum())), (a[~same], b[~same])), shape=(n, n))
    n_regions, region = connected_components(agree, directed=False)
    piece = components(faces)
    # Regions that touch across a disagreeing edge have opposite orientation.
    ra, rb = region[a[same]], region[b[same]]
    links = coo_matrix((np.ones(len(ra)), (ra, rb)), shape=(n_regions, n_regions)).tocsr()
    links = (links + links.T).tocsr()
    parity = np.full(n_regions, -1, dtype=np.int8)
    for start in range(n_regions):
        if parity[start] >= 0:
            continue
        parity[start] = 0
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in links.indices[links.indptr[current]:links.indptr[current + 1]]:
                if parity[neighbour] < 0:
                    parity[neighbour] = 1 - parity[current]
                    stack.append(neighbour)
    face_parity = parity[region].astype(np.int64)
    flipped = np.zeros(n, dtype=bool)
    for label in np.unique(piece):
        members = piece == label
        ones = int(face_parity[members].sum())
        minority = 1 if ones <= members.sum() - ones else 0
        flipped |= members & (face_parity == minority)
    sign = np.where(flipped, -1.0, 1.0)
    volume = float((signed_volume(vertices, faces) * sign).sum())
    return flipped, volume


def rasterize(points: np.ndarray, faces: np.ndarray, width: int, height: int,
              batch: int = 200_000) -> np.ndarray:
    """Union of 2D triangles as a boolean image (x right, y down, in pixels).

    Not `cv2.fillPoly` with many polygons: it fills by the even-odd rule, so where two
    triangles overlap they cancel, and a closed mesh (front and back over every pixel)
    comes out nearly empty. Here a pixel is set when its centre is inside any triangle,
    tested in bulk with numpy; the pixel under each triangle's centre is set too, so
    triangles smaller than a pixel still leave a mark. Triangles more than 8 pixels
    across are drawn one at a time with `cv2.fillConvexPoly`, which has no such rule.
    """
    import cv2

    out = np.zeros((height, width), dtype=bool)
    if len(faces) == 0:
        return out
    tri = np.asarray(points, dtype=np.float64)[np.asarray(faces)]
    centre = np.floor(tri.mean(axis=1)).astype(np.int64)
    inside = (centre[:, 0] >= 0) & (centre[:, 0] < width) & (centre[:, 1] >= 0) & (centre[:, 1] < height)
    out[centre[inside, 1], centre[inside, 0]] = True
    lo = np.ceil(tri.min(axis=1) - 0.5).astype(np.int64)
    hi = np.floor(tri.max(axis=1) - 0.5).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [width - 1, height - 1])
    span = (hi - lo).max(axis=1) + 1
    empty = (hi - lo).min(axis=1) < 0  # no pixel centre inside: the centre mark covers it
    large = (span > 8) & ~empty
    for corners, low, high in zip(tri[large], lo[large], hi[large]):
        canvas = np.zeros((high[1] - low[1] + 1, high[0] - low[0] + 1), dtype=np.uint8)
        local = np.round((corners - low - 0.5) * 16).astype(np.int32)  # cv2 puts pixel centres on integers
        cv2.fillConvexPoly(canvas, local, 1, lineType=cv2.LINE_8, shift=4)
        out[low[1]:high[1] + 1, low[0]:high[0] + 1] |= canvas.astype(bool)
    for size in (1, 2, 4, 8):
        chosen = np.flatnonzero(~large & ~empty & (span > (size // 2 if size > 1 else 0)) & (span <= size))
        if not len(chosen):
            continue
        dy, dx = np.mgrid[0:size, 0:size]
        offsets = np.stack([dx.ravel(), dy.ravel()], axis=1)
        for start in range(0, len(chosen), max(1, batch // (size * size))):
            pick = chosen[start:start + max(1, batch // (size * size))]
            px = lo[pick][:, None, :] + offsets[None]          # (n, k, 2) pixel indices
            centres = px + 0.5
            a, b, c = (tri[pick][:, i][:, None, :] for i in range(3))

            def side(p, q, r):
                return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

            e0, e1, e2 = side(a, b, centres), side(b, c, centres), side(c, a, centres)
            hit = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
            hit &= (px[..., 0] <= hi[pick][:, None, 0]) & (px[..., 1] <= hi[pick][:, None, 1])
            out[px[hit][:, 1], px[hit][:, 0]] = True
    return out


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of every triangle; works for 2D (UV) and 3D points."""
    p0, p1, p2 = (points[faces[:, i]] for i in range(3))
    if points.shape[1] == 2:
        return 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
