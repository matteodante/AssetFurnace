"""Conservative seam candidates from opposed, unique boundary edges.

Equal positions alone never weld vertices. This map is local to one primitive;
ambiguous overlaps are reported, not resolved by nearest-neighbour guesses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class SeamMap:
    render_to_geometric: np.ndarray
    representatives: np.ndarray
    paired_edges: int
    ambiguous_vertices: np.ndarray

    def as_dict(self):
        return {"algorithm": "unique-opposed-boundary-v1", "paired_edges": self.paired_edges,
                "geometric_vertices": len(self.representatives),
                "ambiguous_vertices": self.ambiguous_vertices.tolist()}


def seam_map(positions, triangles, joints=None, weights=None):
    positions = np.asarray(positions, dtype=np.float64)
    triangles = np.asarray(triangles)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu"
            or (triangles < 0).any() or (triangles >= len(positions)).any()):
        raise ValueError("invalid seam input")
    if (joints is None) != (weights is None):
        raise ValueError("joints and weights must be provided together")
    if joints is not None:
        joints, weights = np.asarray(joints), np.asarray(weights)
        if joints.shape != (len(positions), 4) or weights.shape != joints.shape:
            raise ValueError("invalid skin arrays for seam comparison")
    parents = np.arange(len(positions), dtype=np.int64)

    def root(vertex):
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex

    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)

    def same_weights(a, b):
        if joints is None:
            return True
        def vector(vertex):
            result = defaultdict(float)
            for joint, weight in zip(joints[vertex], weights[vertex]):
                if weight > 0:
                    result[int(joint)] += float(weight)
            return result
        left, right = vector(a), vector(b)
        return all(abs(left[j] - right[j]) < 1e-5 for j in left.keys() | right.keys())

    # Face orientation matters: overlapping front/back layers must stay separate.
    points = positions[triangles]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    length = np.linalg.norm(normals, axis=1)
    valid = length > 1e-15
    normals[valid] /= length[valid, None]
    edges = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    sorted_edges = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
    boundary = np.flatnonzero(counts[inverse] == 1)
    candidates = defaultdict(list)
    for edge_index in boundary:
        a, b = map(int, edges[edge_index])
        pa, pb = tuple(positions[a]), tuple(positions[b])
        if pa == pb:
            continue
        candidates[tuple(sorted((pa, pb)))].append((a, b, edge_index // 3, pa, pb))
    ambiguous, paired = set(), 0
    for group in candidates.values():
        if len(group) == 1:
            continue  # Native open boundary, not a seam to infer.
        if len(group) != 2:
            ambiguous.update(v for edge in group for v in edge[:2])
            continue
        a, b, fa, pa, pb = group[0]
        c, d, fb, pc, pd = group[1]
        compatible = (pa == pd and pb == pc and fa != fb and valid[fa] and valid[fb]
                      and normals[fa] @ normals[fb] > -0.25
                      and same_weights(a, d) and same_weights(b, c))
        if not compatible:
            ambiguous.update((a, b, c, d))
            continue
        union(a, d)
        union(b, c)
        paired += 1
    # A vertex participating in any ambiguous edge is not welded indirectly by a
    # different edge at the same corner. Protect the entire candidate component.
    ambiguous_roots = {root(v) for v in ambiguous}
    roots = np.array([root(i) for i in range(len(positions))])
    blocked = np.flatnonzero(np.isin(roots, list(ambiguous_roots)))
    roots[blocked] = blocked
    representatives, mapping = np.unique(roots, return_inverse=True)
    return SeamMap(mapping.astype(np.uint32), representatives.astype(np.uint32), paired,
                   blocked.astype(np.uint32))
