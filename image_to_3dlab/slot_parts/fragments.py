"""Small bits floating away from a piece, found to be dropped.

Pixal3D and the retopology leave the odd stray bit beside a piece made on its own: the first
real set's glove carried a 56-face fragment a fifth of the glove's size away from it, which
hung in the air next to the hand in Unreal. A part is dropped only when it is both small
(under `share` of the faces) and apart from the largest part (further than `gap` of the
piece's largest size): a rivet touching the piece, or a strap of its own size, stays.
"""

from __future__ import annotations

import numpy as np

SHARE = 0.05
GAP = 0.01


def floating(vertices, faces, share: float = SHARE, gap: float = GAP) -> tuple[np.ndarray, dict]:
    """(faces to keep, what was dropped). Parts are joined where they share a position; a small
    part is measured to the surface of every part of a fair size, so a buckle on the second
    boot of a pair (one file holds both sides) stays, and so does a stud in the middle of a
    large triangle."""
    import trimesh

    from image_to_3dlab.checks import geometry

    v, f = np.asarray(vertices, float), np.asarray(faces, np.int64)
    keep = np.ones(len(f), bool)
    if not len(f):
        return keep, {"dropped": 0, "faces": 0}
    _, welded = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    labels = geometry.components(welded.reshape(-1)[f])
    counts = np.bincount(labels)
    big = counts >= share * len(f)
    big[int(np.argmax(counts))] = True   # the largest part counts, however small the share
    anchors = trimesh.Trimesh(vertices=v, faces=f[big[labels]], process=False)
    far = gap * float(np.ptp(v, axis=0).max())
    dropped = 0
    for part in np.flatnonzero(counts):
        if big[part]:
            continue
        points = v[np.unique(f[labels == part])]
        _, distance, _ = trimesh.proximity.closest_point(anchors, points)
        if float(distance.min()) > far:
            keep[labels == part] = False
            dropped += 1
    return keep, {"dropped": dropped, "faces": int((~keep).sum())}
