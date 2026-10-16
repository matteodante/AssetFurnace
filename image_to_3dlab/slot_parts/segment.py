"""Route B0, the fallback: label every face of a whole-character model with its slot.

The concept is made into one model with Pixal3D; its faces are then split by slot.
Visible faces (first hit from the camera) read the slot map at their projected centre;
hidden ones take the slot of the nearest bone of the base body, carried into the model's
frame through the two pictures (the body's and the concept's share their framing). Two
rounds of majority over face neighbours clean the speckle at the borders. Measured in the
first trial: the slot map needed no touch-ups, the back side came out right only with the
bones. `scripts/blender_segment_parts.py` does the ray tests, the split and the export.

Coordinates are Blender's (Z up, the camera on the -Y side), like `slot_parts.fit`.
"""

from __future__ import annotations

import numpy as np

from image_to_3dlab.slot_parts import fit, skeleton, slots
from image_to_3dlab.slot_parts.camera import Camera


def project(cam: Camera, points: np.ndarray) -> np.ndarray:
    """Blender points (N, 3) to source-picture pixels (N, 2)."""
    p = np.asarray(points, float)
    return cam.project(np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1))  # Blender (x, y, z) -> glTF (x, z, -y)


def visible_labels(centres: np.ndarray, visible: np.ndarray, cam: Camera, slot_map: np.ndarray) -> np.ndarray:
    """Slot index for each visible face from the slot map; -1 for hidden or off-picture faces."""
    labels = np.full(len(centres), -1, int)
    uv = np.round(project(cam, centres)).astype(int)
    h, w = slot_map.shape
    ok = np.asarray(visible, bool) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    labels[ok] = slot_map[uv[ok, 1], uv[ok, 0]]
    return labels


def bones_in_model(body: dict, model_cam: Camera, concept_size, y_centre: float) -> list[skeleton.Segment]:
    """The base body's bone segments carried into the whole-character model's frame.

    x and z come through the pictures: body bone -> body picture -> concept picture ->
    back out through the model's camera at its depth. Depth (y) is the bone's depth from
    the body's root, scaled by the ratio of the two cameras' sizes, around the model's
    centre `y_centre`.
    """
    from image_to_3dlab.slot_parts import body as body_mod

    body_cam = Camera.from_dict(body["camera"])
    segs_px = body_mod.segments_px(body, concept_size)
    heads = {b: np.asarray(h, float) for b, h in body["skeleton"]["heads"].items()}
    root = next(b for b, p in body["skeleton"]["parents"].items() if p is None)
    f_body = (body_cam.size / 2) / np.tan(body_cam.fov / 2)
    f_model = (model_cam.size / 2) / np.tan(model_cam.fov / 2)
    k = (f_body * body_cam.scale) / (f_model * model_cam.scale)
    root_depth = -heads[root][2]  # glTF z forward -> Blender depth y = -z

    def carry(px, gltf_point):
        x, z = fit.unproject(model_cam, float(px[0]), float(px[1]), 0.0)
        return np.array([x, (-gltf_point[2] - root_depth) * k + y_centre, z])

    parents = body["skeleton"]["parents"]
    roles = {b: tuple(r) for b, r in body["roles"].items()}
    raw = skeleton.joint_segments(parents, heads, roles)
    return [skeleton.Segment(s.bone, s.role, s.side, carry(p.head, s.head), carry(p.end, s.end))
            for s, p in zip(raw, segs_px)]


def hidden_labels(centres: np.ndarray, segments: list[skeleton.Segment]) -> np.ndarray:
    """Slot index of the nearest bone segment, for each point."""
    p = np.asarray(centres, float)
    best = np.full(len(p), np.inf)
    out = np.full(len(p), slots.label("uncovered"), int)
    for seg in segments:
        d = seg.end - seg.head
        t = np.clip(((p - seg.head) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
        dist2 = ((p - (seg.head + t[:, None] * d)) ** 2).sum(1)
        closer = dist2 < best
        best[closer] = dist2[closer]
        out[closer] = [slots.label(skeleton.slot_for(seg.role, x)) for x in t[closer]]
    return out


def smooth(labels: np.ndarray, neighbours: list[np.ndarray], rounds: int = 2) -> np.ndarray:
    """Majority of each face and its neighbours, `rounds` times."""
    out = np.asarray(labels, int).copy()
    for _ in range(rounds):
        new = out.copy()
        for i, n in enumerate(neighbours):
            values, counts = np.unique(np.concatenate([out[n], [out[i]]]), return_counts=True)
            new[i] = values[np.argmax(counts)]
        out = new
    return out


def label_faces(centres, visible, model_cam: Camera, slot_map, body: dict, concept_size, y_centre: float,
                neighbours: list[np.ndarray], chosen: set[str] | None = None) -> np.ndarray:
    """Every face's slot index: visible from the map, the rest from the bones, then smoothed."""
    labels = visible_labels(centres, visible, model_cam, slot_map)
    hidden = labels <= 0
    if hidden.any():
        labels[hidden] = hidden_labels(np.asarray(centres)[hidden], bones_in_model(body, model_cam, concept_size, y_centre))
    labels = smooth(labels, neighbours)
    if chosen is not None:
        keep = {slots.label(s) for s in chosen}
        labels[~np.isin(labels, list(keep))] = slots.label("uncovered")
    return labels
