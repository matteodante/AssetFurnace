"""The slot map of a concept: every subject pixel takes the slot of its nearest bone.

The concept is drawn on the base body (an edit of the body picture that keeps its pose
and outline), so the body's skeleton, projected with the body's camera, lands on it. The
slot then follows from the bone and from where along it the pixel falls. Geometry only:
which pixels are subject comes from the matte, never from colours.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from image_to_3dlab.slot_parts import skeleton, slots

# One colour per label, for the preview only.
COLOURS = np.array([(0, 0, 0), (230, 200, 40), (200, 40, 40), (40, 160, 220), (160, 60, 200),
                    (40, 180, 80), (240, 130, 30), (120, 120, 120)], np.uint8)


def slot_labels(foreground: np.ndarray, segments: Iterable[skeleton.Segment], chosen: set[str] | None = None,
                shoulder_t: float = skeleton.SHOULDER_T, boot_t: float = skeleton.BOOT_T) -> np.ndarray:
    """A uint8 map of `slots.LABELS` indices. Slots not in `chosen` stay bare body."""
    fg = np.asarray(foreground, bool)
    ys, xs = np.nonzero(fg)
    points = np.stack([xs, ys], 1).astype(float)
    best = np.full(len(points), np.inf)
    names = np.full(len(points), "uncovered", dtype=object)
    for seg in segments:
        head, end = np.asarray(seg.head, float)[:2], np.asarray(seg.end, float)[:2]
        d = end - head
        t = np.clip(((points - head) @ d) / max(float(d @ d), 1e-9), 0.0, 1.0)
        dist2 = ((points - (head + t[:, None] * d)) ** 2).sum(1)
        closer = dist2 < best
        if not closer.any():
            continue
        best[closer] = dist2[closer]
        names[closer] = [skeleton.slot_for(seg.role, x, shoulder_t, boot_t) for x in t[closer]]
    if chosen is not None:
        names[~np.isin(names, list(chosen))] = "uncovered"
    out = np.zeros(fg.shape, np.uint8)
    out[ys, xs] = [slots.label(n) for n in names]
    return out


def overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The concept half-blended with the slot colours, to check the map by eye."""
    rgb = np.asarray(rgb, np.uint8)[..., :3]
    tint = COLOURS[np.asarray(labels)]
    out = rgb.copy()
    on = np.asarray(labels) > 0
    out[on] = (rgb[on].astype(np.uint16) + tint[on]) // 2
    return out.astype(np.uint8)
