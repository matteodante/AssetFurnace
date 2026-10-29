"""A picture of every skeleton the learned rigger drew, so a person can pick another one.

The score keeps the draw that fits the body best, and it can be wrong in a way an eye sees at
once (a wing hung on a leg, a neck drawn in a zig-zag). `draws_sheet` lays each draw out from
the side and from the front over the model's silhouette, with its score, the kept one marked,
and the studio's Rig step offers "Use draw N" under it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TILE = 360          # pixels per view
SILHOUETTE = (110, 110, 110)
BONE = (60, 200, 255)
KEPT = (90, 220, 120)
JOINT = (255, 255, 255)


def _views(points: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """2D views of Z-up points facing -Y: the side (forward to the left) and the front."""
    return [("side", np.stack([points[:, 1], -points[:, 2]], axis=1)),
            ("front", np.stack([-points[:, 0], -points[:, 2]], axis=1))]


def _fit(points2d: np.ndarray, lo: np.ndarray, span: float) -> np.ndarray:
    return ((points2d - lo) / span * (TILE * 0.86) + TILE * 0.07).astype(np.int32)


def draws_sheet(work: Path, out: Path | None = None) -> Path | None:
    """`W/draws-sheet.png` from `W/shell.npz`, `W/draws/draw*.json` and `W/joints.json`;
    None when there is nothing to draw."""
    import cv2

    work = Path(work)
    files = sorted((work / "draws").glob("draw*.json"), key=lambda p: int(p.stem[4:] or 0))
    shell = work / "shell.npz"
    if not files or not shell.is_file():
        return None
    verts = np.load(shell)["model_verts"].astype(np.float64).reshape(-1, 3)
    if len(verts) > 20000:
        verts = verts[np.random.default_rng(0).choice(len(verts), 20000, replace=False)]
    joints = json.loads((work / "joints.json").read_text(encoding="utf-8")) if (work / "joints.json").is_file() else {}
    chosen = joints.get("chosen")
    records = joints.get("draws") or []
    rows = []
    for k, path in enumerate(files):
        d = json.loads(path.read_text(encoding="utf-8"))
        heads, tails = np.asarray(d["heads"], float), np.asarray(d["tails"], float)
        colour = KEPT if k == chosen else BONE
        tiles = []
        for (_, body2d), (_, heads2d), (_, tails2d) in zip(_views(verts), _views(heads), _views(tails)):
            lo = body2d.min(0)
            span = float(np.ptp(body2d, 0).max()) or 1.0
            img = np.full((TILE, TILE, 3), 34, np.uint8)
            for x, y in _fit(body2d, lo, span):
                img[min(max(y, 0), TILE - 1), min(max(x, 0), TILE - 1)] = SILHOUETTE
            for a, b in zip(_fit(heads2d, lo, span), _fit(tails2d, lo, span)):
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), colour, 2, cv2.LINE_AA)
                cv2.circle(img, tuple(int(v) for v in a), 3, JOINT, -1, cv2.LINE_AA)
            tiles.append(img)
        row = np.hstack(tiles)
        record = records[k] if k < len(records) else {}
        score = record.get("score")
        label = f"draw {k}" + (" (kept)" if k == chosen else "") + \
            (f"  score {score:.1f}" if isinstance(score, (int, float)) else "  could not be used")
        cv2.putText(row, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
        rows.append(row)
    out = Path(out) if out else work / "draws-sheet.png"
    cv2.imwrite(str(out), np.vstack(rows))
    return out
