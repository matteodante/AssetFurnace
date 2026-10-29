"""Is a generated creature straight? Its mirror gap: how far the body lands from itself mirrored.

Two-camera capture needs a body whose left and right mirror each other: the rig's sides,
the limb colours and the turn all come from that plane (`rigging.py`). A picture drawn in
a dramatic pose (one wing up, the other down, the head turned) gives a 3D model that has
no such plane, and nothing after it can fix that. So the studio measures the model as soon
as it is built and says so, before Finish and the rig spend minutes on it.

The measure: points on the surface, turned so the mirror plane is X = 0 (the plane's place
is searched too, since a lopsided body's centre is off it), then each point's distance to
the nearest point of the mirrored body. The median, as a share of the model's size, is the
mirror gap. The median, not the mean: a tail curled to one side moves a tenth of the body
and leaves the median alone, while wings held unevenly move most of it.

Measured on 2026-10-29 (6000 points): a dragon drawn rearing with one wing up 3.1%; the
wolf and AnyMove's dragon (tail curled sideways) 1.7% and 1.8%, both captured well; three
dragons drawn with the creature recipe 0.7% to 1.0%; a knight in an A-pose 0.7%.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .rigging import _rot_z, symmetry_yaw

LOPSIDED = 0.025          # median mirror gap above this share of the size: warn
POINTS = 6000
OFFSETS = np.linspace(-0.05, 0.05, 21)   # where the plane may sit, as a share of the size


def mirror_gap(points: np.ndarray) -> dict[str, float]:
    """The median mirror gap of a Z-up point cloud, as a share of its size, and the turn
    (degrees) that puts its mirror plane on X = 0."""
    from scipy.spatial import cKDTree

    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 10:
        raise ValueError("need at least 10 points in 3D")
    size = float(np.ptp(p, 0).max())
    if size <= 0:
        raise ValueError("the points are all in one place")
    yaw = symmetry_yaw(p, samples=1500)
    turned = (p - p.mean(0)) @ _rot_z(yaw).T
    best = math.inf
    for offset in OFFSETS * size:
        q = turned - [offset, 0.0, 0.0]
        gaps, _ = cKDTree(q).query(q * [-1.0, 1.0, 1.0])
        best = min(best, float(np.median(gaps)))
    return {"median": best / size, "turn_deg": math.degrees(yaw)}


def surface_points(model: Path, count: int = POINTS, seed: int = 0) -> np.ndarray:
    """Points spread evenly over a GLB's surface, Z up (glTF is Y up)."""
    import trimesh

    mesh = trimesh.load(str(model), force="mesh")   # a scene's meshes joined, their transforms applied
    if not hasattr(mesh, "sample") or len(getattr(mesh, "faces", [])) == 0:
        raise ValueError(f"{Path(model).name} has no surface to measure")
    points = np.asarray(mesh.sample(count, seed=seed), float)
    return points[:, [0, 2, 1]] * [1.0, -1.0, 1.0]


def check(model: Path) -> dict[str, Any]:
    """The mirror gap of a GLB, in percent, and whether it is lopsided."""
    gap = mirror_gap(surface_points(model))
    return {"mirror_gap_pct": round(100 * gap["median"], 2), "limit_pct": round(100 * LOPSIDED, 2),
            "lopsided": gap["median"] > LOPSIDED}


def report_path(model: Path) -> Path:
    """`dragon.glb` -> `dragon.symmetry.json`, kept beside the model it measured."""
    return Path(model).with_suffix(".symmetry.json")


def write_report(model: Path, report: dict[str, Any]) -> Path:
    path = report_path(model)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return path


def read_report(model: Path) -> dict[str, Any] | None:
    try:
        report = json.loads(report_path(model).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) else None
