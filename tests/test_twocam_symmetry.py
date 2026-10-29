"""The mirror gap that tells a straight generated creature from a lopsided one."""

from __future__ import annotations

import math

import numpy as np
import pytest

from image_to_3dlab.twocam import symmetry


def _box(centre, size, n, rng):
    return np.asarray(centre) + (rng.random((n, 3)) - 0.5) * np.asarray(size)


def _creature(rng, *, tail_curl=0.0, yaw_deg=0.0):
    """A winged four-legged body, Z up, facing -Y: points inside boxes, mirrored in X."""
    parts = [_box([0, 0, 0.6], [0.4, 1.0, 0.3], 1200, rng), _box([0, -0.8, 0.95], [0.25, 0.4, 0.25], 500, rng)]
    for x in (-0.15, 0.15):
        for y, thick in ((-0.3, 0.07), (0.35, 0.12)):
            parts.append(_box([x, y, 0.22], [thick, thick, 0.45], 150, rng))
    for side in (-1, 1):
        parts.append(_box([side * 0.6, -0.15, 0.85], [0.8, 0.6, 0.05], 1000, rng))
    tail = _box([0, 1.0, 0.5], [0.08, 1.0, 0.08], 400, rng)
    tail[:, 0] += tail_curl * np.clip(tail[:, 1] - 0.9, 0, None)
    parts.append(tail)
    return _turned(np.concatenate(parts), yaw_deg)


def _turned(p, yaw_deg):
    a = math.radians(yaw_deg)
    turn = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return p @ turn.T


def _lopsided(rng, yaw_deg=0.0):
    """Three unequal parts placed so no upright plane mirrors them onto each other."""
    parts = [_box([0, 0, 0.5], [0.4, 1.0, 0.3], 1500, rng),
             _box([0.7, -0.3, 1.2], [0.6, 0.3, 0.9], 1500, rng),
             _box([0.5, 0.75, 0.3], [0.15, 0.15, 0.6], 1000, rng)]
    return _turned(np.concatenate(parts), yaw_deg)


@pytest.mark.parametrize("yaw", [0.0, 23.0, 65.0, 140.0])
def test_a_straight_creature_has_a_small_gap_at_any_turn(yaw):
    gap = symmetry.mirror_gap(_creature(np.random.default_rng(0), yaw_deg=yaw))
    assert gap["median"] < symmetry.LOPSIDED
    # the plane is found: the turn undoes the creature's own (both modulo half a turn)
    assert min((gap["turn_deg"] + yaw) % 180, 180 - (gap["turn_deg"] + yaw) % 180) < 3


@pytest.mark.parametrize("yaw", [0.0, 30.0])
def test_a_body_with_no_mirror_plane_is_lopsided(yaw):
    gap = symmetry.mirror_gap(_lopsided(np.random.default_rng(0), yaw))
    assert gap["median"] > symmetry.LOPSIDED


def test_a_tail_curled_to_one_side_does_not():
    """AnyMove's dragon curls its tail sideways and captures well: the median ignores a tenth of the body."""
    gap = symmetry.mirror_gap(_creature(np.random.default_rng(0), tail_curl=1.2))
    assert gap["median"] < symmetry.LOPSIDED


def test_too_few_points_or_a_single_point_are_refused():
    with pytest.raises(ValueError):
        symmetry.mirror_gap(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        symmetry.mirror_gap(np.zeros((50, 3)))


def test_a_glb_is_read_y_up_and_its_report_kept_beside_it(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pieces = []
    for x in (-0.3, 0.3):                      # two legs and a body, mirrored in X
        pieces.append(trimesh.creation.box([0.1, 0.6, 0.1], trimesh.transformations.translation_matrix([x, 0.3, 0])))
    pieces.append(trimesh.creation.box([0.8, 0.2, 0.3], trimesh.transformations.translation_matrix([0, 0.7, 0])))
    model = tmp_path / "beast.glb"
    trimesh.util.concatenate(pieces).export(model)
    report = symmetry.check(model)
    assert report["lopsided"] is False and report["mirror_gap_pct"] < 1.0
    assert report["limit_pct"] == 2.5
    path = symmetry.write_report(model, report)
    assert path == tmp_path / "beast.symmetry.json"
    assert symmetry.read_report(model) == report
    assert symmetry.read_report(tmp_path / "other.glb") is None
