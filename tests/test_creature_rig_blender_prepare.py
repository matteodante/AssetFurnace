"""The bpy-free parts of `scripts/blender_creature_rig_prepare.py`: which way a creature faces,
where its mirror plane is, and what `shell.npz` holds. A real Blender run is in
`test_creature_rig_blender_run.py`."""

import math

import blender_creature_rig_prepare as prep
import numpy as np
import pytest


def box(lo, hi):
    """A closed box: 8 corners, 12 triangles."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def join(*parts):
    verts, faces, base = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + base)
        base += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def quadruped(toes=True, tail_on_floor=False):
    """Four legs, toes reaching forward (-Y), the head raised at the front: about 0.8 tall."""
    parts = [box((-0.15, -0.4, 0.35), (0.15, 0.4, 0.6)),        # body
             box((-0.08, -0.58, 0.5), (0.08, -0.38, 0.8))]       # head, up at the front
    for x in (-0.12, 0.12):
        for y in (-0.3, 0.3):
            parts.append(box((x - 0.03, y - 0.03, 0.03), (x + 0.03, y + 0.03, 0.36)))
            reach = 0.1 if toes else 0.035
            parts.append(box((x - 0.035, y - reach, 0.0), (x + 0.035, y + 0.035, 0.035)))
    if tail_on_floor:
        parts.append(box((-0.03, 0.38, 0.0), (0.03, 1.0, 0.05)))
    else:
        parts.append(box((-0.03, 0.38, 0.3), (0.03, 0.8, 0.4)))
    return join(*parts)


def points(count=4000, **kw):
    v, f = quadruped(**kw)
    return prep.surface_samples(v, f, count)


def test_arguments():
    opts = prep.parse_args(["blender", "-b", "--", "dragon.glb", "W", "--turn-deg", "20"])
    assert opts["model"].name == "dragon.glb" and opts["work"].name == "W" and opts["turn_deg"] == 20.0
    assert prep.parse_args(["x", "--", "dragon.obj", "W"])["turn_deg"] is None
    with pytest.raises(SystemExit, match="cannot read"):
        prep.parse_args(["--", "dragon.usd", "W"])
    with pytest.raises(SystemExit, match="usage"):
        prep.parse_args(["--", "dragon.glb"])
    with pytest.raises(SystemExit, match="needs a value"):
        prep.parse_args(["--", "dragon.glb", "W", "--turn-deg"])


def test_surface_points_lie_on_the_triangles_and_follow_their_area():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [5.1, 0, 0], [5, 0.1, 0]], float)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    p = prep.surface_samples(v, f, 5000)
    assert np.allclose(p[:, 2], 0)
    small = p[:, 0] >= 5
    assert 0 < small.sum() < 5000 * 0.03     # the small triangle has 1% of the area
    big = p[~small]
    assert (big[:, 0] >= -1e-9).all() and (big[:, 1] >= -1e-9).all() and (big.sum(1) <= 1 + 1e-9).all()
    with pytest.raises(ValueError, match="no surface"):
        prep.surface_samples(v, np.zeros((0, 3), int))


def test_the_mirror_plane_is_found_where_it_is():
    p = points() + [0.03, 0.0, 0.0]
    x, gap = prep.mirror_plane(p)
    assert x == pytest.approx(0.03, abs=0.012)
    assert gap < 0.02


def test_toes_say_the_front():
    assert prep.facing_sign(points()) == (1.0, "toes")
    assert prep.facing_sign(points() * [1, -1, 1]) == (-1.0, "toes")


def test_without_toes_the_higher_end_is_the_head():
    assert prep.facing_sign(points(toes=False)) == (1.0, "head")
    assert prep.facing_sign(points(toes=False) * [1, -1, 1]) == (-1.0, "head")


def test_a_tail_lying_on_the_floor_does_not_vote():
    """No leg stands on a tail: it points backwards and must not turn the creature round."""
    assert prep.facing_sign(points(tail_on_floor=True)) == (1.0, "toes")


def test_a_creature_standing_anyhow_is_turned_to_face_minus_y():
    p = points(1000)
    for angle in (30.0, 200.0):
        turned = p @ prep.rot_z(math.radians(angle)).T + [0.4, -0.2, 0.3]
        out = prep.facing(turned)
        q = turned @ out["matrix"][:3, :3].T + out["matrix"][:3, 3]
        assert out["cue"] == "toes"
        assert (out["turn_deg"] + angle) % 360 == pytest.approx(0, abs=2) or \
            (out["turn_deg"] + angle) % 360 == pytest.approx(360, abs=2)
        assert q[:, 2].min() == pytest.approx(0, abs=1e-9)
        assert (q[:, 1].min() + q[:, 1].max()) / 2 == pytest.approx(0, abs=1e-9)
        assert abs(q[:, 0].mean()) < 0.02
        head = q[q[:, 2] > 0.7]
        assert head[:, 1].mean() < -0.3          # the raised head is at -Y


def test_a_given_turn_is_used_as_it_is():
    out = prep.facing(points(1500), turn_deg=90.0)
    assert out["cue"] == "given" and out["turn_deg"] == pytest.approx(90.0)
    assert np.allclose(out["matrix"][:3, :3], prep.rot_z(math.pi / 2))


def test_the_shell_file_holds_what_the_skeleton_step_reads(tmp_path):
    v, f = box((0, 0, 0), (1, 1, 1))
    arrays = prep.shell_arrays(v * 1.01, f, v, f, 1.0, 22.5, 0.012)
    np.savez_compressed(tmp_path / "shell.npz", **arrays)
    got = np.load(tmp_path / "shell.npz")
    assert {"shell_verts", "shell_faces", "model_verts", "model_faces", "height", "turn_deg"} <= set(got.files)
    assert got["shell_verts"].dtype == np.float32 and got["shell_faces"].dtype == np.int32
    assert got["model_verts"].dtype == np.float32 and got["model_faces"].dtype == np.int32
    assert float(got["height"]) == 1.0 and float(got["turn_deg"]) == 22.5
