"""Tests for `image_to_3dlab/twocam/motion.py`: rotation maths and the motion.json checks."""

import json
import math

import numpy as np
import pytest

from image_to_3dlab.twocam import motion as mo


def _axis_angle_matrix(axis, angle):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 1, 1), (-2, 1, 3)])
def test_half_turns_survive_the_quaternion_round_trip(axis):
    """The copysign shortcut used before picked wrong signs here: a half turn about
    (1, -1, 0) came back as one about (1, 1, 0)."""
    r = _axis_angle_matrix(axis, math.pi)
    np.testing.assert_allclose(mo.quaternions_to_matrices(mo.matrices_to_quaternions(r)), r, atol=1e-9)


def test_random_rotations_round_trip():
    rng = np.random.default_rng(0)
    rs = np.array([_axis_angle_matrix(rng.normal(size=3), rng.uniform(0, math.pi)) for _ in range(200)])
    q = mo.matrices_to_quaternions(rs)
    assert (q[:, 0] >= 0).all()
    np.testing.assert_allclose(mo.quaternions_to_matrices(q), rs, atol=1e-9)


def test_a_turn_past_half_way_stays_continuous():
    angles = np.radians(np.arange(0, 300, 10))
    q = mo.matrices_to_quaternions(np.array([_axis_angle_matrix((0, 0, 1), a) for a in angles]))
    aa = mo.quaternions_to_axis_angle(mo.continuous_quaternions(q))
    np.testing.assert_allclose(aa[:, 2], angles, atol=1e-9)
    assert np.abs(np.diff(aa, axis=0)).max() < np.radians(10) + 1e-9


def test_euler_round_trips_with_y_inside_ninety_degrees():
    rng = np.random.default_rng(1)
    for _ in range(50):
        e = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
        np.testing.assert_allclose(mo.matrix_to_euler_xyz(mo.euler_xyz_matrix(e)), e, atol=1e-9)


def test_the_root_location_moves_the_head_by_the_world_offset():
    rest = np.eye(4)
    rest[:3, :3] = _axis_angle_matrix((1, 2, 3), 0.7)
    trans = np.array([0.1, -0.4, 0.25])
    np.testing.assert_allclose(rest[:3, :3] @ mo.root_location(rest, trans), trans, atol=1e-12)


def _motion(frames=3, bones=("root", "a")):
    return {"fps": 24, "bones": list(bones), "trans": [[0, 0, 0]] * frames,
            "root_quat": [[1, 0, 0, 0]] * frames, "euler": [[[0, 0, 0]] * len(bones)] * frames}


def test_a_good_motion_passes_and_counts_its_frames():
    assert mo.validate_motion(_motion(5), ["root", "a", "b"]) == 5


@pytest.mark.parametrize("change, message", [
    (lambda m: m.pop("euler"), "missing euler"),
    (lambda m: m.update(fps=0), "fps must be positive"),
    (lambda m: m.update(trans=[[0, 0, 0]] * 2), "disagree"),
    (lambda m: m["euler"].__setitem__(0, [[float("nan"), 0, 0], [0, 0, 0]]), "NaN"),
])
def test_a_broken_motion_is_refused_before_it_is_keyed(change, message):
    m = json.loads(json.dumps(_motion()))
    change(m)
    with pytest.raises(mo.MotionError, match=message):
        mo.validate_motion(m)


def test_a_motion_for_bones_the_rig_lacks_is_refused():
    with pytest.raises(mo.MotionError, match="no bones named a"):
        mo.validate_motion(_motion(), ["root"])


def test_write_and_load_round_trip(tmp_path):
    path = tmp_path / "motion.json"
    mo.write_motion(_motion(), path)
    assert mo.load_motion(path)["bones"] == ["root", "a"]
