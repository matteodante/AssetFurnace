"""Deformation invariants for the editor, independent of Blender or generated assets."""

import numpy as np
import pytest

from image_to_3dlab.edit_math import (
    EditMathError, calibrate_local_rotation, certify_bind, posed_delta_to_rest,
    rebind_inverse, skin_matrices,
)


def translated(x=0, y=0, z=0):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


def rotation(axis, degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_reposition_then_change_weights_preserves_rest_with_nonidentity_bind():
    old = np.array([translated(2), translated(2, 1)])
    common = translated(0, 0, 3)
    inverse = np.linalg.inv(old) @ common
    new = old.copy()
    new[1, :3, 3] += [0.25, 0.1, 0]
    updated = rebind_inverse(old, new, inverse)
    assert np.allclose(certify_bind(new, updated), common)
    indices = np.array([[0, 1, 0, 0]])
    for weights in ([0.4, 0.6, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]):
        blended = skin_matrices(indices, np.array([weights]), new @ updated)
        assert np.allclose(blended[0] @ [1, 2, 3, 1], [1, 2, 6, 1])
    assert np.array_equal(inverse, np.linalg.inv(old) @ common)


def test_inconsistent_bind_is_rejected_even_when_fixed_weights_would_hide_it():
    with pytest.raises(EditMathError, match="common bind"):
        certify_bind(np.array([np.eye(4), translated(0.2)]), np.array([np.eye(4)] * 2))


def test_inverse_of_blended_matrix_recovers_requested_posed_displacement():
    matrices = np.array([translated(5), translated(-2)])
    matrices[0, :3, :3] = rotation("z", 55) @ np.diag([2, 1, 0.5])
    matrices[1, :3, :3] = rotation("x", 20)
    indices = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    weights = np.array([[0.4, 0.6, 0, 0], [0.75, 0.25, 0, 0]])
    delta = np.array([[0.2, -0.1, 0.3], [0, 0.5, 0]])
    result = posed_delta_to_rest(delta, indices, weights, matrices)
    blend = skin_matrices(indices, weights, matrices)[:, :3, :3]
    assert np.allclose(np.einsum("nij,nj->ni", blend, result), delta)


def test_collapsed_blend_refuses_the_entire_gesture_and_identifies_vertex():
    matrices = np.array([np.eye(4), np.diag([-1., -1., 1., 1.])])
    with pytest.raises(EditMathError) as error:
        posed_delta_to_rest([[0.1, 0, 0]], [[0, 1, 0, 0]], [[0.5, 0.5, 0, 0]], matrices)
    assert error.value.code == "unstable_pose"
    assert error.value.vertices == [0]


def test_distance_limit_catches_uniform_compression_even_if_condition_number_is_one():
    matrix = np.diag([1e-6, 1e-6, 1e-6, 1])
    with pytest.raises(EditMathError, match="displacement"):
        posed_delta_to_rest([[1, 0, 0]], [[0, 0, 0, 0]], [[1, 0, 0, 0]],
                           [matrix], max_rest_distance=0.2)


@pytest.mark.parametrize("weights", [[[0.5, 0, 0, 0]], [[-1, 2, 0, 0]], [[np.nan, 0, 0, 0]]])
def test_invalid_weights_are_not_silently_normalized(weights):
    with pytest.raises(EditMathError):
        skin_matrices([[0, 0, 0, 0]], weights, [np.eye(4)])


@pytest.mark.parametrize("indices", [[[-1, 0, 0, 0]], [[1, 0, 0, 0]], [[0.5, 0, 0, 0]]])
def test_invalid_joint_indices_are_not_truncated_or_wrapped(indices):
    with pytest.raises(EditMathError):
        skin_matrices(indices, [[1, 0, 0, 0]], [np.eye(4)])


def test_calibration_changes_the_surface_axis_and_not_the_rest_rotation():
    rest = rotation("z", 30)
    delta = rotation("x", 45)
    calibration = rotation("z", 90)
    result = calibrate_local_rotation(rest, rest @ delta, calibration)
    # Local X bend reoriented to Y moves +Z toward +X instead of -Y.
    surface = rest.T @ result @ [0, 0, 1]
    assert np.allclose(surface, [np.sqrt(0.5), 0, np.sqrt(0.5)])
    assert np.allclose(calibrate_local_rotation(rest, rest, calibration), rest)
    assert np.allclose(calibrate_local_rotation(rest, rest @ delta, np.eye(3)), rest @ delta)


def test_reflection_is_not_a_valid_motion_calibration():
    with pytest.raises(EditMathError, match="proper rotation"):
        calibrate_local_rotation(np.eye(3), np.eye(3), np.diag([-1, 1, 1]))


def test_nonfinite_or_nonaffine_bind_is_rejected():
    bad = np.eye(4)
    bad[3, 0] = 1
    with pytest.raises(EditMathError, match="affine"):
        certify_bind([bad], [np.eye(4)])
    with pytest.raises(EditMathError):
        rebind_inverse([np.eye(4)], [np.zeros((4, 4))], [np.eye(4)])
