"""UV seams can join; close surfaces and ambiguous coincident layers cannot."""

import numpy as np

from image_to_3dlab.edit_topology import seam_map


def square(z=0):
    points = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 0, z], [1, 1, z], [0, 1, z]], float)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    return points, faces


def test_unique_opposed_uv_edge_joins_only_its_duplicate_endpoints():
    points, faces = square()
    result = seam_map(points, faces)
    assert result.paired_edges == 1
    assert result.render_to_geometric.tolist() == [0, 1, 2, 0, 2, 3]
    assert len(result.ambiguous_vertices) == 0


def test_nearby_cloth_layer_is_not_welded_by_proximity():
    points, faces = square()
    other, _ = square(1e-8)
    result = seam_map(np.concatenate([points, other]), np.concatenate([faces, faces + 6]))
    assert len(result.representatives) == 8
    assert not set(result.render_to_geometric[:6]) & set(result.render_to_geometric[6:])


def test_coincident_layers_are_ambiguous_instead_of_position_welded():
    points, faces = square()
    result = seam_map(np.concatenate([points, points]), np.concatenate([faces, faces + 6]))
    assert len(result.representatives) == 12
    assert len(result.ambiguous_vertices) == 12


def test_opposite_surface_orientation_does_not_weld_front_and_back():
    points, faces = square()
    result = seam_map(np.concatenate([points, points]), np.concatenate([faces, (faces + 6)[:, ::-1]]))
    assert len(result.representatives) == 12
    assert len(result.ambiguous_vertices) == 12


def test_different_skin_influences_block_a_candidate_seam():
    points, faces = square()
    joints = np.zeros((6, 4), dtype=np.uint16)
    joints[3:, 0] = 1
    weights = np.zeros((6, 4))
    weights[:, 0] = 1
    result = seam_map(points, faces, joints, weights)
    assert len(result.representatives) == 6
    assert set(result.ambiguous_vertices) == {0, 2, 3, 4}


def test_reordered_joint_slots_with_equal_mass_are_compatible():
    points, faces = square()
    joints = np.array([[0, 1, 0, 0]] * 3 + [[1, 0, 0, 0]] * 3)
    weights = np.array([[0.25, 0.75, 0, 0]] * 3 + [[0.75, 0.25, 0, 0]] * 3)
    assert len(seam_map(points, faces, joints, weights).representatives) == 4
