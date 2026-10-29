"""The shell on its voxel grid: inside, off the skin, the middle plane, floor contacts."""

from __future__ import annotations

import numpy as np
import pytest
from creature_rig_shapes import box, quadruped, robed, work_folder

from image_to_3dlab.creature_rig.shell import Shell, voxelize


def test_a_closed_box_is_filled_and_about_a_hundred_cells_tall():
    solid, _ = voxelize([box([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])], 0.01)
    assert 100 <= solid.shape[2] <= 106
    share = solid.sum() / 100 ** 3
    assert 0.97 < share < 1.08                                  # the cube's volume, give or take its skin


def test_big_triangles_leave_no_gap_for_the_fill_to_leak_through():
    import trimesh

    m = trimesh.creation.box(extents=[1.0, 1.0, 1.0])            # twelve triangles across the whole cube
    m.apply_translation([0, 0, 0.5])
    solid, _ = voxelize([(m.vertices, m.faces)], 0.01)
    assert solid[50, 50, 50 + 2]


def test_inside_and_off_skin():
    shell = Shell([box([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])], 1.0)
    assert shell.inside([[0, 0, 0.5]]).all()
    assert not shell.inside([[0, 0, 1.3]]).any()
    off = shell.off_skin([[0, 0, 0.5], [0, 0, 1.2], [0.8, 0, 0.5]])
    assert off[0] == 0.0
    assert off[1] == pytest.approx(0.2, abs=0.015)
    assert off[2] == pytest.approx(0.3, abs=0.015)


def test_one_sided_cloth_counts_as_body():
    shell = Shell(robed(), 1.2)
    hem = np.array([[0.4, 0.0, 0.005], [0.0, -0.27, 0.31]])     # on the robe's sheet
    assert (shell.off_skin(hem) <= 1.5 * shell.size).all()
    assert not shell.inside([[0.0, 0.0, 0.2]]).any()             # an open robe is not filled


def test_the_middle_plane_is_the_median_of_the_body_not_its_bounding_box():
    parts = [box([-0.2, -0.2, 0.0], [0.2, 0.2, 1.0]), box([0.2, -0.02, 0.5], [0.9, 0.02, 0.54])]   # a thin spear
    shell = Shell(parts, 1.0)
    assert abs(shell.mid) < 0.02
    assert shell.mirror([[0.3, 0.1, 0.2]])[0] == pytest.approx([2 * shell.mid - 0.3, 0.1, 0.2])


def test_four_feet_make_four_contacts():
    shell = Shell(quadruped(), 1.05)
    assert len(shell.contacts) == 4
    centres = sorted((round(float(g[:, 0].mean()), 1), round(float(g[:, 1].mean()), 1)) for g in shell.contacts)
    assert centres == [(-0.1, -0.4), (-0.1, 0.4), (0.1, -0.4), (0.1, 0.4)]


def test_a_robe_hem_on_the_floor_is_no_foot():
    assert Shell(robed(), 1.2).contacts == []


def test_legs_inside_a_robe_still_stand():
    shell = Shell(robed(legs=True), 1.2)
    assert len(shell.contacts) == 2
    assert all(abs(g[:, 0].mean()) < 0.15 for g in shell.contacts)


def test_a_contact_is_reached_by_a_bone_point_low_and_close_beside_it():
    shell = Shell(quadruped(), 1.05)
    front_left = next(k for k, g in enumerate(shell.contacts) if g[:, 0].mean() > 0 and g[:, 1].mean() < 0)
    feet = np.array([[x, y, 0.05] for x in (-0.1, 0.1) for y in (-0.38, 0.38)])
    assert shell.unreached(feet) == []
    without = feet[~((feet[:, 0] > 0) & (feet[:, 1] < 0))]
    assert shell.unreached(without) == [front_left]
    high = feet + [0, 0, 0.3]                                   # above 15% of the height: reaches nothing
    assert len(shell.unreached(high)) == 4


def test_symmetric_near_tells_a_mirrored_body_from_a_lopsided_one():
    shell = Shell(quadruped() + [box([0.15, -0.3, 0.5], [0.45, 0.3, 0.7])], 1.05)   # a pack on the left only
    centre = np.array([[0.0, 0.0, 0.6]])
    assert shell.symmetric_near(centre, 0.05) > 0.95
    assert shell.symmetric_near([[0.3, 0.0, 0.6]], 0.05) < 0.2


def test_load_reads_the_prepare_steps_shell(tmp_path):
    work = work_folder(tmp_path / "w", quadruped())
    shell = Shell.load(work)
    assert shell.height == pytest.approx(1.05, abs=1e-3)
    assert shell.size == pytest.approx(0.0105, abs=1e-4)
    assert shell.floor == pytest.approx(0.0, abs=1e-6)
    assert shell.depth.max() > 10
