"""Joint trees, named bones, polylines and the centre lines of a voxel body."""

from __future__ import annotations

import numpy as np
import pytest
from creature_rig_shapes import biped, box, quadruped

from image_to_3dlab.creature_rig.shell import Shell
from image_to_3dlab.creature_rig.skeleton import (
    Bone,
    Tree,
    below,
    centre_line,
    in_order,
    place,
    resample,
    side_of,
    stem_of,
)


def _chain_tree():
    # root 0 -> 1 -> 2 (tip); root 0 -> 3 (tip)
    return Tree([[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 0]], [-1, 0, 1, 0], [False, False, True, True])


def test_from_bones_makes_a_joint_per_head_and_a_tip_per_childless_tail():
    tree = Tree.from_bones([-1, 0, 0], [[0, 0, 0], [0, 0, 1], [1, 0, 0]], [[0, 0, 1], [0, 0, 2], [2, 0, 0]])
    assert tree.n == 5
    assert tree.joints() == 3
    tips = tree.points[tree.tip]
    assert sorted(map(tuple, tips)) == [(0, 0, 2), (2, 0, 0)]


def test_from_bones_keeps_the_biggest_of_several_roots():
    tree = Tree.from_bones([-1, -1, 1, 2], [[9, 9, 9], [0, 0, 0], [0, 0, 1], [0, 0, 2]],
                           [[9, 9, 8], [0, 0, 1], [0, 0, 2], [0, 0, 3]])
    assert tree.joints() == 3
    assert not np.any(np.all(tree.points == 9, axis=1))


def test_path_subtree_and_keep():
    tree = _chain_tree()
    assert tree.path(2, 3) == [2, 1, 0, 3]
    assert sorted(tree.subtree(1)) == [1, 2]
    smaller = tree.remove([1])                          # 2 now hangs from the root
    assert smaller.n == 3
    assert list(smaller.parent) == [-1, 0, 0]
    with pytest.raises(ValueError):
        tree.remove([0])


def test_rerooted_keeps_the_joints_and_turns_the_parents_round():
    tree = _chain_tree().rerooted(2)
    assert tree.root == 2
    assert tree.path(2, 3) == [2, 1, 0, 3]
    assert tree.parent[0] == 1


def test_samples_cover_every_bone():
    pts = _chain_tree().samples(11)
    assert len(pts) == 3 * 11
    assert np.isclose(pts[:, 2], 1.5).any()


def test_bone_names_sides_and_order():
    assert side_of("Thigh.L") == "L" and side_of("Head") == ""
    assert stem_of("Thigh1.R") == "Thigh1"
    bones = [Bone("Shin.L", "Thigh.L", [0, 0, 0], [0, 0, 1], "hind", "L"),
             Bone("Root", None, [0, 0, 0], [0, 0, 1], "root", "", False),
             Bone("Thigh.L", "Root", [0, 0, 0], [0, 0, 1], "hind", "L")]
    assert [b.name for b in in_order(bones)] == ["Root", "Thigh.L", "Shin.L"]
    assert [b.name for b in below(bones, "Thigh.L")] == ["Thigh.L", "Shin.L"]
    out = bones[1].to_json()
    assert out["deform"] is False and out["parent"] is None and out["head"] == [0.0, 0.0, 0.0]


def test_place_finds_the_bend_near_a_planned_share():
    line = np.concatenate([np.linspace([0, 0, 1.0], [0, 0, 0.4], 40), np.linspace([0, 0, 0.4], [0, -0.4, 0.0], 30)[1:]])
    line = resample(line, 0.01)
    i = place(line, [0.5], window=3)[0]
    assert line[i] == pytest.approx([0, 0, 0.4], abs=0.02)        # the knee, not the halfway point
    straight = resample(np.array([[0, 0, 0], [0, 0, 1.0]]), 0.01)
    assert straight[place(straight, [0.3], window=3)[0]][2] == pytest.approx(0.3, abs=0.01)


def test_centre_lines_of_a_quadruped_end_at_its_feet_head_and_tail():
    shell = Shell(quadruped(), 1.05)
    tree = centre_line(shell)
    ends = tree.points[tree.ends()]
    assert len(ends) == 6
    feet = ends[ends[:, 2] < 0.1]
    assert len(feet) == 4
    assert ends[:, 1].max() > 0.9                       # the tail tip
    assert ends[:, 1].min() < -0.6                      # the head
    assert tree.radius is not None and tree.radius.max() > 0.1


def test_centre_lines_fold_fingers_into_one_end():
    hand = [box([0.62, -0.06 + 0.035 * k, 0.86], [0.75, -0.04 + 0.035 * k, 0.88]) for k in range(4)]
    shell = Shell(biped() + hand, 1.2)
    tree = centre_line(shell)
    right_side = tree.points[tree.ends()]
    assert (right_side[:, 0] > 0.55).sum() == 1        # one end for the hand, not four fingers
