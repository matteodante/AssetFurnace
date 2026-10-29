"""The bpy-free parts of `scripts/blender_creature_rig_read.py`: carrying UniRig's bones from
its own box back into the model's frame, parents first."""

import blender_creature_rig_prepare as prep
import blender_creature_rig_read as read
import numpy as np
import pytest


def test_arguments_from_blender_and_from_a_plain_python():
    opts = read.parse_args(["blender", "-b", "--python", "x.py", "--", "W", "a.fbx", "b.fbx", "--start", "2"])
    assert opts["work"].name == "W" and [p.name for p in opts["files"]] == ["a.fbx", "b.fbx"]
    assert opts["start"] == 2
    opts = read.parse_args(["blender_creature_rig_read.py", "W", "a.fbx"])
    assert opts["start"] == 0 and len(opts["files"]) == 1
    with pytest.raises(SystemExit, match="usage"):
        read.parse_args(["x.py", "W"])


def test_unirigs_box_maps_back_onto_the_model():
    """UniRig centres the model and scales its largest side to 2 (from -1 to 1)."""
    lo, hi = np.array([-0.3, -0.5, 0.0]), np.array([0.3, 0.5, 0.8])
    drawn_lo, drawn_hi = np.array([-0.6, -1.0, -0.8]), np.array([0.6, 1.0, 0.8])
    m = read.box_map(lo, hi, drawn_lo, drawn_hi)
    assert m["scale"] == pytest.approx(0.5) and m["axis_gap"] == pytest.approx(0.0)
    assert np.allclose(read.to_model([drawn_lo, drawn_hi, [0, 0, 0]], m), [lo, hi, [0, 0, 0.4]])


def test_a_copy_that_disagrees_on_one_axis_is_flagged():
    lo, hi = np.array([-0.3, -0.5, 0.0]), np.array([0.3, 0.5, 0.8])
    turned = read.box_map(lo, hi, [-1.0, -0.6, -0.8], [1.0, 0.6, 0.8])   # X and Y swapped
    assert turned["axis_gap"] > read.AXIS_TOLERANCE
    assert "warning" in read.draw_record(["bone_0"], [None], [[0, 0, 0]], turned)
    fine = read.box_map(lo, hi, [-0.6, -1.0, -0.8], [0.6, 1.0, 0.79])
    assert "warning" not in read.draw_record(["bone_0"], [None], [[0, 0, 0]], fine)


def test_bones_are_listed_parents_first():
    names = ["c", "a", "b", "d"]
    parents = ["b", None, "a", "a"]
    order = read.parents_first(names, parents)
    assert [names[i] for i in order] == ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="loop"):
        read.parents_first(["a", "b"], ["b", "a"])


def test_tails_are_rebuilt_the_way_unirig_makes_them():
    """FBX keeps no bone lengths: an importer's tails are guesses, UniRig's rule is not."""
    heads = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 0, 2]], float)
    tails = read.unirig_tails(heads, [-1, 0, 1, 1, 3], 0.1)
    assert np.allclose(tails[0], [0, 0, 0.1])        # the root: straight up, whatever its children
    assert np.allclose(tails[1], [0, 0, 1.1])        # two children: on along the line from its parent
    assert np.allclose(tails[2], [0, 1.1, 1])        # an end: the same
    assert np.allclose(tails[3], [1, 0, 2])          # one child: ends at it
    assert np.allclose(read.unirig_tails(np.zeros((2, 3)), [-1, 0], 0.1)[1], [0, 0, 0.1])


def test_a_draw_file_holds_the_bones_in_the_model_frame():
    m = read.box_map([0, 0, 0], [2, 2, 2], [-1, -1, -1], [1, 1, 1])
    record = read.draw_record(["tip", "root"], ["root", None], [[0, 0, 1], [0, 0, -1]], m,
                              "out/draw0/skeleton.fbx")
    assert record["names"] == ["root", "tip"] and record["parents"] == [-1, 0]
    assert np.allclose(record["heads"], [[1, 1, 0], [1, 1, 2]])
    assert np.allclose(record["tails"], [[1, 1, 0.03], [1, 1, 2.03]])   # 0.03 of UniRig's box, scaled
    assert record["file"].endswith("skeleton.fbx")


def test_the_model_box_comes_from_the_shell_file(tmp_path):
    v = np.array([[-1, -2, 0], [1, 2, 3], [0, 0, 1]], float)
    f = np.array([[0, 1, 2]])
    np.savez_compressed(tmp_path / "shell.npz", **prep.shell_arrays(v, f, v, f, 3.0, 0.0))
    lo, hi = read.model_box(tmp_path)
    assert np.allclose(lo, [-1, -2, 0]) and np.allclose(hi, [1, 2, 3])
