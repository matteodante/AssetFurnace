"""Tests for the bpy-free parts of `scripts/blender_twocam_prepare.py` and
`scripts/blender_twocam_dataset.py`. Both run on a real rig by hand (docs/twocam-mocap.md)."""

import math

import blender_twocam_dataset as dataset
import blender_twocam_prepare as prepare
import numpy as np
import pytest


def test_prepare_needs_the_creatures_real_size():
    with pytest.raises(SystemExit, match="length-m"):
        prepare.parse_args(["--", "wolf.glb", "out"])
    opts = prepare.parse_args(["--", "wolf.glb", "out", "--length-m", "1.6", "--yaw-deg", "90", "--no-render"])
    assert opts["length_m"] == 1.6 and opts["yaw"] == pytest.approx(math.pi / 2) and not opts["render"]


def test_prepare_refuses_an_unknown_axis():
    with pytest.raises(SystemExit, match="axis"):
        prepare.parse_args(["--", "wolf.glb", "out", "--length-m", "1", "--axis", "w"])


def test_dataset_arguments():
    opts = dataset.parse_args(["--", "rig_info.json", "ds", "100", "50", "7"])
    assert (opts["start"], opts["count"], opts["seed"], opts["scale"]) == (100, 50, 7, 0.6)


def test_limbs_and_loose_ends_are_the_fast_parts():
    assert dataset.fast_bone("limb1_2_L", leaf=False)
    assert dataset.fast_bone("bone_7", leaf=True)
    assert not dataset.fast_bone("bone_3", leaf=False)


def test_poses_stay_inside_every_bones_limits():
    rng = np.random.default_rng(0)
    limits = {"root": np.radians([[-30, 30], [-20, 20], [-30, 30]]), "knee": np.radians([[-10, 110], [0, 0], [-5, 5]])}
    for _ in range(200):
        pose = dataset.sample_pose(rng, ["root", "knee"], limits, "root")
        k = pose["knee"]
        assert (k >= limits["knee"][:, 0] - 1e-12).all() and (k <= limits["knee"][:, 1] + 1e-12).all()
        assert pose["root"][0] >= math.radians(-60) - 1e-12
