"""Tests for `image_to_3dlab/checks/rig_checks.py`: `rig.deformation` measures every clip
of a rig, passes a clean bend and stops a clip that tears or crosses the feet."""

import json

import numpy as np

import _skinned_glb as fx
from image_to_3dlab import checks
from image_to_3dlab.checks import rig_checks


def run(glb):
    # warn_only off: these fixtures are far from any calibration question
    return checks.run_checks(checks.Asset(glb=glb), profile="creature", only=["rig.deformation"],
                             overrides={"rig.deformation": {"warn_only": False}})


def test_needs_a_clip(tmp_path):
    report = run(fx.write(tmp_path / "still_rigged.glb"))
    assert report["results"] == [] and "animation" in report["skipped"][0]["reason"]


def test_a_rigid_bend_passes(tmp_path):
    glb = fx.write(tmp_path / "bend.glb", {"bend": [(0.0, fx.turn_z(0)), (1.0, fx.turn_z(60))]})
    found = run(glb)["results"][0]
    assert found["status"] == "pass", found
    assert found["details"]["clips"]["bend"]["stretched_percent"] == 0.0


def test_a_tearing_bend_fails(tmp_path):
    # the middle row follows the lower bone, so turning the upper one shears the middle faces
    glb = fx.write(tmp_path / "tear.glb", {"tear": [(0.0, fx.turn_z(0)), (1.0, fx.turn_z(170))]})
    found = run(glb)["results"][0]
    assert found["status"] == "fail" and found["details"]["clips"]["tear"]["flipped_percent"] > 0


def test_sibling_clips_of_a_rig_are_measured(tmp_path):
    rig = fx.write(tmp_path / "strip_rigged.glb")
    fx.write(tmp_path / "strip_wave.glb", {"wave": [(0.0, fx.turn_z(0)), (1.0, fx.turn_z(30))]})
    fx.write(tmp_path / "other_wave.glb", {"x": [(0.0, fx.turn_z(0)), (1.0, fx.turn_z(170))]})
    assert [p.name for p in rig_checks.clip_files(rig)] == ["strip_rigged.glb", "strip_wave.glb"]
    found = run(rig)["results"][0]
    assert found["status"] == "pass" and set(found["details"]["clips"]) == {"strip_wave.glb: wave"}


def test_feet_that_swap_sides_fail(tmp_path):
    # the whole strip turns 135 degrees about its foot, carrying bone_1 to the other side of
    # bone_0 along a slanted "left" -- pretend the two bones are the two feet
    glb = fx.write(tmp_path / "duo_rigged.glb", {"swap": [(0.0, fx.turn_z(0)), (1.0, fx.turn_z(135))]}, node=1)
    left = [0.7071, 0.7071, 0]
    family = {"family": "digitigrade", "frame": {"up": [0, 1, 0], "forward": [0, 0, 1], "left": left},
              "legs": [{"bones": ["bone_0"], "side": "R"}, {"bones": ["x", "bone_1"], "side": "L"}]}
    (tmp_path / "duo_rigged.family.json").write_text(json.dumps(family), encoding="utf-8")
    still = np.zeros((2, 3))
    assert rig_checks.feet_swap(np.array([[1.0, 0, 0], [-1.0, 0, 0]]), still, np.array([1.0, 0, 0]))
    assert not rig_checks.feet_swap(np.array([[1.0, 0, 0], [0.2, 0, 5]]), still, np.array([1.0, 0, 0]))
    found = run(glb)["results"][0]
    assert found["status"] == "fail" and "cross" in found["message"]
    assert found["details"]["clips"]["swap"]["stretched_percent"] == 0.0  # rigid: only the feet are wrong
