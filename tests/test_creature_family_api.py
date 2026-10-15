"""Tests for `viewer/creature_family_api.py`: the family choice, the clip step and its
provenance, and what the Library is told."""

import json

import pytest

import autorig_api as aa
import creature_family_api as api
from image_to_3dlab import creature_family as cf


def family_file(rigged, family="digitigrade", clips=("digitigrade-walk",), chosen=False):
    found = {"family": family, "chosen": chosen, "clips": list(clips), "clips_error": None}
    cf.write_family(rigged, found)
    for name in clips:
        rigged.with_name(rigged.stem.removesuffix("_rigged") + f"_{name}.glb").write_bytes(b"glTF")


def test_the_choice_is_a_known_family_or_auto():
    assert api.parse_family(None) == "auto" and api.parse_family(" Quadruped ") == "quadruped"
    with pytest.raises(ValueError, match="unknown family"):
        api.parse_family("kraken")


def test_the_rig_job_ends_with_the_clip_step(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    job = manager.create_rig("wolf.glb", b"glTF", None, "uploaded", family="digitigrade")
    (rig, _), (clean, _), (clips, cwd) = job.commands
    assert clips[1].endswith("creature_clips.py") and clips[clips.index("--rig") + 1] == str(job.result)
    assert clips[-2:] == ["--family", "digitigrade"] and cwd == api.REPO
    assert aa.rig_commands(tmp_path / "s.glb", tmp_path / "r_rigged.glb")[-1][0][-1] == "auto"


def test_clips_get_the_rigs_licence_and_their_own(tmp_path):
    rigged = tmp_path / "wolf_rigged.glb"
    family_file(rigged)
    rig_record = {"license": {"name": "MIT"}, "rig": {"tool": "SkinTokens"}}
    path, = api.record_clips(rigged, rig_record)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "wolf_digitigrade-walk.provenance.json"
    assert record["license"] == {"name": "MIT"} and record["rig"] == {"tool": "SkinTokens"}
    assert record["motion"]["clip"] == "digitigrade-walk" and record["motion"]["license"]["name"] == "MIT"


def test_facts_and_the_closing_message(tmp_path):
    rigged = tmp_path / "wolf_rigged.glb"
    assert api.facts(rigged)["family"] is None and api.done_message({}, rigged) is None
    family_file(rigged)
    assert api.facts(rigged)["family_label"].startswith("Two legs")
    assert "digitigrade-walk" in api.done_message({"humanoid": True}, rigged)
    assert "Preset moves fit it too" in api.done_message({"humanoid": True}, rigged)
    family_file(rigged, family="plant", clips=())
    assert "No test clips" in api.done_message({}, rigged)
    family_file(rigged, family="humanoid", clips=())
    assert api.done_message({"humanoid": True}, rigged) is None  # the usual message stands
