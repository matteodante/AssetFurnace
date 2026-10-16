"""Tests for `scripts/creature_clips.py`'s pure parts: where clips go and which are made.
The `bpy` half is run on a real rig by hand (see docs/plans/2026-10-15-creature-families-design.md)."""

from pathlib import Path

import creature_clips


def test_clip_files_sit_beside_the_rig_under_its_base_name():
    rig = Path("out/wolf_rigged.glb")
    assert creature_clips.clip_path(rig, "digitigrade-walk") == Path("out/wolf_digitigrade-walk.glb")


def test_humanoids_get_test_clips_only_when_asked():
    assert creature_clips.wanted_motions("digitigrade", chosen=False) == ("walk", "attack")
    assert creature_clips.wanted_motions("quadruped", chosen=False) == ("trot", "attack")
    assert creature_clips.wanted_motions("humanoid", chosen=False) == ()
    assert creature_clips.wanted_motions("humanoid", chosen=True) == ("walk", "attack")
    assert creature_clips.wanted_motions("plant", chosen=False) == ("sway", "attack")
    assert creature_clips.wanted_motions("kraken", chosen=True) == ()  # not a family


def test_clips_made_from_the_command_line_get_provenance(tmp_path):
    import json

    rig = tmp_path / "wolf_rigged.glb"
    rig.write_bytes(b"glTF")
    rig.with_suffix(".provenance.json").write_text(json.dumps({"license": {"name": "MIT"}}), encoding="utf-8")
    clip = creature_clips.clip_path(rig, "digitigrade-walk")
    clip.write_bytes(b"glTF")
    path = creature_clips.write_provenance(rig, clip, "digitigrade-walk", "digitigrade")
    assert path == tmp_path / "wolf_digitigrade-walk.provenance.json"
    assert json.loads(path.read_text(encoding="utf-8"))["license"] == {"name": "MIT"}


def test_a_rig_that_cannot_be_read_is_reported_not_failed(tmp_path):
    rig = tmp_path / "broken_rigged.glb"
    rig.write_bytes(b"not a glb")
    found = creature_clips.recognise(rig, None)
    assert found["family"] is None and "could not" in found["clips_error"]


def test_library_clips_come_first_and_replace_procedural_ones_of_the_same_name():
    plan = creature_clips.plan("digitigrade", chosen=False, legs=2, installed=True)
    assert plan[:5] == [("idle", "kaiju"), ("walk", "kaiju"), ("attack", "kaiju"), ("hit", "kaiju"),
                        ("death", "kaiju")]
    assert plan[5:] == []  # walk and attack are the library's now
    quadruped = creature_clips.plan("quadruped", chosen=False, legs=4, installed=True)
    assert ("trot", None) in quadruped and ("attack", "fox") in quadruped and ("attack", None) not in quadruped


def test_without_the_library_the_procedural_clips_are_made_as_before():
    assert creature_clips.plan("digitigrade", chosen=False, legs=2, installed=False) == [("walk", None),
                                                                                         ("attack", None)]


def test_a_winged_creature_takes_the_dragon_on_four_legs_and_the_bird_on_two():
    assert ("fly", "dragon") in creature_clips.plan("winged", chosen=False, legs=4, installed=True)
    assert ("fly", "bird") in creature_clips.plan("winged", chosen=False, legs=2, installed=True)


def test_plants_and_humanoids_get_no_library_clips():
    assert all(src is None for _, src in creature_clips.plan("plant", chosen=False, legs=0, installed=True))
    assert creature_clips.plan("humanoid", chosen=False, legs=2, installed=True) == []


def test_a_library_clip_records_where_its_motion_came_from(tmp_path):
    import json

    from image_to_3dlab import clip_library as cl

    rig = tmp_path / "wolf_rigged.glb"
    rig.write_bytes(b"glTF")
    clip = creature_clips.clip_path(rig, "digitigrade-walk")
    clip.write_bytes(b"glTF")
    made = cl.motion_record("kaiju", "walk")
    path = creature_clips.write_provenance(rig, clip, "digitigrade-walk", "digitigrade", made=made)
    motion = json.loads(path.read_text(encoding="utf-8"))["motion"]
    assert motion["license"]["name"] == "CC0-1.0" and motion["source_clip"] == "Walk"
    assert motion["source_sha256"] == cl.MESH2MOTION["files"]["static/animations/kaiju-animations.glb"][0]
    assert motion["clip"] == "digitigrade-walk"
