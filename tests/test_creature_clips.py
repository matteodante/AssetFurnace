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
