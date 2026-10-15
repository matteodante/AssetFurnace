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
    assert creature_clips.wanted_motions("plant", chosen=True) == ()  # none yet
