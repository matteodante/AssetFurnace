"""The pure parts of scripts/blender_fit_parts.py: arguments, the job file, which bones a
slot sits on and which one a rigid piece follows."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_fit_parts.py"


def _load():
    spec = importlib.util.spec_from_file_location("blender_fit_parts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # bpy is only imported inside main()
    return module


bf = _load()

ROLES = {
    "b0": ("hips", 0, ""), "b1": ("spine", 1, ""), "b2": ("neck", 0, ""), "b3": ("head", 1, ""),
    "b4": ("clavicle", 0, "R"), "b5": ("upperarm", 1, "R"), "b6": ("forearm", 2, "R"), "b7": ("hand", 3, "R"),
    "b8": ("hand", 4, "R"), "b9": ("clavicle", 0, "L"), "b10": ("forearm", 2, "L"), "b11": ("hand", 3, "L"),
}


def test_parse_args_wants_one_job_file():
    assert bf.parse_args(["blender", "-b", "--python", "x.py", "--", "job.json"]) == Path("job.json")
    with pytest.raises(SystemExit):
        bf.parse_args(["blender", "--", "a", "b"])


def test_load_job_checks_what_it_needs(tmp_path):
    job = {"body": "b.json", "concept_size": [1024, 1024], "out": "o",
           "pieces": [{"slot": "helmet", "glb": "h.glb", "concept_box": [0, 0, 1, 1]}]}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert bf.load_job(path)["pieces"][0]["slot"] == "helmet"
    for broken, words in ((dict(job, pieces=[{"slot": "cape", "concept_box": [0, 0, 1, 1]}]), "cape"),
                          (dict(job, pieces=[{"slot": "helmet"}]), "concept_box"),
                          ({k: v for k, v in job.items() if k != "out"}, "out")):
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match=words):
            bf.load_job(path)


def test_a_paired_slot_takes_only_its_side():
    assert sorted(bf.slot_bones(ROLES, "gloves", "R")) == ["b6", "b7", "b8"]
    assert sorted(bf.slot_bones(ROLES, "gloves", "L")) == ["b10", "b11"]
    assert sorted(bf.slot_bones(ROLES, "helmet", None)) == ["b2", "b3"]


def test_rigid_pieces_follow_the_first_head_bone_or_the_right_hand():
    assert bf.rigid_bone(ROLES, "helmet") == "b3"
    assert bf.rigid_bone(ROLES, "weapon") == "b7"
    assert bf.rigid_bone(ROLES, "shield") == "b10"  # the left forearm unless told the right
    assert bf.rigid_bone(ROLES, "shield", side="R") == "b6"
    assert bf.rigid_bone(ROLES, "torso") is None
    no_head = {b: r for b, r in ROLES.items() if r[0] != "head"}
    assert bf.rigid_bone(no_head, "helmet") == "b2"
