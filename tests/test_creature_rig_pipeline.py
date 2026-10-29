"""The skeleton step end to end: shell.npz and draws in, joints.json out."""

from __future__ import annotations

import json

import pytest
from creature_rig_shapes import (
    biped,
    quadruped,
    quadruped_draw,
    robed,
    spider,
    work_folder,
    write_draws,
)

from image_to_3dlab.creature_rig import pipeline

DRAW_KEYS = {"score", "missed_contacts", "contacts", "mirror_gap", "far_share", "joints", "file"}
BONE_KEYS = {"name", "parent", "head", "tail", "kind", "side", "deform"}
KINDS = {"root", "spine", "neck", "head", "tail", "jaw", "hind", "front", "biped", "arm", "wing", "chain", "part"}


def _check_contract(result: dict, work) -> None:
    assert json.loads((work / "joints.json").read_text(encoding="utf-8")) == result
    assert {"body", "height", "facing", "bones", "draws", "chosen", "notes"} <= set(result)
    names = [b["name"] for b in result["bones"]]
    assert len(names) == len(set(names))
    seen = set()
    for b in result["bones"]:
        assert set(b) == BONE_KEYS
        assert b["kind"] in KINDS
        assert b["parent"] is None or b["parent"] in seen          # parents first
        assert b["side"] == (b["name"][-1] if b["name"][-2:] in (".L", ".R") else "")
        seen.add(b["name"])
    root = result["bones"][0]
    assert root["name"] == "Root" and root["deform"] is False and root["parent"] is None


@pytest.fixture(scope="module")
def quad_work(tmp_path_factory):
    return work_folder(tmp_path_factory.mktemp("quad"), quadruped())


def test_the_best_draw_is_kept_named_and_written(quad_work):
    lopsided = quadruped_draw(move={"knee.L": (0.1, 0.3, 0.36), "hock.L": (0.1, 0.45, 0.16),
                                    "elbow.L": (0.1, -0.3, 0.36), "wrist.L": (0.1, -0.45, 0.16)})
    paths = write_draws(quad_work, [lopsided, quadruped_draw()])
    result = pipeline.joints_from_draws(quad_work, paths)
    _check_contract(result, quad_work)
    assert result["body"] == "quadruped" and result["facing"] == "-Y"
    assert result["chosen"] == 1
    assert result["height"] == pytest.approx(1.05, abs=1e-3)
    assert [d["file"] for d in result["draws"]] == ["draw0.json", "draw1.json"]
    assert all(DRAW_KEYS <= set(d) for d in result["draws"])
    assert result["draws"][0]["score"] > result["draws"][1]["score"]
    names = {b["name"] for b in result["bones"]}
    assert {"Hips", "Tail1", "Spine1", "Head", "Thigh.L", "HindToes.R", "UpperArm.L", "FrontToes.R"} <= names
    assert result["notes"][0].startswith("draw 1 kept")


def test_a_broken_draw_is_skipped(quad_work):
    paths = write_draws(quad_work, [quadruped_draw()])
    bad = quad_work / "draws" / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    result = pipeline.joints_from_draws(quad_work, [bad, *paths])
    assert result["draws"][0]["score"] is None and "error" in result["draws"][0]
    assert result["chosen"] == 1


def test_no_usable_draw_falls_back_to_the_centre_lines(quad_work):
    bad = quad_work / "draws" / "broken.json"
    bad.write_text(json.dumps({"names": [], "parents": [], "heads": [], "tails": []}), encoding="utf-8")
    result = pipeline.joints_from_draws(quad_work, [bad])
    _check_contract(result, quad_work)
    assert result["body"] == "centreline" and result["plan"] == "quadruped" and result["chosen"] is None
    assert any("centre-line" in n for n in result["notes"])


def test_a_missed_contact_gets_a_leg(quad_work):
    paths = write_draws(quad_work, [quadruped_draw(drop=("shoulder.L",))])
    result = pipeline.joints_from_draws(quad_work, paths)
    _check_contract(result, quad_work)
    names = {b["name"] for b in result["bones"]}
    assert result["body"] == "quadruped" and {"FrontToes.L", "FrontToes.R"} <= names
    assert result["draws"][0]["missed_contacts"] == 1
    assert any("floor contact without a leg" in n for n in result["notes"])
    assert not any("still without" in n for n in result["notes"])


def test_without_unirig_the_centre_lines_rig_it(tmp_path):
    work = work_folder(tmp_path / "w", biped())
    result = pipeline.joints_without_unirig(work)
    _check_contract(result, work)
    assert result["body"] == "centreline" and result["plan"] == "biped"
    assert result["draws"] == [] and result["chosen"] is None
    assert {"Thigh.L", "Toes.R", "Arm.L", "Head"} <= {b["name"] for b in result["bones"]}


def test_many_legs_skip_the_draws_both_ways(tmp_path):
    work = work_folder(tmp_path / "w", spider())
    paths = write_draws(work, [quadruped_draw()])
    result = pipeline.joints_from_draws(work, paths)
    _check_contract(result, work)
    assert result["body"] == "manylegs" and result["draws"] == [] and result["chosen"] is None
    assert any("drawn skeletons not used" in n for n in result["notes"])
    names = {b["name"] for b in result["bones"]}
    assert {"Coxa1.L", "Tarsus4.R"} <= names
    again = pipeline.joints_without_unirig(work)
    assert again["body"] == "manylegs"


def test_a_robe_on_the_floor_gets_no_legs(tmp_path):
    work = work_folder(tmp_path / "w", robed())
    torso = {"names": ["hips", "chest", "head", "arm.L", "arm.R"], "parents": [-1, 0, 1, 1, 1],
             "heads": [[0, 0, 0.65], [0, 0, 0.9], [0, 0, 1.05], [0.1, 0, 0.95], [-0.1, 0, 0.95]],
             "tails": [[0, 0, 0.9], [0, 0, 1.05], [0, 0, 1.18], [0.14, 0, 0.7], [-0.14, 0, 0.7]]}
    paths = write_draws(work, [torso])
    result = pipeline.joints_from_draws(work, paths)
    assert result["draws"][0]["contacts"] == 0
    assert not any("floor contact" in n for n in result["notes"])
    assert not any(b["kind"] in ("hind", "front", "biped") for b in result["bones"])
