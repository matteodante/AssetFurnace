"""Carrying a fitted set over to another body, and the set for Unreal: the pure parts, the
jobs, and the export with the exporter stood in for."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab.slot_parts import fit
from tests.test_blender_fit_parts import bf
from tests.test_slot_parts_cli import _body, cli


def test_region_transfer_maps_box_onto_box():
    m = fit.region_transfer(((0, 0, 0), (1, 2, 1)), ((10, 10, 10), (12, 13, 11)))
    corners = np.array([[0, 0, 0], [1, 2, 1], [0.5, 1, 0.5]])
    moved = corners @ m[:3, :3].T + m[:3, 3]
    assert np.allclose(moved, [[10, 10, 10], [12, 13, 11], [11, 11.5, 10.5]])


def _fit_dir(tmp_path):
    folder = tmp_path / "set" / "first" / "fit"
    folder.mkdir(parents=True)
    files = ["set.glb", "preview.glb", "body_masked.glb", "body_trimmed.glb", "helmet.glb", "gloves_R.glb", "gloves_L.glb"]
    for name in files:
        (folder / name).write_bytes(b"glb")
    (folder / "fit.json").write_text(json.dumps({"files": files, "pieces": {}, "covered_share": 0.7}))
    return folder


def test_refit_job_carries_every_fitted_piece(tmp_path):
    folder = _fit_dir(tmp_path)
    second, *_ = _body(tmp_path)
    job = cli.refit_job(folder, tmp_path / "first.json", second, tmp_path / "out")
    assert job["clips"] == []
    assert [(p["slot"], Path(p["glb"]).name) for p in job["pieces"]] == [
        ("helmet", "helmet.glb"), ("gloves", "gloves_R.glb"), ("gloves", "gloves_L.glb")]
    assert all(p["transfer"] for p in job["pieces"]) and job["source_body"].endswith("first.json")
    assert [p["split"] for p in job["pieces"]] == [False, True, True]  # an _L/_R pair stays two files


def test_a_fit_from_before_weapons_were_their_own_is_refused_not_half_carried(tmp_path):
    """Weapons and shields became assets of their own on 17/10/2026; a fit that still lists
    one is made again, never carried or exported with the weapon quietly left out."""
    folder = _fit_dir(tmp_path)
    report = json.loads((folder / "fit.json").read_text())
    (folder / "weapon.glb").write_bytes(b"glb")
    (folder / "fit.json").write_text(json.dumps(dict(report, files=[*report["files"], "weapon.glb"])))
    second, *_ = _body(tmp_path)
    with pytest.raises(ValueError, match="run the set again"):
        cli.refit_job(folder, tmp_path / "first.json", second, tmp_path / "out")
    with pytest.raises(ValueError, match="run the set again"):
        cli.unreal_set(folder, second, tmp_path / "unreal", exporter=lambda *a, **k: {"glb": "x", "problems": []})


def test_a_transferred_piece_needs_the_first_body(tmp_path):
    job = {"body": "b.json", "concept_size": [1, 1], "out": "o", "pieces": [{"slot": "helmet", "glb": "h.glb", "transfer": True}]}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="source_body"):
        bf.load_job(path)
    path.write_text(json.dumps(dict(job, source_body="first.json")))
    assert bf.load_job(path)["source_body"] == "first.json"


def test_the_set_for_unreal_shares_one_skeleton(tmp_path):
    folder = _fit_dir(tmp_path)
    body_json, *_ = _body(tmp_path)
    calls = []

    def exporter(source, **kwargs):
        calls.append((Path(source).name, kwargs))
        return {"glb": str(tmp_path / f"{kwargs['name']}.glb"), "problems": []}

    path = cli.unreal_set(folder, body_json, tmp_path / "unreal", size_cm=172, exporter=exporter)
    summary = json.loads(path.read_text())
    assert calls[0][0] == "body_masked.glb" and calls[0][1]["kind"] == "creature" and calls[0][1]["size_cm"] == 172
    assert {c[0] for c in calls[1:]} == {"helmet.glb", "gloves_R.glb", "gloves_L.glb"}
    assert all(c[1]["kind"] == "equipment" and c[1]["body_name"] == "Stick" for c in calls[1:])
    assert summary["skeleton"] == "SK_Stick_Skeleton" and set(summary["meshes"]) == {"body", "helmet", "gloves_R", "gloves_L"}
    assert any("SetLeaderPoseComponent" in line for line in summary["how"])
    assert any("Opacity Mask" in line for line in summary["how"])


def test_each_piece_goes_to_unreal_with_its_material(tmp_path):
    folder = _fit_dir(tmp_path)
    body_json, *_ = _body(tmp_path)
    parts = {"set": "first", "slots": {"helmet": {"prompt": "a leather hood"}, "gloves": {"prompt": ""}},
             "pieces": {"helmet": {}, "gloves": {}}, "concept": {"prompt": "steel gauntlets", "outfit": "steel gauntlets"}}
    (folder.parent / "first.parts.json").write_text(json.dumps(parts))
    calls = {}

    def exporter(source, **kwargs):
        calls[Path(source).name] = kwargs.get("material")
        return {"glb": str(tmp_path / f"{kwargs['name']}.glb"), "problems": []}

    path = cli.unreal_set(folder, body_json, tmp_path / "unreal", exporter=exporter)
    assert calls == {"body_masked.glb": None, "helmet.glb": "leather", "gloves_R.glb": "metal", "gloves_L.glb": "metal"}
    summary = json.loads(path.read_text())
    assert summary["materials"]["helmet"] == {"material": "leather", "by": "rule",
                                              "reason": "the piece's words say 'leather'"}


def test_a_set_without_its_record_gets_the_slot_defaults(tmp_path):
    folder = _fit_dir(tmp_path)
    body_json, *_ = _body(tmp_path)
    calls = {}

    def exporter(source, **kwargs):
        calls[Path(source).name] = kwargs.get("material")
        return {"glb": "x.glb", "problems": []}

    cli.unreal_set(folder, body_json, tmp_path / "unreal", exporter=exporter)
    assert calls["helmet.glb"] == "metal"


def test_the_cli_shows_and_corrects_a_sets_materials(tmp_path, capsys):
    folder = tmp_path / "set" / "first"
    folder.mkdir(parents=True)
    (folder / "first.parts.json").write_text(json.dumps(
        {"set": "first", "slots": {"helmet": {"prompt": ""}}, "pieces": {"helmet": {}}, "concept": {"outfit": ""}}))
    assert cli.main(["materials", str(folder), "--set", "helmet=cloth"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["helmet"]["material"] == "cloth" and shown["helmet"]["by"] == "person"
    assert cli.main(["materials", str(folder), "--clear", "helmet"]) == 0
    assert json.loads(capsys.readouterr().out)["helmet"]["by"] == "rule"
    assert cli.main(["materials", str(folder), "--set", "helmet=steel"]) == 1


def test_the_sets_body_alone_is_exported_with_its_opacity_mask(tmp_path):
    folder = _fit_dir(tmp_path)
    body_json, *_ = _body(tmp_path)
    calls = {}

    def exporter(source, **kwargs):
        calls[Path(source).name] = kwargs.get("opacity_mask", False)
        return {"glb": str(tmp_path / f"{kwargs['name']}.glb"), "problems": []}

    cli.unreal_set(folder, body_json, tmp_path / "unreal", exporter=exporter)
    assert calls["body_masked.glb"] is True and not any(v for k, v in calls.items() if k != "body_masked.glb")
