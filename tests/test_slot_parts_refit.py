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
