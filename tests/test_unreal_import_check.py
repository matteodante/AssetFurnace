"""`scripts/unreal_import_check.py`, the part that runs outside Unreal.

What Unreal does inside is exercised by the acceptance run; here the project it makes,
the commands it launches and how measurements become pass/fail files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import unreal_import_check as uc

from image_to_3dlab import unreal_profile as up


def _sidecar(kind, mesh, **extra):
    names = up.asset_names(kind, mesh[3:])
    side = up.sidecar(kind=kind, names=names, glb=f"{mesh}.glb", size_cm=extra.pop("size_cm", 60),
                      measure="longest" if kind == "prop" else "height", lods=extra.pop("lods", 1),
                      collision="box" if kind == "prop" else None, humanoid=kind != "prop",
                      skeleton_from=extra.pop("skeleton_from", None), animations=[], tint=None, problems=[])
    side.update(extra)
    return side


def _export(tmp_path, kind, mesh, **extra):
    folder = tmp_path / mesh
    folder.mkdir()
    (folder / f"{mesh}.glb").write_bytes(b"glb")
    (folder / f"{mesh}.unreal.json").write_text(json.dumps(_sidecar(kind, mesh, **extra)))
    return folder


def test_the_project_carries_only_what_the_check_needs():
    plugins = {p["Name"] for p in uc.project_document()["Plugins"]}
    assert {"PythonScriptPlugin", "Interchange", "IKRig"} <= plugins


def test_ensure_project_copies_manny_from_the_engine_once(tmp_path):
    engine = tmp_path / "UE_5.8"
    manny = engine / uc.MANNY_TEMPLATE / "Meshes"
    manny.mkdir(parents=True)
    (manny / "SKM_Manny_Simple.uasset").write_bytes(b"x")
    project = uc.ensure_project(tmp_path / "P" / "P.uproject", engine, need_manny=True)
    assert json.loads(project.read_text())["Plugins"]
    assert (project.parent / "Content/Characters/Mannequins/Meshes/SKM_Manny_Simple.uasset").is_file()
    uc.ensure_project(project, engine, need_manny=True)  # a second run leaves it alone


def test_ensure_project_says_when_manny_is_missing(tmp_path):
    with pytest.raises(SystemExit, match="Manny"):
        uc.ensure_project(tmp_path / "P" / "P.uproject", tmp_path / "UE", need_manny=True)


def test_equipment_is_imported_after_its_body(tmp_path):
    exports = uc.load_exports([_export(tmp_path, "equipment", "SK_Helm", skeleton_from="SK_Body"),
                               _export(tmp_path, "creature", "SK_Body"), _export(tmp_path, "prop", "SM_Box")])
    assert [e["sidecar"]["mesh"] for e in exports][-1] == "SK_Helm"


def test_a_folder_without_one_sidecar_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        uc.load_exports([tmp_path])


def test_unreal_runs_once_to_import_then_once_to_render(tmp_path):
    editor = tmp_path / "UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe"
    first, second = uc.editor_commands(editor, tmp_path / "P.uproject", tmp_path / "job.json")
    assert Path(first[0]).name == "UnrealEditor-Cmd.exe"
    assert "-run=pythonscript" in first and "-NullRHI" in first
    assert any(a.startswith("-script=") and "--inside import" in a for a in first)
    assert Path(second[0]).name == "UnrealEditor.exe"
    assert "-RenderOffscreen" in second
    assert any(a.startswith("-ExecCmds=py ") and "--inside capture" in a for a in second)


def test_results_are_written_beside_each_export(tmp_path):
    exports = uc.load_exports([_export(tmp_path, "prop", "SM_Box", lods=1)])
    picture = tmp_path / "shot.png"
    picture.write_bytes(b"png")
    measured = {"engine": "5.8.3", "assets": {"SM_Box": {
        "class": "StaticMesh", "bounds_cm": [60, 20, 20], "min_z": 0.0, "lods": 1,
        "collision": {"box_elems": 1, "convex_elems": 0}, "names": ["SM_Box"], "orm_srgb": [False]}}}
    assert uc.write_results(exports, measured, picture) is True
    written = json.loads((tmp_path / "SM_Box" / "unreal_check.json").read_text())
    assert written["ok"] and written["picture"] == "unreal_check.png"
    assert (tmp_path / "SM_Box" / "unreal_check.png").read_bytes() == b"png"


def test_an_asset_unreal_did_not_import_fails(tmp_path):
    exports = uc.load_exports([_export(tmp_path, "prop", "SM_Box")])
    assert uc.write_results(exports, {"assets": {}}, None) is False
    written = json.loads((tmp_path / "SM_Box" / "unreal_check.json").read_text())
    assert written["results"][0]["check"] == "import"


def test_check_runs_both_stages_and_reads_what_unreal_measured(tmp_path):
    folder = _export(tmp_path, "prop", "SM_Box", lods=1)
    editor = tmp_path / "UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe"
    editor.parent.mkdir(parents=True)
    editor.write_bytes(b"")
    calls = []

    jobs = []

    def run(command, **kwargs):
        calls.append(command)
        job_path = Path((command[3] if len(calls) == 1 else command[2]).split()[-1])
        job = json.loads(job_path.read_text())
        jobs.append(job)
        if len(calls) == 1:
            Path(job["measured"]).write_text(json.dumps({"sun": [1, 0, -1], "assets": {"SM_Box": {
                "class": "StaticMesh", "bounds_cm": [60, 1, 1], "min_z": 0, "lods": 1,
                "collision": {"box_elems": 1}, "names": ["SM_Box"], "orm_srgb": [False]}}}))
        else:
            Path(job["picture"]).write_bytes(b"idle")
            Path(job["picture_walk"]).write_bytes(b"walk")

    assert uc.check([folder], project=tmp_path / "P" / "P.uproject", editor=editor, run=run) is True
    assert len(calls) == 2
    assert "stage" not in jobs[0] and jobs[1]["stage"]["order"] == ["SM_Box"]  # placed from what Unreal measured
    assert (folder / "unreal_check_walk.png").read_bytes() == b"walk"


def test_the_picture_stands_manny_first_and_wears_equipment_on_its_body(tmp_path):
    exports = uc.load_exports([_export(tmp_path, "creature", "SK_Body"), _export(tmp_path, "prop", "SM_Box"),
                               _export(tmp_path, "equipment", "SK_Helm", skeleton_from="SK_Body")])
    measured = {"sun": [0.0, -1.0, -0.5], "assets": {
        "SK_Body": {"bounds_cm": [90, 40, 220]}, "SM_Box": {"bounds_cm": [80, 50, 40]}, "SK_Helm": {"bounds_cm": [1, 1, 1]}}}
    stage = uc.stage_for(exports, measured, manny=True)
    assert stage["order"] == ["Manny", "SK_Body", "SM_Box"]
    assert len(stage["positions"]) == 3
    assert stage["camera"][2] == 110  # half the tallest figure


def test_an_asset_that_failed_to_import_is_left_out_of_the_picture(tmp_path):
    exports = uc.load_exports([_export(tmp_path, "prop", "SM_Box")])
    stage = uc.stage_for(exports, {"assets": {"SM_Box": {"error": "boom"}}}, manny=False)
    assert stage["order"] == []


def test_each_asset_starts_from_an_empty_folder_in_the_check_project(tmp_path):
    folder = _export(tmp_path, "prop", "SM_Box")
    project = tmp_path / "P" / "P.uproject"
    stale = uc.content_folder(project, "SM_Box") / "Old.uasset"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    neighbour = uc.content_folder(project, "SM_Other") / "Keep.uasset"
    neighbour.parent.mkdir(parents=True)
    neighbour.write_bytes(b"keep")
    editor = tmp_path / "UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe"
    editor.parent.mkdir(parents=True)
    editor.write_bytes(b"")
    with pytest.raises(SystemExit):  # the fake Unreal measures nothing
        uc.check([folder], project=project, editor=editor, run=lambda *a, **k: None)
    assert not stale.exists() and neighbour.exists()
    assert uc.content_folder(project, "SM_Box") == project.parent / "Content" / "AF" / "SM_Box"


# --- review round 1 ----------------------------------------------------------------------

def test_a_project_someone_else_made_is_never_cleared(tmp_path):
    folder = _export(tmp_path, "prop", "SM_Box")
    game = tmp_path / "Game" / "Game.uproject"
    game.parent.mkdir()
    game.write_text(json.dumps({"FileVersion": 3, "Description": "my game"}))
    existing = uc.content_folder(game, "SM_Box") / "SM_Box.uasset"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"real asset")
    editor = tmp_path / "UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe"
    editor.parent.mkdir(parents=True)
    editor.write_bytes(b"")
    with pytest.raises(SystemExit, match="will not overwrite"):
        uc.check([folder], project=game, editor=editor, run=lambda *a, **k: None)
    assert existing.read_bytes() == b"real asset"


def test_a_picture_unreal_never_saved_fails_the_check(tmp_path):
    exports = uc.load_exports([_export(tmp_path, "prop", "SM_Box", lods=1)])
    measured = {"assets": {"SM_Box": {
        "class": "StaticMesh", "bounds_cm": [60, 20, 20], "min_z": 0.0, "lods": 1,
        "collision": {"box_elems": 1, "convex_elems": 0}, "names": ["SM_Box"], "orm_srgb": [False]}}}
    assert uc.write_results(exports, measured, tmp_path / "never-written.png") is False
    written = json.loads((tmp_path / "SM_Box" / "unreal_check.json").read_text())
    assert [r for r in written["results"] if r["check"] == "picture"][0]["ok"] is False
