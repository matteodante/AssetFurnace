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

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            job = json.loads(Path(command[3].split()[-1]).read_text())
            Path(job["measured"]).write_text(json.dumps({"assets": {"SM_Box": {
                "class": "StaticMesh", "bounds_cm": [60, 1, 1], "min_z": 0, "lods": 1,
                "collision": {"box_elems": 1}, "names": ["SM_Box"], "orm_srgb": [False]}}}))

    assert uc.check([folder], project=tmp_path / "P" / "P.uproject", editor=editor, run=run) is True
    assert len(calls) == 2
