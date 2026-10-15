"""`scripts/export_unreal.py`: the request is checked, Blender's half is wired, files land.

Blender itself is faked except in the last test, which runs the real thing on one
triangle when Blender is installed: it is what catches an exporter option that a new
Blender renamed, in seconds rather than after a batch.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import export_unreal as ex
import pytest

from image_to_3dlab import unreal_profile as up
from image_to_3dlab.blender import find_blender
from image_to_3dlab.glb_turn import build_glb, parse_glb


def _triangle_glb(material="MI_Thing", size=1.0, node="Thing") -> bytes:
    positions = struct.pack("<9f", 0, 0, 0, size, 0, 0, 0, 0, size)
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": node, "mesh": 0}],
        "meshes": [{"name": "Thing", "primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{"name": material}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [size, 0, size]}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}],
    }
    return build_glb(document, positions)


@pytest.fixture
def files(tmp_path):
    rig = tmp_path / "anim"
    rig.mkdir()
    for name in ("Wolf_9k_rigged.glb", "Wolf_9k_idle.glb", "Wolf_9k_hit_react.glb"):
        (rig / name).write_bytes(_triangle_glb())
    (rig / "Wolf_9k_rigged.provenance.json").write_text(json.dumps(
        {"license": {"name": "MIT"}, "classification": "commercial"}))
    props = tmp_path / "props" / "prop_01"
    props.mkdir(parents=True)
    for i in range(3):
        (props / f"prop_01_LOD{i}.glb").write_bytes(_triangle_glb())
    (props / "prop_01_LOD1.web.glb").write_bytes(b"x")
    part = tmp_path / "Helmet.glb"
    part.write_bytes(_triangle_glb())
    return {"rig": rig, "props": props, "part": part, "root": tmp_path}


def test_size_is_required_because_a_generated_model_has_none(files):
    with pytest.raises(ValueError, match="size-cm"):
        ex.plan(files["part"], kind="prop", size_cm=None)


@pytest.mark.parametrize("kwargs, message", [
    ({"kind": "equipment"}, "--body"),
    ({"kind": "prop", "animations": ["x"]}, "animations"),
    ({"kind": "vehicle"}, "--kind"),
    ({"kind": "prop", "measure": "width"}, "--measure"),
    ({"kind": "prop", "collision": "sphere"}, "--collision"),
])
def test_requests_that_cannot_work_are_refused_before_blender(files, kwargs, message):
    with pytest.raises(ValueError, match=message):
        ex.plan(files["part"], size_cm=50, **kwargs)


def test_a_missing_input_is_named(files):
    with pytest.raises(ValueError, match="does not exist"):
        ex.plan(files["part"], kind="creature", size_cm=50, animations=[files["root"] / "nope.glb"])


def test_a_creature_is_named_after_its_model_and_each_move_after_itself(files):
    rig = files["rig"]
    p = ex.plan(rig / "Wolf_9k_rigged.glb", kind="creature", size_cm=120,
                animations=[rig / "Wolf_9k_idle.glb", rig / "Wolf_9k_hit_react.glb"], out_dir=files["root"] / "out")
    assert p["names"]["mesh"] == "SK_Wolf9k"
    assert [a["name"] for a in p["job"]["animations"]] == ["AS_Wolf9k_Idle", "AS_Wolf9k_HitReact"]
    assert p["job"]["collision"] == "none"
    assert p["glb"] == files["root"] / "out" / "SK_Wolf9k.glb"


def test_an_object_picks_up_its_baked_lods_and_gets_a_box(files):
    p = ex.plan(files["props"] / "prop_01_LOD0.glb", kind="prop", size_cm=60, measure="longest")
    assert p["names"]["mesh"] == "SM_Prop01"
    assert [Path(x).name for x in p["job"]["lods"]] == ["prop_01_LOD1.glb", "prop_01_LOD2.glb"]
    assert p["job"]["collision"] == "box"
    assert p["out_dir"].parent == ex.OUTPUT_ROOT


def test_equipment_shares_the_body_skeleton_name(files):
    p = ex.plan(files["part"], kind="equipment", size_cm=180, body=files["rig"] / "Wolf_9k_rigged.glb")
    assert p["skeleton_from"] == "SK_Wolf9k"
    assert p["job"]["body"].endswith("Wolf_9k_rigged.glb")
    named = ex.plan(files["part"], kind="equipment", size_cm=180, body=files["rig"] / "Wolf_9k_rigged.glb",
                    body_name="Wolf")
    assert named["skeleton_from"] == "SK_Wolf"


def test_blender_runs_headless_with_factory_settings(tmp_path):
    command = ex.blender_command(Path("blender"), tmp_path / "job.json")
    assert command[:3] == ["blender", "-b", "--factory-startup"]
    assert command[-2:] == ["--", str(tmp_path / "job.json")]
    assert str(ex.BLENDER_SCRIPT) in command


def _fake_blender(report_extra=None, material="MI_Wolf9k"):
    def run(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        mesh = up.asset_names(job["kind"], job["base"])["mesh"]
        Path(job["out"]).write_bytes(_triangle_glb(material, node=mesh))
        report = {"ok": True, "lods": 1, "humanoid": True, "animations": [a["name"] for a in job["animations"]],
                  "scale": 2.0}
        report.update(report_extra or {})
        Path(job["report"]).write_text(json.dumps(report), encoding="utf-8")
    return run


def test_export_writes_glb_sidecar_and_provenance(files, tmp_path):
    rig = files["rig"]
    tint = tmp_path / "mask.png"
    from PIL import Image

    Image.new("RGB", (4, 4), (255, 0, 0)).save(tint)
    result = ex.export_unreal(rig / "Wolf_9k_rigged.glb", kind="creature", size_cm=120,
                              animations=[rig / "Wolf_9k_idle.glb"], tint=tint, out_dir=tmp_path / "out",
                              blender=Path("blender"), run=_fake_blender(), log=lambda m: None)
    out = tmp_path / "out"
    assert result["ok"] and result["problems"] == []
    assert (out / "SK_Wolf9k.glb").is_file()
    sidecar = json.loads((out / "SK_Wolf9k.unreal.json").read_text(encoding="utf-8"))
    assert sidecar["animations"] == ["AS_Wolf9k_Idle"]
    assert sidecar["tint"] == "T_Wolf9k_Tint.png" and (out / "T_Wolf9k_Tint.png").is_file()
    assert sidecar["expect"]["manny_compatible"] is True
    record = json.loads((out / "SK_Wolf9k.provenance.json").read_text(encoding="utf-8"))
    assert record["license"] == {"name": "MIT"}
    assert [s["file"] for s in record["sources"]] == ["Wolf_9k_rigged.glb", "Wolf_9k_idle.glb"]
    assert record["sources"][1]["record_file"] == "Wolf_9k_rigged.provenance.json"  # the run's record
    assert not (out / ".work").exists()
    document, _ = parse_glb((out / "SK_Wolf9k.glb").read_bytes())
    assert document["asset"]["extras"][up.EXTRAS_KEY]["size_cm"] == 120


def test_problems_are_reported_and_blenders_files_kept(files, tmp_path):
    result = ex.export_unreal(files["part"], kind="prop", size_cm=40, out_dir=tmp_path / "out",
                              blender=Path("blender"), run=_fake_blender(material="Material.001"),
                              log=lambda m: None)
    assert not result["ok"]
    assert any("Material.001" in p for p in result["problems"])
    assert (tmp_path / "out" / ".work" / "blender.glb").is_file()


def test_a_blender_failure_is_raised_with_its_reason(files, tmp_path):
    def run(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        Path(job["report"]).write_text(json.dumps({"ok": False, "error": "ValueError: no mesh"}))
    with pytest.raises(RuntimeError, match="no mesh"):
        ex.export_unreal(files["part"], kind="prop", size_cm=40, out_dir=tmp_path / "out",
                         blender=Path("blender"), run=run, log=lambda m: None)


def test_cli_refuses_a_bad_request_without_a_traceback(files, capsys):
    assert ex.main([str(files["part"]), "--kind", "equipment", "--size-cm", "10"]) == 2
    assert "--body" in capsys.readouterr().err


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_sizes_and_grounds_an_object(tmp_path):
    source = tmp_path / "thing.glb"
    source.write_bytes(_triangle_glb(size=0.5))
    result = ex.export_unreal(source, kind="prop", size_cm=200, measure="longest", out_dir=tmp_path / "out",
                              log=lambda m: None)
    assert result["ok"], result["problems"]
    sidecar = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))
    lo, hi = sidecar["blender"]["box_after_m"]
    assert max(h - low for low, h in zip(lo, hi)) == pytest.approx(2.0, rel=1e-4)
    assert lo[2] == pytest.approx(0.0, abs=1e-5)
    document, _ = parse_glb(Path(result["glb"]).read_bytes())
    assert {n["name"] for n in document["nodes"]} == {"SM_Thing", "UBX_SM_Thing_00"}


@pytest.mark.parametrize("alias, kind", [("object", "prop"), ("equipment_part", "equipment"), ("creature", "creature")])
def test_the_kind_names_the_other_steps_use_are_understood(files, alias, kind):
    body = files["rig"] / "Wolf_9k_rigged.glb" if kind == "equipment" else None
    assert ex.plan(files["part"], kind=alias, size_cm=50, body=body)["job"]["kind"] == kind


def test_the_material_steps_tint_mask_and_record_are_picked_up(files, tmp_path):
    from PIL import Image

    source = files["root"] / "game" / "Barrel.glb"
    source.parent.mkdir()
    source.write_bytes(_triangle_glb())
    Image.new("RGB", (4, 4), (0, 255, 0)).save(source.parent / "T_Barrel_Tint.png")
    (source.parent / "Barrel.materials.json").write_text(json.dumps({"tint": {"zones": []}}))
    p = ex.plan(source, kind="prop", size_cm=50)
    assert p["tint_source"] == source.parent / "T_Barrel_Tint.png"
    result = ex.export_unreal(source, kind="prop", size_cm=50, out_dir=tmp_path / "out", blender=Path("blender"),
                              run=_fake_blender(material="MI_Barrel"), log=lambda m: None)
    out = tmp_path / "out"
    assert (out / "T_Barrel_Tint.png").is_file()
    assert json.loads((out / "SM_Barrel.materials.json").read_text()) == {"tint": {"zones": []}}
    sidecar = json.loads((out / "SM_Barrel.unreal.json").read_text())
    assert sidecar["tint"] == "T_Barrel_Tint.png" and sidecar["materials_record"] == "SM_Barrel.materials.json"
    assert result["ok"]


def test_an_ambiguous_tint_mask_is_not_guessed(files):
    from PIL import Image

    for name in ("T_A_Tint.png", "T_B_Tint.png"):
        Image.new("RGB", (2, 2)).save(files["part"].parent / name)
    assert ex.plan(files["part"], kind="prop", size_cm=50)["tint_source"] is None
