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
        {"license": {"name": "MIT"}, "classification": "commercial-clear"}))
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


def _skinned_glb(stretch_x: float = 1.0) -> bytes:
    """Two bones and two triangles, skinned, with the origin in the middle of the body
    the way SkinTokens leaves it: feet at y = -0.5, head at y = +0.5."""
    import numpy as np

    positions = np.array([[-0.2, -0.5, 0], [0.2, -0.5, 0], [0, 0.5, 0.05],
                          [-0.2, -0.5, 0.1], [0.2, -0.5, 0.1], [0, 0.5, 0.15]], "<f4")
    positions[:, 0] *= stretch_x
    joints = np.zeros((6, 4), "u1")
    joints[[2, 5], 0] = 1
    weights = np.zeros((6, 4), "<f4")
    weights[:, 0] = 1.0
    indices = np.array([0, 1, 2, 3, 4, 5], "<u2")
    inverse = np.stack([np.eye(4, dtype="<f4"), np.eye(4, dtype="<f4")])
    inverse[0, 3, 1] = 0.0  # column-major translation of the hip at y = 0
    inverse[1, 3, 1] = -0.3  # the chest bone sits at y = 0.3
    chunks = [positions.tobytes(), joints.tobytes(), weights.tobytes(), indices.tobytes(),
              inverse.transpose(0, 2, 1).reshape(-1).astype("<f4").tobytes()]
    views, binary = [], b""
    for chunk in chunks:
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(chunk)})
        binary += chunk
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 3]}],
        "nodes": [{"name": "Armature", "children": [1]},
                  {"name": "hip", "children": [2]},
                  {"name": "chest", "translation": [0, 0.3, 0]},
                  {"name": "Body", "mesh": 0, "skin": 0}],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": 4, "skeleton": 1}],
        "meshes": [{"name": "Body", "primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                                                   "indices": 3, "material": 0}]}],
        "materials": [{"name": "Skin"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3",
             "min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
            {"bufferView": 1, "componentType": 5121, "count": 6, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 6, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
            {"bufferView": 4, "componentType": 5126, "count": 2, "type": "MAT4"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    return build_glb(document, binary)


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_puts_a_creatures_root_bone_on_the_floor(tmp_path):
    # Found the expensive way: a root left at the source's origin (mid-body) lifts the
    # character off the ground in Unreal whenever an animation locks root motion.
    source = tmp_path / "critter.glb"
    source.write_bytes(_skinned_glb())
    result = ex.export_unreal(source, kind="creature", size_cm=150, out_dir=tmp_path / "out", log=lambda m: None)
    sidecar = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))
    lo, hi = sidecar["blender"]["box_after_m"]
    assert lo[2] == pytest.approx(0.0, abs=1e-4)
    assert hi[2] - lo[2] == pytest.approx(1.5, rel=1e-3)
    root = sidecar["blender"]["root_head_m"]
    assert root == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)
    assert result["ok"], result["problems"]


# --- review round 1 ----------------------------------------------------------------------

def test_classification_is_read_where_generation_records_keep_it(tmp_path):
    source = tmp_path / "knight.glb"
    source.write_bytes(_triangle_glb())
    (tmp_path / "knight.provenance.json").write_text(json.dumps(
        {"license": {"name": "MIT"}, "output": {"classification": "commercial-conditional"}}))
    record = ex.provenance([source], source)
    assert record["classification"] == "commercial-conditional"
    assert record["license"] == {"name": "MIT"}


def test_a_source_without_a_record_says_its_licence_is_unknown(tmp_path):
    source = tmp_path / "lonely" / "thing.glb"
    source.parent.mkdir()
    source.write_bytes(_triangle_glb())
    record = ex.provenance([source], source)
    assert record["license"] is None and "unknown" in record["note"]


def test_a_failed_rerun_never_publishes_the_previous_result(files, tmp_path):
    out = tmp_path / "out"
    ex.export_unreal(files["part"], kind="prop", size_cm=40, out_dir=out, blender=Path("blender"),
                     run=_fake_blender(material="Material.001"), log=lambda m: None)  # keeps .work
    assert (out / ".work" / "blender.json").is_file()

    def crash(command, **kwargs):
        pass  # Blender dies before writing anything

    with pytest.raises(RuntimeError, match="no report"):
        ex.export_unreal(files["part"], kind="prop", size_cm=40, out_dir=out, blender=Path("blender"),
                         run=crash, log=lambda m: None)


def test_equipment_is_not_checked_without_its_body(files, capsys):
    code = ex.main([str(files["part"]), "--kind", "equipment", "--size-cm", "180",
                    "--body", str(files["rig"] / "Wolf_9k_rigged.glb"), "--check-in-unreal"])
    assert code == 2
    assert "BODY_EXPORT_DIR PART_EXPORT_DIR" in capsys.readouterr().err


def _wrapped_in_an_empty(size=0.5) -> bytes:
    """The triangle under an empty node that moves and doubles it, as some GLBs do."""
    document, binary = parse_glb(_triangle_glb(size=size))
    document["nodes"] = [{"name": "SceneRoot", "children": [1], "translation": [0, 3, 0], "scale": [2, 2, 2]},
                         {"name": "Thing", "mesh": 0}]
    document["scenes"] = [{"nodes": [0]}]
    return build_glb(document, binary)


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_reshapes_a_model_hung_under_an_empty(tmp_path):
    source = tmp_path / "wrapped.glb"
    source.write_bytes(_wrapped_in_an_empty())
    result = ex.export_unreal(source, kind="prop", size_cm=200, measure="longest", out_dir=tmp_path / "out",
                              log=lambda m: None)
    lo, hi = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))["blender"]["box_after_m"]
    assert max(h - low for low, h in zip(lo, hi)) == pytest.approx(2.0, rel=1e-4)
    assert lo[2] == pytest.approx(0.0, abs=1e-5)
    document, _ = parse_glb(Path(result["glb"]).read_bytes())
    assert "SceneRoot" not in {n.get("name") for n in document["nodes"]}


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_renames_bones_whose_names_collide(tmp_path):
    # Cleaning `bone.0` gives `bone_0`, which the other bone already owns until it is renamed.
    source = tmp_path / "collide.glb"
    document, binary = parse_glb(_skinned_glb())
    document["nodes"][1]["name"], document["nodes"][2]["name"] = "bone.0", "bone_0"
    source.write_bytes(build_glb(document, binary))
    result = ex.export_unreal(source, kind="creature", size_cm=150, out_dir=tmp_path / "out", log=lambda m: None)
    bones = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))["blender"]["bones"]
    assert sorted(bones.values()) == ["bone_0", "bone_0_1"]
    names = {n.get("name") for n in parse_glb(Path(result["glb"]).read_bytes())[0]["nodes"]}
    assert {"bone_0", "bone_0_1", "root"} <= names
    assert not any(name and ".00" in name for name in names)


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_refuses_an_animation_file_without_an_animation(tmp_path):
    source = tmp_path / "critter.glb"
    source.write_bytes(_skinned_glb())
    still = tmp_path / "critter_idle.glb"
    still.write_bytes(_skinned_glb())
    with pytest.raises(RuntimeError, match="critter_idle.glb holds no animation"):
        ex.export_unreal(source, kind="creature", size_cm=150, animations=[still], out_dir=tmp_path / "out",
                         log=lambda m: None)


# --- review round 2 ----------------------------------------------------------------------

def test_an_export_never_overwrites_its_own_source(tmp_path):
    source = tmp_path / "SM_Chair.glb"
    source.write_bytes(_triangle_glb())
    with pytest.raises(ValueError, match="overwrite"):
        ex.plan(source, kind="prop", size_cm=50, out_dir=tmp_path)


def test_a_lone_tint_mask_of_another_model_is_not_taken(tmp_path):
    from PIL import Image

    source = tmp_path / "Chair.glb"
    source.write_bytes(_triangle_glb())
    Image.new("RGB", (2, 2)).save(tmp_path / "T_Table_Tint.png")
    assert ex.plan(source, kind="prop", size_cm=50)["tint_source"] is None
    Image.new("RGB", (2, 2)).save(tmp_path / "T_Chair_Tint.png")
    assert ex.plan(source, kind="prop", size_cm=50)["tint_source"] == tmp_path / "T_Chair_Tint.png"


def test_none_is_no_longer_a_collision_choice(files):
    with pytest.raises(ValueError, match="--collision"):
        ex.plan(files["part"], kind="prop", size_cm=50, collision="none")
    assert ex.plan(files["part"], kind="prop", size_cm=50, collision="unreal")["job"]["collision"] == "unreal"


# --- review round 3 ----------------------------------------------------------------------

def test_the_export_is_as_restricted_as_its_strictest_input(files, tmp_path):
    rig = files["rig"]
    (rig / "Wolf_9k_idle.provenance.json").write_text(json.dumps(
        {"license": {"name": "Research"}, "classification": "research-only"}))
    record = ex.provenance([rig / "Wolf_9k_rigged.glb", rig / "Wolf_9k_idle.glb"], rig / "Wolf_9k_rigged.glb")
    assert record["classification"] == "research-only"
    assert {"name": "Research"} in record["licenses"] and {"name": "MIT"} in record["licenses"]


def test_a_blender_that_hangs_is_stopped_with_a_reason(files, tmp_path):
    import subprocess

    def hang(command, **kwargs):
        assert kwargs.get("timeout")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(RuntimeError, match="stopped"):
        ex.export_unreal(files["part"], kind="prop", size_cm=40, out_dir=tmp_path / "out", blender=Path("blender"),
                         run=hang, log=lambda m: None)


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_keeps_a_prop_that_happens_to_be_called_icosphere(tmp_path):
    source = tmp_path / "ball.glb"
    source.write_bytes(_triangle_glb(node="Icosphere"))
    result = ex.export_unreal(source, kind="prop", size_cm=30, measure="longest", out_dir=tmp_path / "out",
                              log=lambda m: None)
    assert result["ok"], result["problems"]
    assert json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))["blender"]["removed"] == []


def _sphere_soup(count=240) -> bytes:
    """Triangles between random points on a sphere: enough for a hull over 60 faces."""
    import numpy as np

    rng = np.random.default_rng(1)
    points = rng.normal(size=(count, 3)).astype("<f4")
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    document, _ = parse_glb(_triangle_glb())
    binary = points.tobytes()
    document["accessors"][0].update(count=count, min=points.min(0).tolist(), max=points.max(0).tolist())
    document["bufferViews"][0]["byteLength"] = len(binary)
    document["buffers"][0]["byteLength"] = len(binary)
    return build_glb(document, binary)


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_writes_one_clean_convex_hull(tmp_path):
    import numpy as np

    source = tmp_path / "rock.glb"
    source.write_bytes(_sphere_soup())
    result = ex.export_unreal(source, kind="prop", size_cm=50, collision="convex", out_dir=tmp_path / "out",
                              log=lambda m: None)
    document, binary = parse_glb(Path(result["glb"]).read_bytes())
    node = next(n for n in document["nodes"] if n.get("name", "").startswith("UCX_"))
    primitive = document["meshes"][node["mesh"]]["primitives"][0]
    position = document["accessors"][primitive["attributes"]["POSITION"]]
    view = document["bufferViews"][position["bufferView"]]
    start = view.get("byteOffset", 0) + position.get("byteOffset", 0)
    points = np.frombuffer(binary[start:start + position["count"] * 12], "<f4").reshape(-1, 3)
    corners = len(np.unique(np.round(points, 5), axis=0))
    triangles = document["accessors"][primitive["indices"]]["count"] // 3
    # A closed convex shell made of triangles has exactly 2V - 4 of them; an old surface
    # left inside it would add more.
    assert triangles == 2 * corners - 4


def test_a_creature_faces_the_way_its_family_file_says(files):
    """A drake facing between +X and +Z in glTF: without its family's forward, Unreal would
    get it turned by half a right angle. glTF (x, y, z) is Blender (x, -z, y)."""
    from image_to_3dlab import creature_family as cf

    rig = files["rig"] / "Wolf_9k_rigged.glb"
    cf.write_family(rig, {"family": "winged", "frame": {"up": [0, 1, 0], "forward": [-0.6, 0, 0.8],
                                                         "left": [0.8, 0, 0.6]}})
    p = ex.plan(rig, kind="creature", size_cm=120)
    assert p["job"]["forward"] == pytest.approx([-0.6, -0.8, 0.0])
    assert p["job"]["family"] == "winged"
    told = ex.plan(rig, kind="creature", size_cm=120, forward=(1.0, 0.0, 0.0))
    assert told["job"]["forward"] == [1.0, 0.0, 0.0]  # a forward given by hand wins


def test_a_creature_without_a_family_file_keeps_the_default_facing(files):
    p = ex.plan(files["rig"] / "Wolf_9k_rigged.glb", kind="creature", size_cm=120)
    assert p["job"]["forward"] == [0.0, -1.0, 0.0] and p["job"]["family"] is None


def test_an_objects_facing_ignores_any_family_file(files):
    p = ex.plan(files["props"] / "prop_01_LOD0.glb", kind="prop", size_cm=60, measure="longest")
    assert p["job"]["forward"] == [0.0, -1.0, 0.0]


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_real_blender_sizes_a_turned_creature_after_turning_it(tmp_path):
    """A long creature facing half-way between two axes: sized on its box before the turn,
    it came into Unreal 180 cm long instead of 150."""
    source = tmp_path / "drake.glb"
    source.write_bytes(_skinned_glb(stretch_x=8.0))  # 3.2 m along x, 1 m tall
    result = ex.export_unreal(source, kind="creature", size_cm=150, measure="longest", out_dir=tmp_path / "out",
                              forward=(0.7071, -0.7071, 0.0), log=lambda m: None)
    lo, hi = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))["blender"]["box_after_m"]
    assert max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) == pytest.approx(1.5, rel=1e-3)


def test_a_creature_that_is_not_a_humanoid_is_checked_for_facing_by_its_own_bones(files, tmp_path):
    """Unreal measures a humanoid's facing from Manny's feet; a drake has no `foot_l`, so its
    family's hips and head, by the names the export gave them, say which way it faces."""
    from image_to_3dlab import creature_family as cf

    rig = files["rig"] / "Wolf_9k_rigged.glb"
    cf.write_family(rig, {"family": "winged", "spine": ["bone_0", "bone_1"], "neck": ["bone_4", "bone_5"],
                          "body": [], "frame": {"up": [0, 1, 0], "forward": [0, 0, 1], "left": [1, 0, 0]}})
    bones = {"bone_0": "pelvis", "bone_1": "spine_01", "bone_4": "neck_01", "bone_5": "head"}
    result = ex.export_unreal(rig, kind="creature", size_cm=120, out_dir=tmp_path / "out", blender=Path("blender"),
                              run=_fake_blender({"humanoid": False, "bones": bones}), log=lambda m: None)
    expect = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))["expect"]
    assert expect["facing"] == "+Y" and expect["facing_bones"] == ["pelvis", "head"]

