"""The Unreal export profile's pure parts: names, bones, scale, ORM, GLB rewrite, checks.

Everything Blender or Unreal does is a thin shell around these, so a wrong name, a lost
texture or a mis-scaled animation is caught here in milliseconds instead of after a
two-minute Unreal import.
"""

from __future__ import annotations

import io
import json
import math
import struct

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import unreal_profile as up
from image_to_3dlab.glb_turn import build_glb, parse_glb

# --- names -----------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("SK_BrennholdWarrior_9k", "BrennholdWarrior9k"),
    ("prop_01_LOD0", "Prop01"),
    ("prop_01_LOD0.glb", "Prop01"),
    ("a-testuggine romana!", "ATestuggineRomana"),
    ("SM_Barrel", "Barrel"),
    ("MI_Rock_02", "Rock02"),
    ("9lives", "Asset9lives"),
])
def test_base_name_is_an_unreal_safe_camel_case_word(raw, expected):
    assert up.base_name(raw) == expected


def test_base_name_refuses_a_name_with_nothing_usable():
    with pytest.raises(ValueError):
        up.base_name("__--!!")


def test_base_name_is_kept_short_for_windows_paths():
    assert len(up.base_name("x" * 200)) <= up.MAX_NAME


def test_asset_names_follow_unreal_prefixes():
    names = up.asset_names("prop", "Barrel")
    assert names["mesh"] == "SM_Barrel"
    assert names["material"] == "MI_Barrel"
    assert names["orm"] == "T_Barrel_ORM"
    assert names["base_color"] == "T_Barrel_BaseColor"
    assert names["normal"] == "T_Barrel_N"
    assert names["tint"] == "T_Barrel_Tint"
    assert names["collision"] == {"box": "UBX_SM_Barrel_00", "convex": "UCX_SM_Barrel_00"}
    assert up.asset_names("creature", "Wolf")["mesh"] == "SK_Wolf"
    assert up.asset_names("equipment", "Helmet")["mesh"] == "SK_Helmet"


def test_asset_names_refuse_an_unknown_kind():
    with pytest.raises(ValueError):
        up.asset_names("vehicle", "Cart")


def test_lod_nodes_use_the_pattern_interchange_groups():
    assert up.lod_node("SM_Barrel", 0) == "LOD0_SM_Barrel"
    assert up.lod_node("SM_Barrel", 2) == "LOD2_SM_Barrel"
    assert up.lod_material("Barrel", 0) == "MI_Barrel"
    assert up.lod_material("Barrel", 1) == "MI_Barrel_LOD1"


@pytest.mark.parametrize("material, role, expected", [
    ("MI_Barrel", "base_color", "T_Barrel_BaseColor"),
    ("MI_Barrel_LOD1", "orm", "T_Barrel_LOD1_ORM"),
    ("MI_Barrel", "normal", "T_Barrel_N"),
    ("Material.001", "orm", "T_Material001_ORM"),
])
def test_texture_names_follow_their_material(material, role, expected):
    assert up.texture_name(material, role) == expected


def test_sibling_lods_are_found_in_order_and_skip_compressed_copies(tmp_path):
    for name in ("chest_LOD0.glb", "chest_LOD1.glb", "chest_LOD2.glb", "chest_LOD1.web.glb", "chest_LOD4.glb"):
        (tmp_path / name).write_bytes(b"x")
    assert up.sibling_lods(tmp_path / "chest_LOD0.glb") == [tmp_path / "chest_LOD1.glb", tmp_path / "chest_LOD2.glb"]
    assert up.sibling_lods(tmp_path / "chest_LOD1.glb") == []
    assert up.sibling_lods(tmp_path / "other.glb") == []


def test_model_prefix_comes_from_the_rigged_file_or_what_the_moves_share():
    assert up.model_prefix(["SK_W_9k_rigged", "SK_W_9k_idle"]) == "SK_W_9k"
    assert up.model_prefix(["SK_W_9k_idle", "SK_W_9k_hit_react"]) == "SK_W_9k"
    assert up.model_prefix(["big_wolf"]) == "big_wolf"  # one file: nothing to compare with
    assert up.model_prefix(["wolf_a", "bear_b"]) == ""


@pytest.mark.parametrize("stem, expected", [
    ("SK_W_9k_idle", "AS_W_Idle"),
    ("SK_W_9k_hit_react", "AS_W_HitReact"),
    ("SK_W_9k", "AS_W_Anim"),
    ("walk cycle", "AS_W_WalkCycle"),
])
def test_clip_names_drop_the_model_prefix(stem, expected):
    assert up.clip_name(stem, "SK_W_9k", "W") == expected


def test_provenance_is_the_files_own_record_first(tmp_path):
    run = tmp_path / "run"
    (run / "finished" / "prop_01").mkdir(parents=True)
    glb = run / "finished" / "prop_01" / "prop_01_LOD0.glb"
    glb.write_bytes(b"")
    (run / "props.provenance.json").write_text("{}")
    assert up.find_provenance(glb) == run / "props.provenance.json"
    own = glb.with_name("prop_01_LOD0.provenance.json")
    own.write_text("{}")
    assert up.find_provenance(glb) == own
    assert up.find_provenance(tmp_path / "nowhere" / "x.glb", levels=1) is None


# --- size ------------------------------------------------------------------------------

def test_scale_factor_by_height_uses_blender_up_axis():
    assert up.scale_factor((0.48, 0.23, 0.9), 180, "height") == pytest.approx(2.0)


def test_scale_factor_by_longest_side():
    assert up.scale_factor((0.15, 0.32, 0.175), 64, "longest") == pytest.approx(2.0)


@pytest.mark.parametrize("dims, size, measure", [
    ((1, 1, 0), 100, "height"),
    ((1, 1, 1), 0, "height"),
    ((1, 1, 1), 100, "width"),
])
def test_scale_factor_refuses_what_it_cannot_measure(dims, size, measure):
    with pytest.raises(ValueError):
        up.scale_factor(dims, size, measure)


def test_ground_offset_puts_the_base_centre_on_the_origin():
    assert up.ground_offset((-1.0, 2.0, -0.45), (3.0, 4.0, 0.45)) == pytest.approx((-1.0, -3.0, 0.45))


def test_ground_offset_can_centre_on_the_feet_instead_of_the_box():
    assert up.ground_offset((-1, -1, -0.5), (1, 1, 0.5), centre=(0.2, -0.1)) == pytest.approx((-0.2, 0.1, 0.5))


@pytest.mark.parametrize("forward, expected_deg", [
    ((0, -1, 0), 0), ((1, 0, 0), -90), ((0, 1, 0), 180), ((-1, 0, 0), 90),
])
def test_yaw_to_face_minus_y_turns_any_forward_to_blender_front(forward, expected_deg):
    yaw = up.yaw_to_face_front(forward)
    c, s = math.cos(yaw), math.sin(yaw)
    turned = (c * forward[0] - s * forward[1], s * forward[0] + c * forward[1])
    assert turned == pytest.approx((0, -1), abs=1e-9)
    assert math.degrees(yaw) % 360 == pytest.approx(expected_deg % 360)


def test_yaw_refuses_a_vertical_forward():
    with pytest.raises(ValueError):
        up.yaw_to_face_front((0, 0, 1))


# --- bones -----------------------------------------------------------------------------

def _warrior_like():
    order = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
    parents = {"b0": None, "b1": "b0", "b2": "b1", "b3": "b2", "b4": "b3",
               "b5": "b4", "b6": "b5", "b7": "b5", "b8": "b0"}
    soma = {"b0": "Hips", "b1": "Spine1", "b2": "Chest", "b3": "LeftShoulder", "b4": "LeftArm",
            "b5": "LeftHand", "b6": "LeftHandMiddle2", "b7": "LeftHandMiddle2", "b8": "LeftLeg"}
    return order, parents, soma


def test_bone_plan_gives_mannequin_names_and_numbers_the_extras():
    order, parents, soma = _warrior_like()
    plan = up.bone_rename_plan(order, parents, soma)
    assert plan["b0"] == "pelvis"
    assert plan["b1"] == "spine_01"
    assert plan["b2"] == "spine_02"  # numbered from the hips up, as Manny's are
    assert plan["b3"] == "clavicle_l"
    assert plan["b5"] == "hand_l"
    assert plan["b6"] == "middle_01_l"
    assert plan["b7"] == "middle_01_l_extra0"
    assert plan["b8"] == "thigh_l"
    assert len(set(plan.values())) == len(plan)
    assert up.ROOT_BONE not in plan.values()


def test_bone_plan_names_bones_that_follow_a_mapped_parent_as_extras():
    order = ["h", "s", "hair"]
    parents = {"h": None, "s": "h", "hair": "s"}
    plan = up.bone_rename_plan(order, parents, {"h": "Hips", "s": "Head", "hair": "Head"})
    assert plan == {"h": "pelvis", "s": "head", "hair": "head_extra0"}


def test_unmapped_skeletons_keep_clean_unique_names():
    plan = up.clean_bone_names(["bone.0", "bone 0", "root", "Tail-1"])
    assert plan == {"bone.0": "bone_0", "bone 0": "bone_0_1", "root": "root_1", "Tail-1": "Tail_1"}


def test_every_soma_joint_the_mapper_can_return_has_a_mannequin_name():
    from kimodo_retarget import SOMA77

    produced = {n for n in SOMA77 if not n.endswith(("End", "Eye", "Jaw")) and "Hand" not in n[-5:]}
    for name in produced:
        if name.endswith(("Index1", "Middle1", "Ring1", "Pinky1")):
            continue  # SkinTokens fingers start at the knuckle (2), never the metacarpal
        assert name in up.SOMA_TO_UE, name


@pytest.mark.parametrize("path, expected", [
    ('pose.bones["bone_3"].rotation_quaternion', 'pose.bones["spine_03"].rotation_quaternion'),
    ('pose.bones["bone_30"].location', 'pose.bones["thigh_r"].location'),
    ('pose.bones["unknown"].location', 'pose.bones["unknown"].location'),
    ("location", "location"),
])
def test_animation_paths_follow_renamed_bones(path, expected):
    assert up.renamed_path(path, {"bone_3": "spine_03", "bone_30": "thigh_r"}) == expected


@pytest.mark.parametrize("path, expected", [
    ('pose.bones["pelvis"].location', True),
    ("location", True),
    ('pose.bones["pelvis"].rotation_quaternion', False),
    ('pose.bones["pelvis"].scale', False),
])
def test_only_translations_scale_with_the_model(path, expected):
    assert up.is_translation_path(path) is expected


# --- ORM -------------------------------------------------------------------------------

def test_pack_orm_takes_occlusion_red_roughness_green_metal_blue():
    mr = np.zeros((2, 2, 4), np.uint8)
    mr[..., 1], mr[..., 2], mr[..., 0] = 200, 30, 77
    ao = np.full((2, 2, 3), 90, np.uint8)
    orm = up.pack_orm(mr, ao)
    assert orm.shape == (2, 2, 3) and orm.dtype == np.uint8
    assert orm[0, 0].tolist() == [90, 200, 30]


def test_pack_orm_without_occlusion_is_unshadowed():
    mr = np.zeros((1, 1, 3), np.uint8)
    mr[..., 1], mr[..., 2] = 128, 255
    assert up.pack_orm(mr, None)[0, 0].tolist() == [255, 128, 255]


def test_pack_orm_reads_a_greyscale_occlusion_map():
    mr = np.zeros((1, 1, 3), np.uint8)
    assert up.pack_orm(mr, np.full((1, 1), 40, np.uint8))[0, 0, 0] == 40


def test_pack_orm_refuses_maps_of_different_sizes():
    with pytest.raises(ValueError):
        up.pack_orm(np.zeros((2, 2, 3), np.uint8), np.zeros((4, 4), np.uint8))


# --- GLB rewrite -----------------------------------------------------------------------

def _png(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def _glb(materials, images, extra_json=None) -> bytes:
    """A minimal GLB: one triangle, the given materials, embedded PNG images."""
    binary = b""
    views = []
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    views.append({"buffer": 0, "byteOffset": 0, "byteLength": len(positions)})
    binary += positions
    image_entries = []
    for name, data in images:
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary += data
        image_entries.append({"name": name, "mimeType": "image/png", "bufferView": len(views) - 1})
    document = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "SM_Test", "mesh": 0}],
        "meshes": [{"name": "SM_Test", "primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [1, 1, 0]}],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "materials": materials,
        "textures": [{"source": i} for i in range(len(images))],
        "images": image_entries,
    }
    document.update(extra_json or {})
    return build_glb(document, binary)


def _mr_image():
    mr = np.zeros((4, 4, 3), np.uint8)
    mr[..., 1], mr[..., 2] = 180, 20
    return mr


def test_finalize_packs_orm_and_names_every_texture():
    ao = np.full((4, 4), 100, np.uint8)
    data = _glb(
        [{"name": "MI_Barrel", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                       "metallicRoughnessTexture": {"index": 1}},
          "normalTexture": {"index": 2}, "occlusionTexture": {"index": 3}}],
        [("retopo_albedo", _png(np.full((4, 4, 3), 50, np.uint8))), ("detail_mr", _png(_mr_image())),
         ("detail_normal", _png(np.full((4, 4, 3), 128, np.uint8))), ("ao", _png(ao))])
    out, info = up.finalize_glb(data)
    document, binary = parse_glb(out)
    material = document["materials"][0]
    orm_texture = material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
    assert material["occlusionTexture"]["index"] == orm_texture
    names = sorted(image["name"] for image in document["images"])
    assert names == ["T_Barrel_BaseColor", "T_Barrel_N", "T_Barrel_ORM"]
    orm_image = document["images"][document["textures"][orm_texture]["source"]]
    view = document["bufferViews"][orm_image["bufferView"]]
    pixels = np.array(Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])))
    assert pixels[0, 0].tolist() == [100, 180, 20]
    assert info["materials"]["MI_Barrel"]["orm"] == "T_Barrel_ORM"
    assert info["materials"]["MI_Barrel"]["occlusion"] == "baked"


def test_finalize_without_occlusion_still_writes_one_orm_texture():
    data = _glb([{"name": "MI_Rock", "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}}}],
                [("mr", _png(_mr_image()))])
    out, info = up.finalize_glb(data)
    document, _ = parse_glb(out)
    assert [image["name"] for image in document["images"]] == ["T_Rock_ORM"]
    assert document["materials"][0]["occlusionTexture"]["index"] == 0
    assert info["materials"]["MI_Rock"]["occlusion"] == "none"


def test_finalize_leaves_a_material_without_textures_alone():
    data = _glb([{"name": "MI_Plain", "pbrMetallicRoughness": {"metallicFactor": 0.0}}], [])
    out, info = up.finalize_glb(data)
    document, _ = parse_glb(out)
    assert "occlusionTexture" not in document["materials"][0]
    assert info["materials"]["MI_Plain"]["orm"] is None


def test_finalize_keeps_geometry_bytes_and_accessors_valid():
    data = _glb([{"name": "MI_Rock", "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}}}],
                [("mr", _png(_mr_image()))])
    out, _ = up.finalize_glb(data)
    document, binary = parse_glb(out)
    view = document["bufferViews"][document["accessors"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    assert struct.unpack("<9f", binary[start:start + 36]) == (0, 0, 0, 1, 0, 0, 0, 1, 0)
    assert document["buffers"][0]["byteLength"] == len(binary)
    for view in document["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(binary)


def test_finalize_stamps_the_profile_in_asset_extras():
    data = _glb([{"name": "MI_Plain"}], [])
    document, _ = parse_glb(up.finalize_glb(data, stamp={"kind": "prop"})[0])
    assert document["asset"]["extras"][up.EXTRAS_KEY] == {"kind": "prop"}


# --- checks ----------------------------------------------------------------------------

def _document(**overrides):
    document = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": "LOD0_SM_Barrel", "mesh": 0}, {"name": "LOD1_SM_Barrel", "mesh": 1},
                  {"name": "UBX_SM_Barrel_00", "mesh": 2}],
        "meshes": [{}, {}, {}],
        "materials": [{"name": "MI_Barrel"}, {"name": "MI_Barrel_LOD1"}],
        "images": [{"name": "T_Barrel_ORM"}],
    }
    document.update(overrides)
    return document


def test_a_clean_prop_document_passes():
    assert up.check_document(_document(), up.asset_names("prop", "Barrel")) == []


def test_check_refuses_extensions_interchange_cannot_read():
    problems = up.check_document(_document(extensionsRequired=["EXT_meshopt_compression"]),
                                 up.asset_names("prop", "Barrel"))
    assert any("EXT_meshopt_compression" in p for p in problems)


def test_check_flags_unprefixed_names():
    document = _document(materials=[{"name": "Material.001"}], images=[{"name": "detail_mr"}])
    problems = up.check_document(document, up.asset_names("prop", "Barrel"))
    assert any("Material.001" in p for p in problems)
    assert any("detail_mr" in p for p in problems)


def test_check_flags_a_stray_mesh_node():
    document = _document(nodes=[{"name": "Icosphere", "mesh": 0}])
    assert any("Icosphere" in p for p in up.check_document(document, up.asset_names("prop", "Barrel")))


def test_check_wants_a_root_bone_on_skinned_meshes():
    names = up.asset_names("creature", "Wolf")
    document = {"asset": {}, "nodes": [{"name": "SK_Wolf", "mesh": 0, "skin": 0}, {"name": "pelvis"}],
                "meshes": [{}], "skins": [{"joints": [1]}], "materials": [], "images": []}
    assert any("root" in p for p in up.check_document(document, names))
    document["nodes"].append({"name": "root", "children": [1]})
    document["skins"][0]["joints"] = [2, 1]
    assert up.check_document(document, names) == []


# --- sidecar and Unreal report ---------------------------------------------------------

def test_sidecar_records_what_the_importer_needs():
    sidecar = up.sidecar(kind="creature", names=up.asset_names("creature", "Wolf"), glb="SK_Wolf.glb",
                         size_cm=180, measure="height", lods=3, collision=None, humanoid=True,
                         skeleton_from=None, animations=["AS_Wolf_Idle"], tint=None, problems=[])
    assert sidecar["schema"] == up.SIDECAR_SCHEMA
    assert sidecar["mesh"] == "SK_Wolf"
    assert sidecar["skinned"] is True
    assert sidecar["expect"]["size_cm"] == 180
    assert sidecar["expect"]["facing"] == "+Y"
    json.dumps(sidecar)


def _measured(**overrides):
    measured = {"class": "StaticMesh", "bounds_cm": [15.0, 32.0, 17.5], "min_z": 0.1, "lods": 3,
                "collision": {"box_elems": 1, "convex_elems": 0}, "names": ["SM_Barrel", "MI_Barrel", "T_Barrel_ORM"],
                "orm_srgb": [False]}
    measured.update(overrides)
    return measured


def _prop_sidecar(**overrides):
    sidecar = up.sidecar(kind="prop", names=up.asset_names("prop", "Barrel"), glb="SM_Barrel.glb", size_cm=32,
                         measure="longest", lods=3, collision="box", humanoid=False, skeleton_from=None,
                         animations=[], tint=None, problems=[])
    sidecar.update(overrides)
    return sidecar


def test_unreal_report_passes_a_prop_that_matches():
    results = up.evaluate_unreal(_measured(), _prop_sidecar())
    assert all(r["ok"] for r in results), results


@pytest.mark.parametrize("override, failing", [
    ({"bounds_cm": [1.5, 3.2, 1.75]}, "size"),
    ({"min_z": -45.0}, "base"),
    ({"lods": 1}, "lods"),
    ({"collision": {"box_elems": 0, "convex_elems": 1}}, "collision"),
    ({"orm_srgb": [True]}, "orm"),
])
def test_unreal_report_catches_each_failure(override, failing):
    results = {r["check"]: r["ok"] for r in up.evaluate_unreal(_measured(**override), _prop_sidecar())}
    assert results[failing] is False


def test_unreal_report_checks_a_character_faces_like_manny_and_is_recognised():
    sidecar = up.sidecar(kind="creature", names=up.asset_names("creature", "Wolf"), glb="SK_Wolf.glb",
                         size_cm=180, measure="height", lods=3, collision=None, humanoid=True,
                         skeleton_from=None, animations=["AS_Wolf_Idle"], tint=None, problems=[])
    measured = {"class": "SkeletalMesh", "bounds_cm": [90, 40, 180.5], "min_z": 0.0, "lods": 3,
                "names": ["SK_Wolf"], "orm_srgb": [False], "forward": [0.0, 1.0, 0.0],
                "ik_recognised": True, "retarget": "ok", "walk": "ok", "animations": ["AS_Wolf_Idle"]}
    results = {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}
    assert results == {"size": True, "base": True, "names": True, "lods": True, "orm": True,
                       "facing": True, "skeleton": True, "retarget": True, "walk": True, "animations": True}
    measured["forward"] = [1.0, 0.0, 0.0]
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}["facing"] is False


def test_unreal_report_checks_equipment_reuses_the_body_skeleton():
    sidecar = up.sidecar(kind="equipment", names=up.asset_names("equipment", "Helm"), glb="SK_Helm.glb",
                         size_cm=180, measure="height", lods=1, collision=None, humanoid=True,
                         skeleton_from="SK_Warrior", animations=[], tint=None, problems=[])
    measured = {"class": "SkeletalMesh", "bounds_cm": [30, 30, 40], "min_z": 140.0, "lods": 1,
                "names": ["SK_Helm"], "orm_srgb": [False], "skeleton": "/Game/X/SK_Warrior_Skeleton",
                "new_skeletons": 0}
    results = {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}
    assert results["skeleton"] is True
    assert "size" not in results and "base" not in results  # a helmet is measured on its body
    measured["new_skeletons"] = 1
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}["skeleton"] is False


# --- finding Unreal --------------------------------------------------------------------

def test_find_unreal_prefers_the_environment_variable(tmp_path):
    editor = tmp_path / "UnrealEditor.exe"
    editor.write_bytes(b"")
    assert up.find_unreal_editor({up.UNREAL_ENV: str(editor)}, roots=[]) == editor


def test_find_unreal_picks_the_newest_5_x_install(tmp_path):
    for version in ("UE_5.6", "UE_5.8", "UE_5.10"):
        binary = tmp_path / version / "Engine/Binaries/Win64/UnrealEditor.exe"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"")
    found = up.find_unreal_editor({}, roots=[tmp_path])
    assert found.parts[-5] == "UE_5.10"


def test_find_unreal_reports_nothing_when_missing(tmp_path):
    assert up.find_unreal_editor({}, roots=[tmp_path]) is None
    assert up.find_unreal_editor({up.UNREAL_ENV: str(tmp_path / "nope.exe")}, roots=[]) is None


def test_unreal_report_checks_the_tint_mask_arrives_linear():
    sidecar = _prop_sidecar(tint="T_Barrel_Tint.png")
    ok = {r["check"]: r["ok"] for r in up.evaluate_unreal(_measured(tint_srgb=False), sidecar)}
    assert ok["tint"] is True
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(_measured(tint_srgb=True), sidecar)}["tint"] is False
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(_measured(), sidecar)}["tint"] is False
    assert "tint" not in {r["check"] for r in up.evaluate_unreal(_measured(), _prop_sidecar())}


# --- the check picture ---------------------------------------------------------------

def test_stage_puts_the_camera_on_the_suns_side_and_turns_everyone_to_it():
    stage = up.stage_layout([100, 100, 100], 200, sun=(1.0, 0.0, -1.0), side_angle=0)
    assert stage["camera_yaw"] == pytest.approx(0.0)  # looking the way the light travels
    cx, cy, cz = stage["camera"]
    assert cx < 0 and cy == pytest.approx(0) and cz == pytest.approx(100)
    # Unreal turns local +Y (the way characters face) by the actor yaw: it must point back at the camera.
    yaw = math.radians(stage["actor_yaw"])
    assert (-math.sin(yaw), math.cos(yaw)) == pytest.approx((-1.0, 0.0), abs=1e-9)
    xs = [p[0] for p in stage["positions"]]
    ys = [p[1] for p in stage["positions"]]
    assert xs == pytest.approx([0, 0, 0]) and ys == pytest.approx([-100, 0, 100])


def test_stage_frames_the_whole_row_and_the_tallest_figure():
    tall = up.stage_layout([60], 400, sun=(0, -1, -1))
    wide = up.stage_layout([60] * 20, 100, sun=(0, -1, -1))
    def distance(s):
        return math.hypot(s["camera"][0], s["camera"][1])
    half_h = math.tan(math.radians(tall["fov"]) / 2)
    assert distance(tall) * half_h / (16 / 9) >= 200  # half the height fits vertically
    assert distance(wide) * half_h >= 600  # half the row fits horizontally


def test_stage_without_a_usable_sun_looks_from_the_front():
    stage = up.stage_layout([50], 100, sun=(0, 0, -1), side_angle=0)
    assert stage["camera"][1] > 0 and stage["camera_yaw"] == pytest.approx(-90)


@pytest.mark.parametrize("samples, expected", [
    ([0.1, 0.4, 0.2], 0.4),
    ([], 0.0),
])
def test_stride_is_the_widest_step_of_the_cycle(samples, expected):
    assert up.stride(samples) == expected


def test_unreal_report_checks_the_walk_too():
    sidecar = up.sidecar(kind="creature", names=up.asset_names("creature", "Wolf"), glb="SK_Wolf.glb",
                         size_cm=180, measure="height", lods=1, collision=None, humanoid=True,
                         skeleton_from=None, animations=[], tint=None, problems=[])
    measured = {"bounds_cm": [1, 1, 180], "min_z": 0, "lods": 1, "names": ["SK_Wolf"], "orm_srgb": [False],
                "forward": [0, 1, 0], "ik_recognised": True, "retarget": "ok", "walk": "legs barely move"}
    results = {r["check"]: r for r in up.evaluate_unreal(measured, sidecar)}
    assert results["walk"]["ok"] is False and "legs" in results["walk"]["detail"]
    measured["walk"] = "ok"
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}["walk"] is True
    del measured["walk"]  # a humanoid whose walk was never retargeted does not pass
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}["walk"] is False


def test_a_closeup_frames_one_figure_from_the_same_side_as_the_wide_shot():
    x, y, z = up.closeup_camera((100.0, 50.0), camera_yaw=0.0, width=80, height=200, fov=35)
    assert y == pytest.approx(50.0) and x < 100.0 and z == pytest.approx(100.0)
    half_v = math.tan(math.radians(35) / 2) / (16 / 9)
    assert (100.0 - x) * half_v >= 100  # the whole figure fits vertically
    near = up.closeup_camera((0.0, 0.0), camera_yaw=90.0, width=40, height=40, fov=35)
    far = up.closeup_camera((0.0, 0.0), camera_yaw=90.0, width=40, height=400, fov=35)
    assert near[1] < 0 and far[1] < near[1]  # looking +Y from behind, further for taller


def test_unreal_report_wants_the_root_bone_on_the_floor():
    # A root at hip height lifts the whole character off the ground in any animation that
    # locks root motion to the reference pose (Manny's walks do).
    sidecar = up.sidecar(kind="creature", names=up.asset_names("creature", "Wolf"), glb="SK_Wolf.glb",
                         size_cm=180, measure="height", lods=1, collision=None, humanoid=False,
                         skeleton_from=None, animations=[], tint=None, problems=[])
    measured = {"bounds_cm": [1, 1, 180], "min_z": 0, "lods": 1, "names": ["SK_Wolf"], "orm_srgb": [False],
                "forward": [0, 1, 0], "root_cm": [0.0, 0.2, 92.9]}
    results = {r["check"]: r for r in up.evaluate_unreal(measured, sidecar)}
    assert results["root"]["ok"] is False and "92.9" in results["root"]["detail"]
    measured["root_cm"] = [0.0, 0.0, 0.0]
    assert {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}["root"] is True


# --- review round 1 ----------------------------------------------------------------------

def test_a_folder_of_other_assets_records_is_not_guessed(tmp_path):
    run = tmp_path / "animate"
    run.mkdir()
    (run / "a.provenance.json").write_text("{}")
    (run / "b.provenance.json").write_text("{}")
    model = run / "c.glb"
    model.write_bytes(b"")
    assert up.find_provenance(model) is None


def test_the_unreal_check_skips_orm_when_none_was_exported():
    sidecar = _prop_sidecar(orm=False)
    results = {r["check"] for r in up.evaluate_unreal(_measured(orm_srgb=[]), sidecar)}
    assert "orm" not in results
    assert up.sidecar(kind="prop", names=up.asset_names("prop", "B"), glb="SM_B.glb", size_cm=1, measure="height",
                      lods=1, collision=None, humanoid=False, skeleton_from=None, animations=[], tint=None,
                      problems=[], orm=False)["orm"] is False


def test_problems_the_export_reported_fail_the_unreal_check():
    results = {r["check"]: r for r in up.evaluate_unreal(_measured(), _prop_sidecar(problems=["material 'X'"]))}
    assert results["export"]["ok"] is False and "material 'X'" in results["export"]["detail"]


# --- review round 2 ----------------------------------------------------------------------

def test_extra_bone_names_never_collide_with_a_cleaned_name():
    order = ["h", "x", "s", "hair"]
    parents = {"h": None, "x": "h", "s": "h", "hair": "s"}
    plan = up.bone_rename_plan(order, parents, {"h": "Hips", "x": "head_extra0", "s": "Head", "hair": "Head"})
    assert plan["x"] == "head_extra0"
    assert len(set(plan.values())) == len(plan)


def test_orm_keeps_each_slots_settings_and_a_differently_mapped_occlusion_apart():
    transform = {"KHR_texture_transform": {"scale": [2, 2]}}
    data = _glb(
        [{"name": "MI_Barrel", "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0, "extensions": transform}},
          "occlusionTexture": {"index": 1, "texCoord": 1, "strength": 0.5}}],
        [("mr", _png(_mr_image())), ("ao", _png(np.full((4, 4), 100, np.uint8)))])
    out, info = up.finalize_glb(data)
    document, _ = parse_glb(out)
    material = document["materials"][0]
    mr = material["pbrMetallicRoughness"]["metallicRoughnessTexture"]
    assert mr["extensions"] == transform
    occlusion = material["occlusionTexture"]
    assert occlusion["texCoord"] == 1 and occlusion["strength"] == 0.5 and occlusion["index"] != mr["index"]
    assert info["materials"]["MI_Barrel"]["occlusion"] == "separate"
    assert sorted(i["name"] for i in document["images"]) == ["T_Barrel_AO", "T_Barrel_ORM"]


def test_orm_keeps_the_occlusion_strength_when_packed():
    data = _glb([{"name": "MI_Rock", "pbrMetallicRoughness": {"metallicRoughnessTexture": {"index": 0}},
                  "occlusionTexture": {"index": 1, "strength": 0.7}}],
                [("mr", _png(_mr_image())), ("ao", _png(np.full((4, 4), 90, np.uint8)))])
    material = parse_glb(up.finalize_glb(data)[0])[0]["materials"][0]
    assert material["occlusionTexture"]["strength"] == 0.7
    assert material["occlusionTexture"]["index"] == material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]


def test_a_root_unreal_could_not_measure_fails():
    sidecar = up.sidecar(kind="equipment", names=up.asset_names("equipment", "Helm"), glb="SK_Helm.glb",
                         size_cm=180, measure="height", lods=1, collision=None, humanoid=True,
                         skeleton_from="SK_W", animations=[], tint=None, problems=[])
    measured = {"lods": 1, "names": ["SK_Helm"], "orm_srgb": [False], "skeleton": "/Game/SK_W_Skeleton",
                "new_skeletons": 0, "root_error": "RuntimeError('no root')"}
    results = {r["check"]: r for r in up.evaluate_unreal(measured, sidecar)}
    assert results["root"]["ok"] is False and "no root" in results["root"]["detail"]


def test_a_humanoid_without_its_walk_retarget_fails():
    sidecar = up.sidecar(kind="creature", names=up.asset_names("creature", "W"), glb="SK_W.glb", size_cm=180,
                         measure="height", lods=1, collision=None, humanoid=True, skeleton_from=None,
                         animations=[], tint=None, problems=[])
    measured = {"bounds_cm": [1, 1, 180], "min_z": 0, "lods": 1, "names": ["SK_W"], "orm_srgb": [False],
                "forward": [0, 1, 0], "root_cm": [0, 0, 0], "ik_recognised": True, "retarget": "ok"}
    results = {r["check"]: r["ok"] for r in up.evaluate_unreal(measured, sidecar)}
    assert results["walk"] is False


def test_letting_unreal_make_the_collision_expects_its_convex_hull():
    results = {r["check"]: r["ok"] for r in up.evaluate_unreal(
        _measured(collision={"box_elems": 0, "convex_elems": 1}), _prop_sidecar(collision="unreal"))}
    assert results["collision"] is True
    assert "none" not in up.COLLISIONS and "unreal" in up.COLLISIONS


# --- review round 3 ----------------------------------------------------------------------

@pytest.mark.parametrize("classes, expected", [
    (["commercial-clear", "commercial-conditional"], "commercial-conditional"),
    (["commercial-conditional", "research-only", "territory-restricted"], "research-only"),
    (["commercial-clear", None], "commercial-clear"),
    ([None, None], None),
    (["commercial-clear", "made-up-class"], "made-up-class"),  # an unknown class is never loosened
])
def test_the_strictest_licence_class_of_all_inputs_wins(classes, expected):
    assert up.strictest_class(classes) == expected


@pytest.mark.parametrize("mesh", ["SM_Box", "SK_Wolf9k"])
def test_import_folder_names_are_plain_unreal_names(mesh):
    assert up.safe_mesh_name(mesh) == mesh


@pytest.mark.parametrize("mesh", ["../../Windows", "C:/Users", "SM_..", "SM_a/b", "", "Box"])
def test_import_folder_names_that_could_escape_are_refused(mesh):
    with pytest.raises(ValueError):
        up.safe_mesh_name(mesh)
