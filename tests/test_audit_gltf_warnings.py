"""The Unreal export without the glTF validator's warnings the audit found on the real set
(17/10/2026): a normal map without tangents (MESH_PRIMITIVE_GENERATED_TANGENT_SPACE), and a
skinned mesh under its armature instead of a root node (NODE_SKINNED_MESH_NON_ROOT, which the
glTF spec asks for). Real Blender; the validator too when it is installed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab.blender import find_blender
from image_to_3dlab.glb_turn import parse_glb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import export_unreal  # noqa: E402

needs_blender = pytest.mark.skipif(find_blender() is None, reason="needs Blender")


def _validator_codes(glb: Path) -> set[str] | None:
    from image_to_3dlab.checks import asset, validator

    binary = asset.find_validator() if hasattr(asset, "find_validator") else None
    if binary is None:
        return None
    report = validator.run_validator(binary, glb)
    return {m.get("code") for m in (report.get("issues") or {}).get("messages", [])}


def _normal_mapped(path: Path) -> Path:
    box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    uv = np.random.default_rng(0).random((len(box.vertices), 2))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (8, 8), (180, 120, 60)),
        normalTexture=Image.new("RGB", (8, 8), (128, 128, 255)))
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    box.export(path)
    return path


@needs_blender
def test_a_normal_mapped_model_carries_its_tangents(tmp_path):
    result = export_unreal.export_unreal(_normal_mapped(tmp_path / "crate.glb"), kind="prop", size_cm=50,
                                         out_dir=tmp_path / "out", log=lambda m: None)
    document, _ = parse_glb(Path(result["glb"]).read_bytes())
    meshes = [document["meshes"][n["mesh"]] for n in document["nodes"] if "mesh" in n and n["name"] == "SM_Crate"]
    assert meshes and all("TANGENT" in p["attributes"] for p in meshes[0]["primitives"])
    codes = _validator_codes(Path(result["glb"]))
    if codes is not None:
        assert "MESH_PRIMITIVE_GENERATED_TANGENT_SPACE" not in codes


@needs_blender
def test_a_skinned_mesh_is_a_root_node(tmp_path):
    from tests.test_export_unreal import _skinned_glb

    source = tmp_path / "critter.glb"
    source.write_bytes(_skinned_glb())
    result = export_unreal.export_unreal(source, kind="creature", size_cm=150, out_dir=tmp_path / "out",
                                         log=lambda m: None)
    document, _ = parse_glb(Path(result["glb"]).read_bytes())
    roots = set(document["scenes"][document.get("scene", 0)]["nodes"])
    skinned = [i for i, n in enumerate(document["nodes"]) if "skin" in n]
    assert skinned and all(i in roots for i in skinned)
    assert result["ok"], result["problems"]
    codes = _validator_codes(Path(result["glb"]))
    if codes is not None:
        assert "NODE_SKINNED_MESH_NON_ROOT" not in codes


# ------------------------------------------------------------- review round (Codex, #115)

def _with_moving_armature() -> bytes:
    """The two-bone critter, its Armature node moving 0 -> 3 m over a second."""
    from image_to_3dlab.glb_turn import build_glb
    from tests.test_export_unreal import _skinned_glb

    document, binary = parse_glb(_skinned_glb())
    times = np.array([0.0, 1.0], "<f4").tobytes()
    moves = np.array([[0, 0, 0], [3, 0, 0]], "<f4").tobytes()
    for chunk in (times, moves):
        binary += b"\0" * (-len(binary) % 4)
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(chunk)})
        binary += chunk
    views = len(document["bufferViews"])
    document["accessors"] += [
        {"bufferView": views - 2, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
        {"bufferView": views - 1, "componentType": 5126, "count": 2, "type": "VEC3"}]
    accessors = len(document["accessors"])
    document["animations"] = [{"name": "drift", "samplers": [{"input": accessors - 2, "output": accessors - 1}],
                               "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]}]
    document["buffers"][0]["byteLength"] = len(binary)
    return build_glb(document, binary)


@needs_blender
def test_a_moving_armature_does_not_leave_the_mesh_behind(tmp_path):
    source = tmp_path / "drifter.glb"
    source.write_bytes(_with_moving_armature())
    result = export_unreal.export_unreal(source, kind="creature", size_cm=150, out_dir=tmp_path / "out",
                                         log=lambda m: None)
    document, _ = parse_glb(Path(result["glb"]).read_bytes())
    mesh_node = next(n for n in document["nodes"] if "skin" in n)
    box = document["accessors"][document["meshes"][mesh_node["mesh"]]["primitives"][0]["attributes"]["POSITION"]]
    # the critter is centred on its skeleton (x from -0.3 to 0.3 m): not left where the armature was a frame in
    assert (box["min"][0] + box["max"][0]) / 2 == pytest.approx(0.0, abs=0.01), box


def _two_uv_normal_map(path: Path) -> Path:
    """A normal-mapped box whose normal map reads its second UV map, turned 90 degrees."""
    from image_to_3dlab.glb_turn import build_glb

    document, binary = parse_glb(_normal_mapped(path).read_bytes())
    primitive = document["meshes"][0]["primitives"][0]
    uv = document["accessors"][primitive["attributes"]["TEXCOORD_0"]]
    count = uv["count"]
    view = document["bufferViews"][uv["bufferView"]]
    first = np.frombuffer(binary, "<f4", count * 2, view.get("byteOffset", 0) + uv.get("byteOffset", 0)).reshape(-1, 2)
    turned = np.stack([first[:, 1], first[:, 0]], 1).astype("<f4").tobytes()
    binary += b"\0" * (-len(binary) % 4)
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(turned)})
    binary += turned
    document["accessors"].append({"bufferView": len(document["bufferViews"]) - 1, "componentType": 5126,
                                  "count": count, "type": "VEC2"})
    primitive["attributes"]["TEXCOORD_1"] = len(document["accessors"]) - 1
    material = document["materials"][primitive["material"]]
    material["normalTexture"]["texCoord"] = 1
    document["buffers"][0]["byteLength"] = len(binary)
    path.write_bytes(build_glb(document, binary))
    return path


@needs_blender
def test_tangents_are_not_made_from_another_uv_map_than_the_normal_maps(tmp_path):
    result = export_unreal.export_unreal(_two_uv_normal_map(tmp_path / "crate.glb"), kind="prop", size_cm=50,
                                         out_dir=tmp_path / "out", log=lambda m: None)
    document, _ = parse_glb(Path(result["glb"]).read_bytes())
    mesh = next(document["meshes"][n["mesh"]] for n in document["nodes"] if n.get("name") == "SM_Crate")
    assert all("TANGENT" not in p["attributes"] for p in mesh["primitives"])   # readers make them from the right one
