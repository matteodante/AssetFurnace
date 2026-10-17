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
