"""Roughness and metallic estimated where the asset has none, with Marigold-IID Appearance.

Blender and Marigold are fakes, as in test_delight: a view that sees the atlas flat, and a
model whose answer we choose. What is tested is the carrying back, what is written into the
GLB, and that an asset's own map is never replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_delight import SIZE, fake_render, painted
from test_materials import make_glb

from image_to_3dlab import material_models as mm
from image_to_3dlab import materials as m
from image_to_3dlab import pbr_estimate as pe
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import parse_glb


def fake_appearance(rough, metal):
    """An Appearance runner answering `rough(colour)` and `metal(colour)` per picture."""
    def run(images, outs, model, **_):
        for image, out in zip(images, outs):
            colour = np.asarray(Image.open(image).convert("RGB")) / 255.0
            material = np.dstack([rough(colour), metal(colour), np.zeros(colour.shape[:2])])
            np.savez(out, albedo=colour.astype(np.float16), material=material.astype(np.float16))
        return {"target_properties": {"target_names": ["albedo", "material"],
                                      "material": {"prediction_space": "stack",
                                                   "sub_target_names": ["roughness", "metallicity", None]}},
                "device": "fake", "seconds": 0.0, "steps": 4, "resolution": 768, "ensemble": 1}
    return run


def left_metal(colour):
    out = np.zeros(colour.shape[:2])
    out[:, :SIZE // 2] = 1.0
    return out


def rough_ramp(colour):
    return np.tile(np.linspace(0.2, 0.8, SIZE), (SIZE, 1))


def mr_of(path: Path) -> np.ndarray:
    glb = EditGLB(path.read_bytes())
    info = glb.document["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]
    return m._decode_image(glb, m.texture_source(glb.document["textures"][info["index"]]))


def test_estimated_maps_land_on_their_texels_in_gltf_channels(tmp_path):
    asset = make_glb(tmp_path / "crate.glb", painted())
    result = pe.estimate_pbr(asset, tmp_path / "out", render=fake_render(painted()),
                             intrinsics=fake_appearance(rough_ramp, left_metal))
    mr = mr_of(result.glb_path).astype(float) / 255.0
    np.testing.assert_allclose(mr[:, 10, 1], 0.2 + 0.6 * 10.5 / SIZE, atol=0.03)       # G roughness
    np.testing.assert_allclose(mr[:, 120, 1], 0.2 + 0.6 * 120 / SIZE, atol=0.03)
    assert (mr[:, :50, 2] > 0.95).all() and (mr[:, 78:, 2] < 0.05).all()               # B metallic
    pbr = parse_glb(result.glb_path.read_bytes())[0]["materials"][0]["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 1.0 and pbr["roughnessFactor"] == 1.0


def test_an_assets_own_map_is_kept(tmp_path):
    own = np.zeros((SIZE, SIZE, 3), np.uint8)
    own[..., 1] = 140
    asset = make_glb(tmp_path / "knight.glb", painted(), mr=own, metallic_factor=1.0)
    asked = []
    result = pe.estimate_pbr(asset, tmp_path / "out", render=fake_render(painted()),
                             intrinsics=lambda *a, **k: asked.append(a))
    assert asked == []                                        # nothing to estimate: no model run
    assert result.glb_path is None and result.estimated == []


def test_with_force_an_own_map_is_replaced_and_said_so(tmp_path):
    own = np.zeros((SIZE, SIZE, 3), np.uint8)
    asset = make_glb(tmp_path / "knight.glb", painted(), mr=own, metallic_factor=1.0)
    result = pe.estimate_pbr(asset, tmp_path / "out", force=True, render=fake_render(painted()),
                             intrinsics=fake_appearance(rough_ramp, left_metal))
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["materials"][0]["replaced_own_map"] is True
    assert mr_of(result.glb_path)[:, :50, 2].min() > 240


def test_the_result_records_the_model_its_restrictions_and_that_it_is_not_admitted(tmp_path):
    asset = make_glb(tmp_path / "crate.glb", painted())
    result = pe.estimate_pbr(asset, tmp_path / "out", render=fake_render(painted()),
                             intrinsics=fake_appearance(rough_ramp, left_metal))
    provenance = json.loads(result.glb_path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    component = provenance["components"][-1]
    assert component["component"] == f"{mm.MARIGOLD_APPEARANCE.repo}@{mm.MARIGOLD_APPEARANCE.revision}"
    assert component["conditions"] == list(mm.OPENRAIL_CONDITIONS)
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["admitted"] is False
    assert sidecar["materials"][0]["metallic_share"] == pytest.approx(0.5, abs=0.05)
    assert (tmp_path / "out" / "T_crate_MR.png").is_file()


def test_the_appearance_answer_is_read_by_its_declared_channels():
    record = {"target_properties": {"material": {"prediction_space": "stack",
                                                 "sub_target_names": ["metallicity", "roughness", None]}}}
    material = np.dstack([np.full((2, 2), 0.9), np.full((2, 2), 0.3), np.zeros((2, 2))])
    rough, metal = pe.roughness_metallic(material, record)
    assert rough[0, 0] == pytest.approx(0.3) and metal[0, 0] == pytest.approx(0.9)
