"""The Poly Haven trial for estimated roughness and metallic: pick, pack, strip, score, decide."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from test_delight import SIZE, painted
from test_materials import make_glb

from image_to_3dlab import pbr_trial as pt
from image_to_3dlab.glb_turn import parse_glb


def catalogue():
    entries = {}
    for i in range(40):
        entries[f"metal_thing_{i:02d}"] = {"categories": ["props"], "tags": ["metal", "tool"], "polycount": 5000}
        entries[f"wood_thing_{i:02d}"] = {"categories": ["furniture"], "tags": ["wood"], "polycount": 5000}
    entries["big_tree"] = {"categories": ["nature", "trees"], "tags": ["metal"], "polycount": 5000}
    entries["huge_engine"] = {"categories": ["industrial"], "tags": ["metal"], "polycount": 900_000}
    return entries


def test_half_the_trial_set_is_metal_and_nature_and_huge_scans_are_left_out():
    chosen = pt.select_assets(catalogue(), count=30)
    assert len(chosen) == 30 and len(set(chosen)) == 30
    assert sum(name.startswith("metal") for name in chosen) == 15
    assert "big_tree" not in chosen and "huge_engine" not in chosen
    assert pt.select_assets(catalogue(), count=30) == chosen          # the same set every time


def test_a_gltf_with_files_beside_it_becomes_one_glb(tmp_path):
    glb = make_glb(tmp_path / "chair.glb", painted())
    document, binary = parse_glb(glb.read_bytes())
    image_view = document["bufferViews"][document["images"][0]["bufferView"]]
    start = image_view["byteOffset"]
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "chair_diff_1k.png").write_bytes(binary[start:start + image_view["byteLength"]])
    document["images"][0] = {"uri": "textures/chair_diff_1k.png", "mimeType": "image/png"}
    document["buffers"][0]["uri"] = "chair.bin"
    (tmp_path / "chair.bin").write_bytes(binary)
    (tmp_path / "chair.gltf").write_text(json.dumps(document), encoding="utf-8")
    packed, _ = parse_glb(pt.pack_gltf(tmp_path / "chair.gltf"))
    assert "uri" not in packed["images"][0] and "uri" not in packed["buffers"][0]
    np.testing.assert_array_equal(pt.true_maps(pt.pack_gltf(tmp_path / "chair.gltf"))[0][0],
                                  pt.true_maps(glb.read_bytes())[0][0])


def test_a_data_uri_buffer_is_refused_rather_than_guessed(tmp_path):
    document = {"asset": {"version": "2.0"}, "buffers": [{"uri": "data:application/octet-stream;base64,"
                                                                 + base64.b64encode(b"x").decode()}]}
    (tmp_path / "x.gltf").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="embedded"):
        pt.pack_gltf(tmp_path / "x.gltf")


def test_the_true_maps_come_out_scaled_by_their_factors_and_the_copy_has_none(tmp_path):
    mr = np.zeros((SIZE, SIZE, 3), np.uint8)
    mr[..., 1], mr[:, :64, 2] = 200, 255
    glb = make_glb(tmp_path / "axe.glb", painted(), mr=mr, metallic_factor=1.0, roughness_factor=0.5)
    truth = pt.true_maps(glb.read_bytes())
    rough, metal = truth[0]
    assert rough[0, 0] == pytest.approx(200 / 255 * 0.5) and metal[0, 0] == 1.0 and metal[0, 100] == 0.0
    stripped, _ = parse_glb(pt.strip_maps(glb.read_bytes()))
    assert "metallicRoughnessTexture" not in stripped["materials"][0]["pbrMetallicRoughness"]


def test_scores_pool_by_area_and_the_constant_is_learnt_on_the_other_assets():
    truth = np.full((4, 4), 0.8)
    rows = [pt.score(truth, np.zeros((4, 4)), np.full((4, 4), 0.7), np.zeros((4, 4)), np.ones((4, 4), bool)),
            pt.score(np.full((4, 4), 0.4), np.ones((4, 4)), np.full((4, 4), 0.4), np.ones((4, 4)),
                     np.ones((4, 4), bool))]
    rows[0]["asset"], rows[1]["asset"] = "a", "b"
    report = pt.summarise(rows)
    assert report["roughness"]["rmse"] == pytest.approx(np.sqrt((16 * 0.01 + 0) / 32))
    # the constant for "a" is "b"'s mean (0.4) and the other way round
    assert report["roughness"]["constant_rmse"] == pytest.approx(np.sqrt((16 * 0.16 + 16 * 0.16) / 32))
    assert report["metallic"]["f1"] == pytest.approx(1.0) and report["metallic"]["wrong_share"] == 0.0


def test_the_verdict_follows_the_design_thresholds():
    rows = []
    for name, (est, metal_est) in {"a": (0.75, 0.0), "b": (0.45, 1.0)}.items():
        truth_metal = np.zeros((4, 4)) if name == "a" else np.ones((4, 4))
        truth = np.full((4, 4), 0.8 if name == "a" else 0.4)
        row = pt.score(truth, truth_metal, np.full((4, 4), est), np.full((4, 4), metal_est), np.ones((4, 4), bool))
        row["asset"] = name
        rows.append(row)
    verdict = pt.summarise(rows)["verdict"]
    assert verdict["roughness_rmse"] and verdict["roughness_beats_constant"]
    assert verdict["metallic_f1"] and verdict["metallic_wrong_area"]
    assert verdict["blind_preference"] is None and verdict["admitted"] is False     # needs the user's eyes


def test_the_best_metal_threshold_is_reported_as_an_optimistic_bound():
    # Metal says 0.7, plastic says 0.6: useless at 0.5, separable in hindsight at 0.65.
    truth = np.zeros((4, 8))
    truth[:, :4] = 1.0
    said = np.where(truth > 0, 0.7, 0.6)
    row = pt.score(np.full((4, 8), 0.5), truth, np.full((4, 8), 0.5), said, np.ones((4, 8), bool))
    row["asset"] = "a"
    metal = pt.summarise([row])["metallic"]
    assert metal["f1"] == pytest.approx(2 * 16 / (2 * 16 + 16))
    assert metal["best_f1"] == pytest.approx(1.0) and 0.6 < metal["best_threshold"] <= 0.7
