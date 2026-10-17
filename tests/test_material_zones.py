"""Tint zones by material: the vote on the UVs and the plumbing around the segmenters.

Blender, Qwen3.5 and SAM are fakes: a view that sees the atlas flat, and answers we choose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_delight import SIZE, fake_render, painted
from test_materials import make_glb

from image_to_3dlab import material_models as mm
from image_to_3dlab import material_zones as mz
from image_to_3dlab import materials as m
from image_to_3dlab.edit_glb import EditGLB

THREE = ["leather", "metal", "cloth"]


def bands(phrases=THREE) -> np.ndarray:
    """HxWxP scores: columns split in three bands, one material each (left to right)."""
    out = np.zeros((SIZE, SIZE, len(phrases)))
    edges = [0, 60, 100, SIZE]
    for p in range(3):
        out[:, edges[p]:edges[p + 1], p] = 1.0
    return out


def surface_of(path: Path):
    return next(iter(m.surfaces(EditGLB(path.read_bytes())).values()))


def test_answers_are_read_against_the_closed_list_with_synonyms():
    assert mz.parse_vocabulary("Steel, leather and fabric, wood\n") == ["metal", "leather", "cloth", "wood"]
    assert mz.parse_vocabulary("plasma, unicorn") == []
    assert mz.parse_vocabulary("cloth, metal, wood, stone, bone, gem, glass, horn") == \
        ["cloth", "metal", "wood", "stone", "bone", "gem"]


def test_the_vocabulary_is_the_words_the_user_labels_with():
    from image_to_3dlab.material_labels import MATERIALS

    assert set(mz.VOCABULARY) == set(MATERIALS) - {"other"}
    assert set(mz.DEFAULT_PBR) == set(mz.VOCABULARY)


def test_several_views_answers_merge_by_how_often_a_material_is_named():
    merged = mz.merge_vocabularies([["metal", "cloth"], ["cloth", "leather"], ["cloth", "metal", "gem"]])
    assert merged[:2] == ["cloth", "metal"] and set(merged) == {"cloth", "metal", "leather", "gem"}


def test_each_texel_takes_the_material_its_views_saw(tmp_path):
    surface = surface_of(make_glb(tmp_path / "a.glb", painted()))
    views = fake_render(painted())(None, tmp_path / "v", size=SIZE, around=2)
    zone = mz.vote(views, surface, {v.index: bands() for v in views}, THREE)
    assert zone.material_at((0.2, 0.5)) == "leather" and zone.material_at((0.6, 0.5)) == "metal"
    assert zone.material_at((0.9, 0.5)) == "cloth"
    assert zone.zones() == ["leather", "metal", "cloth"]
    mask = zone.mask()
    assert mask[64, 10].tolist() == [255, 0, 0] and mask[64, 120].tolist() == [0, 0, 255]


def test_the_assets_own_metal_tips_a_close_vote(tmp_path):
    surface = surface_of(make_glb(tmp_path / "a.glb", painted()))
    views = fake_render(painted())(None, tmp_path / "v", size=SIZE, around=2)
    unsure = np.zeros((SIZE, SIZE, 2))
    unsure[..., 0], unsure[..., 1] = 0.55, 0.45                   # leather, narrowly
    hint = np.zeros((SIZE, SIZE))
    hint[:, :64] = 1.0
    zone = mz.vote(views, surface, {v.index: unsure for v in views}, ["leather", "metal"], metal_hint=hint)
    assert zone.material_at((0.2, 0.5)) == "metal" and zone.material_at((0.8, 0.5)) == "leather"


def test_zone_counts_follow_the_v1_rule_only_for_creatures_and_equipment():
    assert mz.status(3, "creature")[0] == "pass"
    assert mz.status(2, "equipment_part")[0] == "warn"
    assert mz.status(1, "creature")[0] == "fail"
    assert mz.status(1, "prop")[0] == "pass"


def test_zone_records_name_the_material_and_its_mean_colour(tmp_path):
    surface = surface_of(make_glb(tmp_path / "a.glb", painted()))
    views = fake_render(painted())(None, tmp_path / "v", size=SIZE, around=2)
    zone = mz.vote(views, surface, {v.index: bands() for v in views}, THREE)
    records = mz.zone_records(zone, painted())
    assert [r["channel"] for r in records] == ["R", "G", "B"]
    assert [r["material"] for r in records] == ["leather", "metal", "cloth"]
    assert sum(r["area"] for r in records) == pytest.approx(1.0, abs=0.02)


def test_default_maps_come_from_each_texels_material(tmp_path):
    surface = surface_of(make_glb(tmp_path / "a.glb", painted()))
    views = fake_render(painted())(None, tmp_path / "v", size=SIZE, around=2)
    rough, metal = mz.default_maps(mz.vote(views, surface, {v.index: bands() for v in views}, THREE))
    assert metal[64, 80] == 1.0 and rough[64, 80] == mz.DEFAULT_PBR["metal"][0]
    assert metal[64, 10] == 0.0 and rough[64, 10] == mz.DEFAULT_PBR["leather"][0]


def test_named_regions_become_scores_with_small_regions_on_top():
    regions = np.zeros((2, 8, 8), bool)
    regions[0] = True                                   # the whole object: cloth
    regions[1, 2:4, 2:4] = True                         # a buckle: metal
    scores = mz.region_scores(regions, ["cloth", "metal"], ["cloth", "metal"])
    assert scores[0, 0].tolist() == [1.0, 0.0] and scores[3, 3].tolist() == [0.0, 1.0]


def fake_runner(calls, phrases_answer="leather, metal, cloth"):
    def run(task, job, work, model):
        calls.append((task, model.key))
        work.mkdir(parents=True, exist_ok=True)
        if task == "name_materials":
            if "outline" in job["question"]:
                return {"answers": ["metal" if "region_001" in i["path"] else "cloth" for i in job["images"]]}
            return {"answers": [phrases_answer for _ in job["images"]]}
        if task == "segment_sam3":
            for item in job["images"]:
                np.savez(item["out"], scores=np.moveaxis(bands(job["phrases"]), -1, 0).astype(np.float16))
            return {"images": len(job["images"])}
        if task == "regions_sam2":
            for item in job["images"]:
                masks = np.zeros((2, SIZE, SIZE), bool)
                masks[0] = True
                masks[1, :, :64] = True
                np.savez(item["out"], masks=masks)
            return {"images": len(job["images"])}
        raise AssertionError(task)
    return run


def test_sam3_finds_zones_after_qwen_names_the_materials(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", painted())
    calls = []
    zones, record = mz.find_zones(asset, tmp_path / "work", "sam3", render=fake_render(painted(), count=4),
                                  around=4, runner=fake_runner(calls))
    assert [c[0] for c in calls] == ["name_materials", "segment_sam3"]
    assert zones[0].zones() == ["leather", "metal", "cloth"]
    assert record["vocabulary"] == ["leather", "metal", "cloth"] and record["admitted"] is False
    assert {c["component"].split("@")[0] for c in record["components"]} == {mm.QWEN35.repo, mm.SAM3.repo}


def test_the_fallback_names_sam2_regions_with_qwen(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", painted())
    calls = []
    zones, record = mz.find_zones(asset, tmp_path / "work", "sam2", vocabulary=["cloth", "metal"],
                                  render=fake_render(painted()), around=2, runner=fake_runner(calls))
    assert [c[0] for c in calls] == ["regions_sam2", "name_materials"]
    assert zones[0].material_at((0.2, 0.5)) == "metal" and zones[0].material_at((0.8, 0.5)) == "cloth"
    assert {c["component"].split("@")[0] for c in record["components"]} == {mm.QWEN35.repo, mm.SAM21.repo}


def test_an_unknown_segmenter_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown segmenter"):
        mz.segment([], tmp_path, ["cloth"], "sam9", tmp_path)


def test_the_models_run_in_their_environment_with_the_gpu_held_only_meanwhile(tmp_path, monkeypatch):
    from contextlib import contextmanager

    monkeypatch.setattr(mm, "require", lambda model, hub=None: tmp_path / "snap")
    monkeypatch.setattr(mm, "require_env", lambda repo=None: tmp_path / "python.exe")
    held = []

    @contextmanager
    def lock(session):
        held.append("in")
        yield
        held.append("out")

    def run(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        assert held == ["in"] and job["task"] == "segment_sam3" and job["phrases"] == ["cloth"]
        Path(job["result"]).write_text(json.dumps({"ok": True}), encoding="utf-8")

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()

    assert mz._run("segment_sam3", {"phrases": ["cloth"], "images": []}, tmp_path / "w", mm.SAM3,
                   run=run, lock=lock) == {"ok": True}
    assert held == ["in", "out"]


def zoned(tmp_path, phrases=THREE, scores=None):
    asset = make_glb(tmp_path / "knight.glb", painted())
    views = fake_render(painted())(None, tmp_path / "v", size=SIZE, around=2)
    zone = mz.vote(views, surface_of(asset), {v.index: (bands(phrases) if scores is None else scores) for v in views},
                   phrases)
    record = {"segmenter": "sam3", "vocabulary": phrases, "admitted": False, "components": mz.components("sam3")}
    return asset, zone, record


def test_the_game_materials_carry_the_tint_mask_and_the_zones_names(tmp_path):
    from image_to_3dlab.glb_turn import parse_glb

    asset, zone, record = zoned(tmp_path)
    result = m.make_game_materials(asset, tmp_path / "out", zones={0: zone}, zone_record=record, profile="creature")
    assert result.status == "pass" and result.textures["tint"].name == "T_knight_Tint.png"
    document, _ = parse_glb(result.glb_path.read_bytes())
    extras = document["materials"][0]["extras"]["assetfurnace"]
    assert extras["zones"] == ["leather", "metal", "cloth"]
    glb = EditGLB(result.glb_path.read_bytes())
    tint = m._decode_image(glb, m.texture_source(document["textures"][extras["tintMask"]["index"]]))
    assert tint[64, 10].tolist() == [255, 0, 0] and tint[64, 120].tolist() == [0, 0, 255]
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert [z["material"] for z in sidecar["zones"]] == ["leather", "metal", "cloth"]
    assert sidecar["checks"][0]["check"] == "tint_zones" and sidecar["textures"]["srgb"]["tint"] is False
    provenance = json.loads(result.glb_path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert any(c["component"].startswith(mm.SAM3.repo) for c in provenance["components"])


def test_a_creature_with_two_zones_is_warned(tmp_path):
    two = np.zeros((SIZE, SIZE, 2))
    two[:, :64, 0], two[:, 64:, 1] = 1.0, 1.0
    asset, zone, record = zoned(tmp_path, ["leather", "metal"], two)
    result = m.make_game_materials(asset, tmp_path / "out", zones={0: zone}, zone_record=record, profile="creature")
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert result.status == "warn" and sidecar["checks"][0]["status"] == "warn"


def test_without_zones_nothing_changes(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", painted())
    result = m.make_game_materials(asset, tmp_path / "out")
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["zones"] == [] and sidecar["checks"] == [] and result.status == "pass"
    assert not (tmp_path / "out" / "T_knight_Tint.png").exists()


def test_points_agree_where_the_vote_put_their_material(tmp_path):
    _, zone, _ = zoned(tmp_path)
    points = [{"uv": [0.2, 0.5], "material": 0, "label": "leather"},
              {"uv": [0.6, 0.5], "material": 0, "label": "metal"},
              {"uv": [0.9, 0.5], "material": 0, "label": "wood"},          # wrong: cloth there
              {"uv": [0.9, 0.5], "material": 0, "label": "other"},         # left out
              {"uv": [0.5, 0.5], "material": 3, "label": "metal"}]         # no zones for it
    result = mz.point_agreement({0: zone}, points)
    assert result["points"] == 3 and result["agreement"] == pytest.approx(2 / 3)
    assert result["zone_points"] == 2 and result["zone_agreement"] == 1.0


def test_zones_default_to_sam31(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("game_materials_cli", Path(__file__).resolve().parents[1]
                                                  / "scripts" / "game_materials.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    asked = []

    def fake_find(asset, work, segmenter, vocabulary=None):
        asked.append(segmenter)
        _, zone, record = zoned(tmp_path / "z")
        return {0: zone}, record

    monkeypatch.setattr(mz, "find_zones", fake_find)
    (tmp_path / "z").mkdir()
    asset = make_glb(tmp_path / "knight.glb", painted())
    assert cli.main([str(asset), str(tmp_path / "out"), "--zones"]) == 0
    assert asked == ["sam31"]
    monkeypatch.undo()
    import inspect

    assert inspect.signature(mz.find_zones).parameters["segmenter"].default == "sam31"


def test_a_zones_mean_colour_includes_the_base_colour_factor(tmp_path):
    _, zone, _ = zoned(tmp_path)
    white = np.full((SIZE, SIZE, 3), 255, np.uint8)
    records = mz.zone_records(zone, white, factor=(0.5, 0.5, 0.5))
    assert records[0]["mean_linear"] == pytest.approx([0.5, 0.5, 0.5])


def test_sam31_segments_through_the_same_task_and_is_named_in_the_provenance(tmp_path):
    asset = make_glb(tmp_path / "knight.glb", painted())
    calls = []
    zones, record = mz.find_zones(asset, tmp_path / "work", "sam31", vocabulary=["leather", "metal", "cloth"],
                                  render=fake_render(painted()), around=2, runner=fake_runner(calls))
    assert calls == [("segment_sam3", "sam31")]
    assert zones[0].zones() == ["leather", "metal", "cloth"]
    assert {c["component"].split("@")[0] for c in record["components"]} == {mm.QWEN35.repo, mm.SAM31.repo}
    assert all(c["commercial_status"] == "commercial-conditional" for c in record["components"]
               if c["component"].startswith(mm.SAM31.repo))
