"""The coordinator's second final review (Codex, d6f7057..106fd90, 17/10/2026), the batches
part: one test per defect confirmed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from batch_fakes import install_fake_generator, tiny_glb, tiny_png

from image_to_3dlab import batch
from image_to_3dlab.batch_steps import checks as checks_step
from image_to_3dlab.batch_steps import finish as finish_step
from image_to_3dlab.batch_steps import generate as generate_step
from tests.test_batch_steps import _context, _row


def test_a_commercial_batch_is_held_to_the_picture_models_licence(tmp_path):
    spec = tmp_path / "lot.json"
    spec.write_text(json.dumps({"schema_version": 1, "name": "lot", "use_case": "game", "distribution": "worldwide",
                                "commercial_intent": True, "output": {"directory": str(tmp_path / "out")},
                                "rows": [{"id": "SM_A", "prompt": "a red barrel", "steps": ["picture"]}]}))
    with pytest.raises(batch.BatchError, match="commercial"):   # Qwen-Image is research-only
        batch.load_batch(spec)


def test_checks_measure_the_size_on_the_scaled_export_not_on_the_source(tmp_path, monkeypatch):
    seen = []

    def fake_checks(asset, **kwargs):
        seen.append(asset)
        return {"status": "pass", "results": []}

    monkeypatch.setattr(checks_step, "run_checks", fake_checks)
    source = tiny_glb(tmp_path / "finish" / "crate.glb")
    row = _row(tmp_path, steps=("export_unreal", "checks"), asset=source, images=(), size_cm=90.0, measure="height")
    ctx = replace(_context(tmp_path, row, source, None, {"reference": False}), step_dir=tmp_path / "rows" / "SM_Cat" / "checks")
    checks_step.run(ctx)
    assert seen[-1].reference_numbers is None   # no export yet: a 2 m source is not "90 cm, 122% over"
    exported = tiny_glb(tmp_path / "rows" / "SM_Cat" / "export_unreal" / "SM_Cat.glb")
    checks_step.run(ctx)
    assert seen[-1].glb == exported and seen[-1].reference_numbers == {"height": 0.9}


def test_a_reference_named_like_the_staged_concept_does_not_overwrite_it(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    concept = tiny_png(tmp_path / "drawn" / "chair.png", 7)
    first, second = tiny_png(tmp_path / "refs" / "chair.png", 1), tiny_png(tmp_path / "refs" / "drawn_chair.png", 2)
    row = _row(tmp_path, images=(first, second))
    ctx = replace(_context(tmp_path, row), picture=concept)
    generate_step.run(ctx)
    staged = sorted((ctx.row_dir / "input").glob("*.png"))
    assert len(staged) == 3   # the concept and both references, none written over another
    manifest = json.loads((ctx.row_dir / "manifest.json").read_text())
    assert Path(manifest["input"]["path"]).read_bytes() == concept.read_bytes()   # the model is made from it


def test_a_concept_first_row_passes_finishs_preflight(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    shape = tiny_png(tmp_path / "shape.png")
    row = _row(tmp_path, images=(), steps=("concept", "generate", "finish"),
               options={"concept": {"shape": [str(shape)]}})
    assert not any("source image" in problem for problem in finish_step.preflight([row]))


def test_an_edited_detail_level_found_beside_the_model_reruns_its_export(tmp_path):
    lod0 = tiny_glb(tmp_path / "crate_LOD0.glb", "a")
    lod1 = tiny_glb(tmp_path / "crate_LOD1.glb", "b")
    row = _row(tmp_path, images=(), asset=lod0, steps=("export_unreal",), size_cm=50.0)
    before = row.step_fingerprints()["export_unreal"]
    tiny_glb(lod1, "an edited detail level")
    assert row.step_fingerprints()["export_unreal"] != before
