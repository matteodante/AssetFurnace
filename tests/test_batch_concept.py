"""The `concept` step (feature 2 in a batch): a concept drawn from the row's pictures,
with candidates and picks like `picture`; the studio, `reference_concept.py` and the step
share `image_to_3dlab/concept/requests.py`. The drawing itself is a stand-in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from batch_fakes import install_fake_generator, tiny_png

from image_to_3dlab import batch, mcp_server, step_catalog
from image_to_3dlab.batch_steps import concept as concept_step
from image_to_3dlab.batch_steps import picture as picture_step
from image_to_3dlab.concept import requests, sidecar


@pytest.fixture
def drawn(monkeypatch):
    """requests.draw replaced: a PNG per seed with the records the real one writes."""
    calls: list[dict] = []

    def draw(plan, seed, output_path, **kwargs):
        calls.append({"mode": plan["mode"], "prompt": plan["prompt"], "seed": seed,
                      "references": [(p.name, role) for p, role in plan["references"]]})
        tiny_png(output_path, seed=seed)
        output_path.with_suffix(".provenance.json").write_text(json.dumps({"concept": {"mode": plan["mode"]}}))
        sidecar.write(output_path, mode=plan["mode"], prompt=plan["prompt"], seed=seed, command=[],
                      references=plan["references"])
        return output_path

    def picture(prompt, settings, output_path, **kwargs):
        tiny_png(output_path, seed=settings["seed"])
        output_path.with_suffix(".provenance.json").write_text("{}")
        return output_path

    monkeypatch.setattr(concept_step.requests, "draw", draw)
    monkeypatch.setattr(concept_step, "readiness", list)
    monkeypatch.setattr(picture_step.text_image, "generate", picture)
    monkeypatch.setattr(picture_step, "readiness", list)
    return calls


def _spec(tmp_path: Path, rows: list[dict], steps=("concept", "generate")) -> Path:
    path = tmp_path / "concepts.json"
    path.write_text(json.dumps({
        "schema_version": 1, "name": "concepts", "output": {"directory": "out"},
        "defaults": {"backend": "fake", "steps": list(steps)}, "rows": rows}))
    return path


def test_a_concept_from_the_rows_picture_goes_on_to_3d(tmp_path, monkeypatch, drawn):
    generator = install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "wolf.png", 3)
    spec_path = _spec(tmp_path, [{"id": "SK_Werewolf", "images": ["wolf.png"],
                                  "concept": {"mode": "redraw", "subject": "werewolf", "colours": "grey fur",
                                              "structure": "lineart", "seed": 9}}])
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0, report.failed
    (call,) = drawn
    assert call["mode"] == "redraw" and "werewolf" in call["prompt"] and call["seed"] == 9
    assert call["references"] == [("wolf.png", "shape")]
    folder = spec_path.parent / "out" / "concepts" / "rows" / "SK_Werewolf"
    assert (folder / "concept" / "SK_Werewolf.png.concept.json").is_file()
    # the generator drew from the concept, not from the reference
    assert generator.calls == ["SK_Werewolf"]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert Path(manifest["input"]["path"]).name == "SK_Werewolf.png"
    assert [Path(p).name for p in manifest["batch"]["references"]] == ["wolf.png"]


def test_attempts_wait_for_a_pick_like_pictures(tmp_path, monkeypatch, drawn):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "wolf.png", 3)
    rows = [{"id": "SK_Wolf", "images": ["wolf.png"],
             "concept": {"mode": "other_view", "view": "back", "seed": 1, "count": 3}}]
    spec_path = _spec(tmp_path, rows)
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.waiting == ["SK_Wolf"] and [c["seed"] for c in drawn] == [1, 2, 3]
    monkeypatch.setattr(mcp_server, "resume_batch", lambda ref, retry_failed=False: {"started": True})
    assert mcp_server.pick_candidate(str(spec_path), "SK_Wolf", "concept", 2)["picked"]
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    assert len(drawn) == 3   # the pick drew nothing


def test_a_concept_after_a_drawn_picture_reads_it(tmp_path, monkeypatch, drawn):
    install_fake_generator(monkeypatch)
    spec_path = _spec(tmp_path, [{"id": "SM_Shield", "prompt": "a round wooden shield",
                                  "concept": {"mode": "other_view", "view": "side"}}],
                      steps=("picture", "concept", "generate"))
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    assert drawn[0]["references"] == [("SM_Shield.png", "shape")]


def test_shape_and_style_pictures_from_the_spec(tmp_path, monkeypatch, drawn):
    install_fake_generator(monkeypatch)
    for name in ("body.png", "style.png"):
        tiny_png(tmp_path / name)
    spec_path = _spec(tmp_path, [{"id": "SK_Knight", "images": ["body.png"],
                                  "concept": {"mode": "dress", "outfit": "plate armour",
                                              "style_refs": ["style.png"]}}])
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    assert drawn[0]["references"] == [("body.png", "shape"), ("style.png", "style")]
    # a new style picture is a new concept, and a new 3D model
    tiny_png(tmp_path / "style.png", seed=8)
    batch.run_batch(batch.load_batch(spec_path))
    assert len(drawn) == 2


def test_concept_rows_are_checked_before_the_batch(tmp_path, monkeypatch, drawn):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "wolf.png")
    with pytest.raises(batch.BatchError, match="concept needs a picture"):
        batch.load_batch(_spec(tmp_path, [{"id": "A", "concept": {"mode": "redraw"}, "asset": None}]))
    with pytest.raises(batch.BatchError, match="mode must be one of"):
        batch.load_batch(_spec(tmp_path, [{"id": "A", "images": ["wolf.png"], "concept": {"mode": "paint"}}]))
    spec = batch.load_batch(_spec(tmp_path, [{"id": "A", "images": ["wolf.png"], "concept": {"subject": "x"}}]))
    assert any("mode" in p for p in concept_step.preflight(spec.rows))
    spec = batch.load_batch(_spec(tmp_path, [{"id": "B", "images": ["wolf.png"], "concept": {"mode": "redraw"}}]))
    assert any("subject" in p for p in concept_step.preflight(spec.rows))


def test_the_concept_model_is_held_to_the_licence_policy(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "wolf.png")
    path = tmp_path / "strict.json"
    path.write_text(json.dumps({"schema_version": 1, "license_policy": {"allow_conditional": False},
                                "defaults": {"backend": "fake", "steps": ["concept", "generate"]},
                                "rows": [{"id": "A", "images": ["wolf.png"], "concept": {"mode": "other_view",
                                                                                       "view": "back"}}]}))
    with pytest.raises(batch.BatchError, match="qwen-image"):
        batch.load_batch(path)


def test_the_studio_builds_its_request_with_the_same_plan(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
    import concept_api

    tiny_png(tmp_path / "images" / "wolf.png")
    request = {"mode": "redraw", "subject": "werewolf", "colours": "grey", "shape": ["images/wolf.png"]}
    assert concept_api.build(tmp_path, request) == requests.plan(
        "redraw", [(tmp_path / "images" / "wolf.png").resolve()], [], request)


def test_the_concept_step_is_described_with_its_command_line():
    spec = step_catalog.get("concept")
    assert spec.candidates == "count" and spec.feature == 2 and spec.cli == "scripts/reference_concept.py make"
    assert spec.param("count").flag == "--attempts" and spec.param("style_refs").flag == "--style-ref"
