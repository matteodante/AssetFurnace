"""A batch row that starts from words: the `picture` step, its candidates and the pick,
and the MCP tools that drive them (docs/plans/2026-10-17-step-parity-design.md, sections 4
and 6). The picture model and the generator are stand-ins; nothing loads a model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from batch_fakes import FakeFinish, fake_step, install_fake_generator, tiny_png

from image_to_3dlab import batch, mcp_server, step_catalog, text_image
from image_to_3dlab.batch_steps import picture as picture_step


@pytest.fixture
def drawn(monkeypatch):
    """text_image.generate replaced: a PNG per seed, and a list of the seeds drawn."""
    seeds: list[int] = []

    def generate(prompt, settings, output_path, **kwargs):
        seeds.append(settings["seed"])
        tiny_png(output_path, seed=settings["seed"])
        output_path.with_suffix(".provenance.json").write_text(json.dumps(
            {"prompt": prompt, "settings": settings, "license": {"name": text_image.LICENSE_NAME}}))
        return output_path

    monkeypatch.setattr(picture_step.text_image, "generate", generate)
    monkeypatch.setattr(picture_step, "readiness", list)
    return seeds


@pytest.fixture
def generator(monkeypatch):
    generator = install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    return generator


def _spec(tmp_path: Path, rows: list[dict], **top) -> Path:
    spec = {"schema_version": 1, "name": "words", "use_case": "game", "distribution": "private",
            "commercial_intent": True, "license_policy": {"allow_conditional": True},
            "output": {"directory": "out"},
            "defaults": {"backend": "fake", "steps": ["picture", "generate"], "parameters": {"seed": 1}},
            "rows": rows, **top}
    path = tmp_path / "words.json"
    path.write_text(json.dumps(spec))
    return path


def _state(spec_path: Path) -> dict:
    return json.loads((spec_path.parent / "out" / "words" / batch.STATE_FILE).read_text())


def test_a_row_made_from_words_goes_on_to_3d(tmp_path, drawn, generator):
    spec_path = _spec(tmp_path, [{"id": "SM_Barrel", "prompt": "a wooden barrel", "picture": {"seed": 5}}])
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0 and drawn == [5]
    picture = spec_path.parent / "out" / "words" / "rows" / "SM_Barrel" / "picture" / "SM_Barrel.png"
    assert picture.is_file() and picture.with_suffix(".provenance.json").is_file()
    assert generator.calls == ["SM_Barrel"]   # the generator saw the picture the step made
    manifest = json.loads((picture.parents[1] / "manifest.json").read_text())
    assert Path(manifest["input"]["path"]).name == "SM_Barrel.png"


def test_several_pictures_wait_for_a_pick_and_the_rest_of_the_batch_goes_on(tmp_path, drawn, generator):
    spec_path = _spec(tmp_path, [
        {"id": "SM_Barrel", "prompt": "a wooden barrel", "picture": {"seed": 5, "count": 3}},
        {"id": "SM_Crate", "prompt": "a crate"}])
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 3 and report.waiting == ["SM_Barrel"]
    assert drawn == [5, 6, 7, 42] and generator.calls == ["SM_Crate"]
    entry = _state(spec_path)["rows"]["SM_Barrel"]
    assert entry["status"] == "waiting" and entry["steps"]["picture"]["status"] == "waiting"
    choices = [c["choice"] for c in entry["steps"]["picture"]["candidates"]]
    assert choices == [5, 6, 7]
    status = mcp_server.batch_status(str(spec_path))
    (row,) = [r for r in status["rows"] if r["id"] == "SM_Barrel"]
    assert row["status"] == "waiting" and len(row["waiting"]["candidates"]) == 3
    assert all(Path(c["picture"]).is_file() for c in row["waiting"]["candidates"])
    # resumed with no pick: still waiting, and no picture is drawn again
    batch.run_batch(batch.load_batch(spec_path))
    assert drawn == [5, 6, 7, 42]


def test_a_pick_in_the_spec_goes_on_without_drawing_again(tmp_path, drawn, generator):
    rows = [{"id": "SM_Barrel", "prompt": "a wooden barrel", "picture": {"seed": 5, "count": 3}}]
    spec_path = _spec(tmp_path, rows)
    batch.run_batch(batch.load_batch(spec_path))
    rows[0]["picture"]["pick"] = 6
    spec_path = _spec(tmp_path, rows)
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0 and drawn == [5, 6, 7] and generator.calls == ["SM_Barrel"]
    step = _state(spec_path)["rows"]["SM_Barrel"]["steps"]["picture"]
    assert step["record"] == {"seed": 6, "candidates": [5, 6, 7]} and step["key"]
    # another pick: the picture step only copies, the 3D runs again
    rows[0]["picture"]["pick"] = 7
    spec_path = _spec(tmp_path, rows)
    batch.run_batch(batch.load_batch(spec_path))
    assert drawn == [5, 6, 7] and generator.calls == ["SM_Barrel", "SM_Barrel"]


def test_a_pick_out_of_the_candidates_fails_the_row(tmp_path, drawn, generator):
    spec_path = _spec(tmp_path, [{"id": "SM_Barrel", "prompt": "a barrel", "picture": {"count": 2, "pick": 99}}])
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 1 and "99" in report.failed["SM_Barrel"]


def test_an_agent_picks_and_the_batch_goes_on(tmp_path, drawn, generator, monkeypatch):
    spec_path = _spec(tmp_path, [{"id": "SM_Barrel", "prompt": "a wooden barrel", "picture": {"seed": 5, "count": 3}}])
    batch.run_batch(batch.load_batch(spec_path))
    resumed = []
    monkeypatch.setattr(mcp_server, "resume_batch", lambda ref, retry_failed=False: resumed.append(ref) or
                        {"started": True})
    refused = mcp_server.pick_candidate(str(spec_path), "SM_Barrel", "picture", 99)
    assert refused["picked"] is False and "5, 6, 7" in refused["problem"]
    assert mcp_server.pick_candidate(str(spec_path), "SM_Crate", "picture", 5)["picked"] is False
    picked = mcp_server.pick_candidate(str(spec_path), "SM_Barrel", "picture", 6)
    assert picked["picked"] is True and picked["resumed"] == {"started": True}
    # the runner the agent started reads the pick
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0 and drawn == [5, 6, 7] and generator.calls == ["SM_Barrel"]
    assert batch.load_batch(spec_path).rows[0].options["picture"]["pick"] == 6


def test_rows_from_words_are_checked_before_the_batch(tmp_path, generator):
    cases = [
        ({"id": "A", "prompt": "x", "steps": ["generate", "picture"]}, "picture makes the row's picture"),
        ({"id": "B", "prompt": "x", "steps": ["generate"]}, "prompt is for the picture step"),
        ({"id": "C", "steps": ["picture", "generate"]}, "picture needs a prompt"),
        ({"id": "D", "prompt": "x", "images": ["a.png"], "steps": ["picture", "generate"]}, "not both"),
        ({"id": "E", "prompt": "x", "picture": {"count": 99}}, "count must be"),
    ]
    tiny_png(tmp_path / "a.png")
    for row, words in cases:
        with pytest.raises(batch.BatchError, match=words):
            batch.load_batch(_spec(tmp_path, [row]))


def test_the_picture_model_is_held_to_the_licence_policy(tmp_path, generator):
    spec_path = _spec(tmp_path, [{"id": "A", "prompt": "a barrel"}],
                      license_policy={"allow_conditional": False})
    with pytest.raises(batch.BatchError, match="qwen-image"):
        batch.load_batch(spec_path)


def test_preflight_names_what_is_missing_without_downloading(tmp_path, monkeypatch):
    def missing():
        raise text_image.MissingWeights(["qwen-image-2.1-turbo-Q8_0.gguf"])

    monkeypatch.setattr(picture_step.text_image, "resolve_weights", missing)
    monkeypatch.setattr(picture_step.text_image, "BINARY", tmp_path / "sd-cli")
    row = batch.BatchRow(id="A", images=(), asset=None, kind="prop", faces=None, profile=None,
                         steps=("picture", "generate"), backend="fake", prompt="a barrel")
    problems = picture_step.preflight([row])
    assert any("sd-cli" in p for p in problems) and any("qwen-image-2.1-turbo-Q8_0.gguf" in p for p in problems)


def test_the_picture_step_is_described_for_every_route():
    spec = step_catalog.get("picture")
    assert spec.candidates == "count" and spec.cli == "scripts/text_image.py" and spec.gpu
    assert {p.name for p in spec.options} >= {"seed", "count", "pick", "width", "height"}


def test_an_agents_pick_does_not_outlive_its_pictures(tmp_path, drawn, generator, monkeypatch):
    """New words after a pick: the row waits for a new pick, it does not take the old seed
    for a picture nobody saw (nor fail on a seed that was not drawn)."""
    monkeypatch.setattr(mcp_server, "resume_batch", lambda ref, retry_failed=False: {"started": True})
    rows = [{"id": "SM_Barrel", "prompt": "a knight", "picture": {"seed": 5, "count": 3}}]
    spec_path = _spec(tmp_path, rows)
    batch.run_batch(batch.load_batch(spec_path))
    assert mcp_server.pick_candidate(str(spec_path), "SM_Barrel", "picture", 6)["picked"]
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    rows[0]["prompt"] = "a red dragon"
    spec_path = _spec(tmp_path, rows)
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 3 and report.waiting == ["SM_Barrel"]
    assert generator.calls == ["SM_Barrel"]   # only the first, picked, picture went to 3D
    rows[0]["picture"]["seed"] = 100   # and other seeds: waits, does not fail
    spec_path = _spec(tmp_path, rows)
    assert batch.run_batch(batch.load_batch(spec_path)).waiting == ["SM_Barrel"]


def test_a_pick_the_spec_overrides_is_refused_and_not_kept(tmp_path, drawn, generator, monkeypatch):
    monkeypatch.setattr(mcp_server, "resume_batch", lambda ref, retry_failed=False: {"started": True})
    rows = [{"id": "SM_Barrel", "prompt": "a barrel", "picture": {"seed": 5, "count": 3, "pick": 5}}]
    spec_path = _spec(tmp_path, rows)
    batch.run_batch(batch.load_batch(spec_path))
    refused = mcp_server.pick_candidate(str(spec_path), "SM_Barrel", "picture", 7)
    assert refused["picked"] is False and "spec sets" in refused["problem"]
    assert not (spec_path.parent / "out" / "words" / batch.PICKS_FILE).exists()


def test_one_picture_has_nothing_to_pick(tmp_path, drawn, generator):
    spec_path = _spec(tmp_path, [{"id": "SM_Barrel", "prompt": "a barrel"}])
    batch.run_batch(batch.load_batch(spec_path))
    refused = mcp_server.pick_candidate(str(spec_path), "SM_Barrel", "picture", 42)
    assert refused["picked"] is False and "nothing to pick" in refused["problem"]


def test_a_stopped_drawing_keeps_the_pictures_it_made(tmp_path, drawn, generator, monkeypatch):
    real = picture_step.text_image.generate

    def stop_on_the_third(prompt, settings, output_path, **kwargs):
        if settings["seed"] == 7:
            raise KeyboardInterrupt
        return real(prompt, settings, output_path, **kwargs)

    monkeypatch.setattr(picture_step.text_image, "generate", stop_on_the_third)
    spec_path = _spec(tmp_path, [{"id": "SM_Barrel", "prompt": "a barrel", "picture": {"seed": 5, "count": 3}}])
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    monkeypatch.setattr(picture_step.text_image, "generate", real)
    batch.run_batch(batch.load_batch(spec_path))
    assert drawn == [5, 6, 7]   # 5 and 6 once each


def test_candidates_and_picks_from_the_first_version_still_count(tmp_path, drawn, generator, monkeypatch):
    """Picture rows made before candidate keys were step fingerprints: their drawn pictures
    are not drawn again, and a pick saved then still holds."""
    rows = [{"id": "SM_Barrel", "prompt": "a barrel", "picture": {"seed": 5, "count": 2}}]
    spec_path = _spec(tmp_path, rows)
    batch.run_batch(batch.load_batch(spec_path))
    folder = spec_path.parent / "out" / "words"
    old = picture_step.legacy_key("a barrel", {"seed": 5, "count": 2})
    index = folder / "rows" / "SM_Barrel" / "picture.candidates" / "candidates.json"
    index.write_text(json.dumps({"key": old, "seeds": [5, 6]}))   # as the first version wrote it
    (folder / batch.PICKS_FILE).write_text(json.dumps({"SM_Barrel": {"picture": {"choice": 6, "key": old}}}))
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0 and drawn == [5, 6]
