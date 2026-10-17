"""A step's output replaced by a file of your own: `"replace": {"<step>": "file"}` in a
batch row (docs/plans/2026-10-17-step-parity-design.md, section 4). The step does not run;
the file, checked, becomes its output, and the steps after it run on it."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from batch_fakes import FakeFinish, fake_step, install_fake_generator, tiny_glb, tiny_png

from image_to_3dlab import batch, mcp_server


def skinned(path: Path, note: str = "") -> Path:
    document = json.dumps({"asset": {"version": "2.0"}, "nodes": [{}], "skins": [{"joints": [0]}],
                           "extras": {"note": note}}).encode()
    document += b" " * (-len(document) % 4)
    body = struct.pack("<II", len(document), 0x4E4F534A) + document
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


@pytest.fixture
def steps(monkeypatch):
    """generate (fake generator), finish (fake), and checks/export stand-ins that note the
    asset and clips they were given."""
    generator = install_fake_generator(monkeypatch)
    finish = FakeFinish()
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(finish.run))
    seen: list[tuple[str, str, list[str]]] = []

    def noting(name):
        def run(ctx):
            from image_to_3dlab.batch_steps import StepResult
            seen.append((name, ctx.asset.name, [clip.name for clip in ctx.clips]))
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            note = ctx.step_dir / "seen.txt"
            note.write_text(ctx.asset.name)
            return StepResult(files=[note])
        return fake_step(run)

    for name in ("checks", "export_unreal"):
        monkeypatch.setitem(batch.STEPS, name, noting(name))
    return generator, finish, seen


def _spec(tmp_path: Path, row: dict, **defaults) -> Path:
    tiny_png(tmp_path / "refs" / "knight.png")
    path = tmp_path / "lot.json"
    path.write_text(json.dumps({
        "schema_version": 1, "name": "lot", "output": {"directory": "out"},
        "defaults": {"backend": "fake", "steps": ["generate", "finish", "checks"], **defaults},
        "rows": [{"id": "SK_Knight", "images": ["refs/knight.png"], **row}]}))
    return path


def _state(spec_path: Path) -> dict:
    return json.loads((spec_path.parent / "out" / "lot" / batch.STATE_FILE).read_text())


def test_a_finished_model_of_your_own_goes_on_to_the_checks(tmp_path, steps):
    generator, finish, seen = steps
    mine = tiny_glb(tmp_path / "blender" / "model.glb", "edited in Blender")
    mine.with_suffix(".provenance.json").write_text(json.dumps(
        {"license": {"name": "MIT"}, "classification": "commercial-clear", "manual_edit": {"session_id": "s1"}}))
    spec_path = _spec(tmp_path, {"replace": {"finish": "blender/model.glb"}})
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0, report.failed
    assert generator.calls == ["knight"] and finish.calls == []   # finish did not run
    assert seen == [("checks", "SK_Knight.glb", [])]
    step = _state(spec_path)["rows"]["SK_Knight"]["steps"]["finish"]
    assert step["status"] == "done" and step["record"]["replaced_by"] == str(mine.resolve())
    folder = spec_path.parent / "out" / "lot" / "rows" / "SK_Knight" / "finish"
    assert (folder / "SK_Knight.glb").read_bytes() == mine.read_bytes()
    record = json.loads((folder / "SK_Knight.provenance.json").read_text())
    assert record["license"] == {"name": "MIT"} and record["manual_edit"]["session_id"] == "s1"


def test_a_file_without_a_record_says_so_in_its_provenance(tmp_path, steps):
    tiny_glb(tmp_path / "mine.glb", "mine")
    spec_path = _spec(tmp_path, {"replace": {"finish": "mine.glb"}})
    batch.run_batch(batch.load_batch(spec_path))
    record = json.loads((spec_path.parent / "out" / "lot" / "rows" / "SK_Knight" / "finish"
                         / "SK_Knight.provenance.json").read_text())
    assert record["input"]["source"] == "user-provided" and record["license"] is None
    assert "no licence record" in record["note"]


def test_a_replacing_file_is_held_to_the_licence_policy(tmp_path, steps):
    mine = tiny_glb(tmp_path / "mine.glb", "mine")
    mine.with_suffix(".provenance.json").write_text(json.dumps({"classification": "research-only",
                                                                "license": {"name": "NC"}}))
    path = _spec(tmp_path, {"replace": {"finish": "mine.glb"}})
    spec = json.loads(path.read_text())
    spec.update(use_case="game", distribution="worldwide")
    path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError, match="research-only"):
        batch.load_batch(path)


def test_a_changed_file_reruns_from_its_step_and_keeps_the_ones_before(tmp_path, steps):
    generator, finish, seen = steps
    mine = tiny_glb(tmp_path / "mine.glb", "one")
    spec_path = _spec(tmp_path, {"replace": {"finish": "mine.glb"}})
    batch.run_batch(batch.load_batch(spec_path))
    batch.run_batch(batch.load_batch(spec_path))
    assert len(seen) == 1   # unchanged: done
    tiny_glb(mine, "two")
    generator.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == [] and len(seen) == 2


def test_a_generated_model_replaced_after_a_drawn_picture(tmp_path, steps, monkeypatch):
    generator, finish, seen = steps
    from image_to_3dlab.batch_steps import picture as picture_step

    monkeypatch.setattr(picture_step.text_image, "generate",
                        lambda prompt, settings, out, **k: tiny_png(out) and out.with_suffix(".provenance.json")
                        .write_text("{}") and out)
    monkeypatch.setattr(picture_step, "readiness", list)
    tiny_glb(tmp_path / "sculpted.glb", "by hand")
    path = tmp_path / "words.json"
    path.write_text(json.dumps({
        "schema_version": 1, "name": "lot", "output": {"directory": "out"},
        "defaults": {"backend": "fake", "steps": ["picture", "generate", "finish"]},
        "rows": [{"id": "SM_Well", "prompt": "a stone well", "replace": {"generate": "sculpted.glb"}}]}))
    assert batch.run_batch(batch.load_batch(path)).exit_code == 0
    assert generator.calls == [] and finish.calls == ["SM_Well"]


def test_rig_and_clips_of_your_own(tmp_path, steps):
    generator, finish, seen = steps
    skinned(tmp_path / "rigged.glb")
    clips = [skinned(tmp_path / f"knight_{name}.glb", name) for name in ("idle", "walk")]
    spec_path = _spec(tmp_path, {"kind": "creature", "size_cm": 180, "steps": ["generate", "finish", "rig", "animate",
                                                                               "export_unreal"],
                                 "replace": {"rig": "rigged.glb", "animate": [str(c) for c in clips]}})
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    assert seen == [("export_unreal", "SK_Knight_rigged.glb", ["knight_idle.glb", "knight_walk.glb"])]


@pytest.mark.parametrize("row, words", [
    ({"replace": {"materials": "mine.glb"}}, "materials is not one of its steps"),
    ({"steps": ["generate", "finish", "checks"], "replace": {"checks": "mine.glb"}}, "checks makes nothing a file"),
    ({"replace": {"finish": "refs/knight.png"}}, "finish takes a glb"),
    ({"replace": {"finish": "gone.glb"}}, "does not exist"),
    ({"replace": {"finish": "not-a-glb.glb"}}, "not a GLB"),
    ({"steps": ["generate", "rig"], "replace": {"rig": "mine.glb"}}, "no skeleton"),
    ({"replace": "mine.glb"}, "replace must be an object"),
])
def test_a_bad_replacement_stops_the_batch_before_anything_runs(tmp_path, steps, row, words):
    tiny_glb(tmp_path / "mine.glb")
    (tmp_path / "not-a-glb.glb").write_bytes(b"nope")
    with pytest.raises(batch.BatchError, match=words):
        batch.load_batch(_spec(tmp_path, row))


def test_run_steps_passes_a_replacement(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    monkeypatch.setattr(mcp_server, "AGENT_SPECS", tmp_path / ".agent")
    monkeypatch.setattr(mcp_server, "launch", lambda command, log, env=None: 1)
    monkeypatch.setattr(batch, "DEFAULT_OUTPUT", tmp_path / "batches")
    tiny_png(tmp_path / "knight.png")
    mine = tiny_glb(tmp_path / "mine.glb")
    result = mcp_server.run_steps("SK_Knight", ["generate", "finish"], images=[str(tmp_path / "knight.png")],
                                  backend="fake", replace={"finish": str(mine)})
    spec = json.loads(Path(result["spec"]).read_text())
    assert spec["rows"][0]["replace"] == {"finish": str(mine)}
