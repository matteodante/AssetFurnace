"""A batch row edited in its spec runs again from its first changed step, not from the
start (docs/plans/2026-10-17-step-parity-design.md, section 5). Fake generator, fake steps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from batch_fakes import (
    FakeFinish,
    fake_step,
    install_fake_generator,
    tiny_glb,
    write_spec,
)

from image_to_3dlab import batch


@pytest.fixture
def fake_finish(monkeypatch):
    finish = FakeFinish()
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(finish.run))
    return finish


def _state(spec_path: Path) -> dict:
    return json.loads((spec_path.parent / "out" / "test-batch" / batch.STATE_FILE).read_text())


def _recording_step(calls: list[str], name: str):
    """A step that only notes it ran; it leaves the asset as it was."""
    def run(ctx):
        from image_to_3dlab.batch_steps import StepResult
        calls.append(f"{name}:{ctx.row.id}")
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        note = ctx.step_dir / f"{name}.txt"
        note.write_text(json.dumps(ctx.options, sort_keys=True))
        return StepResult(files=[note])
    return fake_step(run)


def _edit(spec_path: Path, change) -> None:
    spec = json.loads(spec_path.read_text())
    change(spec)
    spec_path.write_text(json.dumps(spec))


def test_a_new_face_budget_reruns_finish_but_not_the_generation(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    _edit(spec_path, lambda spec: spec["rows"][1].update(faces=8000))
    generator.calls.clear()
    fake_finish.calls.clear()
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0
    assert generator.calls == [] and fake_finish.calls == ["SM_row02"]
    entry = _state(spec_path)["rows"]["SM_row02"]
    assert entry["status"] == "done" and "finish" in entry["note"]


def test_a_new_seed_reruns_the_generation_and_everything_after_it(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    _edit(spec_path, lambda spec: spec["rows"][0].update(seed=99))
    generator.calls.clear()
    fake_finish.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == ["row01"] and fake_finish.calls == ["SM_row01"]


def test_a_step_added_at_the_end_runs_alone(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    calls: list[str] = []
    monkeypatch.setitem(batch.STEPS, "export_unreal", _recording_step(calls, "export_unreal"))
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    _edit(spec_path, lambda spec: spec["rows"][0].update(
        steps=["generate", "finish", "export_unreal"], size_cm=90))
    generator.calls.clear()
    fake_finish.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == [] and fake_finish.calls == [] and calls == ["export_unreal:SM_row01"]


def test_changed_options_of_a_late_step_rerun_only_that_step(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    calls: list[str] = []
    monkeypatch.setitem(batch.STEPS, "export_unreal", _recording_step(calls, "export_unreal"))
    monkeypatch.setitem(batch.STEPS, "checks", _recording_step(calls, "checks"))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "export_unreal", "checks"])
    _edit(spec_path, lambda spec: spec["defaults"].update(size_cm=90))
    batch.run_batch(batch.load_batch(spec_path))
    calls.clear()
    generator.calls.clear()
    fake_finish.calls.clear()
    _edit(spec_path, lambda spec: spec["rows"][0].update(checks={"stop_on_fail": True}))
    batch.run_batch(batch.load_batch(spec_path))
    assert calls == ["checks:SM_row01"] and generator.calls == [] and fake_finish.calls == []
    # the real size feeds the export and the checks: both run again, the rest does not
    calls.clear()
    _edit(spec_path, lambda spec: spec["rows"][0].update(size_cm=120))
    batch.run_batch(batch.load_batch(spec_path))
    assert calls == ["export_unreal:SM_row01", "checks:SM_row01"] and fake_finish.calls == []


def test_a_removed_step_leaves_the_row_on_the_step_before_it(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    _edit(spec_path, lambda spec: spec["defaults"].update(steps=["generate"]))
    generator.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == []
    entry = _state(spec_path)["rows"]["SM_row01"]
    assert list(entry["steps"]) == ["generate"]
    assert entry["asset"].startswith("rows/SM_row01/generate/")


def test_moving_a_step_reruns_from_where_the_order_changed(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    calls: list[str] = []
    monkeypatch.setitem(batch.STEPS, "checks", _recording_step(calls, "checks"))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "checks", "finish"])
    batch.run_batch(batch.load_batch(spec_path))
    calls.clear()
    fake_finish.calls.clear()
    _edit(spec_path, lambda spec: spec["defaults"].update(steps=["generate", "finish", "checks"]))
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == ["row01"]   # once, the first time
    assert fake_finish.calls == ["SM_row01"] and calls == ["checks:SM_row01"]


def test_a_state_from_before_step_fingerprints_is_refused_with_what_to_do(tmp_path, monkeypatch, fake_finish):
    """A batch the first runner left (its done steps without fingerprints) is not read: it
    says so and what to do, and runs nothing again, edited or not."""
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    state = _state(spec_path)
    for entry in state["rows"].values():
        for step in entry["steps"].values():
            step.pop("fingerprint")
    (spec_path.parent / "out" / "test-batch" / batch.STATE_FILE).write_text(json.dumps(state))
    generator.calls.clear()
    fake_finish.calls.clear()
    for edit in (lambda spec: None, lambda spec: spec["rows"][1].update(faces=8000)):
        _edit(spec_path, edit)
        with pytest.raises(batch.BatchError) as refused:
            batch.load_batch(spec_path)
        assert len(refused.value.problems) == 1
        problem = refused.value.problems[0]
        assert "before each step kept its fingerprint (rows SM_row01, SM_row02)" in problem
        assert "give the batch a new name to run it again, or remove its folder" in problem
    assert generator.calls == [] and fake_finish.calls == []
    assert _state(spec_path) == state   # left as it was


def test_a_file_option_replaced_in_place_reruns_its_step(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    calls: list[str] = []
    monkeypatch.setitem(batch.STEPS, "export_unreal", _recording_step(calls, "export_unreal"))
    lod = tiny_glb(tmp_path / "lods" / "crate_LOD1.glb", "one")
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "export_unreal"])
    _edit(spec_path, lambda spec: spec["rows"][0].update(
        size_cm=90, export_unreal={"lods": ["lods/crate_LOD1.glb"]}))
    batch.run_batch(batch.load_batch(spec_path))
    batch.run_batch(batch.load_batch(spec_path))
    assert calls == ["export_unreal:SM_row01"]   # unchanged: done
    tiny_glb(lod, "two")
    fake_finish.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert calls == ["export_unreal:SM_row01"] * 2 and fake_finish.calls == []


def test_the_same_files_by_another_path_are_the_same_row(tmp_path):
    one = tiny_glb(tmp_path / "a" / "body.glb", "same")
    two = tiny_glb(tmp_path / "b" / "body.glb", "same")
    rows = [batch.BatchRow(id="SK_A", images=(), asset=None, kind="equipment", faces=None, profile=None,
                           steps=("generate", "export_unreal"), backend="fake",
                           options={"export_unreal": {"body": str(path)}}) for path in (one, two)]
    assert rows[0].fingerprint() == rows[1].fingerprint()
    assert rows[0].step_fingerprints() == rows[1].step_fingerprints()


def test_a_done_rows_missing_file_option_does_not_stop_the_batch(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    calls: list[str] = []
    monkeypatch.setitem(batch.STEPS, "export_unreal", _recording_step(calls, "export_unreal"))
    lod = tiny_glb(tmp_path / "lods" / "crate_LOD1.glb")
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "export_unreal"])
    _edit(spec_path, lambda spec: spec["rows"][0].update(
        size_cm=90, export_unreal={"lods": ["lods/crate_LOD1.glb"]}))
    batch.run_batch(batch.load_batch(spec_path))
    lod.unlink()
    spec = batch.load_batch(spec_path)   # loads: the row is done
    assert batch.run_batch(spec).exit_code == 0 and calls == ["export_unreal:SM_row01"]


def test_a_resume_preflights_only_the_steps_that_will_rerun(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    _edit(spec_path, lambda spec: spec["rows"][0].update(faces=8000))
    spec = batch.load_batch(spec_path)
    work = batch.pending_work(spec, json.loads(
        (spec.directory / batch.STATE_FILE).read_text()), retry_failed=False)
    assert work == {"SM_row01": ("finish",)}


def test_the_step_fingerprints_follow_the_chain():
    row = batch.BatchRow(id="SM_A", images=(), asset=None, kind="prop", faces=3000, profile=None,
                         steps=("generate", "finish", "checks"), backend="fake",
                         parameters={"seed": 1})
    first = row.step_fingerprints()
    assert list(first) == ["generate", "finish", "checks"]
    faces = batch.BatchRow(**{**row.__dict__, "faces": 4000}).step_fingerprints()
    assert faces["generate"] == first["generate"] and faces["finish"] != first["finish"]
    assert faces["checks"] != first["checks"]   # after a changed step, the chain changes
    kind = batch.BatchRow(**{**row.__dict__, "kind": "creature"}).step_fingerprints()
    assert kind["generate"] == first["generate"] and kind["finish"] == first["finish"]
    assert kind["checks"] != first["checks"]


def test_a_file_option_is_fingerprinted_by_its_content(tmp_path):
    lod = tiny_glb(tmp_path / "crate_LOD1.glb", "one")
    row = batch.BatchRow(id="SM_A", images=(), asset=None, kind="prop", faces=None, profile=None,
                         steps=("generate", "export_unreal"), backend="fake",
                         options={"export_unreal": {"lods": [str(lod)]}})
    before = row.step_fingerprints()
    tiny_glb(lod, "two")
    after = row.step_fingerprints()
    assert before["generate"] == after["generate"]
    assert before["export_unreal"] != after["export_unreal"]


def test_a_row_manifest_names_only_what_the_generation_used(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    manifest = json.loads((spec_path.parent / "out" / "test-batch" / "rows" / "SM_row01"
                           / "manifest.json").read_text())
    assert set(manifest["batch"]) == {"id", "references"}


def test_checks_are_a_default_step():
    assert batch.DEFAULT_STEPS == ("generate", "finish", "checks")


def test_file_options_resolve_against_the_spec(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_glb(tmp_path / "lods" / "crate_LOD1.glb")
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "export_unreal"])
    _edit(spec_path, lambda spec: spec["rows"][0].update(
        size_cm=90, export_unreal={"lods": ["lods/crate_LOD1.glb"]}))
    row = batch.load_batch(spec_path).rows[0]
    assert row.options["export_unreal"]["lods"] == [
        str((tmp_path / "lods" / "crate_LOD1.glb").resolve())]


def test_an_unknown_step_option_stops_the_batch_before_the_first_row(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "checks"])
    _edit(spec_path, lambda spec: spec["rows"][0].update(checks={"stop_on_fial": True}))
    with pytest.raises(batch.BatchError, match="stop_on_fial"):
        batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == []
