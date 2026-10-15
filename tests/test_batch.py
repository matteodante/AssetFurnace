"""Batch runner: spec parsing, state and resume, the step registry, and the acceptance
criteria of function 1 (docs/plans/2026-10-15-batch-mcp-design.md), all with a fake
generator. No test here loads a model or starts Blender."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from batch_fakes import (
    FakeFinish,
    FakeGenerator,
    fake_step,
    install_fake_generator,
    tiny_png,
    write_spec,
)

from image_to_3dlab import batch, cli


@pytest.fixture
def fake_finish(monkeypatch):
    finish = FakeFinish()
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(finish.run))
    return finish


def _state(spec_path: Path) -> dict:
    return json.loads((spec_path.parent / "out" / "test-batch" / batch.STATE_FILE).read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- the spec

def test_json_and_csv_specs_give_the_same_rows(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "refs" / "a.png", 1)
    tiny_png(tmp_path / "refs" / "b.png", 2)
    tiny_png(tmp_path / "refs" / "b_side.png", 3)
    (tmp_path / "lot.json").write_text(json.dumps({
        "schema_version": 1, "name": "lot",
        "defaults": {"backend": "fake", "parameters": {"seed": 3}},
        "rows": [
            {"id": "SM_A", "images": ["refs/a.png"], "faces": 3000},
            {"id": "SK_B", "images": ["refs/b.png", "refs/b_side.png"], "kind": "creature",
             "faces": 20000, "profile": "unreal", "steps": ["generate", "finish", "materials"]},
        ],
    }))
    (tmp_path / "lot.csv").write_text(
        "id,images,kind,faces,profile,steps,backend,seed\n"
        "SM_A,refs/a.png,,3000,,,fake,3\n"
        "SK_B,refs/b.png;refs/b_side.png,creature,20000,unreal,generate;finish;materials,fake,3\n"
    )
    from_json = batch.load_batch(tmp_path / "lot.json")
    from_csv = batch.load_batch(tmp_path / "lot.csv")
    assert from_json.rows == from_csv.rows
    first, second = from_json.rows
    assert first.images == ((tmp_path / "refs" / "a.png").resolve(),)
    assert first.steps == batch.DEFAULT_STEPS and first.kind == "prop"
    assert second.images[1].name == "b_side.png"   # references are kept, not dropped
    assert second.parameters == {"seed": 3}


def test_rows_can_be_a_csv_named_by_the_json(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "a.png")
    (tmp_path / "rows.csv").write_text("id,images,backend\nSM_A,a.png,fake\n")
    (tmp_path / "lot.json").write_text(json.dumps(
        {"schema_version": 1, "use_case": "game", "rows": "rows.csv"}))
    spec = batch.load_batch(tmp_path / "lot.json")
    assert [row.id for row in spec.rows] == ["SM_A"]
    assert spec.intent["use_case"] == "game"


def test_paths_resolve_against_the_spec_file_not_the_working_directory(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    monkeypatch.chdir(tmp_path.parent)
    spec = batch.load_batch(spec_path)
    assert spec.rows[0].images[0] == (tmp_path / "refs" / "row01.png").resolve()
    assert spec.directory == (tmp_path / "out" / "test-batch").resolve()


def test_a_csv_spec_lands_under_the_repo_output_folder_by_default(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "a.png")
    (tmp_path / "lot.csv").write_text("id,images,backend\nSM_A,a.png,fake\n")
    spec = batch.load_batch(tmp_path / "lot.csv")
    assert spec.directory == batch.DEFAULT_OUTPUT / "lot"
    assert spec.intent == {"use_case": "showcase", "distribution": "private",
                           "commercial_intent": False}


def test_intent_flags_apply_to_a_csv_and_are_refused_for_json(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "a.png")
    (tmp_path / "lot.csv").write_text("id,images,backend\nSM_A,a.png,fake\n")
    spec = batch.load_batch(tmp_path / "lot.csv", intent={"use_case": "game"})
    assert spec.intent["use_case"] == "game"
    json_spec = write_spec(tmp_path / "j", rows=1)
    with pytest.raises(batch.BatchError, match="in the spec"):
        batch.load_batch(json_spec, intent={"use_case": "game"})


@pytest.mark.parametrize("row, message", [
    ({"id": "SM_A", "images": ["missing.png"]}, "does not exist"),
    ({"id": "bad id!", "images": ["refs/row01.png"]}, "id"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "steps": ["generate", "finsh"]}, "finsh"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "steps": ["finish", "generate"]}, "first"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "steps": ["generate", "generate"]}, "twice"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "faces": 50}, "faces"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "kind": "vehicle"}, "kind"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "backend": "trellis2"}, "trellis2"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "parameters": {"sed": 1}}, "sed"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "colour": "red"}, "colour"),
    ({"id": "SM_A", "steps": ["finish"]}, "asset"),
    ({"id": "SM_A", "images": [], "steps": ["generate"]}, "image"),
])
def test_a_bad_row_stops_the_batch_before_anything_runs(tmp_path, monkeypatch, row, message):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    spec = json.loads(spec_path.read_text())
    spec["rows"].append(row)
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError, match=message):
        batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == []


def test_duplicate_ids_are_refused(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    spec = json.loads(spec_path.read_text())
    spec["rows"].append(dict(spec["rows"][0]))
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError, match="twice"):
        batch.load_batch(spec_path)


def test_the_licence_gate_runs_before_any_row(tmp_path, monkeypatch):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=2, use_case="arcade")
    with pytest.raises(batch.BatchError, match="use_case"):
        batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == []


# ------------------------------------------------------------------ acceptance criteria

def test_ten_rows_run_without_intervention(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=10)
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0
    assert report.counts == {"done": 10}
    assert len(generator.calls) == 10 and len(fake_finish.calls) == 10
    state = _state(spec_path)
    for row_id, entry in state["rows"].items():
        assert entry["status"] == "done", row_id
        asset = spec_path.parent / "out" / "test-batch" / entry["asset"]
        assert asset.is_file() and asset.read_bytes().endswith(b"finished")
        generated = entry["steps"]["generate"]
        glbs = [f["path"] for f in generated["files"] if f["path"].endswith(".glb")]
        sidecars = [f["path"] for f in generated["files"]
                    if f["path"].endswith(".provenance.json")]
        assert len(glbs) == 1 and len(sidecars) == 1


def test_an_interrupted_batch_resumes_at_the_right_row(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch, FakeGenerator(interrupt_on="row06"))
    spec_path = write_spec(tmp_path, rows=10)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    state = _state(spec_path)
    statuses = [entry["status"] for entry in state["rows"].values()]
    assert statuses == ["done"] * 5 + ["running"] + ["queued"] * 4
    assert not (spec_path.parent / "out" / "test-batch" / batch.LOCK_FILE).exists()

    generator.calls.clear()
    fake_finish.calls.clear()
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0
    assert generator.calls == [f"row{i:02d}" for i in range(6, 11)]
    assert fake_finish.calls == [f"SM_row{i:02d}" for i in range(6, 11)]


def test_a_killed_row_resumes_from_its_first_unfinished_step(tmp_path, monkeypatch):
    """As after a hard kill during finish: generate is recorded done, the row is running."""
    generator = install_fake_generator(monkeypatch)
    crashing = FakeFinish(fail_on=None)

    def crash(ctx):
        raise KeyboardInterrupt

    monkeypatch.setitem(batch.STEPS, "finish", fake_step(crash))
    spec_path = write_spec(tmp_path, rows=2)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == ["row01"]

    monkeypatch.setitem(batch.STEPS, "finish", fake_step(crashing.run))
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == ["row01", "row02"]       # row01 was not generated again
    assert crashing.calls == ["SM_row01", "SM_row02"]


def test_a_finished_step_whose_file_vanished_runs_again(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch, FakeGenerator(interrupt_on="row02"))
    spec_path = write_spec(tmp_path, rows=2)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    # row01 is done; pretend row02's half-run left nothing, and row01's GLB was deleted
    state = _state(spec_path)
    entry = state["rows"]["SM_row01"]
    entry["status"] = "running"
    (spec_path.parent / "out" / "test-batch" / state_path_of(entry)).unlink()
    (spec_path.parent / "out" / "test-batch" / batch.STATE_FILE).write_text(json.dumps(state))
    generator.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == ["row01", "row02"]


def state_path_of(entry: dict) -> str:
    return next(f["path"] for f in entry["steps"]["generate"]["files"]
                if f["path"].endswith(".glb"))


def test_batch_outputs_match_ten_single_runs(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=10, steps=["generate"])
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    state = _state(spec_path)

    for index in range(1, 11):
        name = f"row{index:02d}"
        manifest = tmp_path / "single" / f"{name}.json"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps({
            "schema_version": 1, "use_case": "game", "distribution": "private",
            "commercial_intent": True, "license_policy": {"allow_conditional": True},
            "input": {"path": f"../refs/{name}.png"},
            "model": {"backend": "fake", "parameters": {"seed": 7}},
            "output": {"directory": str(tmp_path / "single" / name)},
        }))
        assert cli.main(["--run-manifest", str(manifest)]) == 0
        single_glb = next((tmp_path / "single" / name).glob("clear/*.glb"))
        single_record = json.loads(single_glb.with_suffix(".provenance.json").read_text())

        entry = state["rows"][f"SM_{name}"]
        batch_glb = spec_path.parent / "out" / "test-batch" / entry["asset"]
        batch_record = json.loads(batch_glb.with_suffix(".provenance.json").read_text())
        assert _sha(batch_glb) == _sha(single_glb), name
        assert _comparable(batch_record) == _comparable(single_record), name


def _comparable(record: dict) -> dict:
    """A provenance record without what differs between any two runs: ids, times, paths."""
    record = json.loads(json.dumps(record))
    for key in ("run_id", "created_at", "source_manifest", "software"):
        record.pop(key, None)
    record["input"].pop("path")
    record["output"].pop("path")
    return record


# ------------------------------------------------------------------- state and resume

def test_a_failed_row_records_its_reason_and_the_batch_goes_on(tmp_path, monkeypatch,
                                                                fake_finish):
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    spec_path = write_spec(tmp_path, rows=3)
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 1
    assert report.counts == {"done": 2, "failed": 1}
    entry = _state(spec_path)["rows"]["SM_row02"]
    assert entry["status"] == "failed"
    assert "generate" in entry["reason"] and "fake failure on row02" in entry["reason"]


def test_a_step_that_reports_a_problem_fails_the_row(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish(fail_on="SM_row01").run))
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    assert _state(spec_path)["rows"]["SM_row01"]["reason"] == "finish: mesh exploded"


def test_failed_rows_are_retried_only_when_asked(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    spec_path = write_spec(tmp_path, rows=3)
    batch.run_batch(batch.load_batch(spec_path))
    generator.calls.clear()
    generator.fail_on = None
    report = batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == [] and report.counts == {"done": 2, "failed": 1}
    report = batch.run_batch(batch.load_batch(spec_path), retry_failed=True)
    assert generator.calls == ["row02"] and report.counts == {"done": 3}


def test_a_row_changed_in_the_spec_runs_again(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    spec = json.loads(spec_path.read_text())
    spec["rows"][1]["parameters"] = {"seed": 99}
    spec_path.write_text(json.dumps(spec))
    generator.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == ["row02"]


def test_a_changed_intent_refuses_to_resume(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    spec = json.loads(spec_path.read_text())
    spec["distribution"] = "worldwide"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError, match="intent"):
        batch.run_batch(batch.load_batch(spec_path))


def test_a_live_lock_refuses_a_second_runner(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    monkeypatch.setattr(batch, "process_alive", lambda pid: pid == 4242)
    spec_path = write_spec(tmp_path, rows=1)
    spec = batch.load_batch(spec_path)
    spec.directory.mkdir(parents=True)
    (spec.directory / batch.LOCK_FILE).write_text("4242")
    with pytest.raises(batch.BatchError, match="already running"):
        batch.run_batch(spec)


def test_a_lock_taken_for_this_runner_is_its_own(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec = batch.load_batch(write_spec(tmp_path, rows=1))
    spec.directory.mkdir(parents=True)
    (spec.directory / batch.LOCK_FILE).write_text(str(os.getpid()))
    assert batch.run_batch(spec).exit_code == 0


def test_a_stale_lock_is_ignored(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    monkeypatch.setattr(batch, "process_alive", lambda pid: False)
    spec_path = write_spec(tmp_path, rows=1)
    spec = batch.load_batch(spec_path)
    spec.directory.mkdir(parents=True)
    (spec.directory / batch.LOCK_FILE).write_text("999999")
    assert batch.run_batch(spec).exit_code == 0


def test_another_running_batch_in_the_same_folder_blocks_the_gpu(tmp_path, monkeypatch,
                                                                 fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    other = tmp_path / "out" / "other-batch"
    other.mkdir(parents=True)
    (other / batch.LOCK_FILE).write_text(str(os.getpid()))
    with pytest.raises(batch.BatchError, match="other-batch"):
        batch.run_batch(batch.load_batch(spec_path))


def test_the_state_file_is_never_left_half_written(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    batch.write_json(target, {"a": 1})
    original_replace = os.replace

    def fail(*args):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        batch.write_json(target, {"a": 2})
    monkeypatch.setattr(os, "replace", original_replace)
    assert json.loads(target.read_text()) == {"a": 1}


# -------------------------------------------------------------------- step registry

def test_a_missing_step_is_skipped_with_a_warning(tmp_path, monkeypatch, fake_finish, capsys):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "export_unreal"])
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0
    step = _state(spec_path)["rows"]["SM_row01"]["steps"]["export_unreal"]
    assert step["status"] == "skipped" and "not available" in step["warning"]
    assert "export_unreal" in capsys.readouterr().err


def test_a_step_that_becomes_available_runs_on_resume(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "checks"])
    batch.run_batch(batch.load_batch(spec_path))
    seen = []

    def checks(ctx):
        from image_to_3dlab.batch_steps import StepResult
        seen.append(ctx.asset.name)
        return StepResult()

    monkeypatch.setitem(batch.STEPS, "checks", fake_step(checks))
    generator.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert seen == ["SM_row01.glb"] and generator.calls == []
    assert _state(spec_path)["rows"]["SM_row01"]["steps"]["checks"]["status"] == "done"


def test_the_shipped_steps_are_generate_and_finish():
    available = batch.available_steps()
    assert list(available) == list(batch.STEP_NAMES)
    assert available["generate"] and available["finish"]
    assert not any(available[name] for name in ("materials", "export_unreal", "checks"))


def test_a_step_module_that_fails_to_import_is_an_error_not_a_skip(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    (package / "batch_steps").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "batch_steps" / "__init__.py").write_text("")
    (package / "batch_steps" / "checks.py").write_text("import a_module_that_is_not_there\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(batch, "STEP_PACKAGE", "pkg.batch_steps")
    assert batch.load_step("materials") is None
    with pytest.raises(ModuleNotFoundError):
        batch.load_step("checks")


def test_preflight_problems_stop_the_batch_before_the_first_row(tmp_path, monkeypatch):
    generator = install_fake_generator(monkeypatch, ready=False)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(
        FakeFinish().run, preflight=lambda rows: ["Blender not found"]))
    spec_path = write_spec(tmp_path, rows=3)
    with pytest.raises(batch.BatchError) as raised:
        batch.run_batch(batch.load_batch(spec_path))
    assert "bootstrap_fake.py" in str(raised.value) and "Blender" in str(raised.value)
    assert generator.calls == []


def test_step_options_reach_the_step(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    seen = {}

    def finish(ctx):
        from image_to_3dlab.batch_steps import StepResult
        seen[ctx.row.id] = ctx.options
        return StepResult()

    monkeypatch.setitem(batch.STEPS, "finish", fake_step(finish))
    spec_path = write_spec(tmp_path, rows=2)
    spec = json.loads(spec_path.read_text())
    spec["defaults"]["finish"] = {"atlas": 1024}
    spec["rows"][1]["finish"] = {"paint": True}
    spec_path.write_text(json.dumps(spec))
    batch.run_batch(batch.load_batch(spec_path))
    assert seen == {"SM_row01": {"atlas": 1024}, "SM_row02": {"atlas": 1024, "paint": True}}


def test_an_asset_row_skips_generation(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    asset = tmp_path / "old" / "crate.glb"
    asset.parent.mkdir()
    asset.write_bytes(b"glTFold")
    asset.with_suffix(".provenance.json").write_text(json.dumps({"license": {"name": "MIT"}}))
    (tmp_path / "lot.json").write_text(json.dumps({
        "schema_version": 1, "output": {"directory": "out"},
        "rows": [{"id": "SM_Crate", "asset": "old/crate.glb", "steps": ["finish"]}],
    }))
    report = batch.run_batch(batch.load_batch(tmp_path / "lot.json"))
    assert report.exit_code == 0 and generator.calls == []
    assert fake_finish.calls == ["SM_Crate"]


# ------------------------------------------------------------------ outputs index

def test_the_outputs_index_lists_finished_rows_for_the_review_queue(tmp_path, monkeypatch,
                                                                    fake_finish):
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    spec_path = write_spec(tmp_path, rows=3)
    batch.run_batch(batch.load_batch(spec_path))
    directory = spec_path.parent / "out" / "test-batch"
    index = json.loads((directory / batch.INDEX_FILE).read_text())
    assert index["schema_version"] == 1 and index["batch"] == "test-batch"
    rows = {row["id"]: row for row in index["rows"]}
    assert set(rows) == {"SM_row01", "SM_row03"}
    row = rows["SM_row01"]
    assert row["kind"] == "prop" and row["asset"].endswith("SM_row01.glb")
    assert (directory / row["asset"]).is_file() and (directory / row["sidecar"]).is_file()
    assert row["references"] == ["rows/SM_row01/input/row01.png"]


def test_cli_run_batch_reports_and_returns_the_exit_code(tmp_path, monkeypatch, fake_finish,
                                                          capsys):
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    spec_path = write_spec(tmp_path, rows=2)
    assert cli.main(["--run-batch", str(spec_path)]) == 1
    out = capsys.readouterr()
    assert "1 done" in out.out and "1 failed" in out.out


def test_cli_run_batch_rejects_a_bad_spec_with_code_2(tmp_path, monkeypatch, capsys):
    install_fake_generator(monkeypatch)
    (tmp_path / "lot.json").write_text(json.dumps({"schema_version": 1, "rows": [{"id": "x"}]}))
    assert cli.main(["--run-batch", str(tmp_path / "lot.json")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_run_batch_cannot_be_mixed_with_a_manifest(tmp_path, capsys):
    assert cli.main(["--run-batch", "a.json", "--run-manifest", "b.json"]) == 2


@pytest.mark.parametrize("name", ["example-batch.json", "example-batch.csv"])
def test_the_example_specs_only_lack_your_pictures(name):
    """The templates are valid apart from the placeholder files they point at."""
    with pytest.raises(batch.BatchError) as raised:
        batch.load_batch(batch.REPO / "manifests" / name)
    assert raised.value.problems
    assert all("does not exist" in problem for problem in raised.value.problems)
