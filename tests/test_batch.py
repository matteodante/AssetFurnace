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
    ({"id": "SM_A", "steps": []}, "at least one step"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "parameters": "seed=1"}, "parameters"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "finish": "fast"}, "settings for finish"),
    ({"id": "SM_A", "images": [42]}, "list of file paths"),
    ({"id": "SM_A", "images": ["refs/row01.png"], "backend": {"x": 1}}, "backend must be a name"),
    ({"id": "SM_A", "asset": 42, "steps": ["finish"]}, "asset must be a file path"),
    ({"id": "SM_A", "images": ["refs/row01.png", "other/row01.png"]}, "same file name"),
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
    assert not (spec_path.parent / "out" / batch.LOCK_FILE).exists()

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
    spec = batch.load_batch(write_spec(tmp_path, rows=1))
    batch.acquire_lock(spec.output_dir, spec.name, 4242, "theirs")
    with pytest.raises(batch.BatchError, match="already running"):
        batch.run_batch(spec)


def test_a_lock_reserved_for_this_runner_is_taken_over(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    monkeypatch.setattr(batch, "process_alive", lambda pid: pid == 4242)
    spec = batch.load_batch(write_spec(tmp_path, rows=1))
    batch.acquire_lock(spec.output_dir, spec.name, 4242, "reserved")
    monkeypatch.setenv(batch.LOCK_TOKEN_ENV, "reserved")
    assert batch.run_batch(spec).exit_code == 0
    assert not (spec.output_dir / batch.LOCK_FILE).exists()


def test_a_stale_lock_is_ignored(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    monkeypatch.setattr(batch, "process_alive", lambda pid: False)
    spec = batch.load_batch(write_spec(tmp_path, rows=1))
    batch.acquire_lock(spec.output_dir, spec.name, 999999, "dead")
    assert batch.run_batch(spec).exit_code == 0


def test_another_running_batch_in_the_same_folder_blocks_the_gpu(tmp_path, monkeypatch,
                                                                 fake_finish):
    install_fake_generator(monkeypatch)
    spec = batch.load_batch(write_spec(tmp_path, rows=1))
    batch.acquire_lock(spec.output_dir, "other-batch", os.getpid(), "theirs")
    with pytest.raises(batch.BatchError, match="other-batch"):
        batch.run_batch(spec)


def test_of_two_lockers_at_once_exactly_one_wins(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def take(index):
        try:
            batch.acquire_lock(tmp_path, f"batch{index}", os.getpid(), f"token{index}")
            return True
        except batch.BatchError:
            return False

    with ThreadPoolExecutor(8) as pool:
        won = list(pool.map(take, range(8)))
    assert won.count(True) == 1
    holder = json.loads((tmp_path / batch.LOCK_FILE).read_text())
    assert holder["batch"] == f"batch{won.index(True)}"


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


# ---------------------------------------------------------------- the true size

def test_size_and_measure_reach_the_row_and_its_records(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=2)
    spec = json.loads(spec_path.read_text())
    spec["rows"][0].update(size_cm=180, measure="height")
    spec["rows"][1].update(size_cm=42.5)
    spec_path.write_text(json.dumps(spec))
    loaded = batch.load_batch(spec_path)
    assert (loaded.rows[0].size_cm, loaded.rows[0].measure) == (180, "height")
    assert (loaded.rows[1].size_cm, loaded.rows[1].measure) == (42.5, "height")
    batch.run_batch(loaded)
    directory = spec_path.parent / "out" / "test-batch"
    manifest = json.loads((directory / "rows" / "SM_row01" / "manifest.json").read_text())
    assert manifest["batch"]["size_cm"] == 180 and manifest["batch"]["measure"] == "height"
    index = json.loads((directory / batch.INDEX_FILE).read_text())
    assert index["rows"][1]["size_cm"] == 42.5 and index["rows"][1]["measure"] == "height"


def test_size_and_measure_come_from_csv_columns(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "a.png")
    (tmp_path / "lot.csv").write_text(
        "id,images,backend,size_cm,measure\nSM_A,a.png,fake,250,longest\n")
    row = batch.load_batch(tmp_path / "lot.csv").rows[0]
    assert (row.size_cm, row.measure) == (250, "longest")


@pytest.mark.parametrize("fields, message", [
    ({"size_cm": 0}, "size_cm"),
    ({"size_cm": -3}, "size_cm"),
    ({"size_cm": "tall"}, "size_cm"),
    ({"size_cm": True}, "size_cm"),
    ({"size_cm": 120, "measure": "width"}, "measure"),
    ({"measure": "height"}, "size_cm"),
])
def test_bad_sizes_are_refused(tmp_path, monkeypatch, fields, message):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    spec = json.loads(spec_path.read_text())
    spec["rows"][0].update(fields)
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError, match=message):
        batch.load_batch(spec_path)


# ---------------------------------------------------------------- the review queue

def test_finished_rows_are_queued_for_review(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    spec_path = write_spec(tmp_path, rows=3)
    batch.run_batch(batch.load_batch(spec_path))
    log = tmp_path / batch.REVIEW_LOG
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [event["key"] for event in events] == ["SM_row01", "SM_row03"]
    event = events[0]
    directory = spec_path.parent / "out" / "test-batch"
    asset = directory / "rows" / "SM_row01" / "finish" / "SM_row01.glb"
    # feature 7's format (image_to_3dlab.checks.review.enqueue_for_review)
    assert event["event"] == "queued" and event["asset_sha256"] == _sha(asset)
    assert event["asset"] == str(asset.resolve()) and event["at"]
    assert event["source"] == "batch:test-batch" and event["kind"] == "prop"
    assert event["report"] == "" and event["status"] is None
    assert event["summary"] == {"fail": [], "warn": []}
    # a resumed batch does not queue the same version again
    batch.run_batch(batch.load_batch(spec_path))
    assert len(log.read_text().splitlines()) == 2


def test_the_review_event_carries_the_checks_result_when_there_is_one(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)

    def checks(ctx):
        from image_to_3dlab.batch_steps import StepResult
        report = ctx.asset.with_suffix(".checks.json")
        report.write_text(json.dumps({
            "status": "warn", "profile": "prop", "asset_sha256": _sha(ctx.asset),
            "results": [{"check": "budget.faces", "status": "warn"},
                        {"check": "license.components", "status": "pass"}]}))
        return StepResult(files=[report])

    monkeypatch.setitem(batch.STEPS, "checks", fake_step(checks))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "checks"])
    batch.run_batch(batch.load_batch(spec_path))
    event = json.loads((tmp_path / batch.REVIEW_LOG).read_text().splitlines()[0])
    assert event["status"] == "warn" and event["report"].endswith(".checks.json")
    assert event["summary"] == {"fail": [], "warn": ["budget.faces"]}
    assert event["profile"] == "prop" and event["license_status"] == "pass"


def test_a_report_for_another_file_is_not_attached(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)

    def checks(ctx):
        from image_to_3dlab.batch_steps import StepResult
        report = ctx.asset.with_suffix(".checks.json")
        report.write_text(json.dumps({"status": "fail", "asset_sha256": "an older version"}))
        return StepResult(files=[report])

    monkeypatch.setitem(batch.STEPS, "checks", fake_step(checks))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "checks"])
    batch.run_batch(batch.load_batch(spec_path))
    event = json.loads((tmp_path / batch.REVIEW_LOG).read_text().splitlines()[0])
    assert event["status"] is None and event["report"] == ""


def test_feature_7s_queue_function_is_used_when_it_is_installed(tmp_path, monkeypatch,
                                                                fake_finish):
    """After feature 7 merges, its enqueue_for_review writes the line, not this module."""
    import sys
    import types

    calls = []
    checks = types.ModuleType("image_to_3dlab.checks")
    checks.enqueue_for_review = lambda glb, key, **kw: calls.append((Path(glb).name, key, kw))
    monkeypatch.setitem(sys.modules, "image_to_3dlab.checks", checks)
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    spec = json.loads(spec_path.read_text())
    spec["rows"][0]["kind"] = "equipment"
    spec_path.write_text(json.dumps(spec))
    batch.run_batch(batch.load_batch(spec_path))
    ((name, key, kw),) = calls
    assert (name, key) == ("SM_row01.glb", "SM_row01")
    assert kw["source"] == "batch:test-batch" and kw["kind"] == "equipment_part"
    assert kw["log"] == tmp_path / batch.REVIEW_LOG and kw["report"] is None
    assert not (tmp_path / batch.REVIEW_LOG).exists()



def test_a_malformed_output_or_policy_is_a_problem_not_a_crash(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, output="out", license_policy=True)
    with pytest.raises(batch.BatchError) as raised:
        batch.load_batch(spec_path)
    assert any("output must be an object" in p for p in raised.value.problems)
    assert any("license_policy must be an object" in p for p in raised.value.problems)


def test_rows_dropped_from_the_spec_leave_the_count(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    spec = json.loads(spec_path.read_text())
    spec["rows"] = spec["rows"][:1]
    spec_path.write_text(json.dumps(spec))
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0
    state = _state(spec_path)
    assert batch.summary(state) == {"done": 1}
    assert state["removed_rows"]["SM_row02"]["status"] == "failed"


def test_a_row_left_running_reads_as_queued_once_its_runner_is_gone():
    state = {"rows": {"a": {"status": "done"}, "b": {"status": "running"}}}
    assert batch.summary(state, running=True) == {"done": 1, "running": 1}
    assert batch.summary(state, running=False) == {"done": 1, "queued": 1}
    assert "interrupted" in batch.rows_as_seen(state, running=False)["b"]["note"]


def test_the_review_state_belongs_to_one_version_of_one_asset(tmp_path):
    log = tmp_path / batch.REVIEW_LOG
    lines = [
        {"event": "queued", "key": "SM_Crate", "asset_sha256": "old"},
        {"event": "sent_back", "key": "SM_Crate", "asset_sha256": "old", "note": "too big"},
        {"event": "queued", "key": "SM_Crate", "asset_sha256": "new"},
        {"event": "approved", "key": "SM_Crate", "asset_sha256": "other-batch"},
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")
    assert batch.review_state(log, "SM_Crate", "new")["event"] == "queued"
    assert batch.review_state(log, "SM_Crate", "old")["note"] == "too big"
    assert batch.review_state(log, "SM_Barrel", "new") is None



def test_the_repaint_goes_through_the_licence_gate(tmp_path, monkeypatch):
    """Hunyuan's paint is not cleared for a worldwide game, whatever generated the model."""
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, distribution="worldwide")
    spec = json.loads(spec_path.read_text())
    spec["rows"][0]["finish"] = {"paint": True}
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError, match="the repaint"):
        batch.load_batch(spec_path)
    spec["rows"][0]["finish"] = {}
    spec_path.write_text(json.dumps(spec))
    batch.load_batch(spec_path)   # the fake generator alone is fine worldwide


def test_a_changed_file_in_a_camera_folder_runs_the_step_again(tmp_path, monkeypatch,
                                                               fake_finish):
    generator = install_fake_generator(monkeypatch, FakeGenerator(interrupt_on="row02"))
    spec_path = write_spec(tmp_path, rows=2)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    state = _state(spec_path)
    files = state["rows"]["SM_row01"]["steps"]["generate"]["files"]
    camera = next(f["path"] for f in files if f["path"].endswith("transforms.json"))
    directory = spec_path.parent / "out" / "test-batch"
    (directory / camera).write_text("{}")
    state["rows"]["SM_row01"]["status"] = "running"
    (directory / batch.STATE_FILE).write_text(json.dumps(state))
    generator.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == ["row01", "row02"]


def test_a_row_stopped_after_queueing_is_queued_once_and_finished(tmp_path, monkeypatch,
                                                                  fake_finish):
    install_fake_generator(monkeypatch)
    real = batch.queue_for_review

    def queue_then_stop(spec, row, asset):
        real(spec, row, asset)
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "queue_for_review", queue_then_stop)
    spec_path = write_spec(tmp_path, rows=1)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    assert _state(spec_path)["rows"]["SM_row01"]["status"] == "running"
    monkeypatch.setattr(batch, "queue_for_review", real)
    batch.run_batch(batch.load_batch(spec_path))
    assert _state(spec_path)["rows"]["SM_row01"]["status"] == "done"
    assert len((tmp_path / batch.REVIEW_LOG).read_text().splitlines()) == 1



def test_a_string_for_allow_conditional_never_opens_the_licence_gate(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, license_policy={"allow_conditional": "false"})
    with pytest.raises(batch.BatchError, match="true or false"):
        batch.load_batch(spec_path)


def test_a_stale_lock_is_only_cleared_if_nobody_replaced_it(tmp_path):
    batch.acquire_lock(tmp_path, "old", 999999, "dead")
    stale = batch._read_lock(tmp_path)
    (tmp_path / batch.LOCK_FILE).unlink()
    batch.acquire_lock(tmp_path, "fresh", os.getpid(), "alive")   # someone got there first
    batch._clear_stale_lock(tmp_path, stale)
    assert batch._read_lock(tmp_path)["batch"] == "fresh"


def test_of_many_takers_of_a_stale_lock_exactly_one_wins(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(batch, "process_alive", lambda pid: pid != 999999)
    batch.acquire_lock(tmp_path, "old", 999999, "dead")

    def take(index):
        try:
            batch.acquire_lock(tmp_path, f"batch{index}", 1000 + index, f"token{index}")
            return True
        except batch.BatchError:
            return False

    with ThreadPoolExecutor(8) as pool:
        won = list(pool.map(take, range(8)))
    assert won.count(True) == 1


def test_the_hash_of_an_asset_no_step_replaced(tmp_path):
    asset = tmp_path / "crate.glb"
    asset.write_bytes(b"glTFcrate")
    entry = {"asset": "crate.glb", "steps": {"checks": {"status": "done", "files": []}}}
    assert batch.asset_sha(entry, tmp_path) == _sha(asset)


def test_a_finish_only_rows_source_goes_through_the_licence_gate(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    asset = tmp_path / "old" / "knight.glb"
    asset.parent.mkdir()
    asset.write_bytes(b"glTF")
    asset.with_suffix(".provenance.json").write_text(json.dumps(
        {"model": {"backend": "hunyuan-comfyui"}, "input": {"path": "x"}}))
    (tmp_path / "lot.json").write_text(json.dumps({
        "schema_version": 1, "use_case": "game", "distribution": "worldwide",
        "rows": [{"id": "SK_Knight", "asset": "old/knight.glb", "steps": ["finish"]}]}))
    with pytest.raises(batch.BatchError, match="its source"):
        batch.load_batch(tmp_path / "lot.json")


def test_a_resume_checks_only_what_it_will_run(tmp_path, monkeypatch, fake_finish):
    """A failed row that is not retried does not need its generator installed."""
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    install_fake_generator(monkeypatch, ready=False)
    assert batch.run_batch(batch.load_batch(spec_path)).counts == {"done": 1, "failed": 1}
    with pytest.raises(batch.BatchError, match="bootstrap_fake"):
        batch.run_batch(batch.load_batch(spec_path), retry_failed=True)


def test_a_finish_only_rows_recorded_image_is_part_of_its_fingerprint(tmp_path):
    asset = tmp_path / "crate.glb"
    asset.write_bytes(b"glTF")
    image = tiny_png(tmp_path / "crate.png", 1)
    asset.with_suffix(".provenance.json").write_text(json.dumps({"input": {"path": str(image)}}))
    row = batch.BatchRow(id="SM_Crate", images=(), asset=asset, kind="prop", faces=None,
                         profile=None, steps=("finish",), backend="pixal3d")
    before = row.fingerprint()
    tiny_png(image, 2)
    assert row.fingerprint() != before



def test_ids_differing_only_in_case_are_the_same_folder(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    spec = json.loads(spec_path.read_text())
    spec["rows"].append({"id": "sm_ROW01", "images": ["refs/row01.png"]})
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError, match="ignoring case"):
        batch.load_batch(spec_path)


def test_a_csv_line_with_too_many_cells_is_a_problem(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    tiny_png(tmp_path / "a.png")
    (tmp_path / "lot.csv").write_text("id,images,backend\nSM_A,a.png,fake,oops\n")
    with pytest.raises(batch.BatchError, match="more cells than columns"):
        batch.load_batch(tmp_path / "lot.csv")


def test_the_repaint_is_gated_only_where_finish_runs(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, distribution="worldwide")
    spec = json.loads(spec_path.read_text())
    spec["defaults"]["finish"] = {"paint": True}
    spec["rows"][0]["steps"] = ["generate"]
    spec_path.write_text(json.dumps(spec))
    batch.load_batch(spec_path)   # no finish, no repaint, nothing to refuse


def test_nested_provenance_leads_to_the_source_backend_and_image(tmp_path):
    image = tiny_png(tmp_path / "concept.png")
    asset = tmp_path / "SK_Knight.glb"
    asset.write_bytes(b"glTF")
    asset.with_suffix(".provenance.json").write_text(json.dumps({
        "input": {"path": str(tmp_path / "older.glb")},          # a finish record
        "source_record": {"input": {"path": str(image)},
                          "model": {"backend": "hunyuan-comfyui"}}}))
    assert batch.recorded_backend(asset) == "hunyuan-comfyui"
    assert batch.recorded_image(asset) == image


def test_an_asset_rows_record_counts_even_with_its_own_image(tmp_path):
    asset = tmp_path / "crate.glb"
    asset.write_bytes(b"glTF")
    record = asset.with_suffix(".provenance.json")
    record.write_text(json.dumps({"classification": "commercial-conditional"}))
    row = batch.BatchRow(id="SM_Crate", images=(tiny_png(tmp_path / "c.png"),), asset=asset,
                         kind="prop", faces=None, profile=None, steps=("finish",),
                         backend="pixal3d")
    before = row.fingerprint()
    record.write_text(json.dumps({"classification": "territory-restricted"}))
    assert row.fingerprint() != before


def test_a_done_row_whose_files_went_missing_runs_again(tmp_path, monkeypatch, fake_finish):
    generator = install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    finished = spec_path.parent / "out" / "test-batch" / "rows" / "SM_row01" / "finish"
    (finished / "SM_row01.glb").unlink()
    generator.calls.clear()
    fake_finish.calls.clear()
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.counts == {"done": 2}
    assert generator.calls == [] and fake_finish.calls == ["SM_row01"]
    assert (finished / "SM_row01.glb").is_file()


def test_a_row_reset_to_the_queue_leaves_the_index_at_once(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    spec = json.loads(spec_path.read_text())
    spec["rows"][0]["parameters"] = {"seed": 5}
    spec_path.write_text(json.dumps(spec))

    def stop(ctx):
        index = json.loads((spec_path.parent / "out" / "test-batch" / batch.INDEX_FILE).read_text())
        assert index["rows"] == []   # the old result is gone before the rerun
        raise KeyboardInterrupt

    monkeypatch.setitem(batch.STEPS, "finish", fake_step(stop))
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))


def test_a_finish_only_rows_picture_is_staged_and_indexed(tmp_path, monkeypatch, fake_finish):
    asset = tmp_path / "old" / "crate.glb"
    asset.parent.mkdir()
    asset.write_bytes(b"glTFold")
    tiny_png(tmp_path / "old" / "crate.png")
    (tmp_path / "lot.json").write_text(json.dumps({
        "schema_version": 1, "output": {"directory": "out"},
        "rows": [{"id": "SM_Crate", "asset": "old/crate.glb", "images": ["old/crate.png"],
                  "steps": ["finish"]}]}))
    batch.run_batch(batch.load_batch(tmp_path / "lot.json"))
    index = json.loads((tmp_path / "out" / "lot" / batch.INDEX_FILE).read_text())
    assert index["rows"][0]["references"] == ["rows/SM_Crate/input/crate.png"]


def test_the_generation_records_its_inputs_and_log(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1)
    batch.run_batch(batch.load_batch(spec_path))
    files = [f["path"] for f in _state(spec_path)["rows"]["SM_row01"]["steps"]["generate"]["files"]]
    assert "rows/SM_row01/input/row01.png" in files
    assert all("bytes" in f for f in _state(spec_path)["rows"]["SM_row01"]["steps"]["generate"]["files"])



def test_a_source_backend_without_a_reviewed_licence_is_refused(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    asset = tmp_path / "old" / "thing.glb"
    asset.parent.mkdir()
    asset.write_bytes(b"glTF")
    asset.with_suffix(".provenance.json").write_text(json.dumps(
        {"model": {"backend": "mystery3d"}, "input": {"path": "x"}}))
    (tmp_path / "lot.json").write_text(json.dumps({
        "schema_version": 1,
        "rows": [{"id": "SM_Thing", "asset": "old/thing.glb", "steps": ["finish"]}]}))
    with pytest.raises(batch.BatchError, match="mystery3d"):
        batch.load_batch(tmp_path / "lot.json")


@pytest.mark.parametrize("field, value", [
    ("commercial_intent", "false"), ("use_case", "arcade"), ("distribution", "galaxy")])
def test_the_intent_must_be_real_values_for_every_batch(tmp_path, monkeypatch, field, value):
    install_fake_generator(monkeypatch)
    asset = tmp_path / "crate.glb"
    asset.write_bytes(b"glTF")
    (tmp_path / "lot.json").write_text(json.dumps({
        "schema_version": 1, field: value,
        "rows": [{"id": "SM_Crate", "asset": "crate.glb", "steps": ["finish"]}]}))
    with pytest.raises(batch.BatchError, match=field):
        batch.load_batch(tmp_path / "lot.json")


def test_two_batch_names_that_are_one_folder_are_refused(tmp_path, monkeypatch, fake_finish):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, name="Oak")
    batch.run_batch(batch.load_batch(spec_path))
    spec = json.loads(spec_path.read_text())
    spec["name"] = "oak"
    spec_path.write_text(json.dumps(spec))
    loaded = batch.load_batch(spec_path)
    loaded_dir = loaded.output_dir / "Oak"   # where the first one lives, whatever the OS
    monkeypatch.setattr(batch.BatchSpec, "directory", property(lambda self: loaded_dir))
    with pytest.raises(batch.BatchError, match="ignoring case"):
        batch.run_batch(loaded)


def test_a_finish_only_row_stages_the_picture_its_record_leads_to(tmp_path, monkeypatch,
                                                                  fake_finish):
    asset = tmp_path / "old" / "crate.glb"
    asset.parent.mkdir()
    asset.write_bytes(b"glTFold")
    image = tiny_png(tmp_path / "old" / "crate_concept.png")
    asset.with_suffix(".provenance.json").write_text(json.dumps({"input": {"path": str(image)}}))
    (tmp_path / "lot.json").write_text(json.dumps({
        "schema_version": 1, "output": {"directory": "out"},
        "rows": [{"id": "SM_Crate", "asset": "old/crate.glb", "steps": ["finish"]}]}))
    batch.run_batch(batch.load_batch(tmp_path / "lot.json"))
    index = json.loads((tmp_path / "out" / "lot" / batch.INDEX_FILE).read_text())
    assert index["rows"][0]["references"] == ["rows/SM_Crate/input/crate_concept.png"]


def test_a_reused_pid_does_not_hold_the_lock(tmp_path, monkeypatch):
    batch.acquire_lock(tmp_path, "old", os.getpid(), "dead runner")
    lock = json.loads((tmp_path / batch.LOCK_FILE).read_text())
    assert lock["identity"] == batch.process_identity(os.getpid())
    lock["identity"] = "a process that started long ago"
    (tmp_path / batch.LOCK_FILE).write_text(json.dumps(lock))
    assert batch.running_batches(tmp_path) == []
    batch.acquire_lock(tmp_path, "new", os.getpid(), "fresh")   # taken over as stale
    assert json.loads((tmp_path / batch.LOCK_FILE).read_text())["batch"] == "new"


def test_a_record_leads_to_the_picture_a_finish_used(tmp_path):
    image = tiny_png(tmp_path / "concept.png")
    record = {"input": {"path": str(tmp_path / "older.glb"), "image": {"path": str(image)}}}
    assert batch.image_from_record(record) == image



def test_an_output_directory_must_be_a_path(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, output={"directory": 1})
    with pytest.raises(batch.BatchError, match="output.directory"):
        batch.load_batch(spec_path)
