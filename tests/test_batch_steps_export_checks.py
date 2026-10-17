"""The batch steps `export_unreal` (feature 5) and `checks` (feature 7), with Blender and
the check registry replaced. docs/plans/2026-10-17-step-parity-design.md, section 5."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from batch_fakes import (
    FakeFinish,
    fake_step,
    install_fake_generator,
    tiny_glb,
    tiny_png,
    write_spec,
)

from image_to_3dlab import batch
from image_to_3dlab.batch_steps import StepContext
from image_to_3dlab.batch_steps import checks as checks_step
from image_to_3dlab.batch_steps import export_unreal as export_step
from image_to_3dlab.checks import review


def _row(tmp_path: Path, **fields) -> batch.BatchRow:
    values = {"id": "SM_Cat", "images": (tiny_png(tmp_path / "refs" / "cat.png"),),
              "asset": None, "kind": "prop", "faces": 3000, "profile": None,
              "steps": ("generate", "finish", "export_unreal", "checks"), "backend": "fake",
              "parameters": {"seed": 2}, "options": {}, "size_cm": 90.0, "measure": "height"}
    values.update(fields)
    return batch.BatchRow(**values)


def _context(tmp_path: Path, row: batch.BatchRow, step: str, options=None) -> StepContext:
    row_dir = tmp_path / "rows" / row.id
    asset = tiny_glb(tmp_path / "finish" / f"{row.id}.glb", "finished")
    sidecar = asset.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({"license": {"name": "MIT"}, "classification": "commercial-clear"}))
    return StepContext(row=row, row_dir=row_dir, step_dir=row_dir / step, asset=asset, sidecar=sidecar,
                       intent={"use_case": "game", "distribution": "private", "commercial_intent": True},
                       license_policy={"allow_conditional": True}, options=options or {},
                       log=lambda line: None)


# ---------------------------------------------------------------------- export_unreal

@pytest.fixture
def fake_export(monkeypatch):
    calls = []

    def export(source, **kwargs):
        calls.append({"source": source, **kwargs})
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        glb = tiny_glb(out / "SM_Cat.glb", "unreal")
        sidecar = out / "SM_Cat.unreal.json"
        sidecar.write_text("{}")
        provenance = out / "SM_Cat.provenance.json"
        provenance.write_text("{}")
        problems = kwargs.pop("_problems", [])
        return {"ok": not problems, "out_dir": str(out), "glb": str(glb), "sidecar": str(sidecar),
                "provenance": str(provenance), "tint": None, "problems": problems, "unreal": {}}

    monkeypatch.setattr(export_step, "export_unreal", export)
    monkeypatch.setattr(export_step, "find_blender", lambda: Path("blender.exe"))
    return calls


def test_export_passes_the_rows_size_and_the_step_options(tmp_path, fake_export):
    row = _row(tmp_path, measure="longest")
    lod = tiny_glb(tmp_path / "SM_Cat_LOD1.glb")
    ctx = _context(tmp_path, row, "export_unreal",
                   {"collision": "convex", "lods": [str(lod)], "forward": [1, 0, 0], "name": "Cat"})
    result = export_step.run(ctx)
    (call,) = fake_export
    assert call["source"] == ctx.asset and call["kind"] == "prop"
    assert call["size_cm"] == 90.0 and call["measure"] == "longest"
    assert call["collision"] == "convex" and call["lods"] == [lod]
    assert call["forward"] == (1.0, 0.0, 0.0) and call["name"] == "Cat"
    assert call["out_dir"] == ctx.step_dir
    # the export is an extra output: the row's asset stays the finished GLB
    assert result.ok and result.asset is None
    assert ctx.step_dir / "SM_Cat.glb" in result.files
    assert result.record["glb"] == "SM_Cat.glb" and result.record["sidecar"] == "SM_Cat.unreal.json"


def test_export_names_the_asset_after_the_row_by_default(tmp_path, fake_export):
    export_step.run(_context(tmp_path, _row(tmp_path), "export_unreal"))
    assert fake_export[0]["name"] == "SM_Cat"


def test_an_export_with_problems_fails_the_row(tmp_path, fake_export, monkeypatch):
    original = export_step.export_unreal
    monkeypatch.setattr(export_step, "export_unreal",
                        lambda source, **kw: original(source, **kw, _problems=["no root bone"]))
    result = export_step.run(_context(tmp_path, _row(tmp_path), "export_unreal"))
    assert not result.ok and "no root bone" in result.reason


def test_export_preflight_needs_a_size_blender_and_a_body_for_equipment(tmp_path, monkeypatch):
    monkeypatch.setattr(export_step, "find_blender", lambda: None)
    problems = export_step.preflight([_row(tmp_path, size_cm=None, measure=None),
                                      _row(tmp_path, id="SK_Glove", kind="equipment")])
    assert any("SM_Cat" in p and "size_cm" in p for p in problems)
    assert any("SK_Glove" in p and "body" in p for p in problems)
    assert any("Blender" in p for p in problems)
    monkeypatch.setattr(export_step, "find_blender", lambda: Path("blender.exe"))
    assert export_step.preflight([_row(tmp_path)]) == []


# --------------------------------------------------------------------------- checks

@pytest.fixture
def fake_checks(monkeypatch):
    calls = []

    def run_checks(asset, profile="prop", overrides=None):
        calls.append({"asset": asset, "profile": profile, "overrides": overrides})
        status = (overrides or {}).get("_status", {}).get("value", "warn")
        return {"asset": str(asset.glb), "asset_sha256": asset.sha256, "status": status,
                "profile": profile, "results": [{"check": "license.components", "status": "pass"}],
                "skipped": []}

    monkeypatch.setattr(checks_step, "run_checks", run_checks)
    return calls


def test_checks_use_the_kinds_profile_the_row_size_and_its_picture(tmp_path, fake_checks):
    row = _row(tmp_path, kind="equipment")
    ctx = _context(tmp_path, row, "checks", {"set": {"budget.faces.max": 20000}})
    result = checks_step.run(ctx)
    (call,) = fake_checks
    assert call["profile"] == "equipment_part"
    assert call["overrides"] == {"budget.faces": {"max": 20000}}
    asset = call["asset"]
    assert asset.glb == ctx.asset and asset.key == "SM_Cat" and asset.provenance == ctx.sidecar
    assert asset.reference == row.images[0]
    assert asset.reference_numbers == {"height": 0.9}
    report = ctx.step_dir / "SM_Cat.checks.json"
    assert result.ok and result.asset is None and report in result.files
    assert json.loads(report.read_text())["status"] == "warn"
    assert result.record == {"report": "SM_Cat.checks.json", "status": "warn"}


def test_checks_leave_out_a_size_that_is_not_a_height(tmp_path, fake_checks):
    checks_step.run(_context(tmp_path, _row(tmp_path, measure="longest"), "checks",
                             {"reference": False, "width": 0.5}))
    asset = fake_checks[0]["asset"]
    assert asset.reference is None and asset.reference_numbers == {"width": 0.5}


def test_a_failed_check_stops_the_row_only_when_asked(tmp_path, fake_checks, monkeypatch):
    def failing(asset, profile="prop", overrides=None):
        return {"asset": str(asset.glb), "asset_sha256": asset.sha256, "status": "fail",
                "profile": profile, "results": [], "skipped": []}

    monkeypatch.setattr(checks_step, "run_checks", failing)
    assert checks_step.run(_context(tmp_path, _row(tmp_path), "checks")).ok
    stopped = checks_step.run(_context(tmp_path, _row(tmp_path), "checks", {"stop_on_fail": True}))
    assert not stopped.ok and "fail" in stopped.reason


def test_checks_preflight_catches_a_bad_threshold(tmp_path):
    problems = checks_step.preflight([_row(tmp_path, options={"checks": {"set": {"budget.facez.max": 1}}})])
    assert any("SM_Cat" in p and "budget.facez" in p for p in problems)
    problems = checks_step.preflight([_row(tmp_path, options={"checks": {"profile": "spaceship"}})])
    assert any("spaceship" in p for p in problems)
    assert checks_step.preflight([_row(tmp_path, options={"checks": {"set": {"budget.faces.max": 9}}})]) == []


# ------------------------------------------------------------- inside a whole batch

def test_a_batch_row_arrives_in_the_review_queue_with_its_checks(tmp_path, monkeypatch, fake_checks):
    """Gap 11: a row with the checks step can be approved without running check_asset.py."""
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "checks"])
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0
    log = tmp_path / batch.REVIEW_LOG
    (item,) = review.queue(review.load(log))
    assert item["key"] == "SM_row01" and item["status"] == "warn"
    assert review.approval_blocker(item) is None
    directory = spec_path.parent / "out" / "test-batch"
    (row,) = json.loads((directory / batch.INDEX_FILE).read_text())["rows"]
    assert row["checks"] == {"status": "warn", "report": "rows/SM_row01/checks/SM_row01.checks.json"}


def test_the_index_and_the_mcp_list_the_unreal_export(tmp_path, monkeypatch, fake_export, fake_checks):
    from image_to_3dlab import mcp_server

    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "export_unreal"])
    spec = json.loads(spec_path.read_text())
    spec["defaults"]["size_cm"] = 90
    spec_path.write_text(json.dumps(spec))
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    directory = spec_path.parent / "out" / "test-batch"
    (row,) = json.loads((directory / batch.INDEX_FILE).read_text())["rows"]
    assert row["unreal"] == {"glb": "rows/SM_row01/export_unreal/SM_Cat.glb",
                             "sidecar": "rows/SM_row01/export_unreal/SM_Cat.unreal.json"}
    assert row["asset"].endswith("finish/SM_row01.glb")
    (listed,) = mcp_server.list_outputs(str(directory))["rows"]
    assert listed["unreal"]["glb"] == str((directory / row["unreal"]["glb"]).resolve())


def test_the_shipped_steps_now_include_export_and_checks():
    available = batch.available_steps()
    assert available["export_unreal"] and available["checks"] and available["materials"]


def test_a_version_queued_without_checks_is_queued_again_with_them(tmp_path, monkeypatch, fake_checks):
    """Gap 11 on a finished batch: adding the checks step re-queues the same version with
    its report, so it can be approved; without new checks it is never queued twice."""
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish"])
    batch.run_batch(batch.load_batch(spec_path))
    log = tmp_path / batch.REVIEW_LOG
    (item,) = review.queue(review.load(log))
    assert review.approval_blocker(item) == "its checks have not run yet"
    spec = json.loads(spec_path.read_text())
    spec["defaults"]["steps"] = ["generate", "finish", "checks"]
    spec_path.write_text(json.dumps(spec))
    batch.run_batch(batch.load_batch(spec_path))
    batch.run_batch(batch.load_batch(spec_path))
    events = [line for line in review.load(log) if line["event"] == "queued"]
    assert len(events) == 2 and events[1]["report"].endswith("SM_row01.checks.json")
    (item,) = review.queue(review.load(log))
    assert review.approval_blocker(item) is None


def test_the_real_checks_run_on_a_row(tmp_path):
    """No fakes: the registry's checks read a real (if tiny) GLB without failing the step."""
    row = _row(tmp_path)
    ctx = _context(tmp_path, row, "checks")
    result = checks_step.run(ctx)
    assert result.ok, result.reason
    report = json.loads((ctx.step_dir / "SM_Cat.checks.json").read_text())
    assert report["asset_key"] == "SM_Cat" and report["profile"] == "prop"


def test_export_preflight_checks_its_files_when_it_will_run(tmp_path, monkeypatch):
    monkeypatch.setattr(export_step, "find_blender", lambda: Path("blender.exe"))
    row = _row(tmp_path, options={"export_unreal": {"lods": [str(tmp_path / "gone_LOD1.glb")]}})
    (problem,) = export_step.preflight([row])
    assert "gone_LOD1.glb" in problem and "SM_Cat" in problem


def test_checks_run_again_with_another_result_are_queued_again(tmp_path, monkeypatch, fake_checks):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "checks"])
    batch.run_batch(batch.load_batch(spec_path))

    def failing(asset, profile="prop", overrides=None):
        return {"asset": str(asset.glb), "asset_sha256": asset.sha256, "status": "fail", "profile": profile,
                "results": [{"check": "budget.faces", "status": "fail"},
                            {"check": "license.components", "status": "pass"}], "skipped": []}

    monkeypatch.setattr(checks_step, "run_checks", failing)
    spec = json.loads(spec_path.read_text())
    spec["defaults"]["checks"] = {"set": {"budget.faces.max": 10}}
    spec_path.write_text(json.dumps(spec))
    batch.run_batch(batch.load_batch(spec_path))
    (item,) = review.queue(review.load(tmp_path / batch.REVIEW_LOG))
    assert item["status"] == "fail" and item["summary"]["fail"] == ["budget.faces"]
    batch.run_batch(batch.load_batch(spec_path))   # done: nothing queued again
    assert sum(line["event"] == "queued" for line in review.load(tmp_path / batch.REVIEW_LOG)) == 2


def test_a_batch_row_hangs_its_object_from_a_socket_of_its_body(tmp_path, fake_export):
    body = tiny_glb(tmp_path / "body.glb")
    ctx = _context(tmp_path, _row(tmp_path), "export_unreal", {"body": str(body), "socket": "SOCKET_Weapon=hand_r"})
    assert export_step.run(ctx).ok
    assert fake_export[0]["socket"] == {"name": "SOCKET_Weapon", "bone": "hand_r"}
    bad = _context(tmp_path, _row(tmp_path), "export_unreal", {"body": str(body), "socket": "hand_r"})
    assert "NAME=BONE" in export_step.run(bad).reason
