"""The MCP tools that reach single steps: describe_steps, run_steps (a batch of one row),
rerun_step. docs/plans/2026-10-17-step-parity-design.md, section 6. No runner starts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from batch_fakes import (
    FakeFinish,
    fake_step,
    install_fake_generator,
    tiny_png,
    write_spec,
)

from image_to_3dlab import batch, mcp_server
from image_to_3dlab.batch_steps import picture as picture_step


@pytest.fixture
def launched(monkeypatch, tmp_path):
    calls = []

    def launch(command, log_path, env=None):
        calls.append(command)
        return os.getpid()

    monkeypatch.setattr(mcp_server, "launch", launch)
    monkeypatch.setattr(mcp_server, "AGENT_SPECS", tmp_path / "batches" / ".agent")
    monkeypatch.setattr(batch, "DEFAULT_OUTPUT", tmp_path / "batches")
    monkeypatch.setattr(picture_step, "readiness", list)
    return calls


def test_describe_steps_is_the_catalogue_with_what_is_installed():
    steps = {step["name"]: step for step in mcp_server.describe_steps()["steps"]}
    assert list(steps)[:4] == ["picture", "concept", "generate", "finish"]
    assert steps["finish"]["available"] and steps["picture"]["candidates"] == "count"
    assert any(param["name"] == "faces" for param in steps["finish"]["params"])
    assert steps["materials"] == {"name": "materials", "available": False, "described": False}
    assert json.loads(json.dumps(steps)) == steps


def test_run_steps_is_a_batch_of_one_row(tmp_path, monkeypatch, launched):
    install_fake_generator(monkeypatch)
    result = mcp_server.run_steps("SM_Barrel", ["picture", "generate"], prompt="a wooden barrel",
                                  backend="fake", options={"picture": {"count": 3, "seed": 5}},
                                  use_case="game", commercial=True)
    assert result["started"] is True, result
    spec = json.loads(Path(result["spec"]).read_text(encoding="utf-8"))
    (row,) = spec["rows"]
    assert row == {"id": "SM_Barrel", "steps": ["picture", "generate"], "kind": "prop",
                   "prompt": "a wooden barrel", "backend": "fake", "picture": {"count": 3, "seed": 5}}
    assert spec["use_case"] == "game" and spec["commercial_intent"] is True
    assert launched[0][launched[0].index("--run-batch") + 1] == result["spec"]
    assert result["directory"] == str(tmp_path / "batches" / "SM_Barrel")


def test_run_steps_refuses_what_a_batch_would_refuse(monkeypatch, launched):
    install_fake_generator(monkeypatch)
    assert mcp_server.run_steps("bad name!", ["generate"])["started"] is False
    refused = mcp_server.run_steps("SM_A", ["generate"], prompt="a barrel", backend="fake")
    assert refused["started"] is False and any("picture step" in p for p in refused["problems"])
    assert launched == []


def test_rerun_step_runs_a_finished_step_again_and_what_follows(tmp_path, monkeypatch):
    generator = install_fake_generator(monkeypatch)
    finish = FakeFinish()
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(finish.run))
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    directory = spec_path.parent / "out" / "test-batch"
    resumed = []
    monkeypatch.setattr(mcp_server, "resume_batch", lambda ref, retry_failed=False: resumed.append(ref) or
                        {"started": True})
    assert mcp_server.rerun_step(str(directory), "SM_row02", "checks")["started"] is False
    assert mcp_server.rerun_step(str(directory), "SM_row02", "finish") == {"started": True}
    generator.calls.clear()
    finish.calls.clear()
    batch.run_batch(batch.load_batch(spec_path))
    assert generator.calls == [] and finish.calls == ["SM_row02"]


def test_a_finished_pick_can_be_changed(tmp_path, monkeypatch):
    seeds = []

    def generate(prompt, settings, output_path, **kwargs):
        seeds.append(settings["seed"])
        tiny_png(output_path, seed=settings["seed"])
        output_path.with_suffix(".provenance.json").write_text("{}")
        return output_path

    monkeypatch.setattr(picture_step.text_image, "generate", generate)
    monkeypatch.setattr(picture_step, "readiness", list)
    generator = install_fake_generator(monkeypatch)
    spec_path = tmp_path / "words.json"
    spec_path.write_text(json.dumps({
        "schema_version": 1, "name": "test-batch", "output": {"directory": "out"},
        "defaults": {"backend": "fake", "steps": ["picture", "generate"]},
        "rows": [{"id": "SM_A", "prompt": "a crate", "picture": {"count": 2, "seed": 1}}]}))
    batch.run_batch(batch.load_batch(spec_path))
    monkeypatch.setattr(mcp_server, "resume_batch", lambda ref, retry_failed=False: {"started": True})
    assert mcp_server.pick_candidate(str(spec_path), "SM_A", "picture", 2)["picked"]
    batch.run_batch(batch.load_batch(spec_path))
    assert mcp_server.pick_candidate(str(spec_path), "SM_A", "picture", 1)["picked"]   # changed mind
    batch.run_batch(batch.load_batch(spec_path))
    assert seeds == [1, 2] and len(generator.calls) == 2
