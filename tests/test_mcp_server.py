"""The MCP server's tools, called as plain functions and through the MCP SDK.

Submitting is tested with the launcher replaced: no batch process starts here."""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from batch_fakes import (
    FakeFinish,
    FakeGenerator,
    fake_step,
    install_fake_generator,
    write_spec,
)

from image_to_3dlab import batch, mcp_server


@pytest.fixture
def ran(tmp_path, monkeypatch):
    """A finished batch of three rows, one of them failed."""
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=3)
    batch.run_batch(batch.load_batch(spec_path))
    return spec_path


@pytest.fixture
def launched(monkeypatch):
    calls = []

    def launch(command, log_path):
        calls.append((command, log_path))
        return os.getpid()   # a live pid, as a started runner would have

    monkeypatch.setattr(mcp_server, "launch", launch)
    return calls


def test_validate_reports_problems_without_running(tmp_path, monkeypatch):
    generator = install_fake_generator(monkeypatch, ready=False)
    spec_path = write_spec(tmp_path, rows=2)
    result = mcp_server.validate_batch(str(spec_path))
    assert result["ok"] is False
    assert any("bootstrap_fake.py" in problem for problem in result["problems"])
    assert generator.calls == []


def test_validate_a_good_spec(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    result = mcp_server.validate_batch(str(write_spec(tmp_path, rows=2)))
    assert result["ok"] is True and result["rows"] == 2
    assert result["directory"] == str((tmp_path / "out" / "test-batch").resolve())


def test_submit_starts_the_cli_in_the_background_and_answers_at_once(tmp_path, monkeypatch,
                                                                     launched):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=2)
    result = mcp_server.submit_batch(str(spec_path), retry_failed=True)
    assert result["started"] is True and result["pid"] == os.getpid()
    command, log_path = launched[0]
    assert command[1:] == [str(mcp_server.PIPELINE), "--run-batch", str(spec_path.resolve()),
                           "--retry-failed"]
    assert log_path.parent == (tmp_path / "out" / "test-batch").resolve()
    # The lock is taken at once, so a second submit cannot slip in before the runner starts
    again = mcp_server.submit_batch(str(spec_path))
    assert again["started"] is False and "already running" in again["problems"][0]


def test_submit_refuses_a_spec_that_would_not_start(tmp_path, monkeypatch, launched):
    install_fake_generator(monkeypatch, ready=False)
    result = mcp_server.submit_batch(str(write_spec(tmp_path, rows=1)))
    assert result["started"] is False and launched == []


def test_status_by_folder_or_by_spec(ran):
    by_spec = mcp_server.batch_status(str(ran))
    by_folder = mcp_server.batch_status(str(ran.parent / "out" / "test-batch"))
    assert by_spec == by_folder
    assert by_spec["counts"] == {"done": 2, "failed": 1}
    assert by_spec["active"] is None
    rows = {row["id"]: row for row in by_spec["rows"]}
    assert rows["SM_row02"]["status"] == "failed"
    assert "fake failure" in rows["SM_row02"]["reason"]
    assert rows["SM_row01"]["steps"] == {"generate": "done", "finish": "done"}


def test_status_of_a_batch_that_never_ran(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    result = mcp_server.batch_status(str(write_spec(tmp_path, rows=1)))
    assert result["counts"] == {} and result["rows"] == []


def test_list_outputs_gives_absolute_paths(ran):
    result = mcp_server.list_outputs(str(ran))
    assert [row["id"] for row in result["rows"]] == ["SM_row01", "SM_row03"]
    row = result["rows"][0]
    for key in ("asset", "sidecar"):
        assert os.path.isabs(row[key]) and os.path.isfile(row[key])
    assert all(os.path.isabs(path) for path in row["files"])
    only = mcp_server.list_outputs(str(ran), row_id="SM_row03")
    assert [row["id"] for row in only["rows"]] == ["SM_row03"]
    with pytest.raises(ValueError, match="SM_row02"):
        mcp_server.list_outputs(str(ran), row_id="SM_row02")


def test_cancel_stops_the_runner(ran, monkeypatch):
    directory = ran.parent / "out" / "test-batch"
    assert mcp_server.cancel_batch(str(directory))["cancelled"] is False
    (directory / batch.LOCK_FILE).write_text(str(os.getpid()))
    stopped = []
    monkeypatch.setattr(mcp_server, "terminate_group", stopped.append)
    assert mcp_server.cancel_batch(str(directory)) == {"cancelled": True, "pid": os.getpid()}
    assert stopped == [os.getpid()]


def test_list_steps():
    steps = mcp_server.list_steps()["steps"]
    assert steps["generate"] is True and steps["checks"] is False


def test_the_sdk_serves_every_tool():
    pytest.importorskip("mcp")
    server = mcp_server.build_server()

    async def talk():
        tools = {tool.name for tool in await server.list_tools()}
        result = await server.call_tool("list_steps", {})
        return tools, result

    tools, result = asyncio.run(talk())
    assert tools == {"validate_batch", "submit_batch", "batch_status", "list_outputs",
                     "cancel_batch", "list_steps"}
    assert json.loads(result.content[0].text)["steps"]["finish"] is True
