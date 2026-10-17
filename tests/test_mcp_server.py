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

    def launch(command, log_path, env=None):
        calls.append((command, log_path, env))
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
    command, log_path, env = launched[0]
    assert command[1:] == [str(mcp_server.PIPELINE), "--run-batch", str(spec_path.resolve()),
                           "--retry-failed"]
    assert log_path.parent == (tmp_path / "out" / "test-batch").resolve()
    lock = json.loads((tmp_path / "out" / batch.LOCK_FILE).read_text())
    assert lock["token"] == env[batch.LOCK_TOKEN_ENV] and lock["pid"] == os.getpid()
    # The lock is taken at once, so a second submit cannot slip in before the runner starts
    again = mcp_server.submit_batch(str(spec_path))
    assert again["started"] is False and "already running" in again["problems"][0]


def test_a_launch_that_fails_gives_the_lock_back(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))

    def broken(command, log_path, env=None):
        raise OSError("no python")

    monkeypatch.setattr(mcp_server, "launch", broken)
    result = mcp_server.submit_batch(str(write_spec(tmp_path, rows=1)))
    assert result["started"] is False and "no python" in result["problems"][0]
    assert not (tmp_path / "out" / batch.LOCK_FILE).exists()


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
    batch.acquire_lock(directory.parent, "test-batch", os.getpid(), "runner")
    stopped = []
    monkeypatch.setattr(mcp_server, "terminate_group", stopped.append)
    assert mcp_server.cancel_batch(str(directory)) == {"cancelled": True, "pid": os.getpid()}
    assert stopped == [os.getpid()]


def test_list_steps():
    steps = mcp_server.list_steps()["steps"]
    assert steps["generate"] is True and steps["materials"] is False


def test_the_sdk_serves_every_tool():
    pytest.importorskip("mcp")
    server = mcp_server.build_server()

    async def talk():
        tools = {tool.name for tool in await server.list_tools()}
        result = await server.call_tool("list_steps", {})
        return tools, result

    tools, result = asyncio.run(talk())
    assert tools == {tool.__name__ for tool in mcp_server.TOOLS}
    assert {"validate_batch", "submit_batch", "resume_batch", "batch_status", "list_outputs",
            "cancel_batch", "list_steps", "review_decide", "pieces_run",
            "describe_steps", "run_steps", "pick_candidate"} <= tools
    assert json.loads(result.content[0].text)["steps"]["finish"] is True


def _csv_spec(tmp_path):
    from batch_fakes import tiny_png

    tiny_png(tmp_path / "a.png")
    path = tmp_path / "lot.csv"
    path.write_text("id,images,backend\nSM_A,a.png,fake\n")
    return path


def test_a_csv_takes_its_intent_from_the_call(tmp_path, monkeypatch, launched):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    monkeypatch.setattr(batch, "DEFAULT_OUTPUT", tmp_path / "output" / "batches")
    spec = _csv_spec(tmp_path)
    checked = mcp_server.validate_batch(str(spec), use_case="game", commercial=True)
    assert checked["ok"] and checked["intent"]["use_case"] == "game"
    started = mcp_server.submit_batch(str(spec), use_case="game", distribution="private",
                                      commercial=True)
    assert started["started"] is True
    assert launched[0][0][-5:] == ["--use-case", "game", "--distribution", "private",
                                   "--commercial"]


def test_resume_reuses_the_spec_and_the_intent_it_ran_with(tmp_path, monkeypatch, launched):
    install_fake_generator(monkeypatch, FakeGenerator(interrupt_on="a"))
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    monkeypatch.setattr(batch, "DEFAULT_OUTPUT", tmp_path / "output" / "batches")
    spec = batch.load_batch(_csv_spec(tmp_path), intent={"use_case": "game"})
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(spec)
    result = mcp_server.resume_batch(str(spec.directory), retry_failed=True)
    assert result["started"] is True
    command = launched[0][0]
    assert command[3] == str(spec.path) and "--retry-failed" in command
    assert command[command.index("--use-case") + 1] == "game"


def test_resume_of_a_batch_that_never_ran(tmp_path):
    result = mcp_server.resume_batch(str(tmp_path))
    assert result["started"] is False and "never ran" in result["problems"][0]



def test_status_after_a_stop_shows_the_interrupted_row_queued(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch, FakeGenerator(interrupt_on="row02"))
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=3)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    status = mcp_server.batch_status(str(spec_path))
    assert status["active"] is None and status["counts"] == {"done": 1, "queued": 2}
    row = next(row for row in status["rows"] if row["id"] == "SM_row02")
    assert row["status"] == "queued" and "interrupted" in row["note"]



def test_a_launched_runner_is_waited_for(tmp_path, monkeypatch):
    """On POSIX an exited child nobody waits for stays a zombie and keeps the lock alive."""
    import threading

    waited = threading.Event()

    class Child:
        pid = 4242

        def __init__(self, *args, **kwargs):
            pass

        def wait(self):
            waited.set()
            return 0

    monkeypatch.setattr(mcp_server.subprocess, "Popen", Child)
    assert mcp_server.launch(["python"], tmp_path / "run.log") == 4242
    assert waited.wait(2)



def test_a_resume_submission_does_not_need_what_its_done_rows_needed(tmp_path, monkeypatch,
                                                                     launched):
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=2)
    batch.run_batch(batch.load_batch(spec_path))
    install_fake_generator(monkeypatch, ready=False)
    assert mcp_server.validate_batch(str(spec_path))["ok"] is False   # the full check
    assert mcp_server.submit_batch(str(spec_path))["started"] is True



def test_resume_takes_the_spec_as_well_as_the_folder(tmp_path, monkeypatch, launched):
    install_fake_generator(monkeypatch, FakeGenerator(interrupt_on="row02"))
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=2)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    assert mcp_server.resume_batch(str(spec_path))["started"] is True



def test_status_and_stop_find_a_batch_whose_rows_went_stale(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch, FakeGenerator(interrupt_on="row02"))
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = write_spec(tmp_path, rows=2)
    with pytest.raises(KeyboardInterrupt):
        batch.run_batch(batch.load_batch(spec_path))
    (tmp_path / "refs" / "row02.png").unlink()
    status = mcp_server.batch_status(str(spec_path))
    assert status["name"] == "test-batch" and status["counts"]["done"] == 1
    assert mcp_server.cancel_batch(str(spec_path))["cancelled"] is False   # not running
