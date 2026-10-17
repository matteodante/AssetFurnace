"""Second final review, GPU and processes: whatever stops a job stops what it started before
the shared GPU lock is let go, and a lock is only ever taken away from a dead holder."""

from __future__ import annotations

import importlib.util
import os
import signal
from pathlib import Path

import pytest

from image_to_3dlab import batch, gpu_lock, processes, rigging

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pixal3d_generate.py"


def _pixal3d():
    spec = importlib.util.spec_from_file_location("pixal3d_generate_final2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- SIGTERM is an interrupt

def _send_sigterm():
    """What a SIGTERM does to this process, without sending a real one (on Windows
    os.kill with SIGTERM ends the process outright)."""
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler), "SIGTERM would end the process without any cleanup"
    handler(signal.SIGTERM, None)


def test_sigterm_becomes_an_interrupt_and_the_old_handler_comes_back():
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), processes.sigterm_as_interrupt():
        _send_sigterm()
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_handler_someone_else_set_is_left_alone():
    def theirs(signum, frame):
        pass

    previous = signal.signal(signal.SIGTERM, theirs)
    try:
        with processes.sigterm_as_interrupt():
            assert signal.getsignal(signal.SIGTERM) is theirs
        assert signal.getsignal(signal.SIGTERM) is theirs
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_a_cancelled_batch_is_interrupted_not_killed(monkeypatch, tmp_path):
    # MCP cancel_batch sends SIGTERM to the batch's group: the batch must unwind (rows back
    # in the queue, children stopped) as it does on Ctrl+C
    monkeypatch.setattr(batch, "load_batch", lambda path, intent: object())
    monkeypatch.setattr(batch, "run_batch", lambda spec, retry_failed: _send_sigterm())
    before = signal.getsignal(signal.SIGTERM)
    assert batch.run_from_cli(tmp_path / "b.yaml", {}, False) == 130
    assert signal.getsignal(signal.SIGTERM) is before


class _Child:
    def __init__(self, on_wait=None):
        self.pid, self.on_wait, self.waits = 4242, on_wait, 0

    def wait(self, timeout=None):
        self.waits += 1
        if self.on_wait and self.waits == 1:
            self.on_wait()
        return 0


def test_a_sigterm_while_skintokens_runs_stops_it_before_the_lock_goes(monkeypatch, tmp_path):
    lock = tmp_path / "gpu.lock"
    stopped = []
    monkeypatch.setattr(processes, "terminate_group", lambda pid: stopped.append((pid, lock.exists())))
    child = _Child(on_wait=_send_sigterm)
    with pytest.raises(KeyboardInterrupt):
        rigging.run_commands([(["skintokens"], tmp_path)], tmp_path / "rig.log", gpu={0},
                             popen=lambda *a, **k: child, lock=lock)
    assert stopped == [(4242, True)]          # stopped while the lock was still held
    assert not lock.exists()


# --------------------------------------------------- Pixal3D and the judge stop their child

class _Trellis:
    def __init__(self, lock):
        self.lock, self.calls = lock, []
        self.pid = 5151

        def lines():
            yield "loading\n"
            raise KeyboardInterrupt

        self.stdout = lines()

    def kill(self):
        self.calls.append(("kill", self.lock.exists()))

    def wait(self, timeout=None):
        self.calls.append(("wait", self.lock.exists()))
        return -9

    def poll(self):
        return None


def test_an_interrupted_pixal3d_run_stops_trellis_before_the_lock_goes(monkeypatch, tmp_path):
    px = _pixal3d()
    lock = tmp_path / "gpu.lock"
    made = []
    monkeypatch.setattr(px.subprocess, "Popen", lambda *a, **k: made.append(_Trellis(lock)) or made[-1])
    with pytest.raises(KeyboardInterrupt):
        px.run_trellis(["trellis-cli"], None, lock_path=lock, cwd=tmp_path)
    assert made[0].calls[:2] == [("kill", True), ("wait", True)]
    assert not lock.exists()


def test_a_judge_that_fails_to_start_stops_llama_server_before_the_lock_goes(monkeypatch, tmp_path):
    from image_to_3dlab.style import judge

    lock = tmp_path / "gpu.lock"
    calls = []

    class Server:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            calls.append(("terminate", lock.exists()))

        def wait(self, timeout=None):
            calls.append(("wait", lock.exists()))
            return 0

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(judge, "find_parts", lambda hub=None: {"server": "s", "model": "m", "mmproj": "p"})
    monkeypatch.setattr(judge.subprocess, "Popen", lambda *a, **k: Server())
    monkeypatch.setattr(judge.urllib.request, "urlopen", interrupted)
    with pytest.raises(KeyboardInterrupt), judge.LlamaServer(lock_path=lock):
        pass
    assert calls[:2] == [("terminate", True), ("wait", True)]
    assert not lock.exists()


# ------------------------------------------------------- a stale lock, and only that one

def test_breaking_a_stale_lock_never_removes_a_fresh_one(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("777 first", encoding="utf-8")       # retaken since it was judged stale
    gpu_lock._break(path, "4242 dead")
    assert path.read_text(encoding="utf-8") == "777 first"
    assert [p.name for p in tmp_path.iterdir()] == ["gpu.lock"]


def test_only_one_job_at_a_time_breaks_a_stale_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("4242 dead", encoding="utf-8")
    guard = gpu_lock.breaking(path)
    guard.write_text(str(os.getpid()), encoding="utf-8")  # another job is breaking it
    gpu_lock._break(path, "4242 dead")
    assert path.read_text(encoding="utf-8") == "4242 dead"
    assert guard.exists()
    guard.unlink()
    gpu_lock._break(path, "4242 dead")
    assert not path.exists() and not guard.exists()


def test_a_breaker_that_died_does_not_block_the_next_one(tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("4242 dead", encoding="utf-8")
    guard = gpu_lock.breaking(path)
    guard.write_text("4243", encoding="utf-8")
    gpu_lock._break(path, "4242 dead", alive=lambda pid: False)
    gpu_lock._break(path, "4242 dead", alive=lambda pid: False)
    assert not path.exists() and not guard.exists()


def test_the_interleaving_from_the_review_leaves_one_holder(tmp_path, monkeypatch):
    # two waiters judge the same dead lock; the first breaks it and takes the GPU; the
    # second then breaks with its old judgement: the first's lock must survive, so a third
    # job cannot take the GPU alongside it
    path = tmp_path / "gpu.lock"
    path.write_text("4242 dead", encoding="utf-8")
    judged = gpu_lock.holder(path)[1]
    gpu_lock._break(path, judged)
    assert gpu_lock._take(path, "777 first")
    gpu_lock._break(path, judged)                         # the second waiter, late
    assert not gpu_lock._take(path, "888 third")
    assert path.read_text(encoding="utf-8") == "777 first"


# ------------------------------------- a studio picture cancelled while the rewriter waits

def test_a_picture_cancelled_while_the_rewriter_waits_never_rewrites(tmp_path, monkeypatch):
    import threading
    import time

    from test_image_api import WEIGHTS, api

    from image_to_3dlab import prompt_rewrite, text_image

    lock = tmp_path / "gpu.lock"
    lock.write_text(f"{os.getpid()} pixal3d", encoding="utf-8")   # a live holder: this process
    monkeypatch.setattr(text_image, "GPU_LOCK", lock)
    monkeypatch.setattr(api, "GPU_POLL", 0.05)
    ran = []
    monkeypatch.setattr(prompt_rewrite, "system_prompt_path", lambda: tmp_path / "system.txt")
    monkeypatch.setattr(prompt_rewrite, "llama_completion", lambda: tmp_path / "llama-completion")
    monkeypatch.setattr(prompt_rewrite.subprocess, "run", lambda *a, **k: ran.append(a) or pytest.fail("rewrote"))
    manager = api.ImageJobManager(output_root=tmp_path / "images")
    job = manager.create("a fox", api.clean_settings({"rewrite": True}))
    assert job.settings.get("rewrite")
    worker = threading.Thread(target=api.run_job, args=(job, manager), kwargs={"weights": WEIGHTS}, daemon=True)
    worker.start()
    deadline = time.monotonic() + 10
    while not any("Waiting for the GPU" in str(e.get("message", "")) for e in job.events):
        assert time.monotonic() < deadline, job.events
        time.sleep(0.02)
    assert manager.cancel(job.id)
    worker.join(10)
    assert not worker.is_alive()
    assert job.status == "cancelled" and not ran
    assert lock.read_text(encoding="utf-8") == f"{os.getpid()} pixal3d"
