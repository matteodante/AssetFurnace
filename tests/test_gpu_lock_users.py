"""The GPU lock (AGENTS.md: %TEMP%\\assetfurnace-gpu.lock) is held by everything that computes
on the GPU, not only text to image: Pixal3D, the Qwen3-VL judge, and sd-cli when a set of
pieces draws (the first real set showed none of them took it). Nothing heavy runs here."""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import pixal3d_generate  # noqa: E402
import slot_parts  # noqa: E402

from image_to_3dlab import gpu_lock  # noqa: E402
from image_to_3dlab.style import judge  # noqa: E402


def test_pixal3d_holds_the_lock_while_it_computes(tmp_path):
    lock = tmp_path / "gpu.lock"
    probe = f"import pathlib, sys; sys.exit(0 if pathlib.Path(r'{lock}').exists() else 3)"
    assert pixal3d_generate.run_trellis([sys.executable, "-c", probe], None, lock_path=lock, cwd=tmp_path) == 0
    assert not lock.exists()   # given back afterwards


def test_the_judge_holds_the_lock_while_its_server_runs(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(judge, "find_parts", lambda hub=None: {"server": "s", "model": "m", "mmproj": "p"})
    seen = []
    monkeypatch.setattr(judge.LlamaServer, "_start", lambda self: seen.append(lock.exists()) or self)
    with judge.LlamaServer(lock_path=lock):
        assert lock.read_text().endswith("qwen3-vl judge")
    assert seen == [True] and not lock.exists()

    def broken(self):
        raise RuntimeError("llama-server stopped while loading")

    monkeypatch.setattr(judge.LlamaServer, "_start", broken)
    with pytest.raises(RuntimeError):
        judge.LlamaServer(lock_path=lock).__enter__()
    assert not lock.exists()   # a server that never came up gives the GPU back


def test_a_set_draws_with_the_lock_and_runs_blender_without_it(monkeypatch):
    held = []

    @contextlib.contextmanager
    def gpu(session, **kwargs):
        held.append(session)
        yield

    monkeypatch.setattr(gpu_lock, "gpu", gpu)
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", ""))
    slot_parts._run([str(Path("vendor") / "sdcpp" / "sd-cli.exe"), "-p", "a helmet"])
    slot_parts._run(["blender.exe", "--background"])
    assert held == ["slot_parts sd-cli"]


def test_a_busy_gpu_leaves_the_pick_to_the_silhouette(tmp_path, monkeypatch, capsys):
    from image_to_3dlab.slot_parts import candidates

    class Busy:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            raise gpu_lock.GpuBusy("the GPU is busy (1 other session)")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(judge, "LlamaServer", Busy)
    monkeypatch.setattr(candidates, "score_all", lambda *a, **kw: {"torso": {}})
    scores, used = slot_parts.score_candidates({}, tmp_path, tmp_path / "pieces", [1, 2])
    assert scores == {"torso": {}} and used.startswith("shape only")
    assert "silhouette" in capsys.readouterr().out



def test_a_judge_that_fails_on_the_way_is_treated_alike_by_every_step(tmp_path, monkeypatch, capsys):
    """Code read 18/10/2026: a llama-server that fell over (RuntimeError) stopped a whole run while
    scoring the pieces, yet only skipped the judge for the weapon's scores and its orientation."""
    from image_to_3dlab.slot_parts import candidates, orient

    monkeypatch.undo()   # the real score_held and orient.with_judge, not conftest's stand-ins

    class Broken:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("llama-server exited with code 3")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(judge, "LlamaServer", Broken)
    monkeypatch.setattr(candidates, "score_all", lambda *a, **kw: {"torso": {}})
    scores, used = slot_parts.score_candidates({}, tmp_path, tmp_path / "pieces", [1, 2])
    assert scores == {"torso": {}} and used == "shape only (llama-server exited with code 3)"
    pictures = {1: tmp_path / "s1.png", 2: tmp_path / "s2.png"}
    assert slot_parts.score_held("weapon", "a longsword", pictures) == {1: None, 2: None}
    assert orient.with_judge([tmp_path / "s1.png"]) == {tmp_path / "s1.png": None}
    said = capsys.readouterr().out
    assert said.count("going on without the picture judge (llama-server exited with code 3)") == 3
