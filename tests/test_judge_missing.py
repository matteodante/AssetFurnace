"""Without the picture judge (llama.cpp running Qwen3-VL), a set's pictures are picked by
silhouette only. The person is told: Setup & Status lists the judge with its download, the
run's log says it goes on without it, and the pick panel, the page and agents see it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
sys.path.insert(0, str(REPO / "scripts"))
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def test_setup_lists_the_judge_with_its_download():
    import backend_catalog
    import download_api

    judge = backend_catalog.BY_ID["judge"]
    assert judge.kind == "tool" and "bootstrap_style.py --only llama" in judge.install
    assert "silhouette" in judge.tradeoff and judge.license_name == "MIT"
    command = download_api.COMMANDS["judge"]
    assert command[-3:] == ["--only", "llama", "--yes"]


def test_the_run_says_it_goes_on_without_the_judge(tmp_path, monkeypatch, capsys):
    import slot_parts

    from image_to_3dlab.slot_parts import candidates
    from image_to_3dlab.style import judge, models

    def missing(**kwargs):
        raise models.MissingModel("llama.cpp is not installed; run: python scripts/bootstrap_style.py --only llama")

    monkeypatch.setattr(judge, "LlamaServer", missing)
    monkeypatch.setattr(candidates, "score_all", lambda *a, **kw: {"torso": {}})
    _, used = slot_parts.score_candidates({}, tmp_path, tmp_path / "pieces", [1, 2])
    assert used.startswith("shape only")
    said = capsys.readouterr().out
    assert "without the picture judge" in said and "Setup & Status" in said


def test_the_page_and_agents_see_which_judge_picked(tmp_path):
    from image_to_3dlab.slot_parts import candidates, runs

    folder = tmp_path / "sets" / "red" / "stick"
    candidates.write(folder / "pieces", {"torso": {"candidates": {"1": {"picture": str(folder / "a.png"), "score": 0.4,
                                                                        "shape": {"score": 0.4}}},
                                                   "auto": 1, "picked": 1}}, "shape only (llama.cpp is not installed)")
    (folder / "run_state.json").write_text(json.dumps({"waiting": "picks"}))
    record = {"log": str(folder / "run.log"), "folder": str(folder), "set": "red", "body": "stick", "started": 0}
    status = runs.status(tmp_path, record, running=False)
    assert status["judge"].startswith("shape only")
    later = {**record, "started": 4_000_000_000}   # a run started after that file was written
    assert runs.status(tmp_path, later, running=False)["judge"] is None


@needs_node
def test_the_pick_panel_says_the_judge_is_missing():
    url = (REPO / "viewer" / "studio" / "slot-parts.js").as_uri()
    program = (f"import * as m from {json.dumps(url)};\n"
               "console.log(JSON.stringify([m.judgeNote('shape only (llama.cpp is not installed)'),"
               " m.judgeNote('qwen3-vl-8b'), m.judgeNote(null)]));")
    out = json.loads(subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True, encoding="utf-8").stdout)
    assert "Setup & Status" in out[0] and out[1] == "" and out[2] == ""


def test_setup_does_not_offer_a_download_where_there_is_no_build(monkeypatch):
    import importlib

    from image_to_3dlab.style import models

    monkeypatch.setattr(models, "llama_build", lambda system=None, machine=None: None)
    import backend_catalog

    reloaded = importlib.reload(backend_catalog)
    try:
        assert reloaded.BY_ID["judge"].automated_setup is False
    finally:
        monkeypatch.undo()
        importlib.reload(backend_catalog)


def test_the_bootstrap_says_when_there_is_no_build(monkeypatch, capsys):
    import bootstrap_style

    from image_to_3dlab.style import models

    monkeypatch.setattr(models, "llama_build", lambda system=None, machine=None: None)
    monkeypatch.setattr(sys, "argv", ["bootstrap_style.py", "--only", "llama", "--yes"])
    assert bootstrap_style.main() == 1
    assert "no prebuilt release" in capsys.readouterr().out
