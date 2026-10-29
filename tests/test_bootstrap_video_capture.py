"""scripts/bootstrap_video_capture.py: what Animate from video fetches, and only with a yes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap_video_capture as bv

MISSING = {"ready": False, "download_gb": 29.7, "missing": ["leejet/MiniMax-H3-GGUF/x.gguf (11.4 GB)"]}


def test_the_prompt_names_sizes_the_licence_and_who_must_apply_first():
    text = bv.announcement(MISSING, dino_here=False)
    assert "29.7 GB" in text and "MiniMax H3 Community License" in text and "346 MB" in text
    assert "South Korea" in text and "Apache-2.0" in text


def test_nothing_is_fetched_without_a_yes_or_when_all_is_here(monkeypatch, capsys):
    fetched = []
    monkeypatch.setattr(bv.video, "download", lambda **k: fetched.append("h3"))
    monkeypatch.setattr(bv, "fetch_dino", lambda: fetched.append("dino"))
    monkeypatch.setattr(bv.video, "weights_plan", lambda: MISSING)
    monkeypatch.setattr(bv.video, "dino_cached", lambda: False)
    monkeypatch.setattr(sys, "stdin", None)
    assert bv.main([]) == 1 and fetched == []
    assert bv.main(["--yes"]) == 0 and fetched == ["h3", "dino"]
    monkeypatch.setattr(bv.video, "weights_plan", lambda: {**MISSING, "ready": True, "missing": []})
    monkeypatch.setattr(bv.video, "dino_cached", lambda: True)
    assert bv.main(["--check"]) == 0 and bv.main([]) == 0
    assert "Nothing to fetch" in capsys.readouterr().out
