"""scripts/bootstrap_unirig.py: what it installs, in which order, and when it refuses."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap_unirig as bu


def test_torch_comes_first_and_the_geometric_wheels_match_it_exactly():
    torch, reqs, pyg, spconv = bu.pip_commands("uv", Path("py"), Path("reqs.txt"))
    assert torch[-2:] == ["--index-url", bu.TORCH_INDEX] and "torch==2.11.0" in torch
    assert reqs[-3:] == ["-r", "reqs.txt", "numpy==1.26.4"]
    # torch_scatter/torch_cluster load only against the torch they were built for
    assert "torch-2.11.0+cu128" in pyg[pyg.index("-f") + 1] and "--no-index" in pyg
    assert spconv[-1] == bu.SPCONV


def test_flash_attn_is_left_out_of_upstreams_requirements():
    kept = bu.requirements_without_flash_attn("lightning\nflash_attn\nFlash-Attn==2.5\ntransformers\n")
    assert kept.split() == ["lightning", "transformers"]


@pytest.mark.parametrize("nvidia, driver, word", [(False, None, "NVIDIA"), (True, (12, 4), "12.4"), (True, None, "no CUDA")])
def test_a_machine_that_cannot_run_it_is_told_before_anything_is_fetched(nvidia, driver, word):
    with pytest.raises(SystemExit, match=word):
        bu.check_machine(nvidia, driver)
    bu.check_machine(True, (12, 8))      # the build it installs


def test_without_a_yes_and_nobody_to_ask_nothing_is_downloaded(monkeypatch, capsys):
    monkeypatch.setattr(bu, "check_machine", lambda *a: None)
    monkeypatch.setattr(sys, "stdin", None)
    assert bu.main([]) == 1
    assert "Refusing to download" in capsys.readouterr().out


def test_the_announcement_names_the_licence_size_and_pinned_revisions():
    text = bu.announcement()
    assert "MIT" in text and "1.4 GB" in text and bu.COMMIT[:7] in text and bu.HF_REVISION[:7] in text
