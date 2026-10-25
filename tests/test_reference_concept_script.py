"""The command line of Draw from a reference: the same request the studio sends, and one seed per attempt."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reference_concept.py"
spec = importlib.util.spec_from_file_location("reference_concept", SCRIPT)
script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = script
spec.loader.exec_module(script)


def test_make_arguments_become_the_studio_request():
    args = script.parser().parse_args(["make", "other_view", "--view", "back", "--depth",
                                       "--shape", "images/front.png", "--shape", "concept/maps/a/back_depth_black.png",
                                       "--style-ref", "images/style.png"])
    assert script.request(args, 5) == {"mode": "other_view", "view": "back", "depth": True, "seed": 5,
                                       "shape": ["images/front.png", "concept/maps/a/back_depth_black.png"],
                                       "style_refs": ["images/style.png"]}


def test_attempts_take_consecutive_seeds_and_stay_within_four():
    assert script.seeds(10, 3) == [10, 11, 12]
    assert len(script.seeds(None, 2)) == 2
    with pytest.raises(SystemExit):
        script.seeds(1, 5)


def test_prepare_prints_the_map(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script.concept_api, "prepare", lambda output, source, kind, view: {"path": f"{source}:{kind}:{view}"})
    assert script.main(["--output", str(tmp_path), "prepare", "wolf.glb", "depth", "--view", "side"]) == 0
    assert '"wolf.glb:depth:side"' in capsys.readouterr().out
