"""scripts/check_asset.py and scripts/review_queue.py: the checks and the queue from a terminal."""

from __future__ import annotations

import json

import _check_fixtures as fx
import check_asset
import mark_asset
import pytest
import review_queue

from image_to_3dlab.checks import review


def test_overrides_read_the_check_name_up_to_the_last_dot():
    assert check_asset.parse_overrides(["budget.faces.max=20000", "silhouette.warn_only=false",
                                        "license.components.conditional=warn"]) == {
        "budget.faces": {"max": 20000}, "silhouette": {"warn_only": False},
        "license.components": {"conditional": "warn"}}
    with pytest.raises(ValueError):
        check_asset.parse_overrides(["nonsense"])


def test_a_failing_asset_exits_1_and_lands_in_the_queue(tmp_path, capsys):
    glb = fx.holed(tmp_path / "barrel.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    log = tmp_path / "log.jsonl"
    assert check_asset.main([str(glb), "--key", "barrel_01", "--log", str(log)]) == 1
    out = capsys.readouterr().out
    assert "mesh.holes" in out and "FAIL" in out
    assert (tmp_path / "barrel.checks.json").is_file()
    assert review.queue(review.load(log))[0]["key"] == "barrel_01"


def test_a_clean_asset_exits_0_and_json_is_the_report(tmp_path, capsys):
    glb = fx.clean(tmp_path / "barrel.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    assert check_asset.main([str(glb), "--no-queue", "--json", "--height", "1.8"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] in ("pass", "warn")
    assert any(r["check"] == "measure.height" for r in report["results"])


def test_decisions_reach_the_review_log_and_the_verdicts(tmp_path, capsys):
    glb = fx.clean(tmp_path / "barrel.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    log, register = tmp_path / "log.jsonl", tmp_path / "verdicts.jsonl"
    check_asset.main([str(glb), "--key", "barrel_01", "--log", str(log)])
    item, problem = review_queue.decide("barrel_01", "sent_back", "lid too flat", log, register)
    assert problem is None and item["state"] == "sent_back"
    [verdict] = mark_asset.load_register(register)
    assert verdict["verdict"] == "bad" and verdict["note"] == "lid too flat"
    assert verdict["tags"] == ["review-queue"]


def test_the_cli_refuses_a_send_back_without_a_note(tmp_path, capsys):
    glb = fx.clean(tmp_path / "barrel.glb")
    log = tmp_path / "log.jsonl"
    check_asset.main([str(glb), "--key", "barrel_01", "--log", str(log)])
    assert review_queue.main(["--send-back", "barrel_01", "--log", str(log)]) == 2
    assert "note" in capsys.readouterr().err
    assert review_queue.main(["--list", "--log", str(log)]) == 0
    assert "barrel_01" in capsys.readouterr().out
