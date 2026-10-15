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
    with pytest.raises(ValueError):  # Codex review 7: a setting with no "=" is a typo, not max=""
        check_asset.parse_overrides(["budget.faces.max"])


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
    sha = review.fold(review.load(log))["barrel_01"]["asset_sha256"]
    item, problem = review_queue.decide("barrel_01", "sent_back", "lid too flat", log, register, asset_sha256=sha)
    assert problem is None and item["state"] == "sent_back"
    [verdict] = mark_asset.load_register(register)
    assert verdict["verdict"] == "bad" and verdict["note"] == "lid too flat"
    assert verdict["tags"] == ["review-queue"]


def test_the_cli_refuses_a_send_back_without_a_note(tmp_path, capsys):
    glb = fx.clean(tmp_path / "barrel.glb")
    log = tmp_path / "log.jsonl"
    check_asset.main([str(glb), "--key", "barrel_01", "--log", str(log)])
    sha = review.fold(review.load(log))["barrel_01"]["asset_sha256"]
    capsys.readouterr()
    assert review_queue.main(["--send-back", "barrel_01", "--sha", sha, "--log", str(log)]) == 2
    assert "note" in capsys.readouterr().err
    assert review_queue.main(["--list", "--log", str(log)]) == 0
    assert "barrel_01" in capsys.readouterr().out


def test_a_non_positive_height_is_refused_at_the_command_line(tmp_path, capsys):
    with pytest.raises(SystemExit):
        check_asset.main([str(fx.clean(tmp_path / "a.glb")), "--height", "-1.8"])


def test_a_missing_glb_is_a_clear_error(tmp_path, capsys):
    """Codex review 5: no traceback for a path that is not there."""
    assert check_asset.main([str(tmp_path / "nope.glb"), "--no-queue"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_the_cli_needs_the_version_to_decide_and_lists_it(tmp_path, capsys):
    glb = fx.clean(tmp_path / "barrel.glb")
    log = tmp_path / "log.jsonl"
    check_asset.main([str(glb), "--key", "barrel_01", "--log", str(log)])
    capsys.readouterr()
    assert review_queue.main(["--list", "--log", str(log)]) == 0
    sha = review.fold(review.load(log))["barrel_01"]["asset_sha256"]
    assert sha[:12] in capsys.readouterr().out
    assert review_queue.main(["--send-back", "barrel_01", "--note", "x", "--log", str(log)]) == 2
    assert "--sha" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_a_non_finite_measurement_is_refused_at_the_command_line(tmp_path, value):
    """Codex review 6: `--height inf` printed a traceback."""
    with pytest.raises(SystemExit) as stopped:
        check_asset.main([str(fx.clean(tmp_path / "a.glb")), "--height", value])
    assert stopped.value.code == 2


def test_a_verdict_is_not_written_for_bytes_that_changed_after_the_decision(tmp_path, monkeypatch):
    """Codex review 6: the verdict must describe the version the decision was about."""
    glb = fx.clean(tmp_path / "barrel.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    log, register = tmp_path / "log.jsonl", tmp_path / "verdicts.jsonl"
    check_asset.main([str(glb), "--key", "barrel_01", "--log", str(log)])
    sha = review.fold(review.load(log))["barrel_01"]["asset_sha256"]
    original = review.decide

    def decide_then_overwrite(*args, **kwargs):
        item = original(*args, **kwargs)
        fx.cube(glb)  # a batch writes its next result in place right after the decision
        return item

    monkeypatch.setattr(review, "decide", decide_then_overwrite)
    item, problem = review_queue.decide("barrel_01", "sent_back", "lid", log, register, asset_sha256=sha)
    assert item["state"] == "sent_back"
    assert problem and "changed" in problem
    assert not register.exists()
