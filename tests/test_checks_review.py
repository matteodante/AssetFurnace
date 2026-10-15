"""The review queue: an append-only log where a sent-back asset keeps its note.

Acceptance for feature 7: "a send-back returns to the queue with its note".
"""

from __future__ import annotations

import json

import _check_fixtures as fx
import pytest

from image_to_3dlab import checks
from image_to_3dlab.checks import review

T = [f"2026-10-15T10:{m:02d}:00+00:00" for m in range(60)]


def report_for(sha, status="pass", key="mob_0412", fails=(), warns=()):
    results = [{"check": n, "status": "fail", "value": 1, "threshold": None, "message": "bad"} for n in fails]
    results += [{"check": n, "status": "warn", "value": 1, "threshold": None, "message": "meh"} for n in warns]
    return {"asset": f"output/{key}.glb", "asset_sha256": sha, "asset_key": key, "status": status,
            "results": results, "skipped": [], "profile": "prop"}


@pytest.fixture
def log(tmp_path):
    return tmp_path / "review-log.jsonl"


def test_a_checked_asset_enters_the_queue(log):
    review.enqueue(log, report_for("a" * 64, "warn", warns=["silhouette"]), "output/mob_0412.checks.json", now=T[0])
    [item] = review.queue(review.load(log))
    assert item["key"] == "mob_0412"
    assert item["state"] == "to_review"
    assert item["status"] == "warn"
    assert item["summary"] == {"fail": [], "warn": ["silhouette"]}
    assert item["report"] == "output/mob_0412.checks.json"


def test_a_send_back_needs_a_note(log):
    review.enqueue(log, report_for("a" * 64), "r.json", now=T[0])
    with pytest.raises(ValueError, match="note"):
        review.decide(log, "mob_0412", "sent_back", "   ")
    with pytest.raises(ValueError, match="decision"):
        review.decide(log, "mob_0412", "maybe", "x")
    with pytest.raises(KeyError):
        review.decide(log, "nobody", "approved", "")


def test_a_send_back_returns_to_the_queue_with_its_note(log):
    """The acceptance path: queued -> sent back -> still queued with the note -> new version
    -> to review again, carrying the old note -> approved -> out of the queue."""
    review.enqueue(log, report_for("a" * 64), "r.json", now=T[0])
    review.decide(log, "mob_0412", "sent_back", "horns too short, see reference", now=T[1])

    [item] = review.queue(review.load(log))
    assert item["state"] == "sent_back"
    assert item["notes"][-1]["note"] == "horns too short, see reference"

    # Re-running the checks on the same file does not hide the send-back.
    review.enqueue(log, report_for("a" * 64), "r.json", now=T[2])
    assert review.queue(review.load(log))[0]["state"] == "sent_back"

    # A new version comes back to review, with the note it answers.
    review.enqueue(log, report_for("b" * 64), "r.json", now=T[3])
    [item] = review.queue(review.load(log))
    assert item["state"] == "to_review"
    assert item["asset_sha256"] == "b" * 64
    assert [n["note"] for n in item["notes"]] == ["horns too short, see reference"]

    review.decide(log, "mob_0412", "approved", "", now=T[4])
    assert review.queue(review.load(log)) == []
    final = review.fold(review.load(log))["mob_0412"]
    assert final["state"] == "approved"
    assert [e["event"] for e in final["history"]] == ["queued", "sent_back", "queued", "queued", "approved"]


def test_the_log_is_append_only_json_lines(log):
    review.enqueue(log, report_for("a" * 64), "r.json", now=T[0])
    review.decide(log, "mob_0412", "sent_back", "too shiny", now=T[1])
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["queued", "sent_back"]
    assert json.loads(lines[1])["asset_sha256"] == "a" * 64


def test_the_queue_lists_to_review_first_then_sent_back(log):
    review.enqueue(log, report_for("a" * 64, key="one"), "r.json", now=T[0])
    review.enqueue(log, report_for("b" * 64, key="two"), "r.json", now=T[1])
    review.decide(log, "one", "sent_back", "redo", now=T[2])
    review.enqueue(log, report_for("c" * 64, key="three"), "r.json", now=T[3])
    assert [i["key"] for i in review.queue(review.load(log))] == ["three", "two", "one"]


def test_a_bad_key_is_refused(log):
    with pytest.raises(ValueError, match="key"):
        review.enqueue(log, report_for("a" * 64, key="../etc"), "r.json")


def test_a_damaged_line_does_not_lose_the_rest(log):
    review.enqueue(log, report_for("a" * 64), "r.json", now=T[0])
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    review.decide(log, "mob_0412", "approved", "fine", now=T[1])
    assert review.fold(review.load(log))["mob_0412"]["state"] == "approved"


def test_check_and_enqueue_writes_the_report_and_queues_it(tmp_path, log):
    glb = fx.clean(tmp_path / "barrel.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    report, path = review.check_and_enqueue(checks.Asset(glb=glb), log=log, profile="prop")
    assert path == tmp_path / "barrel.checks.json" and path.is_file()
    [item] = review.queue(review.load(log))
    assert item["key"] == "barrel"          # no key given: the file's name
    assert item["status"] == report["status"]
