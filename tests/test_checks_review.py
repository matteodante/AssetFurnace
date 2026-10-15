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
    if "license.components" not in (*fails, *warns):  # a full run always has the licence check
        results.append({"check": "license.components", "status": "pass", "value": None, "threshold": None,
                        "message": "all clear"})
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


def test_a_batch_output_can_be_queued_before_any_check_ran(tmp_path, log):
    """Feature 1 queues batch outputs; the checks step may not have run yet."""
    glb = fx.clean(tmp_path / "row_07.glb")
    item = review.enqueue_for_review(glb, "row_07", source="batch:props-01", kind="prop", log=log, now=T[0])
    assert item["state"] == "to_review"
    assert item["status"] is None and item["summary"] == {"fail": [], "warn": []}
    assert item["source"] == "batch:props-01" and item["kind"] == "prop"
    assert item["asset"] == str(glb.resolve())
    [line] = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert line["asset_sha256"] == checks.Asset(glb=glb).sha256


def test_queueing_with_a_report_carries_its_status(tmp_path, log):
    glb = fx.clean(tmp_path / "row_08.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    report = checks.run_checks(checks.Asset(glb=glb), only=["mesh.holes"])
    path = checks.write_report(report)
    item = review.enqueue_for_review(glb, "row_08", source="batch:props-01", report=report,
                                     report_file=path, log=log)
    assert item["status"] == "pass" and item["report"] == str(path)


def test_a_report_for_another_file_is_refused(tmp_path, log):
    glb = fx.clean(tmp_path / "a.glb")
    other = checks.run_checks(checks.Asset(glb=fx.cube(tmp_path / "b.glb")), only=["mesh.holes"])
    with pytest.raises(ValueError, match="different file"):
        review.enqueue_for_review(glb, "a", report=other, log=log)


def test_a_decision_is_bound_to_the_version_the_reviewer_saw(log):
    """Codex review: a new version queued meanwhile must not receive the old one's decision."""
    review.enqueue(log, report_for("a" * 64), "r.json", now=T[0])
    review.enqueue(log, report_for("b" * 64), "r.json", now=T[1])
    with pytest.raises(review.ReviewConflict, match="newer version"):
        review.decide(log, "mob_0412", "approved", "", asset_sha256="a" * 64)
    assert review.decide(log, "mob_0412", "approved", "", asset_sha256="b" * 64)["state"] == "approved"


def test_only_a_version_waiting_for_review_can_be_decided(log):
    """Codex review: a sent-back version waits for a new one; an approved one is done."""
    review.enqueue(log, report_for("a" * 64), "r.json", now=T[0])
    review.decide(log, "mob_0412", "sent_back", "redo", now=T[1])
    with pytest.raises(review.ReviewConflict, match="new version"):
        review.decide(log, "mob_0412", "approved", "")
    review.enqueue(log, report_for("b" * 64), "r.json", now=T[2])
    review.decide(log, "mob_0412", "approved", "", now=T[3])
    with pytest.raises(review.ReviewConflict, match="already"):
        review.decide(log, "mob_0412", "sent_back", "changed my mind")


def test_approval_needs_checks_that_ran_and_a_licence_that_passed(log, tmp_path):
    """Codex review 2: nothing unchecked, and nothing whose licence check failed, is approved."""
    glb = fx.clean(tmp_path / "row.glb")
    review.enqueue_for_review(glb, "row", source="batch:x", log=log, now=T[0])
    with pytest.raises(review.ReviewConflict, match="checks"):
        review.decide(log, "row", "approved", "")
    review.enqueue(log, report_for("c" * 64, "fail", key="lic", fails=["license.components"]), "r.json", now=T[1])
    with pytest.raises(review.ReviewConflict, match="licence"):
        review.decide(log, "lic", "approved", "")
    # Sending either back is always possible.
    assert review.decide(log, "lic", "sent_back", "fix the provenance")["state"] == "sent_back"
    assert review.decide(log, "row", "sent_back", "run the checks")["state"] == "sent_back"


def test_approval_needs_a_licence_check_that_ran(log, tmp_path):
    """Codex review 3: a partial report without the licence check is not enough to approve."""
    glb = fx.clean(tmp_path / "part.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    partial = checks.run_checks(checks.Asset(glb=glb), only=["mesh.holes"])
    review.enqueue_for_review(glb, "part", report=partial, report_file="r.json", log=log, now=T[0])
    with pytest.raises(review.ReviewConflict, match="licence check has not run"):
        review.decide(log, "part", "approved", "")
    full = checks.run_checks(checks.Asset(glb=glb), only=["mesh.holes", "license.components"])
    review.enqueue_for_review(glb, "part", report=full, report_file="r.json", log=log, now=T[1])
    assert review.fold(review.load(log))["part"]["license_status"] == "pass"
    assert review.decide(log, "part", "approved", "", now=T[2])["state"] == "approved"


def _queued(tmp_path, log):
    glb = fx.clean(tmp_path / "row.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    report = checks.run_checks(checks.Asset(glb=glb), only=["mesh.holes", "license.components"])
    review.enqueue_for_review(glb, "row", report=report, report_file="r.json", log=log, now=T[0])
    return glb


def test_a_file_replaced_on_disk_after_queueing_cannot_be_decided(tmp_path, log):
    """Codex review 4: the decision must be about the bytes that were checked."""
    glb = _queued(tmp_path, log)
    fx.cube(glb)  # a batch writes its next result in place, without queueing it
    with pytest.raises(review.ReviewConflict, match="changed on disk"):
        review.decide(log, "row", "approved", "")
    with pytest.raises(review.ReviewConflict, match="changed on disk"):
        review.decide(log, "row", "sent_back", "redo")


def test_decisions_take_a_lock_other_processes_see(tmp_path, log):
    """Codex review 4: the studio and the CLI are separate processes."""
    _queued(tmp_path, log)
    lock = log.with_name(log.name + ".lock")
    lock.write_text("held by another process")
    with pytest.raises(review.ReviewConflict, match="busy"):
        review.decide(log, "row", "approved", "", lock_timeout=0.3)
    assert lock.exists()  # someone else's lock is not removed while fresh
    import os
    import time
    old = time.time() - 600
    os.utime(lock, (old, old))   # a lock left by a crashed process is stale
    assert review.decide(log, "row", "approved", "", lock_timeout=0.3)["state"] == "approved"
    assert not lock.exists()
