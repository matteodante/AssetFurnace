"""The review queue: where a person approves an asset or sends it back with a note.

The record is `output/review-log.jsonl`, one JSON line per event, never rewritten (the
same rule as `scripts/mark_asset.py`: a judgement is an observation made at a moment).
Events are keyed by the asset's key (the batch row's id, or the studio's asset id) and
carry the GLB's sha256, so a new version of the same asset is told apart from the old.

| event       | when                               |
|-------------|------------------------------------|
| `queued`    | the checks ran on a version        |
| `approved`  | the person approves it             |
| `sent_back` | the person sends it back, with a note (required) |

An asset's state is decided by its events in order. A sent-back asset stays in the queue,
note in view, until a new version (a different sha256) is checked: that version comes back
to review carrying every earlier note. Checking the same file again changes nothing.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from image_to_3dlab.checks.asset import REPO

LOG = REPO / "output" / "review-log.jsonl"
DECISIONS = ("approved", "sent_back")
KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_WRITE = threading.RLock()


class ReviewConflict(ValueError):
    """The decision does not fit the asset's current state: another version, or already decided."""


def _now(now: str | None) -> str:
    return now or datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 - 3.10


def safe_key(key: str) -> str:
    if not isinstance(key, str) or not KEY.fullmatch(key) or ".." in key:
        raise ValueError(f"not a usable asset key: {key!r} (letters, digits, _ . -)")
    return key


def key_for(glb: Path) -> str:
    """The key used when none is given: the file's name, made safe."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(glb).stem).strip("-.")[:120] or "asset"


def load(log: Path = LOG) -> list[dict]:
    """Every event, oldest first. A damaged line is skipped, not fatal."""
    if not Path(log).is_file():
        return []
    events = []
    for line in Path(log).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("event") and event.get("key"):
            events.append(event)
    return events


def _append(log: Path, event: dict) -> None:
    log = Path(log)
    with _WRITE:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def licence_status(report: dict) -> str | None:
    """How `license.components` came out in this report, or None when it did not run."""
    found = next((r for r in report.get("results", []) if r.get("check") == "license.components"), None)
    return found["status"] if found else None


def licence_digest(report: dict) -> str | None:
    """A short fingerprint of the whole licence result (names and conditions), or None.

    Two results can share a status (MIT and DINOv3 both pass) and still differ in what the
    game must credit; a decided version is reviewed again when this changes.
    """
    import hashlib

    found = next((r for r in report.get("results", []) if r.get("check") == "license.components"), None)
    if found is None:
        return None
    return hashlib.sha256(json.dumps(found.get("value"), sort_keys=True).encode()).hexdigest()[:16]


def summary(report: dict) -> dict[str, list[str]]:
    return {status: [r["check"] for r in report.get("results", []) if r["status"] == status]
            for status in ("fail", "warn")}


def fold(events: list[dict]) -> dict[str, dict]:
    """Every asset's current state, from its events in order."""
    items: dict[str, dict] = {}
    for event in events:
        key = event["key"]
        item = items.get(key)
        if event["event"] == "queued":
            if item is None:
                item = items[key] = {"key": key, "notes": [], "history": []}
            # The same file checked again keeps its decision, unless its licence result
            # changed (new provenance beside the same GLB): then it is reviewed again.
            same_version = (item.get("asset_sha256") == event.get("asset_sha256")
                            and item.get("license_status") == event.get("license_status")
                            and item.get("license_digest") == event.get("license_digest"))
            if not (same_version and item.get("state") in DECISIONS):
                item["state"] = "to_review"
            for field in ("asset", "asset_sha256", "report", "status", "summary", "profile", "license_status",
                          "license_digest", "source", "kind"):
                item[field] = event.get(field)
        elif event["event"] in DECISIONS:
            if item is None or item.get("asset_sha256") != event.get("asset_sha256"):
                continue  # a decision on a version that is no longer the current one
            item["state"] = event["event"]
            if event.get("note"):
                item["notes"].append({"at": event.get("at"), "decision": event["event"],
                                      "note": event["note"], "asset_sha256": event.get("asset_sha256")})
        else:
            continue
        item["updated_at"] = event.get("at")
        item["history"].append({k: event.get(k) for k in ("event", "at", "asset_sha256", "note", "status")})
    return items


def overview(log: Path = LOG, recent: int = 10) -> dict[str, list[dict]]:
    """What waits for a person, and the latest approvals: what the studio's review sheet
    and the agents' `review_queue` both show."""
    events = load(log)
    approved = [i for i in fold(events).values() if i["state"] == "approved"]
    approved.sort(key=lambda i: i.get("updated_at") or "", reverse=True)
    return {"items": queue(events), "approved": approved[:recent]}


def item_with_report(log: Path, key: str) -> dict:
    """One asset's state, notes and history, with its checks report read in (None when it
    has none or the file is gone). KeyError for a key the log does not have."""
    item = fold(load(log)).get(safe_key(key))
    if item is None:
        raise KeyError(key)
    report = None
    if item.get("report") and Path(item["report"]).is_file():
        report = json.loads(Path(item["report"]).read_text(encoding="utf-8"))
    return {**item, "report": report, "report_file": item.get("report")}


def queue(events: list[dict]) -> list[dict]:
    """What waits for a person: to review first, then sent back; newest first in each."""
    items = [i for i in fold(events).values() if i["state"] in ("to_review", "sent_back")]
    items.sort(key=lambda i: i.get("updated_at") or "", reverse=True)
    items.sort(key=lambda i: 0 if i["state"] == "to_review" else 1)
    return items


def enqueue(log: Path, report: dict, report_file: str | Path, key: str | None = None,
            now: str | None = None, source: str | None = None, kind: str | None = None,
            lock_timeout: float = 10.0) -> dict:
    """Record that the checks ran on a version; return the asset's new state.

    Takes the same lock as `decide`, so a new version cannot slip in between a decision's
    read and its write in another process.
    """
    key = safe_key(key or report.get("asset_key") or key_for(Path(report["asset"])))
    with _WRITE, _file_lock(log, timeout=lock_timeout):
        _append(log, {"event": "queued", "key": key, "at": _now(now), "asset": report["asset"],
                      "asset_sha256": report["asset_sha256"], "report": str(report_file),
                      "status": report["status"], "summary": summary(report), "profile": report.get("profile"),
                      "license_status": licence_status(report), "license_digest": licence_digest(report),
                      "source": source, "kind": kind})
        return fold(load(log))[key]


def enqueue_for_review(glb: str | Path, key: str, *, source: str | None = None, kind: str | None = None,
                       report: dict | None = None, report_file: str | Path | None = None,
                       log: Path = LOG, now: str | None = None, lock_timeout: float = 10.0) -> dict:
    """Put a GLB in the review queue, with or without a checks report. Stable: feature 1 calls it.

    `key` is the asset's id (a batch row's id); `source` says who queued it ("batch:<name>"),
    `kind` what it is (prop, creature, equipment_part). Without a report the item's status
    is None: the checks have not run. A report must be for this very file (same sha256).
    """
    from image_to_3dlab.provenance import sha256_file

    glb = Path(glb).resolve()
    sha = sha256_file(glb)
    if report is not None:
        if report.get("asset_sha256") != sha:
            raise ValueError(f"the report is for a different file than {glb.name}")
        return enqueue(log, {**report, "asset": str(glb)}, report_file or "", key, now, source, kind, lock_timeout)
    empty = {"asset": str(glb), "asset_sha256": sha, "status": None, "results": [], "profile": None}
    return enqueue(log, empty, "", key, now, source, kind, lock_timeout)


@contextmanager
def _file_lock(log: Path, timeout: float = 10.0, stale: float = 120.0):
    """An exclusive lock beside the log that other processes see (the studio and the CLI).

    Taken by creating `<log>.lock` exclusively; a lock older than `stale` seconds is left
    by a process that died and is broken. Waiting longer than `timeout` is a conflict.
    """
    lock = Path(log).with_name(Path(log).name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise ReviewConflict("the review log is busy: another decision is being written; try again") from None
            time.sleep(0.05)
            continue
        os.write(handle, str(os.getpid()).encode())
        os.close(handle)
        break
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def approval_blocker(item: dict) -> str | None:
    """Why this version cannot be approved, or None. Sending it back is always possible.

    The licence is the guardrail: an asset whose checks never ran, or whose licence check
    failed, is not approved by a click.
    """
    if item.get("status") is None:
        return "its checks have not run yet"
    licence = item.get("license_status")
    if licence == "fail":
        return "its licence check failed"
    if licence not in ("pass", "warn"):
        return "its licence check has not run (run every check, not a part of them)"
    return None


def decide(log: Path, key: str, decision: str, note: str = "", now: str | None = None,
           asset_sha256: str | None = None, lock_timeout: float = 10.0) -> dict:
    """Approve the version waiting for review, or send it back with a note; return the new state.

    `asset_sha256` is the version the reviewer looked at, and is required: if a newer one was
    queued in the meantime, the decision is refused rather than recorded against a file nobody saw. Only
    a version waiting for review can be decided; a sent-back one waits for its replacement.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    note = (note or "").strip()
    if decision == "sent_back" and not note:
        raise ValueError("sending an asset back needs a note saying what to change")
    # Read, judge and write as one step, across threads and processes, so two reviewers
    # cannot both decide the same version.
    with _WRITE, _file_lock(log, timeout=lock_timeout):
        current = fold(load(log)).get(safe_key(key))
        if current is None:
            raise KeyError(f"{key} is not in the review log")
        if not asset_sha256:
            raise ValueError(f"name the version you looked at (asset_sha256; --sha on the command line); "
                             f"{key} is now at {current.get('asset_sha256')}")
        if asset_sha256 != current.get("asset_sha256"):
            raise ReviewConflict(f"{key} has a newer version in the queue; reload it before deciding")
        on_disk = Path(current.get("asset") or "")
        if on_disk.is_file():
            from image_to_3dlab.provenance import sha256_file

            if sha256_file(on_disk) != current.get("asset_sha256"):
                raise ReviewConflict(f"{key} changed on disk since it was checked; check it again before deciding")
        if current["state"] == "sent_back":
            raise ReviewConflict(f"{key} was sent back; it waits for a new version to be checked")
        if current["state"] != "to_review":
            raise ReviewConflict(f"{key} is already {current['state'].replace('_', ' ')}")
        if decision == "approved":
            blocker = approval_blocker(current)
            if blocker:
                raise ReviewConflict(f"{key} cannot be approved: {blocker}")
        _append(log, {"event": decision, "key": key, "at": _now(now), "note": note,
                      "asset": current.get("asset"), "asset_sha256": current.get("asset_sha256")})
        # Read back under the lock: a version queued right after must not be handed back as decided.
        return fold(load(log))[key]


def check_and_enqueue(asset, log: Path = LOG, profile: str = "prop", overrides: dict | None = None,
                      key: str | None = None) -> tuple[dict, Path]:
    """Run the checks, write `<name>.checks.json` beside the GLB and put it in the queue.

    The one call a batch step needs: `(report, report_path)` come back for its own record.
    """
    from image_to_3dlab.checks.registry import run_checks, write_report

    report = run_checks(asset, profile=profile, overrides=overrides)
    report["asset_key"] = safe_key(key or asset.key or key_for(asset.glb))
    path = write_report(report)
    enqueue(log, report, path)
    return report, path
