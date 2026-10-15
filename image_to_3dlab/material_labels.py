"""The user's material labels: the human judgements every material check is measured against.

Three judgements per asset version (design v2, section 6):

- ``baked_light``: is light painted into the albedo? (yes / no)
- ``materials_look``: do the materials look right under the studio's light? ("right" / "wrong")
- ``points``: points clicked on the model, each with the material it shows: a texel
  (glTF ``uv``, v downward), the glTF material index and a word from ``MATERIALS``.
  Zones are later scored against these.

The record is ``output/material-labels.jsonl``, one JSON line per save, never rewritten
(like the review log): a judgement is an observation made at a moment. Each line names the
asset version (sha256); a new version starts with no labels. Within a version, each
judgement is the latest one given.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

MATERIALS = ("cloth", "leather", "metal", "skin", "fur", "hair", "feathers", "scales", "wood", "stone",
             "bone", "horn", "plant", "gem", "glass", "plastic", "other")
LOOKS = ("right", "wrong")
KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
FIELDS = ("baked_light", "materials_look", "points", "note")
_WRITE = threading.RLock()


def _now(now: str | None) -> str:
    return now or datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 - 3.10


def _point(point) -> dict:
    if not isinstance(point, dict):
        raise TypeError("a point is an object with uv, material and label")
    uv = point.get("uv")
    if (not isinstance(uv, list) or len(uv) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                                         or not 0.0 <= v <= 1.0 for v in uv)):
        raise ValueError(f"a point's uv must be two numbers in 0..1, got {uv!r}")
    label = point.get("label")
    if label not in MATERIALS:
        raise ValueError(f"a point's label must be one of {', '.join(MATERIALS)}, got {label!r}")
    material = point.get("material")
    if material is not None and (isinstance(material, bool) or not isinstance(material, int) or material < 0):
        raise ValueError("a point's material must be a glTF material index or null")
    return {"uv": [round(float(v), 6) for v in uv], "material": material, "label": label}


def validate(fields: dict) -> dict:
    """The judgements in `fields` that were given, checked; refuses anything else."""
    out = {}
    if fields.get("baked_light") is not None:
        if not isinstance(fields["baked_light"], bool):
            raise ValueError("baked_light must be true or false")
        out["baked_light"] = fields["baked_light"]
    if fields.get("materials_look") is not None:
        if fields["materials_look"] not in LOOKS:
            raise ValueError(f"materials_look must be one of {LOOKS}")
        out["materials_look"] = fields["materials_look"]
    if fields.get("points") is not None:
        if not isinstance(fields["points"], list) or len(fields["points"]) > 200:
            raise ValueError("points must be a list of at most 200")
        out["points"] = [_point(p) for p in fields["points"]]
    if fields.get("note") is not None:
        if not isinstance(fields["note"], str) or len(fields["note"]) > 2000:
            raise ValueError("note must be text of at most 2000 characters")
        out["note"] = fields["note"].strip()
    return out


def load(log: Path) -> list[dict]:
    """Every saved label event, oldest first. A damaged line is skipped, not fatal."""
    if not Path(log).is_file():
        return []
    events = []
    for line in Path(log).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("key") and event.get("asset_sha256"):
            events.append(event)
    return events


def fold(events: list[dict]) -> dict[str, dict]:
    """Each asset's labels for its latest version: every judgement as last given."""
    state: dict[str, dict] = {}
    for event in events:
        key = event["key"]
        current = state.get(key)
        if current is None or current["asset_sha256"] != event["asset_sha256"]:
            current = state[key] = {"key": key, "asset": event.get("asset"), "asset_sha256": event["asset_sha256"],
                                    "baked_light": None, "materials_look": None, "points": [], "note": ""}
        for field in FIELDS:
            if field in event:
                current[field] = event[field]
        current["updated_at"] = event.get("at")
    return state


def record(log: Path, key: str, asset: str, asset_sha256: str, now: str | None = None, **fields) -> dict:
    """Save the judgements given for one asset version; return its labels after the save."""
    if not isinstance(key, str) or not KEY.fullmatch(key) or ".." in key:
        raise ValueError(f"not a usable asset key: {key!r}")
    if not isinstance(asset_sha256, str) or not SHA.fullmatch(asset_sha256):
        raise ValueError("asset_sha256 must be the 64-character hash of the version labelled")
    given = validate(fields)
    if not given:
        raise ValueError("nothing to save: give baked_light, materials_look, points or a note")
    event = {"event": "labels", "key": key, "asset": str(asset), "asset_sha256": asset_sha256, "at": _now(now),
             **given}
    log = Path(log)
    with _WRITE:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return fold(load(log))[key]


def is_complete(labels: dict) -> bool:
    """All three judgements given: light, look, and at least one material point."""
    return (labels.get("baked_light") is not None and labels.get("materials_look") is not None
            and bool(labels.get("points")))


def dataset(events: list[dict]) -> list[dict]:
    """The complete labels, one per asset at its latest version: what a check is scored on."""
    return [labels for labels in fold(events).values() if is_complete(labels)]


def progress(events: list[dict], queued: dict[str, str]) -> dict:
    """How far labelling a set has got: `queued` maps each key to the version in the queue.

    Labels of an older version count for nothing: the version in the queue is the one the
    export will include.
    """
    state = fold(events)
    current = [state[k] for k, sha in queued.items() if k in state and state[k]["asset_sha256"] == sha]
    complete = sum(1 for labels in current if is_complete(labels))
    started = len(current) - complete
    return {"total": len(queued), "complete": complete, "started": started,
            "missing": len(queued) - complete - started}
