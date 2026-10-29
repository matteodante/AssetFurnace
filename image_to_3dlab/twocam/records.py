"""Provenance for a two-camera capture: what the motion was read from, and under which terms.

Same pattern as the Animate tab's records (`viewer/autorig_api.py`): the model's own
licence carries over from its sidecar, and the motion's sources are listed beside it,
never mixed in. A capture's video is usually made by a hosted video model (AnyMove used
MiniMax H3 on fal), whose terms are not ours to guess, so they are recorded as declared.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .reader import BACKBONE_ID, BACKBONE_LICENSE

METHOD = {
    "name": "two-camera motion capture",
    "method_by": "@CanItCode",
    "code": "image_to_3dlab/twocam, ported from Patch-gg/anymove",
    "code_license": "Apache-2.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(path: Path) -> dict[str, str]:
    return {"name": Path(path).name, "sha256": sha256_file(path)}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def capture_record(video: Path, reader: Path, rig_info: Path, made_with: str | None = None,
                   video_license: str | None = None) -> dict[str, Any]:
    """Written beside the detections, and copied into the motion the fit makes from them."""
    return {
        "schema_version": 1,
        "created_at": now(),
        "tool": "scripts/twocam_mocap.py read",
        "method": METHOD,
        "video": {**_file(video), "made_with": made_with or "not declared",
                  "license": video_license or "not declared: check the video model's terms"},
        "reader": {**_file(reader), "backbone": BACKBONE_ID, "backbone_license": BACKBONE_LICENSE},
        "rig_info": _file(rig_info),
    }


def source_record_for(model: Path) -> dict[str, Any] | None:
    """The licence record kept beside a model file, if it names a licence."""
    for sidecar in (model.with_suffix(".provenance.json"), model.with_suffix(".json")):
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("license"):
            return record
    return None


def animation_record(blend: Path, capture: dict[str, Any] | None,
                     source: dict[str, Any] | None) -> dict[str, Any]:
    """Provenance for the animated GLB: the model's licence, then the motion's sources."""
    return {
        "schema_version": 1,
        "created_at": now(),
        "tool": "scripts/twocam_mocap.py apply: scripts/blender_twocam_apply.py",
        "input": _file(blend),
        "license": (source or {}).get("license"),
        "classification": ((source or {}).get("output") or {}).get("classification")
        or (source or {}).get("classification"),
        "motion": {"made_with": METHOD["name"], "method": METHOD,
                   "capture": capture or "not recorded: the motion came without a capture record"},
        "source_record": source,
    }


def sidecar_path(target: Path) -> Path:
    """`walk.glb` -> `walk.provenance.json`, the repo's sidecar convention."""
    return Path(target).with_suffix(".provenance.json")


def write_sidecar(target: Path, record: dict[str, Any]) -> Path:
    path = sidecar_path(target)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
