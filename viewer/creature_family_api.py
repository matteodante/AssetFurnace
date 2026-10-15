"""The studio's side of creature families: the choice sent with a rig, the clip step that
follows SkinTokens, provenance for the clips, and what the Library shows about a rig.

The work itself is in `image_to_3dlab/creature_family.py` (which family, which bones) and
`scripts/creature_clips.py` (the test clips). This module only wires them into the rig
job, so `autorig_api`, `generate_api` and `assets_api` each need a line, not a feature.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import creature_family  # noqa: E402

CLIPS_SCRIPT = REPO / "scripts" / "creature_clips.py"
AUTO = "auto"
CHOICES = (AUTO, *creature_family.FAMILIES)
PROCEDURAL_LICENSE = {
    "name": "MIT",
    "note": "Procedural test clip made by AssetFurnace (scripts/creature_motion.py); no motion data or model.",
}


def parse_family(value: Any) -> str:
    """The family a person chose for a rig, or "auto". Anything else is refused."""
    family = str(value or AUTO).strip().lower()
    if family not in CHOICES:
        raise ValueError(f"unknown family {family!r}; choose one of {', '.join(CHOICES)}")
    return family


def clip_command(python: Path, rigged: Path, family: str = AUTO) -> tuple[list[str], Path]:
    """The step after rigging: recognise the family, write the test clips."""
    return [str(python), str(CLIPS_SCRIPT), "--rig", str(rigged), "--family", parse_family(family)], REPO


def clip_files(rigged: Path) -> list[Path]:
    """The test clips the clip step wrote for this rig, as its family file lists them."""
    found = creature_family.read_family(rigged) or {}
    stem = rigged.stem.removesuffix("_rigged")
    return [p for p in (rigged.with_name(f"{stem}_{name}.glb") for name in found.get("clips") or []) if p.is_file()]


def record_clips(rigged: Path, rig_record: dict[str, Any]) -> list[Path]:
    """A provenance sidecar per test clip: the rig's, plus how the motion was made."""
    found = creature_family.read_family(rigged) or {}
    written = []
    for clip in clip_files(rigged):
        record = dict(rig_record)
        record.update({
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": "viewer Animate tab: scripts/creature_clips.py",
            "motion": {"clip": clip.stem.removeprefix(rigged.stem.removesuffix("_rigged") + "_"),
                       "family": found.get("family"), "made_with": "procedural test clip",
                       "license": PROCEDURAL_LICENSE},
        })
        path = clip.with_suffix(".provenance.json")
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        written.append(path)
    return written


def facts(rigged: Path | None) -> dict[str, Any]:
    """What the Library and the rig job report about a rig's family."""
    found = creature_family.read_family(rigged) if rigged else None
    if not found:
        return {"family": None, "family_label": None, "family_chosen": None, "clips_error": None}
    return {"family": found["family"], "family_label": creature_family.LABELS[found["family"]],
            "family_chosen": bool(found.get("chosen")), "clips_error": found.get("clips_error")}


def done_message(fit: dict[str, Any], rigged: Path) -> str | None:
    """The rig job's closing line when the family has something to say, else None."""
    found = creature_family.read_family(rigged)
    if not found or found["family"] == "humanoid":
        return None
    label = creature_family.LABELS[found["family"]]
    clips = found.get("clips") or []
    moves = " Preset moves fit it too." if fit.get("humanoid") else ""
    if clips:
        return f"Rigged as {label.lower()}, with test clips: {', '.join(clips)}.{moves}"
    return f"Rigged as {label.lower()}. No test clips for this family yet.{moves}"
