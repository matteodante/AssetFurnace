"""The studio's side of creature families: the choice sent with a rig, the clip step that
follows SkinTokens, provenance for the clips, and what the Library shows about a rig.

The work itself is in `image_to_3dlab/creature_family.py` (which family, which bones) and
`scripts/creature_clips.py` (the test clips). This module only wires them into the rig
job, so `autorig_api`, `generate_api` and `assets_api` each need a line, not a feature.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import creature_family  # noqa: E402
# shared with the batch steps and scripts/autorig.py
from image_to_3dlab.rigging import clip_command, clip_files, parse_family, record_clips  # noqa: E402,F401

CLIPS_SCRIPT = REPO / "scripts" / "creature_clips.py"
AUTO = "auto"
CHOICES = (AUTO, *creature_family.FAMILIES)
PROCEDURAL_LICENSE = creature_family.PROCEDURAL_LICENSE


def facts(rigged: Path | None) -> dict[str, Any]:
    """What the Library and the rig job report about a rig's family."""
    found = creature_family.read_family(rigged) if rigged else None
    if not found:
        return {"family": None, "family_label": None, "family_chosen": None, "clips_error": None,
                "library_clips": []}
    return {"family": found["family"], "family_label": creature_family.LABELS[found["family"]],
            "family_chosen": bool(found.get("chosen")), "clips_error": found.get("clips_error"),
            # the clips carried over from the CC0 library, as opposed to procedural tests
            "library_clips": sorted(found.get("clip_sources") or {})}


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
