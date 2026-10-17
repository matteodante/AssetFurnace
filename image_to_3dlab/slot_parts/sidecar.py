"""`<set>.parts.json`: what a set of pieces was made from, how, and under which licences.

One record per set and body: the mode, the body (name and the SHA-256 of its picture and
rig), the concept, the chosen slots with their route, each piece's picture, seed and files,
the fit and the check results, and the licence of every model that touched it. It travels
with the files, like every other `.provenance.json` in AssetFurnace.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
MODES = ("dressed_split", "body_and_equipment")  # modes 2 and 3; mode 1 needs no record of its own


def _sha(path) -> str | None:
    from image_to_3dlab.provenance import sha256_file

    path = Path(path)
    return sha256_file(path) if path.is_file() else None


def models(matte: str | None = None) -> list[dict]:
    """The models every set goes through, with their licences from `provenance.LICENSES`."""
    from image_to_3dlab.provenance import LICENSES, MATTE_LICENSES

    out = []
    for key, role in (("qwen-image-2.1", "concept, piece pictures, weapon and shield pictures"),
                      ("pixal3d", "each piece in 3D")):
        profile = LICENSES[key]
        out.append({"id": key, "role": role, "license": profile.license_name, "license_url": profile.license_url,
                    "classification": profile.classification, "conditions": list(profile.conditions)})
    if matte:
        out.append({"id": matte, "role": "cut-out of the concept", "license": MATTE_LICENSES.get(matte, "unknown")})
    out.append({"id": "skintokens", "role": "the base body's rig (made before, recorded with the body)",
                "license": "MIT (code and weights)"})
    return out


def classification() -> str:
    """The set's licence class: the strictest of the models every set goes through (Qwen's
    research-only since its licence was read, 2026-10-17)."""
    from image_to_3dlab.unreal_profile import strictest_class

    return strictest_class(m.get("classification") for m in models()) or "commercial-conditional"


def body_entry(body: dict, body_json: Path) -> dict:
    """The base body a set is on, as its record names it."""
    return {"name": body["name"], "record": str(Path(body_json).resolve()),
            "image_sha256": _sha(body["image"]), "rig_sha256": _sha(body["rigged"]), "labels": body.get("labels", {})}


def now_text(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def parts_record(*, mode: str, set_name: str, body: dict, body_json: Path, concept: Path, concept_info: dict,
                 chosen: dict, pieces: dict, fit: dict | None, checks: dict, matte: str | None,
                 now: datetime | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode is one of {', '.join(MODES)}, got {mode!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "set": set_name, "mode": mode,
        "created": now_text(now),
        "body": body_entry(body, body_json),
        "concept": {"path": str(Path(concept).resolve()), "sha256": _sha(concept), **concept_info},
        "slots": chosen,
        "pieces": pieces,
        "fit": {k: fit[k] for k in ("covered_share", "files") if k in fit} if fit else None,
        "checks": checks,
        "models": models(matte),
        "classification": classification(),
    }


BODY_FILES = ("set.glb", "preview.glb", "body_masked.glb", "body_trimmed.glb")


def fit_provenance(out: Path, job: dict, report: dict) -> list[Path]:
    """`<file>.provenance.json` beside every file the fit wrote: the licences, bound by sha256.

    A piece is Pixal3D's (MIT, with the DINOv3 encoder's licence) from a Qwen picture, skinned
    with the base body's weights; the body files are the body's own.
    """
    from image_to_3dlab.provenance import LICENSES

    out = Path(out)
    sources = {p["slot"]: p["glb"] for p in job["pieces"]}
    sources.update({Path(p["glb"]).stem: p["glb"] for p in job["pieces"]})  # a refit names pieces by side
    written = []
    for name in report.get("files", []):
        glb = out / name
        if not glb.is_file():
            continue
        slot = Path(name).stem.removesuffix("_L").removesuffix("_R")
        source = job["body"] if name in BODY_FILES else sources.get(Path(name).stem, sources.get(slot))
        record = {
            "schema": "slot_parts.fit/1",
            "output": {"path": name, "sha256": _sha(glb)},
            "license": LICENSES["pixal3d"].license_name,
            "classification": classification(),
            "components": [{"role": m["role"], "name": m["id"], "license": m["license"]} for m in models()],
            "source": {"path": str(source), "sha256": _sha(source) if source else None},
            "body": str(job["body"]),
        }
        path = glb.with_name(glb.stem + ".provenance.json")
        path.write_text(json.dumps(record, indent=1) + "\n")
        written.append(path)
    return written


def write(folder: Path, set_name: str, record: dict) -> Path:
    path = Path(folder) / f"{set_name}.parts.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    return path
