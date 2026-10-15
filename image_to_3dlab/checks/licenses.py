"""Licences of every component, read from the provenance that travels with the asset.

Three record shapes are in use, and all are read: `provenance.finalize_output`'s
`<name>.provenance.json`, Pixal3D's `<name>.json`, and the rig's record, which nests the
records of what it was made from under `source_record`. Every licence anywhere in a record
counts, however deep.

Models finished before the batch runner wrote their own sidecar have none beside them.
For those the record is found through the finish run's folder name
(`<generated run>__finish__<time>`) and accepted only when the generated GLB has the same
sha256 as the finish run's `input/source.glb`.

Rules: no record at all fails (provenance is this repo's guardrail); a Tencent/Hunyuan
licence, a territory-restricted class, or a blocked component that was loaded fails; a
licence nobody recognises warns; licences with conditions pass and list their conditions,
so they reach the game's credits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from image_to_3dlab.checks.registry import register, result

FINISH_RUN = re.compile(r"^(?P<source>[A-Za-z0-9_-]{1,200})__finish__\d{8}-\d{6}(?:-\d+)?$")

# Pattern -> (class, condition). Matched anywhere in a licence name, case-insensitive.
KNOWN = [
    (r"tencent|hunyuan", "blocked", "Not licensed in the EU, UK or South Korea."),
    (r"dinov3", "conditional", "DINOv3 License terms apply to the image encoder."),
    (r"qwen", "conditional", "Attribution: say 'Built with Qwen'."),
    (r"stability ai community", "conditional", "Stability AI Community License: registration and revenue cap."),
    (r"\bmit\b", "clear", None),
    (r"apache", "clear", None),
    (r"\bbsd\b", "clear", None),
    (r"\bcc0\b|cc-by|creative commons attribution", "clear", None),
]
BLOCKED_CLASSES = {"territory-restricted", "blocked", "non-commercial"}


def _read(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and _has_licence(record) else None


def _has_licence(record: Any) -> bool:
    if isinstance(record, dict):
        return any(k in ("license", "license_name") for k in record) or any(map(_has_licence, record.values()))
    if isinstance(record, list):
        return any(map(_has_licence, record))
    return False


def _beside(glb: Path) -> list[Path]:
    return [glb.with_name(glb.stem + ".provenance.json"), glb.with_name(glb.stem + ".json")]


def find_records(glb: Path, explicit: Path | None = None,
                 output_root: Path | None = None) -> tuple[list[tuple[Path, dict]], str | None]:
    """Every provenance record for `glb`, and a note on how they were found."""
    from image_to_3dlab.provenance import sha256_file

    if explicit is not None:
        record = _read(explicit)
        return ([(explicit, record)] if record else []), "given"
    found = [(p, r) for p in _beside(glb) if (r := _read(p))]
    if found:
        return found, "beside the model"
    for folder in [glb.parent, *glb.parents][:4]:
        match = FINISH_RUN.match(folder.name)
        if not match:
            continue
        source = folder / "input" / "source.glb"
        root = output_root or folder.parent.parent
        generated = root / match["source"] / f"{match['source']}.glb"
        if not (source.is_file() and generated.is_file()):
            return [], f"finish run, but {generated} is not there to trace it"
        if sha256_file(source) != sha256_file(generated):
            return [], f"finish run, but {generated.name} is not the model it was finished from"
        return [(p, r) for p in _beside(generated) if (r := _read(p))], f"traced to {generated}"
    return [], None


def collect(record: Any, found: dict | None = None) -> dict:
    """Licence names, classes, conditions and loaded blocked components, at any depth."""
    found = found if found is not None else {"licenses": [], "classes": [], "conditions": [], "blocked": []}

    def add(key, value):
        if value and value not in found[key]:
            found[key].append(value)

    if isinstance(record, dict):
        licence = record.get("license", record.get("license_name"))
        if isinstance(licence, str):
            add("licenses", licence)
        elif isinstance(licence, dict):
            add("licenses", licence.get("name"))
            for condition in licence.get("conditions", []) or []:
                add("conditions", condition)
        if isinstance(record.get("classification"), str):
            add("classes", record["classification"])
        if isinstance(record.get("attribution"), str):
            add("conditions", f"Attribution: say '{record['attribution']}'.")
        if record.get("commercial_status") == "blocked" and record.get("loaded", True) is not False:
            add("blocked", record.get("component") or "unnamed component")
        for value in record.values():
            if isinstance(value, (dict, list)):
                collect(value, found)
    elif isinstance(record, list):
        for value in record:
            collect(value, found)
    return found


def classify(name: str) -> tuple[list[str], list[str]]:
    """(classes, conditions) for one licence name; no classes means unknown."""
    classes, conditions = [], []
    for pattern, kind, condition in KNOWN:
        if re.search(pattern, name, re.IGNORECASE):
            classes.append(kind)
            if condition:
                conditions.append(condition)
    return classes, conditions


@register("license.components")
def components(asset, config):
    records, how = find_records(asset.glb, asset.provenance, asset.output_root)
    threshold = {"fail": "no provenance, or a blocked licence", "warn": "an unknown licence"}
    if not records:
        why = f" ({how})" if how else ""
        return result("license.components", "fail", None, threshold,
                      f"no provenance record found for this model{why}")
    found = {"licenses": [], "classes": [], "conditions": [], "blocked": []}
    for _, record in records:
        collect(record, found)
    blocked, unknown, conditional = [], [], []
    for name in found["licenses"]:
        classes, conditions = classify(name)
        if not classes:
            unknown.append(name)
        if "blocked" in classes:
            blocked.append(name)
        if "conditional" in classes:
            conditional.append(name)
        for condition in conditions:
            if condition not in found["conditions"]:
                found["conditions"].append(condition)
    blocked += [c for c in found["classes"] if c in BLOCKED_CLASSES]
    blocked += [f"{c} (blocked, but loaded)" for c in found["blocked"]]
    value = {"licenses": found["licenses"], "conditions": found["conditions"]}
    details = {"records": [str(p) for p, _ in records], "found": how}
    if blocked:
        return result("license.components", "fail", value, threshold,
                      f"blocked for this project: {', '.join(blocked)}", details)
    if unknown:
        return result("license.components", "warn", value, threshold,
                      f"licence not recognised, check it by hand: {', '.join(unknown)}", details)
    if conditional:
        status = "warn" if config.get("conditional") == "warn" else "pass"
        return result("license.components", status, value, threshold,
                      f"usable with conditions: {'; '.join(found['conditions'])}", details)
    return result("license.components", "pass", value, threshold,
                  f"all clear: {', '.join(found['licenses'])}", details)
