"""Run a list of assets, row by row, and pick up where a stopped run left off.

    python pipeline.py --run-batch lot.json        # or lot.csv

A batch spec lists one row per asset: an id, its images, a kind, a face budget, an output
profile and the steps to run. Each row's `generate` step becomes an ordinary run manifest
and goes through `manifest_run.run_manifest`, the code `--run-manifest` uses, so every
output keeps its `.provenance.json`. The design and the reasons behind it are in
`docs/plans/2026-10-15-batch-mcp-design.md`; the how-to is `docs/batches.md`.

On disk, under `<output>/<name>/`:

    batch-state.json   per row: queued / running / done / failed (with the reason), per step
                       the files it wrote and their sha256
    outputs.json       the finished rows and their files, for the studio and the review queue
    spec.json          the spec as it was resolved for the last run
    rows/<id>/         the row's manifest, inputs and one folder per step

and `<output>/.batch.lock` names the one batch running in that folder, with its pid: one
GPU, one batch. It is taken atomically, so two submissions at once cannot both start.

Resuming is running the same spec again. Done rows are skipped; an interrupted row
starts again from its first step whose files are not all still there with the hash
recorded; failed rows stay failed unless `retry_failed`; a row edited in the spec runs
again; a changed intent or licence policy refuses to resume.

Every finished row is also put in the review queue of the checks (feature 7): a `queued`
line in `review-log.jsonl`, in the folder that holds the batches folder (`output/` for the
default `output/batches/`).
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import manifest_run, step_catalog
from .batch_steps import StepContext
from .processes import group_alive, process_alive
from .provenance import validate_run_policy

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "output" / "batches"

# Stable names: functions 5, 7 and 8 plug in under these. Order here is only for listing.
STEP_NAMES = ("picture", "concept", "generate", "finish", "rig", "clips", "animate", "materials", "export_unreal",
              "checks")
STEP_PACKAGE = "image_to_3dlab.batch_steps"
# Steps installed by hand, over the modules: tests use this for fakes.
STEPS: dict[str, Any] = {}

DEFAULT_STEPS = ("generate", "finish", "checks")
DEFAULT_BACKEND = "pixal3d"
KINDS = ("prop", "creature", "equipment")
# What `scripts/retopo_repaint.py` and `blender_retopo_bake.py` accept.
FACE_RANGE = (1000, 200_000)
# Ids become folder and file names: short and plain, so Windows paths stay short too.
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# The real size an asset should have, for the Unreal export: along its height, or along
# its longest side. 100 km is not a game asset.
MEASURES = ("height", "longest")
SIZE_LIMIT_CM = 10_000_000

# The licence profile of the `picture` step's model (Qwen-Image 2.1), for the run policy.
PICTURE_MODEL = "qwen-image-2.1"
# The licence profile of finish's repaint (Hunyuan 2.1 Paint), for the run policy.
REPAINT_BACKEND = "hunyuan-comfyui"

STATE_FILE = "batch-state.json"
INDEX_FILE = "outputs.json"
SPEC_COPY = "spec.json"
PICKS_FILE = "picks.json"   # {row id: {step: {"choice", "key"}}}, written by pick_candidate
# Steps that make the row's picture: they come before generate.
PICTURE_STEPS = ("picture", "concept")
LOCK_FILE = ".batch.lock"   # in the batches folder, not in one batch: one GPU, one batch
# A submitter that reserved the lock for a runner hands it the reservation's token.
LOCK_TOKEN_ENV = "I2L_BATCH_LOCK_TOKEN"
REVIEW_LOG = "review-log.jsonl"

INTENT_KEYS = ("use_case", "distribution", "commercial_intent")
DEFAULT_INTENT = {"use_case": "showcase", "distribution": "private", "commercial_intent": False}
POLICY_KEYS = {"allow_conditional", "allow_research_only"}
TOP_KEYS = {"schema_version", "name", "_about", "license_policy", "output", "defaults",
            "rows", *INTENT_KEYS}
ROW_KEYS = {"id", "images", "asset", "prompt", "kind", "faces", "profile", "steps", "backend",
            "parameters", "seed", "size_cm", "measure", *STEP_NAMES}
CSV_COLUMNS = ("id", "images", "asset", "prompt", "kind", "faces", "profile", "steps", "backend", "seed",
               "size_cm", "measure")


class BatchError(Exception):
    """A batch that cannot start. Holds every problem found, not just the first."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class BatchRow:
    id: str
    images: tuple[Path, ...]       # the first is the generator's input; the rest references
    asset: Path | None             # an existing GLB, for rows without `generate`
    kind: str
    faces: int | None
    profile: str | None
    steps: tuple[str, ...]
    backend: str
    parameters: dict[str, Any] = field(default_factory=dict)
    options: dict[str, dict[str, Any]] = field(default_factory=dict)   # per step
    size_cm: float | None = None   # the asset's real size, for export_unreal
    measure: str | None = None     # what size_cm measures: "height" or "longest"
    # the images as the spec named them, before a moved one was read from its staged copy
    given_images: tuple[str, ...] = ()
    # a row with only an asset: the picture its record leads to, or the copy staged for it
    # when that one has since moved
    concept: Path | None = None
    # what the `picture` step draws, for a row that starts from words
    prompt: str | None = None

    def fingerprint(self) -> str:
        """What the row asks for, files by content: moving the spec does not re-run it."""
        digest = sha256_file   # streamed: a large GLB is not read into memory

        # an asset's record is part of what the row works from (its licence, its models),
        # and without images of its own the row takes its concept from that record too
        # every record the licence gate read (the provenance and the backend's own)
        records = [c for c in ((self.asset.with_suffix(".provenance.json"),
                                self.asset.with_suffix(".json")) if self.asset else ())
                   if c.is_file()]
        image = self.concept if self.asset and not self.images else None
        payload = {
            "images": [digest(image) for image in self.images],
            "asset": digest(self.asset) if self.asset else None,
            "asset_records": [[c.name, digest(c)] for c in records],
            "asset_image": digest(image) if image else None,
            "kind": self.kind, "faces": self.faces, "profile": self.profile,
            "steps": list(self.steps), "backend": self.backend,
            "parameters": self.parameters, "options": self._options_by_content(),
            "size_cm": self.size_cm, "measure": self.measure,
            **({"prompt": self.prompt} if self.prompt is not None else {}),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _options_by_content(self) -> dict[str, dict[str, Any]]:
        """The step options, with the files they name by content: a body or a LOD
        replaced in place is a change, the same files reached by another path are not."""
        out = {}
        for name, options in self.options.items():
            options = dict(options)
            for key in step_catalog.file_options(name, options):
                options[key] = _file_digests(options[key])
            if options.get("bible"):   # the bible by what it says (for this kind): edited, or another active one, is a change
                options["bible"] = {"bible": _bible_digest(options["bible"]), "kind": self.kind}
            out[name] = options
        return out

    def step_fingerprints(self) -> dict[str, str]:
        """One fingerprint per step, chained: each covers the row's source, every step
        before it, and what this step itself depends on (its options and the row fields
        the catalogue names). An edited row runs again from its first step whose
        fingerprint changed; the steps before keep their results."""
        digest = sha256_file
        records = [c for c in ((self.asset.with_suffix(".provenance.json"),
                                self.asset.with_suffix(".json")) if self.asset else ())
                   if c.is_file()]
        image = self.concept if self.asset and not self.images else None
        previous = _hash({
            "images": [digest(image) for image in self.images],
            "asset": digest(self.asset) if self.asset else None,
            "asset_records": [[c.name, digest(c)] for c in records],
            "asset_image": digest(image) if image else None,
            **({"prompt": self.prompt} if self.prompt is not None else {}),
        })
        fingerprints = {}
        by_content = self._options_by_content()
        for name in self.steps:
            options = by_content.get(name, {})
            fields = (step_catalog.CATALOG[name].row_fields if name in step_catalog.CATALOG
                      else UNKNOWN_STEP_FIELDS)
            previous = _hash({"after": previous, "step": name, "options": options,
                              "fields": {key: getattr(self, key) for key in fields}})
            fingerprints[name] = previous
        return fingerprints

    def candidate_key(self, step: str) -> str:
        """What a step that makes candidates makes them from: its fingerprint without the
        pick. The same key, the same candidates; a pick made for one key means nothing
        for another."""
        options = {**self.options, step: {k: v for k, v in self.options.get(step, {}).items() if k != "pick"}}
        return replace(self, options=options).step_fingerprints()[step]


# A step the catalogue does not describe yet depends, as far as anyone knows, on all of them.
UNKNOWN_STEP_FIELDS = ("kind", "faces", "profile", "backend", "parameters", "size_cm", "measure")


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _bible_digest(bible: str) -> Any:
    """A style bible by its name and the hash of its bible.json; as given when it cannot be
    found (the step's preflight says why)."""
    from .style import apply

    try:
        style = apply.resolve(bible)
    except ValueError:
        return bible
    return {"bible": style.name, "sha256": style.sha256}


def _file_digests(value: Any) -> Any:
    """A file option by its content: the same path holding another file is a change."""
    if isinstance(value, list):
        return [_file_digests(item) for item in value]
    if value is False:
        return value   # a file option turned off
    if isinstance(value, str):
        # a file gone is an input gone, as for the row's own images: a done row keeps its
        # result, a row still to run fails on it
        return {"sha256": sha256_file(Path(value))}
    return value


def with_picks(row: BatchRow, picks: dict[str, Any]) -> BatchRow:
    """The row with the picks made through the MCP server or the studio (`picks.json`),
    each only while the candidates it was made among are still the ones the step would
    make (its candidate key): new words, settings or inputs before it make it wait again.
    A pick the spec gives itself always wins. Applied in the row's step order: a step's
    key covers the picks of the steps before it, as it did when the step ran."""
    current = row
    for step in row.steps:
        made = picks.get(step)
        if not isinstance(made, dict) or "pick" in current.options.get(step, {}):
            continue
        if made.get("key") not in (current.candidate_key(step), *_legacy_keys(current, step)):
            continue
        current = replace(current, options={**current.options,
                                            step: {**current.options.get(step, {}), "pick": made.get("choice")}})
    return current


def _legacy_keys(row: BatchRow, step: str) -> tuple[str, ...]:
    """The key a pick on a `picture` step had before keys were step fingerprints (the
    words, the settings and the seeds), so those picks still hold."""
    if step != "picture":
        return ()
    from .batch_steps.picture import legacy_key

    return (legacy_key(row.prompt, row.options.get(step, {})),)


def recorded_sidecar(asset: Path | None) -> Path | None:
    """The licence record kept beside an existing GLB: its provenance, else the run
    manifest some backends write."""
    if asset is None:
        return None
    candidates = [c for c in (asset.with_suffix(".provenance.json"), asset.with_suffix(".json"))
                  if c.is_file() and isinstance(read_json(c), dict) and read_json(c)]
    # a damaged or thin provenance file must not hide the backend's own record: the one
    # finish copies the licence from is the first that has licence data at all
    for candidate in candidates:
        if _has_licence_data(read_json(candidate)):
            return candidate
    return candidates[0] if candidates else None


def _has_licence_data(record: dict[str, Any]) -> bool:
    """Whether a record has what finish inherits: a licence and a licence class."""
    return bool(record.get("license")) and bool(
        record.get("classification") or (record.get("output") or {}).get("classification"))


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
# A finished asset's record nests the record of what it was made from under
# `source_record`, and so on back to the generation; deeper than this is not real.
RECORD_DEPTH = 8


def _records(record: Any):
    """A record and the records it nests, newest first."""
    for _ in range(RECORD_DEPTH):
        if not isinstance(record, dict):
            return
        yield record
        record = record.get("source_record")


def image_from_record(record: Any) -> Path | None:
    """The concept image a provenance record leads back to: a finish record's input is the
    GLB it finished, so the image is found in the records it nests."""
    for level in _records(record):
        given = level.get("input") or {}
        for path in ((given.get("image") or {}).get("path"), given.get("path")):
            if (isinstance(path, str) and Path(path).suffix.lower() in IMAGE_SUFFIXES
                    and Path(path).is_file()):
                return Path(path)
    return None


def recorded_image(asset: Path | None) -> Path | None:
    """The source image an existing GLB's records lead back to, if it still exists."""
    if asset is None:
        return None
    for candidate in (asset.with_suffix(".provenance.json"), asset.with_suffix(".json")):
        image = image_from_record(read_json(candidate)) if candidate.is_file() else None
        if image is not None:
            return image
    return None


def _beside(asset: Path | None) -> list[dict[str, Any]]:
    """Every record kept beside a GLB that parses: its provenance and the backend's own.
    The licence gate reads all of them, so a thin or partial one cannot hide another."""
    if asset is None:
        return []
    found = []
    for candidate in (asset.with_suffix(".provenance.json"), asset.with_suffix(".json")):
        data = read_json(candidate) if candidate.is_file() else None
        if isinstance(data, dict) and data:
            found.append(data)
    return found


def recorded_backend(asset: Path | None) -> str | None:
    """The model an existing GLB was generated with, as its records name it, however many
    finishes ago."""
    for record in _beside(asset):
        for level in _records(record):
            backend = (level.get("model") or {}).get("backend") or level.get("backend")
            if isinstance(backend, str):
                return backend
    return None


KNOWN_CLASSES = ("commercial-clear", "commercial-conditional", "territory-restricted",
                 "research-only")


def recorded_classes(asset: Path | None) -> list[str]:
    """Every licence class an existing GLB's records name, through the whole chain: the
    output's, the record's own, and each component that was actually used."""
    found: list[str] = []
    for level in (level for record in _beside(asset) for level in _records(record)):
        values = [(level.get("output") or {}).get("classification"), level.get("classification")]
        values += [component.get("commercial_status") for component in level.get("components") or []
                   if isinstance(component, dict) and component.get("loaded", True)]
        found += [value for value in values if isinstance(value, str) and value not in found]
    return found


def class_problem(classification: str, intent: dict[str, Any],
                  policy: dict[str, Any]) -> str | None:
    """Why a licence class is not allowed under this intent and policy, the rules of
    `provenance.validate_run_policy` applied to a class rather than a backend."""
    if classification == "commercial-clear":
        return None
    if classification not in KNOWN_CLASSES:
        # a class this repo does not know (a newer tool, a typo, `blocked`) fails closed
        return f"its licence class {classification!r} is not one this repo has reviewed"
    if not policy.get("allow_conditional", True):
        return f"it is {classification}; the policy disallows conditional models"
    if classification == "research-only" and policy.get("allow_research_only") is False:
        return "it is research-only; the policy disallows research-only models"
    if classification == "research-only" and not (
            intent["use_case"] == "showcase" and intent["distribution"] == "private"
            and not intent.get("commercial_intent")):
        return "it is research-only (non-commercial): a private, non-commercial showcase only"
    if (classification == "territory-restricted" and intent["use_case"] == "game"
            and intent["distribution"] == "worldwide"):
        return "it is territory-restricted, not cleared for worldwide game distribution"
    return None


@dataclass
class BatchSpec:
    name: str
    path: Path
    rows: list[BatchRow]
    intent: dict[str, Any]
    license_policy: dict[str, Any]
    output_dir: Path

    @property
    def directory(self) -> Path:
        return self.output_dir / self.name

    def intent_fingerprint(self) -> str:
        payload = {"intent": self.intent, "license_policy": self.license_policy}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class BatchReport:
    directory: Path
    counts: dict[str, int]
    failed: dict[str, str]

    waiting: list[str] = field(default_factory=list)   # rows that wait for a pick

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 3 if self.waiting else 0


# --------------------------------------------------------------------------- the spec

def spec_directory(path: Path) -> Path:
    """The folder a spec's batch lives in, read from its name and output alone: finding a
    running batch must not depend on every row's inputs still being valid."""
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".csv":
        return DEFAULT_OUTPUT / path.stem
    top = read_json(path)
    if not isinstance(top, dict):
        raise BatchError(f"could not read batch spec {path}")
    output = top.get("output") if isinstance(top.get("output"), dict) else {}
    directory = output.get("directory")
    root = (path.parent / directory).resolve() if isinstance(directory, str) and directory else DEFAULT_OUTPUT
    name = top.get("name") if isinstance(top.get("name"), str) else path.stem
    return root / name


def load_batch(path: Path, intent: dict[str, Any] | None = None) -> BatchSpec:
    """Read and check a JSON or CSV spec. Every problem is reported at once.

    `intent` holds the command line's `--use-case/--distribution/--commercial`, which only
    a CSV can take: a JSON spec says it itself, and two sources would disagree.
    """
    path = path.expanduser().resolve()
    intent = {key: value for key, value in (intent or {}).items() if value is not None}
    problems: list[str] = []
    if not path.is_file():
        raise BatchError(f"batch spec does not exist: {path}")
    if path.suffix.lower() == ".csv":
        top: dict[str, Any] = {"rows": path.name}
    else:
        try:
            top = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchError(f"could not read batch spec {path}: {exc}") from exc
        if not isinstance(top, dict) or top.get("schema_version") != 1:
            raise BatchError("a JSON batch spec must be an object with schema_version 1")
        problems += [f"unknown spec field {key!r}" for key in sorted(set(top) - TOP_KEYS)]
        if intent:
            problems.append("--use-case/--distribution/--commercial are for a CSV spec; "
                            "a JSON spec sets them in the spec")
    base = path.parent

    resolved_intent = {key: top.get(key, DEFAULT_INTENT[key]) for key in INTENT_KEYS}
    resolved_intent.update(intent)
    # they go into every manifest and provenance record as they are: they must be real
    if resolved_intent["use_case"] not in ("showcase", "game"):
        problems.append(f"use_case must be showcase or game, not {resolved_intent['use_case']!r}")
    if resolved_intent["distribution"] not in ("private", "public", "worldwide"):
        problems.append("distribution must be private, public or worldwide, not "
                        f"{resolved_intent['distribution']!r}")
    if not isinstance(resolved_intent["commercial_intent"], bool):
        problems.append("commercial_intent must be true or false, not "
                        f"{resolved_intent['commercial_intent']!r}")
    policy = top.get("license_policy", {})
    if not isinstance(policy, dict):
        problems.append("license_policy must be an object")
        policy = {}
    elif set(policy) - POLICY_KEYS:
        # a misspelt key would leave its default in force: the gate must not fail open
        problems.append(f"license_policy: unknown field(s) "
                        f"{', '.join(sorted(set(policy) - POLICY_KEYS))}")
        policy = {**policy, "allow_conditional": False}
    elif not all(isinstance(policy.get(key, True), bool) for key in POLICY_KEYS):
        # "false" as a string is truthy: the licence gate must not fail open on a typo
        problems.append("license_policy.allow_conditional and allow_research_only must be "
                        "true or false")
        policy = {**policy, "allow_conditional": False, "allow_research_only": False}
    output_field = top.get("output", {})
    if not isinstance(output_field, dict):
        problems.append('output must be an object, like {"directory": "../output/batches"}')
        output_field = {}
    elif set(output_field) - {"directory"}:
        # a misspelt directory would silently write to the default folder
        problems.append(f"output: unknown field(s) {', '.join(sorted(set(output_field) - {'directory'}))}")
    output = output_field.get("directory")
    if output is not None and not isinstance(output, str):
        problems.append(f"output.directory must be a folder path, not {output!r}")
        output = None
    output_dir = (base / output).resolve() if output else DEFAULT_OUTPUT
    name = top.get("name", path.stem)
    if not isinstance(name, str):   # null or 7 would become a folder, a lock and a source
        problems.append(f"batch name must be text, not {name!r}")
        name = path.stem
    if not ID_PATTERN.fullmatch(name) or not _usable_folder_name(name):
        problems.append(f"batch name {name!r}: use letters, digits, '_', '.', '-' "
                        "(up to 64), starting with a letter or digit, not ending in a dot, "
                        "and not a name Windows reserves (CON, AUX, NUL, COM1...)")

    raw_rows = top.get("rows")
    if isinstance(raw_rows, str):
        raw_rows, csv_problems = _read_csv(base / raw_rows)
        problems += csv_problems
    if not isinstance(raw_rows, list) or not raw_rows:
        problems.append("the spec has no rows")
        raw_rows = []
    defaults = top.get("defaults", {})
    if not isinstance(defaults, dict):
        problems.append("defaults must be an object")
        defaults = {}
    problems += [f"defaults: unknown field {key!r}"
                 for key in sorted(set(defaults) - (ROW_KEYS - {"id", "images", "asset"}))]

    # the images each row ran with last time, as the spec gave them: a staged copy stands
    # in for a moved image only if the spec still names that same path
    last = read_json(output_dir / name / SPEC_COPY)
    ran_with = {row.get("id"): set(row.get("given_images") or [])
                for row in (last or {}).get("rows", []) if isinstance(row, dict)}
    used = {row.get("id"): row.get("asset")
            for row in (last or {}).get("rows", []) if isinstance(row, dict)}
    # and the pictures staged for each row, for an asset row whose recorded picture moved
    saved = (read_json(output_dir / name / STATE_FILE) or {}).get("rows", {})
    saved = saved if isinstance(saved, dict) else {}
    staged_copies = {row_id: [output_dir / name / record["path"] for record in entry.get("staged", [])]
                     for row_id, entry in saved.items() if isinstance(entry, dict)}
    # the candidates picked through the MCP server or the studio, per row and step
    picks = read_json(output_dir / name / PICKS_FILE)
    picks = picks if isinstance(picks, dict) else {}
    rows, seen = [], set()
    for number, raw in enumerate(raw_rows, start=1):
        row, row_problems = _parse_row(raw, defaults, base, number,
                                       staged=output_dir / name / "rows",
                                       ran_with=ran_with, staged_copies=staged_copies,
                                       saved=saved, used=used)
        problems += row_problems
        if row is None:
            continue
        row = with_picks(row, picks.get(row.id) or {})
        # rows/<id>/ is a folder: on Windows and macOS `Tree` and `tree` are the same one
        if row.id.lower() in seen:
            problems.append(f"row {number}: id {row.id!r} is used twice (ids are compared "
                            "ignoring case, as folder names are)")
        seen.add(row.id.lower())
        rows.append(row)

    for row in rows:
        # every model a row would run is gated: its generator, and the repaint if asked
        # a recorded backend this repo has no reviewed licence for is refused, not skipped;
        # only an asset with no record at all goes without (its finish record says so)
        source = None if "generate" in row.steps else recorded_backend(row.asset)
        gated = ([(row.backend, row.backend)] if "generate" in row.steps else []) + (
            [(source, f"its source ({source})")] if source is not None else []) + (
            [(PICTURE_MODEL, "the picture model")] if {"picture", "concept"} & set(row.steps) else []) + (
            [(REPAINT_BACKEND, "the repaint")]
            if "finish" in row.steps and row.options.get("finish", {}).get("paint") else [])
        for backend, what in gated:
            try:
                validate_run_policy(backend, resolved_intent["use_case"],
                                    resolved_intent["distribution"],
                                    policy.get("allow_conditional", True))
            except ValueError as exc:
                problems.append(f"row {row.id}: license policy for {what}: {exc}")
        if "generate" not in row.steps:
            # a source can carry more than its generator: a repaint, a matte, whatever
            # finished it before. Every class its records name is held to the policy.
            for classification in recorded_classes(row.asset):
                problem = class_problem(classification, resolved_intent, policy)
                if problem:
                    problems.append(f"row {row.id}: license policy for its source: {problem}")
    if problems:
        raise BatchError(problems)
    return BatchSpec(name=name, path=path, rows=rows, intent=resolved_intent,
                     license_policy=policy, output_dir=output_dir)


WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}


def _usable_folder_name(name: str) -> bool:
    """A name every OS makes into its own folder: Windows refuses CON or AUX (with any
    extension) and drops a trailing dot, so `crate.` would be `crate`."""
    return not name.endswith(".") and name.split(".")[0].upper() not in WINDOWS_RESERVED


def _is_glb(path: Path) -> bool:
    """Whether a file is a whole binary glTF 2 (header, declared length, JSON chunk): a
    truncated one fails before the batch, not in Blender's importer after earlier rows."""
    import struct

    try:
        with path.open("rb") as handle:
            header = handle.read(20)
            if len(header) < 20:
                return False
            magic, version, length, chunk_length, chunk_type = struct.unpack("<4sIIII", header)
            if (magic != b"glTF" or version != 2 or length != path.stat().st_size
                    or chunk_type != 0x4E4F534A or 20 + chunk_length > length):
                return False
            gltf = json.loads(handle.read(chunk_length))
    except (OSError, ValueError, struct.error):
        return False
    return isinstance(gltf, dict) and isinstance(gltf.get("asset"), dict)


def _read_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """CSV rows as JSON-shaped rows: `;` splits lists, an empty cell means the default."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            records = list(reader)
    except OSError as exc:
        return [], [f"could not read {path}: {exc}"]
    problems = [f"{path.name}: unknown column {column!r}"
                for column in columns if column not in CSV_COLUMNS]
    rows = []
    for number, record in enumerate(records, start=2):   # line 1 is the header
        row: dict[str, Any] = {}
        if record.get(None):   # DictReader keeps surplus cells under None
            problems.append(f"{path.name} line {number}: more cells than columns (a comma "
                            "inside a value needs quotes)")
        for key, value in record.items():
            if key is None or not isinstance(value, (str, type(None))):
                continue
            value = (value or "").strip()
            if not value or key not in CSV_COLUMNS:
                continue
            if key in ("images", "steps"):
                row[key] = [part.strip() for part in value.split(";") if part.strip()]
            elif key in ("faces", "seed"):
                row[key] = int(value) if value.lstrip("-").isdigit() else value
            elif key == "size_cm":
                try:
                    row[key] = int(value) if value.isdigit() else float(value)
                except ValueError:
                    row[key] = value
            else:
                row[key] = value
        rows.append(row)
    return rows, problems


def _parse_row(raw: Any, defaults: dict[str, Any], base: Path, number: int,
               staged: Path | None = None, ran_with: dict[str, set[str]] | None = None,
               staged_copies: dict[str, list[Path]] | None = None,
               saved: dict[str, Any] | None = None, used: dict[str, str] | None = None,
               ) -> tuple[BatchRow | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [f"row {number}: must be an object"]
    label = f"row {raw.get('id', number)}"
    problems = [f"{label}: unknown field {key!r}" for key in sorted(set(raw) - ROW_KEYS)]
    value = {**defaults, **raw}

    row_id = value.get("id", "")
    if not isinstance(row_id, str):   # 7 or null would become a folder name and a review key
        problems.append(f"{label}: id must be text, not {row_id!r}")
        row_id = ""
    elif not ID_PATTERN.fullmatch(row_id) or not _usable_folder_name(row_id):
        problems.append(f"{label}: id {row_id!r}: use letters, digits, '_', '.', '-' "
                        "(up to 64), starting with a letter or digit, not ending in a dot, "
                        "and not a name Windows reserves (CON, AUX, NUL, COM1...)")

    steps = value.get("steps", list(DEFAULT_STEPS))
    if not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
        problems.append(f"{label}: steps must be a list of names")
        steps = []
    elif not steps:
        problems.append(f"{label}: a row needs at least one step")
    for step in steps:
        if step not in STEP_NAMES:
            problems.append(f"{label}: unknown step {step!r}; steps are "
                            f"{', '.join(STEP_NAMES)}")
        if steps.count(step) > 1:
            problems.append(f"{label}: step {step!r} is listed twice")
    if "picture" in steps and steps[0] != "picture":
        problems.append(f"{label}: picture makes the row's picture, so it must come first")
    if "generate" in steps and any(step not in PICTURE_STEPS for step in steps[:steps.index("generate")]):
        problems.append(f"{label}: generate makes the asset, so it must come first (only a "
                        "step that draws its picture may come before it)")

    raw_images = value.get("images", []) or []
    if not isinstance(raw_images, list) or not all(isinstance(i, str) for i in raw_images):
        problems.append(f"{label}: images must be a list of file paths")
        raw_images = []
    if value.get("asset") is not None and not isinstance(value.get("asset"), str):
        problems.append(f"{label}: asset must be a file path")
        value = {**value, "asset": None}
    images = tuple((base / image).resolve() for image in raw_images)
    given = tuple(str(image) for image in images)
    if staged is not None:
        # an image moved away after its row ran is still in the row's folder: a resume of
        # the rest of the batch reads that copy (same bytes, so the row stays done). Only
        # for the path it ran with: a spec edited to name another path is a new input.
        copies = staged / row_id / "input"
        known = (ran_with or {}).get(row_id, set())
        images = tuple(copies / image.name
                       if not image.is_file() and str(image) in known
                       and (copies / image.name).is_file() else image for image in images)
    problems += [f"{label}: image does not exist: {image}"
                 for image in images if not image.is_file()]
    names = [image.name.lower() for image in images]
    if len(set(names)) < len(names):
        # they are copied into one input folder, where one would overwrite the other
        problems.append(f"{label}: two images have the same file name; rename one")
    asset = (base / value["asset"]).resolve() if value.get("asset") else None
    finished = ((saved or {}).get(row_id, {}).get("status") == "done"
                and asset is not None and str(asset) == (used or {}).get(row_id))
    if asset is not None and not asset.is_file() and finished:
        pass   # its result is in the batch already: a resume of the rest goes on without it
    elif asset is not None and not asset.is_file():
        problems.append(f"{label}: asset does not exist: {asset}")
    elif asset is not None and not _is_glb(asset):
        problems.append(f"{label}: asset is not a GLB file: {asset}")
    prompt = value.get("prompt")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        problems.append(f"{label}: prompt must be words, not {prompt!r}")
        prompt = None
    if "picture" in steps:
        if prompt is None:
            problems.append(f"{label}: picture needs a prompt, the words to draw")
        if raw_images:
            problems.append(f"{label}: picture draws the row's picture from its prompt; give a "
                            "prompt or images, not both")
    elif prompt is not None:
        problems.append(f"{label}: prompt is for the picture step; add \"picture\" to its steps")
    if ("concept" in steps and not images and "picture" not in steps
            and not (raw.get("concept") or defaults.get("concept") or {}).get("shape")):
        problems.append(f"{label}: concept needs a picture to start from: the row's images, a "
                        "picture step before it, or shape pictures in its settings")
    if "generate" in steps:
        if not images and "picture" not in steps and "concept" not in steps:
            problems.append(f"{label}: generate needs an image")
        if asset is not None:
            problems.append(f"{label}: has both an asset and a generate step; pick one")
    elif steps and asset is None and not set(steps) <= set(PICTURE_STEPS):
        # a row that only draws (picture, concept) makes a picture, and needs no model
        problems.append(f"{label}: without generate, the row needs an asset (a GLB)")

    kind = value.get("kind", "prop")
    if kind not in KINDS:
        problems.append(f"{label}: kind {kind!r} is not one of {', '.join(KINDS)}")
    faces = value.get("faces")
    if faces is not None and not (isinstance(faces, int)
                                  and FACE_RANGE[0] <= faces <= FACE_RANGE[1]):
        problems.append(f"{label}: faces must be a whole number from {FACE_RANGE[0]:,} to "
                        f"{FACE_RANGE[1]:,}, not {faces!r}")

    backend = value.get("backend", DEFAULT_BACKEND)
    if not isinstance(backend, str):
        problems.append(f"{label}: backend must be a name, not {backend!r}")
        backend = DEFAULT_BACKEND
    # defaults first (their parameters, then their seed), then the row's: a row's own
    # parameters beat a default seed, and its own seed beats everything
    parameters: dict[str, Any] = {}
    for source, seed in ((defaults.get("parameters"), defaults.get("seed")),
                         (raw.get("parameters"), raw.get("seed"))):
        if source is not None and not isinstance(source, dict):
            problems.append(f"{label}: parameters must be an object, not {source!r}")
        elif source is not None:
            parameters.update(source)
        if seed is not None:
            parameters["seed"] = seed
    if "generate" in steps:
        generator = manifest_run.GENERATORS.get(backend)
        if generator is None:
            problems.append(
                f"{label}: backend {backend!r} cannot run in a batch; available: "
                f"{', '.join(sorted(manifest_run.GENERATORS))}. Batches only run generators "
                "that can check their weights without downloading them")
        else:
            problems += [f"{label}: {backend} does not take parameter {key!r}"
                         for key in sorted(set(parameters) - generator.parameters)]
            problems += [f"{label}: {problem}"
                         for problem in generator.check_parameters(parameters)]

    options = {}
    for step in STEP_NAMES:
        merged: dict[str, Any] = {}
        for source in (defaults.get(step), raw.get(step)):
            if source is None:
                continue
            if not isinstance(source, dict):
                problems.append(f"{label}: the settings for {step} must be an object")
                continue
            # a file named in the spec is found from the spec's folder, like its images
            merged.update(step_catalog.resolve_files(step, source, base))
        if merged:
            options[step] = merged
            if step in step_catalog.CATALOG and step != "generate":
                problems += [f"{label}: {problem}"
                             for problem in step_catalog.check(step, merged, files=False)]
    concept = None
    if asset is not None and asset.is_file() and not images:
        concept = recorded_image(asset) or next(
            (copy for copy in (staged_copies or {}).get(row_id, []) if copy.is_file()), None)
    size_cm, measure = value.get("size_cm"), value.get("measure")
    if size_cm is not None and (isinstance(size_cm, bool) or not isinstance(size_cm, (int, float))
                                or not 0 < size_cm <= SIZE_LIMIT_CM):
        problems.append(f"{label}: size_cm must be a positive number of centimetres, "
                        f"not {size_cm!r}")
    if measure is not None and measure not in MEASURES:
        problems.append(f"{label}: measure {measure!r} is not one of {', '.join(MEASURES)}")
    if measure is not None and size_cm is None:
        problems.append(f"{label}: measure says what size_cm measures; give size_cm too")
    if size_cm is not None and measure is None:
        measure = "height"
    if options.get("generate"):
        # the generator reads `parameters`; settings here would be silently ignored
        problems.append(f"{label}: put generator settings in parameters, not under generate "
                        f"({', '.join(sorted(options['generate']))})")
    profile = value.get("profile")
    if problems:
        return None, problems
    return BatchRow(id=row_id, images=images, asset=asset, kind=kind, faces=faces,
                    profile=str(profile) if profile is not None else None,
                    steps=tuple(steps), backend=backend, parameters=parameters,
                    options=options, size_cm=size_cm, measure=measure, given_images=given,
                    concept=concept, prompt=prompt), []


# --------------------------------------------------------------------- step registry

def load_step(name: str) -> Any | None:
    """The step's module (or a test's stand-in), or None when it does not exist yet."""
    if name in STEPS:
        return STEPS[name]
    module_name = f"{STEP_PACKAGE}.{name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def available_steps() -> dict[str, bool]:
    return {name: load_step(name) is not None for name in STEP_NAMES}


def preflight(spec: BatchSpec, work: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Everything that would stop a row, found before the first one runs. `work` limits it
    to what this run will execute, row by row and step by step (a resume skips the done
    rows, and the done steps of an interrupted one)."""
    problems = []
    for name in STEP_NAMES:
        using = [row for row in spec.rows
                 if name in (row.steps if work is None else work.get(row.id, ()))]
        step = load_step(name) if using else None
        if step is not None and hasattr(step, "preflight"):
            problems += [f"{name}: {problem}" for problem in step.preflight(using)]
    return problems


def _last_run_rows(directory: Path) -> dict[str, dict[str, Any]]:
    """The rows as the last run resolved them (`spec.json`), by id."""
    last = read_json(directory / SPEC_COPY)
    return {row["id"]: row for row in (last or {}).get("rows", [])
            if isinstance(row, dict) and isinstance(row.get("id"), str)}


def _as_it_ran(row: BatchRow, entry: dict[str, Any], last: dict[str, Any] | None) -> BatchRow | None:
    """The row as it was when `entry` was made, if that can be told for certain: rebuilt
    from the last run's spec copy, or this row with the steps it ran, whichever gives back
    the fingerprint the entry recorded. None when neither does."""
    candidates = []
    if last:
        try:
            candidates.append(BatchRow(
                id=row.id, images=tuple(Path(image) for image in last.get("images") or []),
                asset=Path(last["asset"]) if last.get("asset") else None, kind=last.get("kind"),
                faces=last.get("faces"), profile=last.get("profile"), steps=tuple(last.get("steps") or ()),
                backend=last.get("backend"), parameters=last.get("parameters") or {},
                options={name: last[name] for name in STEP_NAMES if isinstance(last.get(name), dict)},
                size_cm=last.get("size_cm"), measure=last.get("measure"), concept=row.concept,
                prompt=last.get("prompt")))
        except (TypeError, ValueError):
            pass
    candidates.append(BatchRow(**{**row.__dict__, "steps": tuple(entry.get("steps", {}))}))
    for candidate in candidates:
        try:
            if candidate.fingerprint() == entry.get("fingerprint"):
                return candidate
        except OSError:   # an input it named is gone: it cannot be told
            continue
    return None


def _with_known_fingerprints(row: BatchRow, entry: dict[str, Any],
                             last: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The entry's steps, those done before steps had fingerprints given the ones they ran
    with, when the row as it ran can be rebuilt. A batch from before this change, edited or
    given a new default step, then keeps what is still valid instead of running whole."""
    steps = {name: dict(step) for name, step in entry.get("steps", {}).items()}
    if not any(step.get("status") == "done" and "fingerprint" not in step for step in steps.values()):
        return steps
    ran = _as_it_ran(row, entry, last)
    if ran is None:
        return steps
    then = ran.step_fingerprints()
    for name, step in steps.items():
        if step.get("status") == "done" and "fingerprint" not in step and name in then:
            step["fingerprint"] = then[name]
    return steps


def pending_work(spec: BatchSpec, state: dict[str, Any] | None,
                 retry_failed: bool) -> dict[str, tuple[str, ...]]:
    """What a run would execute, given the saved state: per pending row, the steps from
    its first unfinished one. Done rows (unless a step they skipped now exists) and
    failed ones (unless retrying) are not there."""
    previous = (state or {}).get("rows", {})
    last = _last_run_rows(spec.directory)
    work: dict[str, tuple[str, ...]] = {}
    for row in spec.rows:
        entry = previous.get(row.id)
        try:
            fingerprint = row.fingerprint()
        except OSError:   # an input gone: a done row keeps its result, others will fail
            if entry is None or not _still_done(entry, spec.directory):
                work[row.id] = ()
            continue
        if entry is None:
            work[row.id] = row.steps
            continue
        if entry.get("fingerprint") != fingerprint:
            # edited: from its first step whose fingerprint changed (failed or not: the
            # spec changed, so it is a new request)
            known = {"steps": _with_known_fingerprints(row, entry, last.get(row.id))}
            work[row.id] = _remaining(row, known, spec.directory, row.step_fingerprints())
            continue
        if _still_done(entry, spec.directory):
            continue
        if entry.get("status") == "failed" and not retry_failed:
            continue
        work[row.id] = _remaining(row, entry, spec.directory)
    return work


def _remaining(row: BatchRow, entry: dict[str, Any], directory: Path,
               fingerprints: dict[str, str] | None = None) -> tuple[str, ...]:
    """The steps from the first one not done with its files still in place, or, given the
    row's step fingerprints, the first one done for another version of the row."""
    for index, name in enumerate(row.steps):
        step = entry.get("steps", {}).get(name, {})
        files = step.get("files", [])
        if (step.get("status") != "done"
                or (fingerprints is not None and step.get("fingerprint") != fingerprints[name])
                or not all((directory / record["path"]).is_file() for record in files)):
            return row.steps[index:]
    return ()


# ------------------------------------------------------------------------ the state

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    """Write whole or not at all: a kill mid-write leaves the previous file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:   # Windows: a reader has the file open for a moment
                if attempt == 9:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, directory: Path) -> str:
    try:
        return path.resolve().relative_to(directory.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _absolute(text: str | None, directory: Path) -> Path | None:
    return None if text is None else directory / text


def _file_records(path: Path, directory: Path) -> list[dict[str, Any]]:
    """A file and its hash; a folder (a Pixal3D camera) as every file in it, so a resume
    notices one of them changed, not only that the folder is gone."""
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return [{"path": _relative(file, directory), "sha256": sha256_file(file),
             "bytes": file.stat().st_size} for file in files]


def _files_intact(files: list[dict[str, Any]], directory: Path) -> bool:
    for record in files:
        path = directory / record["path"]
        if record.get("sha256") is None:
            if not path.exists():
                return False
        elif not path.is_file() or sha256_file(path) != record["sha256"]:
            return False
    return True


def _read_lock(output_dir: Path) -> dict[str, Any] | None:
    lock = read_json(output_dir / LOCK_FILE)
    return lock if isinstance(lock, dict) and isinstance(lock.get("pid"), int) else None


def process_identity(pid: int) -> str | None:
    """When the process `pid` started, as the OS says: with the pid, it tells the process
    apart from a later one that reused the number. None when it cannot be read."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                times = [wintypes.FILETIME() for _ in range(4)]
                if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                    return None
                return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
            finally:
                kernel32.CloseHandle(handle)
        import subprocess

        started = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True,
                                 text=True, timeout=10, check=False).stdout.strip()
        return started or None
    except (OSError, ValueError, AttributeError):
        return None


def _live_lock(output_dir: Path) -> dict[str, Any] | None:
    """The lock, if its process is alive and is the one that took it: a pid the OS has
    since given to another process does not hold it."""
    lock = _read_lock(output_dir)
    if not lock:
        return None
    if not process_alive(lock["pid"]):
        # On POSIX a runner started by the MCP server leads its own process group: a
        # generator or Blender it left behind still works in the row's folder, so the
        # lock holds until the whole group is gone. Windows tracks the one process.
        return lock if os.name != "nt" and group_alive(lock["pid"]) else None
    recorded = lock.get("identity")
    current = process_identity(lock["pid"]) if recorded else None
    return None if recorded and current and current != recorded else lock


def acquire_lock(output_dir: Path, name: str, pid: int, token: str,
                 starting: bool = False) -> None:
    """Take the batches folder's lock for batch `name`, or raise BatchError.

    Created with O_EXCL, so of two processes asking at once exactly one gets it. A lock
    whose process is dead is taken over; one with the same token is a reservation made
    for this runner by whoever started it, and is rewritten with this pid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LOCK_FILE
    # `starting`: a reservation made by a submitter in its own name, until the runner's
    # pid is known; nothing may stop that pid
    record = {"batch": name, "pid": pid, "identity": process_identity(pid), "token": token,
              "since": now(), **({"starting": True} if starting else {})}
    for _ in range(3):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except PermissionError:   # Windows: the old lock is being deleted this instant
            raise BatchError("another batch is starting in this folder; try again") from None
        except FileExistsError:
            lock = _read_lock(output_dir)
            if lock is None and time.time() - _mtime(path) < 5:
                raise BatchError("another batch is starting in this folder; try again") from None
            if lock and lock.get("token") == token and lock.get("batch") == name:
                write_json(path, record)
                return
            if lock and _live_lock(output_dir) == lock:
                if lock.get("batch") == name:
                    raise BatchError(f"batch {name} is already running (pid {lock['pid']})") from None
                raise BatchError(f"another batch is using the GPU: {lock.get('batch')}. "
                                 "One batch runs at a time") from None
            _clear_stale_lock(output_dir, lock)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record))
        return
    raise BatchError("could not take the batch lock; try again")


def _clear_stale_lock(output_dir: Path, stale: dict[str, Any] | None) -> None:
    """Remove a dead process's lock, unless someone replaced it meanwhile.

    Two callers can both see the same stale lock; deleting it blind would let the second
    delete the fresh lock the first just took. Removal happens under a claim file created
    with O_EXCL, and only if the lock is still the one seen stale.
    """
    claim = output_dir / f"{LOCK_FILE}.claim"
    try:
        handle = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except (FileExistsError, PermissionError):   # PermissionError: Windows, being deleted
        if claim.exists() and time.time() - _mtime(claim) > 30:   # killed mid-claim
            _unlink(claim)
        raise BatchError("another batch is starting in this folder; try again") from None
    try:
        if _read_lock(output_dir) == stale and not _unlink(output_dir / LOCK_FILE):
            raise BatchError("another batch is starting in this folder; try again")
    finally:
        os.close(handle)
        _unlink(claim)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _unlink(path: Path) -> bool:
    """Delete a file, retrying while Windows says another process has it open for a
    moment (a status reader). False when it is still there."""
    for _ in range(20):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            time.sleep(0.05)
    return False


def release_lock(output_dir: Path, token: str) -> None:
    lock = _read_lock(output_dir)
    if lock and lock.get("token") == token:
        # if it stays held open, it names this process, about to end: the next taker
        # finds it stale
        _unlink(output_dir / LOCK_FILE)


def live_lock(output_dir: Path) -> dict[str, Any] | None:
    """The batches folder's lock while its process is alive, read once: a caller listing
    many batches compares names against it instead of reading it per batch."""
    return _live_lock(output_dir)


def lock_info(directory: Path) -> dict[str, Any] | None:
    """The live lock of the batch in `directory`, or None."""
    lock = _live_lock(directory.parent)
    return lock if lock and lock.get("batch") == directory.name else None


def lock_holder(directory: Path) -> int | None:
    """The pid running the batch in `directory`, or None (not running, or its process died)."""
    lock = _live_lock(directory.parent)
    return lock["pid"] if lock and lock.get("batch") == directory.name else None


def running_batches(output_dir: Path) -> list[Path]:
    """The batch running in `output_dir`, if any: one GPU, one batch."""
    lock = _live_lock(output_dir)
    return [output_dir / str(lock.get("batch"))] if lock else []


def rows_as_seen(state: dict[str, Any], running: bool) -> dict[str, dict[str, Any]]:
    """The rows as a reader should see them: with no runner alive, a row left `running`
    (a stopped or killed batch) is back in the queue, as resuming will treat it."""
    rows = {}
    for row_id, entry in state.get("rows", {}).items():
        if entry.get("status") == "running" and not running:
            entry = {**entry, "status": "queued", "note": "interrupted; resume to run it"}
        rows[row_id] = entry
    return rows


def summary(state: dict[str, Any], running: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in rows_as_seen(state, running).values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return counts


def asset_sha(entry: dict[str, Any], directory: Path) -> str | None:
    """The sha256 of a row's final asset: as kept when the row finished, as a step recorded
    it, else of the file itself (a row from before the hash was kept)."""
    if entry.get("asset_sha256"):
        return entry["asset_sha256"]
    for step in entry.get("steps", {}).values():
        for record in step.get("files", []):
            if record.get("path") == entry.get("asset"):
                return record.get("sha256")
    asset = entry.get("asset")
    path = directory / asset if asset else None
    return sha256_file(path) if path is not None and path.is_file() else None


# --------------------------------------------------------------------------- running

def run_batch(spec: BatchSpec, retry_failed: bool = False,
              log: Callable[[str], None] | None = None) -> BatchReport:
    """Run every row that is not done. Raises BatchError when the batch cannot start;
    a KeyboardInterrupt leaves the state as it was, ready to resume."""
    log = log or (lambda line: print(line, flush=True))
    problems = preflight(spec, pending_work(spec, read_json(spec.directory / STATE_FILE),
                                            retry_failed))
    if problems:
        raise BatchError(problems)
    directory = spec.directory
    # A submitter (the MCP server, the studio) reserves the lock as it starts this
    # runner and passes the reservation's token; run by hand, there is none.
    token = os.environ.get(LOCK_TOKEN_ENV) or uuid.uuid4().hex
    acquire_lock(spec.output_dir, spec.name, os.getpid(), token)
    try:
        state_path = directory / STATE_FILE
        state = read_json(state_path)
        if state and state.get("name") not in (None, spec.name):
            # on Windows and macOS `Oak` and `oak` are one folder: never two batches in it
            raise BatchError(f"batch {spec.name} would share its folder with batch "
                             f"{state['name']} (names are compared ignoring case); rename it")
        ran_from = (state or {}).get("spec") or {}
        if (ran_from.get("path") and ran_from["path"] != str(spec.path)
                and ran_from.get("sha256") != sha256_file(spec.path)):
            # another file under this batch name would take over its state and its resume
            raise BatchError(f"batch {spec.name} already ran from another spec "
                             f"({ran_from['path']}); give this one its own name")
        if state and state.get("intent_fingerprint") != spec.intent_fingerprint():
            raise BatchError(
                f"the intent or licence policy of batch {spec.name} changed since it ran "
                f"(see {state_path}). Give the batch a new name, or remove its folder")
        directory.mkdir(parents=True, exist_ok=True)
        return _run_rows(spec, state or {}, retry_failed, log)
    finally:
        release_lock(spec.output_dir, token)


def _run_rows(spec: BatchSpec, state: dict[str, Any], retry_failed: bool,
              log: Callable[[str], None]) -> BatchReport:
    directory = spec.directory
    previous = state.get("rows", {})
    last = _last_run_rows(directory)   # read before this run writes its own copy
    rows: dict[str, Any] = {}
    for row in spec.rows:
        entry = previous.get(row.id)
        try:
            fingerprint = row.fingerprint()
        except OSError as exc:
            if entry is not None and _still_done(entry, directory):
                rows[row.id] = entry   # its result is in the batch; the source moved on
                continue
            # an input gone or unreadable since validation fails this row, not the batch
            rows[row.id] = {**(entry or {}), "status": "failed", "steps": (entry or {}).get("steps", {}),
                            "reason": f"input: {exc}", "fingerprint": None}
            continue
        fingerprints = row.step_fingerprints()
        if entry is None:
            entry = {"status": "queued", "fingerprint": fingerprint, "steps": {}}
        elif entry.get("fingerprint") != fingerprint:
            # Edited: the steps before the first changed one keep their results. A step
            # done before steps had fingerprints has none, so it counts as changed.
            kept = {name: step for name, step in
                    _with_known_fingerprints(row, entry, last.get(row.id)).items() if name in row.steps}
            again = _remaining(row, {"steps": kept}, directory, fingerprints)
            entry = {"status": "queued", "fingerprint": fingerprint, "steps": kept,
                     **({"staged": entry["staged"]} if entry.get("staged") else {}),
                     "note": (f"the row changed in the spec; it runs again from {again[0]}"
                              if again else "the row changed in the spec")}
        else:
            # unchanged: steps done before fingerprints existed are stamped with theirs,
            # so a later edit can tell which of them to keep
            for name, step in entry.get("steps", {}).items():
                if step.get("status") == "done" and "fingerprint" not in step and name in fingerprints:
                    step["fingerprint"] = fingerprints[name]
        rows[row.id] = entry
    # rows dropped from the spec keep their record, apart: they are not part of the batch
    removed = {**state.get("removed_rows", {}),
               **{row_id: entry for row_id, entry in previous.items() if row_id not in rows}}
    state = {"schema_version": 1, "name": spec.name,
             "spec": {"path": str(spec.path), "sha256": sha256_file(spec.path)},
             "intent": spec.intent, "license_policy": spec.license_policy,
             "intent_fingerprint": spec.intent_fingerprint(),
             "started_at": now(), "pid": os.getpid(), "rows": rows,
             **({"removed_rows": removed} if removed else {})}

    def save() -> None:
        state["updated_at"] = now()
        write_json(directory / STATE_FILE, state)

    for row in spec.rows:   # a done row that will run again is not finished any more
        if rows[row.id]["status"] == "done" and not _still_done(rows[row.id], directory):
            rows[row.id]["status"] = "queued"

    write_json(directory / SPEC_COPY, _spec_record(spec))
    save()
    # a row reset to the queue (edited, its files gone) leaves the index now, not after
    # its rerun: a stop in between must not show an old result as finished
    write_json(directory / INDEX_FILE, _index(spec, rows))
    for number, row in enumerate(spec.rows, start=1):
        entry = rows[row.id]
        label = f"[batch] {number}/{len(spec.rows)} {row.id}"
        if _still_done(entry, directory):
            continue
        if entry["status"] == "failed" and not retry_failed:
            log(f"{label}: failed before ({entry.get('reason')}); --retry-failed runs it again")
            continue
        _run_row(spec, row, entry, save, lambda line, label=label: log(f"{label}: {line}"))
        write_json(directory / INDEX_FILE, _index(spec, rows))

    write_json(directory / INDEX_FILE, _index(spec, rows))
    counts: dict[str, int] = {}
    for row in spec.rows:
        counts[rows[row.id]["status"]] = counts.get(rows[row.id]["status"], 0) + 1
    failed = {row.id: rows[row.id].get("reason", "")
              for row in spec.rows if rows[row.id]["status"] == "failed"}
    waiting = [row.id for row in spec.rows if rows[row.id]["status"] == "waiting"]
    return BatchReport(directory=directory, counts=counts, failed=failed, waiting=waiting)


def _still_done(entry: dict[str, Any], directory: Path) -> bool:
    """A done row to skip: no step it skipped has appeared since, and every file its steps
    wrote is still there at the size recorded. Size, not hash: hashing every finished GLB
    of a batch of thousands on each resume would take longer than the resume saves. A row
    whose files went missing runs again from its first broken step, which is hashed."""
    if entry.get("status") != "done" or _newly_available(entry):
        return False
    recorded = [record for step in entry.get("steps", {}).values()
                if step.get("status") == "done" for record in step.get("files", [])]
    for record in recorded + entry.get("staged", []):
        path = directory / record["path"]
        if not path.is_file() or ("bytes" in record and path.stat().st_size != record["bytes"]):
            return False
    return True


def _newly_available(entry: dict[str, Any]) -> bool:
    """A done row with a step that was skipped as missing, and now exists."""
    return any(step.get("status") == "skipped" and load_step(name) is not None
               for name, step in entry.get("steps", {}).items())


def _run_row(spec: BatchSpec, row: BatchRow, entry: dict[str, Any],
             save: Callable[[], None], log: Callable[[str], None]) -> None:
    directory = spec.directory
    row_dir = directory / "rows" / row.id
    entry.update(status="running", started_at=now())
    entry.pop("reason", None)
    entry.pop("finished_at", None)
    save()
    # Inputs checked at the start of the batch can change while earlier rows run (another
    # GLB, another record, another licence): the row runs only on what was checked.
    try:
        current = row.fingerprint()
    except OSError as exc:
        _fail(entry, entry.setdefault("steps", {}), "input", str(exc), save, log)
        return
    if current != entry.get("fingerprint"):
        _fail(entry, entry.setdefault("steps", {}), "input",
              "changed since the batch was checked; run it again to check the new one",
              save, log)
        return

    staged = row.images or ((row.concept,) if row.concept else ())
    if "generate" not in row.steps and staged:
        # generate copies a row's images; a finish-only row's (or the picture its asset's
        # record leads to) are staged here, so its folder and outputs.json carry it too
        try:
            (row_dir / "input").mkdir(parents=True, exist_ok=True)
            for image in staged:
                if image.resolve() != (row_dir / "input" / image.name).resolve():
                    shutil.copyfile(image, row_dir / "input" / image.name)
            # recorded on the row, not on a step: a checks-only row has them too
            entry["staged"] = [record for image in staged for record in
                               _file_records(row_dir / "input" / image.name, directory)]
        except OSError as exc:   # gone or unreadable since the preflight: this row fails
            _fail(entry, entry.setdefault("steps", {}), "stage", str(exc), save, log)
            return
    asset = row.asset
    sidecar = recorded_sidecar(asset) if asset else None
    # the row's picture: its first image, the one its asset's record leads to, or what a
    # step that draws one (`picture`) leaves
    picture = row.images[0] if row.images else row.concept
    clips: list[Path] = []   # animated GLBs a step made, for the export
    material: str | None = None   # the one material a materials step chose, for the export
    steps = entry.setdefault("steps", {})
    for name in [name for name in steps if name not in row.steps]:
        del steps[name]   # a step taken out of the row is no longer part of its result
    fingerprints = row.step_fingerprints()
    rerun = False   # once a step runs again, every step after it does too
    for name in row.steps:
        done = steps.get(name, {})
        if (not rerun and done.get("status") == "done"
                and done.get("fingerprint") == fingerprints[name]
                and _files_intact(done.get("files", []), directory)):
            asset = _absolute(done.get("asset"), directory) or asset
            sidecar = _absolute(done.get("sidecar"), directory) or sidecar
            picture = _absolute(done.get("picture"), directory) or picture
            clips = [directory / clip for clip in done.get("clips", [])] or clips
            material = done.get("material") or material
            continue
        rerun = True
        step = load_step(name)
        if step is None:
            warning = f"step {name} is not available yet; skipped"
            print(f"warning: {row.id}: {warning}", file=sys.stderr, flush=True)
            steps[name] = {"status": "skipped", "warning": warning,
                           "fingerprint": fingerprints[name]}
            save()
            continue
        step_dir = row_dir / name
        try:
            if step_dir.exists():
                shutil.rmtree(step_dir)
        except OSError as exc:   # a file held open (Windows): this row fails, not the batch
            _fail(entry, steps, name, f"could not clear {step_dir.name}: {exc}", save, log)
            return
        log(f"{name} ...")
        steps[name] = {"status": "running", "started_at": now()}
        save()
        ctx = StepContext(row=row, row_dir=row_dir, step_dir=step_dir, asset=asset,
                          sidecar=sidecar, intent=spec.intent,
                          license_policy=spec.license_policy,
                          options=row.options.get(name, {}), log=log, picture=picture, clips=clips,
                          material=material,
                          candidate_key=row.candidate_key(name))
        try:
            result = step.run(ctx)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - one row's failure must not stop the batch
            _fail(entry, steps, name, str(exc) or type(exc).__name__, save, log)
            return
        if not result.ok:
            _fail(entry, steps, name, result.reason or "failed", save, log)
            return
        if result.waiting is not None:
            # Several to pick from: the row waits, the batch goes on with the others. A
            # pick (in the spec, or pick_candidate) makes the step finish on the next run.
            candidates = [{**candidate, "file": _relative(Path(candidate["file"]), directory)}
                          for candidate in result.waiting["candidates"]]
            steps[name] = {"status": "waiting", "fingerprint": fingerprints[name],
                           "candidates": candidates, "key": ctx.candidate_key, "finished_at": now()}
            entry.update(status="waiting", finished_at=now(),
                         note=f"{name}: pick one of {len(candidates)} "
                              f"({', '.join(str(c['choice']) for c in candidates)})")
            save()
            log(f"waiting: {entry['note']}")
            return
        asset = result.asset or asset
        picture = result.picture or picture
        clips = result.clips or clips
        material = result.material or material
        sidecar = result.sidecar if result.asset else sidecar
        steps[name] = {
            "status": "done", "finished_at": now(), "fingerprint": fingerprints[name],
            "files": [record for path in result.files
                      for record in _file_records(path, directory)],
            "asset": _relative(result.asset, directory) if result.asset else None,
            "sidecar": _relative(result.sidecar, directory) if result.sidecar else None,
            **({"picture": _relative(result.picture, directory)} if result.picture else {}),
            **({"clips": [_relative(clip, directory) for clip in result.clips]} if result.clips else {}),
            **({"material": result.material} if result.material else {}),
            # a step that made candidates: the key a later pick (pick_candidate) is made for
            **({"key": ctx.candidate_key} if (result.record or {}).get("candidates") else {}),
            **({"record": result.record} if result.record else {}),
        }
        save()

    # Queued first: a runner stopped in between leaves the row running, and its resume
    # queues it (the same version is never queued twice) before recording it done.
    if asset is not None:
        try:
            queue_for_review(spec, row, asset, report_file=checks_report(row_dir, steps))
        except Exception as exc:  # noqa: BLE001 - one row's queue error must not stop the batch
            # the row's steps are done and kept; --retry-failed just queues it
            _fail(entry, steps, "review", f"could not queue it for review: {exc}", save, log)
            return
    entry["asset_sha256"] = sha256_file(asset) if asset is not None else None
    if ": pick one of" in str(entry.get("note")):
        entry.pop("note")   # it waited, and was picked
    entry.update(status="done", finished_at=now(),
                 asset=_relative(asset, directory) if asset else None,
                 sidecar=_relative(sidecar, directory) if sidecar else None)
    save()
    log("done")


def checks_report(row_dir: Path, steps: dict[str, Any]) -> Path | None:
    """The report the row's checks step wrote, if it ran."""
    done = steps.get("checks", {})
    report = (done.get("record") or {}).get("report") if done.get("status") == "done" else None
    return row_dir / "checks" / report if report else None


def _fail(entry, steps, name, reason, save, log) -> None:
    steps[name] = {"status": "failed", "reason": reason, "finished_at": now()}
    entry.update(status="failed", reason=f"{name}: {reason}", finished_at=now())
    save()
    log(f"failed: {entry['reason']}")


# Feature 7's checks profiles name the equipment kind differently.
REVIEW_KINDS = {"prop": "prop", "creature": "creature", "equipment": "equipment_part"}


def queue_for_review(spec: BatchSpec, row: BatchRow, asset: Path,
                     report_file: Path | None = None) -> None:
    """Put a finished row in the review queue of the checks (feature 7).

    The queue's own `image_to_3dlab.checks.enqueue_for_review` writes the line once that
    package is installed; until then this writes the same line itself. The key is the
    row's id, the version the asset's sha256: the same version is never queued twice,
    unless it now comes with a checks report it did not have. The report is the checks
    step's (`report_file`), or a `<asset>.checks.json` beside the asset, when it is for
    this very file.
    """
    log_path = spec.output_dir.parent / REVIEW_LOG
    sha = sha256_file(asset)
    kind = REVIEW_KINDS.get(row.kind, row.kind)
    source = f"batch:{spec.name}"
    report_file = report_file or asset.with_suffix(".checks.json")
    report = read_json(report_file) if report_file.is_file() else None
    if not isinstance(report, dict) or report.get("asset_sha256") != sha:
        report, report_file = None, None
    # the key is the asset's id (feature 7's contract): the same bytes under it are the
    # same version. Queued again only when who made it, what it is, or its report changed
    # (checks run again with other thresholds write the same file with another result).
    ours = [line for line in _lines(log_path) if line.get("event") == "queued"
            and line.get("key") == row.id and line.get("source") == source]
    if any(line.get("asset_sha256") == sha and line.get("kind") == kind for line in ours) and (
            report_file is None or _same_report(ours[-1], report_file, report)):
        return
    try:
        from image_to_3dlab.checks import enqueue_for_review
    except ModuleNotFoundError as exc:
        if exc.name != "image_to_3dlab.checks":
            raise
    else:
        enqueue_for_review(asset, row.id, source=source, kind=kind, report=report,
                           report_file=report_file, log=log_path)
        return
    results = (report or {}).get("results", [])
    licence = next((r.get("status") for r in results if r.get("check") == "license.components"),
                   None)
    event = {
        "event": "queued", "key": row.id, "at": now(), "asset": str(asset.resolve()),
        "asset_sha256": sha, "report": str(report_file) if report_file else "",
        "status": (report or {}).get("status"),
        "summary": {status: [r.get("check") for r in results if r.get("status") == status]
                    for status in ("fail", "warn")},
        "profile": (report or {}).get("profile"), "license_status": licence,
        "source": source, "kind": kind,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _same_report(line: dict[str, Any], report_file: Path, report: dict[str, Any]) -> bool:
    """Whether a queued line already carries this report's result."""
    from image_to_3dlab.checks.review import licence_status, summary

    return (line.get("report") == str(report_file) and line.get("status") == report.get("status")
            and line.get("summary") == summary(report) and line.get("profile") == report.get("profile")
            and line.get("license_status") == licence_status(report))


def _lines(path: Path) -> list[dict[str, Any]]:
    """The JSON lines of a log; a damaged line is skipped, not fatal."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = []
    for raw in text.splitlines():
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, dict):
            lines.append(value)
    return lines


def review_index(log_path: Path) -> dict[tuple[str, str | None], tuple[int, dict[str, Any]]]:
    """The latest event per (key, version), read once for a whole batch's rows."""
    index: dict[tuple[str, str | None], tuple[int, dict[str, Any]]] = {}
    for number, line in enumerate(_lines(log_path)):
        if isinstance(line.get("key"), str):
            index[(line["key"], line.get("asset_sha256"))] = (number, line)
    return index


def review_in(index: dict[tuple[str, str | None], tuple[int, dict[str, Any]]], asset_key: str,
              sha256: str | None) -> dict[str, Any] | None:
    """The latest event in `index` for this version, or one that names no version."""
    found = [hit for hit in (index.get((asset_key, sha256)), index.get((asset_key, None))) if hit]
    return max(found, key=lambda hit: hit[0])[1] if found else None


def review_state(log_path: Path, asset_key: str, sha256: str | None) -> dict[str, Any] | None:
    """The latest review event for this version of an asset: queued, approved or
    sent_back (with its note). Another batch's row with the same id, or an older version,
    has another sha256 and does not count; an event that names no hash goes by the key."""
    latest = None
    for line in _lines(log_path):
        if line.get("key") == asset_key and line.get("asset_sha256") in (None, sha256):
            latest = line
    return latest


def _spec_record(spec: BatchSpec) -> dict[str, Any]:
    return {
        "schema_version": 1, "name": spec.name, "source": str(spec.path),
        "intent": spec.intent, "license_policy": spec.license_policy,
        "rows": [{"id": row.id, "images": [str(image) for image in row.images],
                  "given_images": list(row.given_images or [str(i) for i in row.images]),
                  "asset": str(row.asset) if row.asset else None, "kind": row.kind,
                  **({"prompt": row.prompt} if row.prompt is not None else {}),
                  "faces": row.faces, "profile": row.profile, "steps": list(row.steps),
                  "backend": row.backend, "parameters": row.parameters,
                  "size_cm": row.size_cm, "measure": row.measure,
                  **row.options} for row in spec.rows],
    }


def _index(spec: BatchSpec, rows: dict[str, Any]) -> dict[str, Any]:
    """The finished rows, with paths relative to the batch folder: what a review queue or
    the studio's library reads, without having to understand the state file."""
    directory = spec.directory
    listed = []
    for row in spec.rows:
        entry = rows[row.id]
        if entry["status"] != "done":
            continue
        inputs = directory / "rows" / row.id / "input"
        # this run's staged pictures, not whatever an earlier run left in the folder
        pictures = row.images or tuple(directory / record["path"]
                                       for record in entry.get("staged", []))
        listed.append({
            "id": row.id, "kind": row.kind, "profile": row.profile, "faces": row.faces,
            "size_cm": row.size_cm, "measure": row.measure,
            "asset": entry.get("asset"), "sidecar": entry.get("sidecar"),
            "references": [_relative(inputs / image.name, directory)
                           for image in pictures if (inputs / image.name).is_file()],
            "steps": {name: step.get("status") for name, step in entry["steps"].items()},
            "files": [record["path"] for step in entry["steps"].values()
                      for record in step.get("files", [])],
            "finished_at": entry.get("finished_at"),
            **_drawn(directory, entry["steps"]),
            **_extra_outputs(directory, directory / "rows" / row.id, entry["steps"]),
        })
    return {"schema_version": 1, "batch": spec.name, "updated_at": now(), "rows": listed}


def _drawn(directory: Path, steps: dict[str, Any]) -> dict[str, Any]:
    """The picture the row's last drawing step (picture, concept) chose, if any."""
    drawn = [step.get("picture") for name, step in steps.items()
             if name in PICTURE_STEPS and step.get("status") == "done" and step.get("picture")]
    return {"picture": drawn[-1]} if drawn else {}


def _extra_outputs(directory: Path, row_dir: Path, steps: dict[str, Any]) -> dict[str, Any]:
    """What a step left beside the asset: the Unreal export and the checks report."""
    extra: dict[str, Any] = {}
    export = steps.get("export_unreal", {})
    if export.get("status") == "done" and export.get("record"):
        folder = row_dir / "export_unreal"
        extra["unreal"] = {key: _relative(folder / export["record"][key], directory)
                           for key in ("glb", "sidecar") if export["record"].get(key)}
    report = checks_report(row_dir, steps)
    if report is not None:
        extra["checks"] = {"status": steps["checks"]["record"].get("status"),
                           "report": _relative(report, directory)}
    return extra


# ------------------------------------------------------------------------------ CLI

def run_from_cli(spec_path: Path, intent: dict[str, Any], retry_failed: bool) -> int:
    """`pipeline.py --run-batch`: 0 all done, 1 some rows failed, 2 the batch cannot start,
    3 some rows wait for a pick (none failed),
    130 interrupted (run the same command again to resume)."""
    try:
        spec = load_batch(spec_path, intent=intent)
        report = run_batch(spec, retry_failed=retry_failed)
    except BatchError as exc:
        for problem in exc.problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: run the same command again to resume", file=sys.stderr)
        return 130
    counts = ", ".join(f"{count} {status}" for status, count in sorted(report.counts.items()))
    print(f"[batch] {spec.name}: {counts} -> {report.directory}")
    for row_id, reason in report.failed.items():
        print(f"[batch] failed {row_id}: {reason}")
    for row_id in report.waiting:
        print(f"[batch] waiting {row_id}: pick a candidate (pick_candidate, or \"pick\" in "
              "the step's settings), then run the same command again")
    return report.exit_code
