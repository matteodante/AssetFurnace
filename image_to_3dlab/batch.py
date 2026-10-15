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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import manifest_run
from .batch_steps import StepContext
from .processes import process_alive
from .provenance import validate_run_policy

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "output" / "batches"

# Stable names: functions 5, 7 and 8 plug in under these. Order here is only for listing.
STEP_NAMES = ("generate", "finish", "materials", "export_unreal", "checks")
STEP_PACKAGE = "image_to_3dlab.batch_steps"
# Steps installed by hand, over the modules: tests use this for fakes.
STEPS: dict[str, Any] = {}

DEFAULT_STEPS = ("generate", "finish")
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

# The licence profile of finish's repaint (Hunyuan 2.1 Paint), for the run policy.
REPAINT_BACKEND = "hunyuan-comfyui"

STATE_FILE = "batch-state.json"
INDEX_FILE = "outputs.json"
SPEC_COPY = "spec.json"
LOCK_FILE = ".batch.lock"   # in the batches folder, not in one batch: one GPU, one batch
# A submitter that reserved the lock for a runner hands it the reservation's token.
LOCK_TOKEN_ENV = "I2L_BATCH_LOCK_TOKEN"
REVIEW_LOG = "review-log.jsonl"

INTENT_KEYS = ("use_case", "distribution", "commercial_intent")
DEFAULT_INTENT = {"use_case": "showcase", "distribution": "private", "commercial_intent": False}
TOP_KEYS = {"schema_version", "name", "_about", "license_policy", "output", "defaults",
            "rows", *INTENT_KEYS}
ROW_KEYS = {"id", "images", "asset", "kind", "faces", "profile", "steps", "backend",
            "parameters", "seed", "size_cm", "measure", *STEP_NAMES}
CSV_COLUMNS = ("id", "images", "asset", "kind", "faces", "profile", "steps", "backend", "seed",
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

    def fingerprint(self) -> str:
        """What the row asks for, files by content: moving the spec does not re-run it."""
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        payload = {
            "images": [digest(image) for image in self.images],
            "asset": digest(self.asset) if self.asset else None,
            "kind": self.kind, "faces": self.faces, "profile": self.profile,
            "steps": list(self.steps), "backend": self.backend,
            "parameters": self.parameters, "options": self.options,
            "size_cm": self.size_cm, "measure": self.measure,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


# --------------------------------------------------------------------------- the spec

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
    policy = top.get("license_policy", {})
    if not isinstance(policy, dict):
        problems.append("license_policy must be an object")
        policy = {}
    output_field = top.get("output", {})
    if not isinstance(output_field, dict):
        problems.append('output must be an object, like {"directory": "../output/batches"}')
        output_field = {}
    output = output_field.get("directory")
    output_dir = (base / str(output)).resolve() if output else DEFAULT_OUTPUT
    name = str(top.get("name", path.stem))
    if not ID_PATTERN.fullmatch(name):
        problems.append(f"batch name {name!r}: use letters, digits, '_', '.', '-' "
                        "(up to 64), starting with a letter or digit")

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

    rows, seen = [], set()
    for number, raw in enumerate(raw_rows, start=1):
        row, row_problems = _parse_row(raw, defaults, base, number)
        problems += row_problems
        if row is None:
            continue
        if row.id in seen:
            problems.append(f"row {number}: id {row.id!r} is used twice")
        seen.add(row.id)
        rows.append(row)

    for row in rows:
        # every model a row would run is gated: its generator, and the repaint if asked
        gated = ([row.backend] if "generate" in row.steps else []) + (
            [REPAINT_BACKEND] if row.options.get("finish", {}).get("paint") else [])
        for backend in gated:
            try:
                validate_run_policy(backend, resolved_intent["use_case"],
                                    resolved_intent["distribution"],
                                    policy.get("allow_conditional", True))
            except ValueError as exc:
                what = "the repaint" if backend == REPAINT_BACKEND else backend
                problems.append(f"row {row.id}: license policy for {what}: {exc}")
    if problems:
        raise BatchError(problems)
    return BatchSpec(name=name, path=path, rows=rows, intent=resolved_intent,
                     license_policy=policy, output_dir=output_dir)


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
    for record in records:
        row: dict[str, Any] = {}
        for key, value in record.items():
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


def _parse_row(raw: Any, defaults: dict[str, Any], base: Path,
               number: int) -> tuple[BatchRow | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [f"row {number}: must be an object"]
    label = f"row {raw.get('id', number)}"
    problems = [f"{label}: unknown field {key!r}" for key in sorted(set(raw) - ROW_KEYS)]
    value = {**defaults, **raw}

    row_id = str(value.get("id", ""))
    if not ID_PATTERN.fullmatch(row_id):
        problems.append(f"{label}: id {row_id!r}: use letters, digits, '_', '.', '-' "
                        "(up to 64), starting with a letter or digit")

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
    if "generate" in steps and steps[0] != "generate":
        problems.append(f"{label}: generate makes the asset, so it must come first")

    images = tuple((base / image).resolve() for image in value.get("images", []) or [])
    problems += [f"{label}: image does not exist: {image}"
                 for image in images if not image.is_file()]
    names = [image.name.lower() for image in images]
    if len(set(names)) < len(names):
        # they are copied into one input folder, where one would overwrite the other
        problems.append(f"{label}: two images have the same file name; rename one")
    asset = (base / value["asset"]).resolve() if value.get("asset") else None
    if asset is not None and not asset.is_file():
        problems.append(f"{label}: asset does not exist: {asset}")
    if "generate" in steps:
        if not images:
            problems.append(f"{label}: generate needs an image")
        if asset is not None:
            problems.append(f"{label}: has both an asset and a generate step; pick one")
    elif steps and asset is None:
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
    parameters: dict[str, Any] = {}
    for source in (defaults.get("parameters"), raw.get("parameters")):
        if source is None:
            continue
        if not isinstance(source, dict):
            problems.append(f"{label}: parameters must be an object, not {source!r}")
            continue
        parameters.update(source)
    if "seed" in value:
        parameters["seed"] = value["seed"]
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

    options = {}
    for step in STEP_NAMES:
        merged: dict[str, Any] = {}
        for source in (defaults.get(step), raw.get(step)):
            if source is None:
                continue
            if not isinstance(source, dict):
                problems.append(f"{label}: the settings for {step} must be an object")
                continue
            merged.update(source)
        if merged:
            options[step] = merged
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
    profile = value.get("profile")
    if problems:
        return None, problems
    return BatchRow(id=row_id, images=images, asset=asset, kind=kind, faces=faces,
                    profile=str(profile) if profile is not None else None,
                    steps=tuple(steps), backend=backend, parameters=parameters,
                    options=options, size_cm=size_cm, measure=measure), []


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


def preflight(spec: BatchSpec) -> list[str]:
    """Everything that would stop a row, found before the first one runs."""
    problems = []
    for name in STEP_NAMES:
        rows = [row for row in spec.rows if name in row.steps]
        step = load_step(name) if rows else None
        if step is not None and hasattr(step, "preflight"):
            problems += [f"{name}: {problem}" for problem in step.preflight(rows)]
    return problems


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
    return [{"path": _relative(file, directory), "sha256": sha256_file(file)} for file in files]


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


def _live_lock(output_dir: Path) -> dict[str, Any] | None:
    lock = _read_lock(output_dir)
    return lock if lock and process_alive(lock["pid"]) else None


def acquire_lock(output_dir: Path, name: str, pid: int, token: str) -> None:
    """Take the batches folder's lock for batch `name`, or raise BatchError.

    Created with O_EXCL, so of two processes asking at once exactly one gets it. A lock
    whose process is dead is taken over; one with the same token is a reservation made
    for this runner by whoever started it, and is rewritten with this pid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LOCK_FILE
    record = {"batch": name, "pid": pid, "token": token, "since": now()}
    for _ in range(3):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            lock = _read_lock(output_dir)
            if lock is None and time.time() - _mtime(path) < 5:
                raise BatchError("another batch is starting in this folder; try again") from None
            if lock and lock.get("token") == token and lock.get("batch") == name:
                write_json(path, record)
                return
            if lock and process_alive(lock["pid"]):
                if lock.get("batch") == name:
                    raise BatchError(f"batch {name} is already running (pid {lock['pid']})") from None
                raise BatchError(f"another batch is using the GPU: {lock.get('batch')}. "
                                 "One batch runs at a time") from None
            path.unlink(missing_ok=True)   # left by a process that is gone
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(record))
        return
    raise BatchError("could not take the batch lock; try again")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def release_lock(output_dir: Path, token: str) -> None:
    lock = _read_lock(output_dir)
    if lock and lock.get("token") == token:
        (output_dir / LOCK_FILE).unlink(missing_ok=True)


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


def asset_sha(entry: dict[str, Any]) -> str | None:
    """The sha256 recorded for a row's final asset."""
    for step in entry.get("steps", {}).values():
        for record in step.get("files", []):
            if record.get("path") == entry.get("asset"):
                return record.get("sha256")
    return None


# --------------------------------------------------------------------------- running

def run_batch(spec: BatchSpec, retry_failed: bool = False,
              log: Callable[[str], None] | None = None) -> BatchReport:
    """Run every row that is not done. Raises BatchError when the batch cannot start;
    a KeyboardInterrupt leaves the state as it was, ready to resume."""
    log = log or (lambda line: print(line, flush=True))
    problems = preflight(spec)
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
    rows: dict[str, Any] = {}
    for row in spec.rows:
        entry = previous.get(row.id)
        fingerprint = row.fingerprint()
        if entry is None or entry.get("fingerprint") != fingerprint:
            entry = {"status": "queued", "fingerprint": fingerprint, "steps": {},
                     **({"note": "the row changed in the spec"} if entry else {})}
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

    write_json(directory / SPEC_COPY, _spec_record(spec))
    save()
    for number, row in enumerate(spec.rows, start=1):
        entry = rows[row.id]
        label = f"[batch] {number}/{len(spec.rows)} {row.id}"
        if entry["status"] == "done" and not _newly_available(entry):
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
    return BatchReport(directory=directory, counts=counts, failed=failed)


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

    asset = row.asset
    sidecar = _sidecar_for(asset) if asset else None
    steps = entry.setdefault("steps", {})
    rerun = False   # once a step runs again, every step after it does too
    for name in row.steps:
        done = steps.get(name, {})
        if (not rerun and done.get("status") == "done"
                and _files_intact(done.get("files", []), directory)):
            asset = _absolute(done.get("asset"), directory) or asset
            sidecar = _absolute(done.get("sidecar"), directory) or sidecar
            continue
        rerun = True
        step = load_step(name)
        if step is None:
            warning = f"step {name} is not available yet; skipped"
            print(f"warning: {row.id}: {warning}", file=sys.stderr, flush=True)
            steps[name] = {"status": "skipped", "warning": warning}
            save()
            continue
        step_dir = row_dir / name
        if step_dir.exists():
            shutil.rmtree(step_dir)
        log(f"{name} ...")
        steps[name] = {"status": "running", "started_at": now()}
        save()
        ctx = StepContext(row=row, row_dir=row_dir, step_dir=step_dir, asset=asset,
                          sidecar=sidecar, intent=spec.intent,
                          license_policy=spec.license_policy,
                          options=row.options.get(name, {}), log=log)
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
        asset = result.asset or asset
        sidecar = result.sidecar if result.asset else sidecar
        steps[name] = {
            "status": "done", "finished_at": now(),
            "files": [record for path in result.files
                      for record in _file_records(path, directory)],
            "asset": _relative(result.asset, directory) if result.asset else None,
            "sidecar": _relative(result.sidecar, directory) if result.sidecar else None,
            **({"record": result.record} if result.record else {}),
        }
        save()

    # Queued first: a runner stopped in between leaves the row running, and its resume
    # queues it (the same version is never queued twice) before recording it done.
    if asset is not None:
        queue_for_review(spec, row, asset)
    entry.update(status="done", finished_at=now(),
                 asset=_relative(asset, directory) if asset else None,
                 sidecar=_relative(sidecar, directory) if sidecar else None)
    save()
    log("done")


def _fail(entry, steps, name, reason, save, log) -> None:
    steps[name] = {"status": "failed", "reason": reason, "finished_at": now()}
    entry.update(status="failed", reason=f"{name}: {reason}", finished_at=now())
    save()
    log(f"failed: {entry['reason']}")


def queue_for_review(spec: BatchSpec, row: BatchRow, asset: Path) -> None:
    """Put a finished row in the review queue: one `queued` line, never rewritten.

    The key is the row's id plus the asset's sha256, so a new version of the same asset
    is told apart and the same version is not queued twice. When a `checks` step left a
    `<asset>.checks.json` beside it, its overall status comes along.
    """
    log_path = spec.output_dir.parent / REVIEW_LOG
    sha = sha256_file(asset)
    for line in _lines(log_path):
        if (line.get("event") == "queued" and line.get("asset_key") == row.id
                and line.get("sha256") == sha):
            return
    report = asset.with_suffix(".checks.json")
    checks = read_json(report) if report.is_file() else None
    event = {
        "event": "queued", "time": now(), "asset_key": row.id, "sha256": sha,
        "asset": _relative(asset, log_path.parent), "kind": row.kind,
        "profile": row.profile, "source": f"batch:{spec.name}",
        "status": checks.get("status") if isinstance(checks, dict) else None,
        "checks": _relative(report, log_path.parent) if checks is not None else None,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


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


def review_state(log_path: Path, asset_key: str, sha256: str | None) -> dict[str, Any] | None:
    """The latest review event for this version of an asset: queued, approved or
    sent_back (with its note). Another batch's row with the same id, or an older version,
    has another sha256 and does not count; an event that names no hash goes by the key."""
    latest = None
    for line in _lines(log_path):
        if line.get("asset_key") == asset_key and line.get("sha256") in (None, sha256):
            latest = line
    return latest


def _sidecar_for(asset: Path) -> Path | None:
    """The licence record kept beside an existing GLB, if it has one."""
    for candidate in (asset.with_suffix(".provenance.json"), asset.with_suffix(".json")):
        if candidate.is_file():
            return candidate
    return None


def _spec_record(spec: BatchSpec) -> dict[str, Any]:
    return {
        "schema_version": 1, "name": spec.name, "source": str(spec.path),
        "intent": spec.intent, "license_policy": spec.license_policy,
        "rows": [{"id": row.id, "images": [str(image) for image in row.images],
                  "asset": str(row.asset) if row.asset else None, "kind": row.kind,
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
        listed.append({
            "id": row.id, "kind": row.kind, "profile": row.profile, "faces": row.faces,
            "size_cm": row.size_cm, "measure": row.measure,
            "asset": entry.get("asset"), "sidecar": entry.get("sidecar"),
            "references": [_relative(inputs / image.name, directory)
                           for image in row.images if (inputs / image.name).is_file()],
            "steps": {name: step.get("status") for name, step in entry["steps"].items()},
            "files": [record["path"] for step in entry["steps"].values()
                      for record in step.get("files", [])],
            "finished_at": entry.get("finished_at"),
        })
    return {"schema_version": 1, "batch": spec.name, "updated_at": now(), "rows": listed}


# ------------------------------------------------------------------------------ CLI

def run_from_cli(spec_path: Path, intent: dict[str, Any], retry_failed: bool) -> int:
    """`pipeline.py --run-batch`: 0 all done, 1 some rows failed, 2 the batch cannot start,
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
    return report.exit_code
