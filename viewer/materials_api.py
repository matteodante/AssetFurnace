"""Studio endpoints for the Game materials panel: the materials step on the model shown.

    GET  /api/materials?file=        the runs made from that model (packed GLB, tint mask, zones)
    POST /api/materials/jobs         start one: {file, zones, materials, zones_file, zones_mask}
    GET  /api/materials/jobs/<id>    its state

`zones` is "off" (pack the model's own maps as an ORM, no tint mask), a segmenter
("sam31", "sam3", "sam2": the GPU, under its lock; Qwen3.5 names the materials unless
`materials` lists them), or "file": zones of the person's own, `zones_file` ({"name",
"text"}: a JSON {zone: material}, or an earlier run's `.materials.json` corrected by hand)
with `zones_mask` ({"name", "data"}: base64 of the PNG of its zone indices in UV space, or
that run's `_Tint.png`). The work is `image_to_3dlab.materials.make_game_materials`, as
`scripts/game_materials.py` and the batch step `materials` run it, into
`output/materials/<model>__materials__<time>/`; the files the person gave are kept there
under `input/`. Export for Unreal offers the tint masks made for the model it exports.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import blender_edit_api

from image_to_3dlab import material_zones
from image_to_3dlab.materials import SIDECAR_SCHEMA, make_game_materials
from image_to_3dlab.piece_materials import MATERIALS

HEADER = "materials"
FOLDER = "materials"
RUN = re.compile(r"^[A-Za-z0-9_-]{1,80}__materials__\d{8}-\d{6}(?:-\d+)?$")
ZONES = ("off", "sam31", "sam3", "sam2", "file")
MAX_BODY = 48 * 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()


def resolve(output: Path, relative: Any) -> Path:
    """A GLB of the Library, by the path the page gives (relative to output/)."""
    if not isinstance(relative, str) or not relative or "\\" in relative or relative.startswith("/"):
        raise ValueError("Choose a model from the Library")
    root = output.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".glb" or not path.is_file():
        raise ValueError(f"{relative} is not a model in the Library")
    return path


def _upload(given: Any, key: str, suffix: str) -> tuple[str, bytes]:
    if not isinstance(given, dict) or not isinstance(given.get("name"), str):
        raise ValueError(f"{key} is missing")
    name = Path(given["name"]).name
    if not name.lower().endswith(suffix):
        raise ValueError(f"the {key.replace('_', ' ')} must be a {suffix} file")
    if "text" in given:
        return name, str(given["text"]).encode("utf-8")
    try:
        return name, base64.b64decode(str(given.get("data") or ""), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"the {key.replace('_', ' ')} could not be read") from None


def parse(output: Path, data: Any) -> dict[str, Any]:
    """The page's request, checked: what `start` runs. Raises ValueError with what is wrong."""
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    model = resolve(output, data.get("file"))
    zones = data.get("zones") or "off"
    if zones not in ZONES:
        raise ValueError(f"zones is one of {', '.join(ZONES)}")
    words = [w.strip() for w in str(data.get("materials") or "").split(",") if w.strip()]
    wrong = [w for w in words if w not in MATERIALS]
    if wrong:
        raise ValueError(f"{', '.join(wrong)} is not one of the materials ({', '.join(MATERIALS)})")
    request = {"model": model, "file": data["file"], "zones": zones, "materials": words or None}
    if zones == "file":
        if not data.get("zones_file"):
            raise ValueError("choose your zones file: a JSON of zones, or an earlier run's .materials.json")
        if not data.get("zones_mask"):
            raise ValueError("choose the zones' mask: the PNG of the zones in UV space, or that run's _Tint.png")
        request["zones_file"] = _upload(data["zones_file"], "zones_file", ".json")
        request["zones_mask"] = _upload(data["zones_mask"], "zones_mask", ".png")
    elif zones != "off":
        problems = material_zones.readiness(zones, ask=not words)
        if problems:
            raise ValueError("; ".join(problems))
    return request


def _folder(output: Path, stem: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:60] or "model"
    folder = output / FOLDER / f"{slug}__materials__{stamp}"
    suffix = 2
    while folder.exists():
        folder = output / FOLDER / f"{slug}__materials__{stamp}-{suffix}"
        suffix += 1
    return folder


def _work(output: Path, job: dict[str, Any], request: dict[str, Any]) -> None:
    folder = Path(job["folder"])
    try:
        zones = record = None
        if request["zones"] == "file":
            given = {}
            for key in ("zones_file", "zones_mask"):
                name, data = request[key]
                path = folder / "input" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                given[key] = path
            zones, record = material_zones.zones_from_file(request["model"], given["zones_file"], given["zones_mask"])
        elif request["zones"] != "off":
            job["message"] = "Finding the tint zones (the GPU, a few minutes)"
            zones, record = material_zones.find_zones(request["model"], folder / "work", request["zones"],
                                                      request["materials"])
        job["message"] = "Packing the maps"
        result = make_game_materials(request["model"], folder, zones=zones, zone_record=record)
        job.update(status="done", message="", run=_describe(output, result.sidecar_path))
    except (ValueError, RuntimeError, OSError) as exc:
        job.update(status="error", error=str(exc))
        shutil.rmtree(folder, ignore_errors=True)   # nothing half-made is left or offered


def start(output: Path, request: dict[str, Any], background: bool = True) -> dict[str, Any]:
    """Start one run (one at a time); `background=False` runs it here, for tests."""
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise RuntimeError("a Game materials run is already working; wait for it")
        folder = _folder(output, request["model"].stem)
        folder.mkdir(parents=True)
        job = {"id": uuid.uuid4().hex, "status": "running", "file": request["file"], "folder": str(folder),
               "message": "Starting", "error": None, "run": None}
        JOBS[job["id"]] = job
    if background:
        threading.Thread(target=_work, args=(output, job, request), daemon=True, name=f"materials-{job['id'][:8]}").start()
    else:
        _work(output, job, request)
    return status(job["id"])


def status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    return {k: v for k, v in job.items() if k != "folder"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe(output: Path, record_path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SIDECAR_SCHEMA:
        return None
    folder = record_path.parent
    glb = folder / str(record.get("glb") or "")
    tint = (record.get("textures") or {}).get("tint")
    if not glb.is_file():
        return None

    def rel(path: Path) -> str:
        return path.resolve().relative_to(output.resolve()).as_posix()

    return {"folder": rel(folder), "glb": rel(glb), "record": rel(record_path),
            "tint": rel(folder / tint) if tint and (folder / tint).is_file() else None,
            "zones": [z.get("material") for z in record.get("zones") or [] if isinstance(z, dict)],
            "status": record.get("status"), "source_sha256": record.get("source_sha256"),
            "created": record_path.stat().st_mtime}


def runs_for(output: Path, relative: str) -> list[dict[str, Any]]:
    """The finished runs made from this model (by its bytes), newest first."""
    sha = _sha(resolve(output, relative))
    found = []
    root = output / FOLDER
    for folder in root.iterdir() if root.is_dir() else []:
        if not (folder.is_dir() and RUN.fullmatch(folder.name)):
            continue
        for record_path in folder.glob("*.materials.json"):
            run = _describe(output, record_path)
            if run and run["source_sha256"] == sha:
                found.append(run)
    return sorted(found, key=lambda r: r["created"], reverse=True)


def _check(handler, mutation: bool) -> None:
    blender_edit_api.check_request(handler, mutation=False)
    if mutation and (not handler.headers.get("Origin") or handler.headers.get("X-AssetFurnace-Request") != HEADER):
        raise PermissionError("Use this Studio page to make game materials")


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    """Answer an /api/materials request; False for any other path, so the caller goes on."""
    path = urlsplit(handler.path).path
    if path != "/api/materials" and not path.startswith("/api/materials/"):
        return False
    parts = path.strip("/").split("/")
    try:
        _check(handler, mutation)
        if not mutation and parts == ["api", "materials"]:
            query = parse_qs(urlsplit(handler.path).query)
            handler._send_json(200, {"runs": runs_for(output, (query.get("file") or [""])[0]),
                                     "materials": list(MATERIALS)})
        elif mutation and parts == ["api", "materials", "jobs"]:
            if handler.headers.get_content_type() != "application/json":
                raise ValueError("Expected JSON")
            handler._send_json(202, start(output, parse(output, json.loads(blender_edit_api.body(handler, MAX_BODY)))))
        elif not mutation and len(parts) == 4 and parts[2] == "jobs":
            handler._send_json(200, status(parts[3]))
        else:
            handler._send_json(404, {"error": "Unknown Game materials endpoint"})
    except PermissionError as exc:
        handler.close_connection = True
        handler._send_json(403, {"error": str(exc)})
    except KeyError:
        handler._send_json(404, {"error": "No such Game materials run"})
    except RuntimeError as exc:
        handler._send_json(409, {"error": str(exc)})
    except (ValueError, TypeError, OSError) as exc:
        handler.close_connection = True
        handler._send_json(422, {"error": str(exc)})
    return True
