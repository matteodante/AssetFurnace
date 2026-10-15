"""Studio endpoints for game-ready materials: run the step on one GLB, read what it made.

    GET  /api/materials?file=<path under output/>   -> {"file", "run", "job"}
    POST /api/materials/run  {"file": ...}          -> 202 {"job"}

The step is ``scripts/game_materials.py`` in its own process: it is CPU work on 4K
textures and should not hold the server. Each source GLB has one stable run folder,
``output/materials/<stem>__<hash>/``, so running it again replaces the old result.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from blender_edit_api import body, check_request, private_file

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "game_materials.py"
FOLDER = "materials"
HEADER = "materials"
JOBS: dict[str, dict] = {}
STATES = {0: "done", 2: "sent_back"}


def source_path(output: Path, relative: str) -> Path:
    """The GLB a request names, refused unless it is an existing .glb inside output/."""
    if not relative or relative.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", relative):
        raise ValueError("Pick a model from the Library")
    root = output.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".glb" or not path.is_file():
        raise ValueError("That model is not in the Library")
    if private_file(output, root / relative):
        raise ValueError("That file is a private editing copy; use a saved version instead")
    return path


def run_dir(output: Path, relative: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_-]", "-", Path(relative).stem)[:60]
    digest = hashlib.sha1(relative.replace("\\", "/").encode()).hexdigest()[:8]
    return output / FOLDER / f"{stem}__{digest}"


def latest_run(output: Path, relative: str) -> dict | None:
    """The sidecar of this GLB's last run, with every file as a path under output/."""
    folder = run_dir(output, relative)
    sidecar = folder / f"{Path(relative).stem}.materials.json"
    if not sidecar.is_file():
        return None
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    prefix = folder.relative_to(output).as_posix()
    files = {key: f"{prefix}/{name}" for key, name in (data.get("textures") or {}).items()
             if isinstance(name, str)}
    if data.get("glb"):
        files["glb"] = f"{prefix}/{data['glb']}"
        provenance = folder / f"{Path(data['glb']).stem}.provenance.json"
        if provenance.is_file():
            files["provenance"] = f"{prefix}/{provenance.name}"
    files["sidecar"] = f"{prefix}/{sidecar.name}"
    data["files"] = files
    for material in data.get("materials") or []:
        material["files"] = {key: f"{prefix}/{name}" for key, name in (material.get("textures") or {}).items()}
    data["made"] = sidecar.stat().st_mtime
    return data


def job_state(output: Path, relative: str) -> dict:
    job = JOBS.get(relative)
    if job is None:
        return {"state": "idle"}
    code = job["process"].poll()
    state = "running" if code is None else STATES.get(code, "failed")
    result = {"state": state, "started": job["started"]}
    if state == "failed":
        log = run_dir(output, relative) / "run.log"
        tail = log.read_text(encoding="utf-8", errors="replace")[-1500:] if log.is_file() else ""
        result["error"] = tail.strip().splitlines()[-1] if tail.strip() else "The materials step stopped."
        result["log"] = tail
    return result


def start(output: Path, relative: str, popen=None) -> dict:
    source = source_path(output, relative)
    job = JOBS.get(relative)
    if job is not None and job["process"].poll() is None:
        raise RuntimeError("Materials are already being made for this model")
    folder = run_dir(output, relative)
    folder.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(SCRIPT), str(source), str(folder)]
    with (folder / "run.log").open("wb") as log:
        process = (popen or subprocess.Popen)(args, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=subprocess.STDOUT)
    JOBS[relative] = {"process": process, "started": time.time()}
    return job_state(output, relative)


def _discard_body(handler, limit: int = 65536) -> None:
    """Read a small request body a refusal left unread, so closing does not reset the
    client mid-send. Only for refusals raised before `body` ran."""
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        return
    if 0 < length <= limit:
        handler.rfile.read(length)


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    url = urlsplit(handler.path)
    if not url.path.startswith("/api/materials"):
        return False
    try:
        check_request(handler)
        if mutation and handler.headers.get("X-AssetFurnace-Request") != HEADER:
            raise PermissionError("Use this Studio page to make materials")
        if not mutation and url.path == "/api/materials":
            relative = parse_qs(url.query).get("file", [""])[0]
            source_path(output, relative)
            handler._send_json(200, {"file": relative, "run": latest_run(output, relative),
                                     "job": job_state(output, relative)})
        elif mutation and url.path == "/api/materials/run":
            if handler.headers.get_content_type() != "application/json":
                raise ValueError("Expected JSON")
            relative = json.loads(body(handler, 4096)).get("file", "")
            handler._send_json(202, {"file": relative, "job": start(output, relative)})
        else:
            handler._send_json(404, {"error": "Unknown materials endpoint"})
    except PermissionError as exc:
        _discard_body(handler)
        handler.close_connection = True
        handler._send_json(403, {"error": str(exc)})
    except RuntimeError as exc:
        handler._send_json(409, {"error": str(exc)})
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        handler.close_connection = True
        handler._send_json(422, {"error": str(exc)})
    return True
