"""Studio endpoints for Export for Unreal: start an export, follow it, take the files.

    GET  /api/unreal-export                  what this machine can do (Blender, Unreal)
    POST /api/unreal-export/jobs             start one: {file, kind, size_cm, measure, ...}
    GET  /api/unreal-export/jobs/<id>        its state, then the files and the checks
    GET  /api/unreal-export/jobs/<id>/zip    every file it wrote, in one download
    POST /api/unreal-export/jobs/<id>/reveal open the folder (only on this computer)

The work is `scripts/export_unreal.py`'s `export_unreal()`, then optionally
`scripts/unreal_import_check.py`; this file only checks the request and runs them in the
background. Every path the page sends is relative to `output/` and must stay inside it.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import blender_edit_api

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_to_3dlab import unreal_profile as up  # noqa: E402
from image_to_3dlab.blender import find_blender  # noqa: E402

HEADER = "unreal-export"
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
MAX_BODY = 16384


def resolve(output: Path, relative: str) -> Path:
    """`relative` (as the Library spells it) inside `output/`, a GLB that exists."""
    if not isinstance(relative, str) or not relative or "\\" in relative or relative.startswith("/"):
        raise ValueError("Choose a model from the Library")
    root = output.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".glb" or not path.is_file():
        raise ValueError(f"{relative} is not a model in the Library")
    return path


def parse(output: Path, data: dict) -> dict:
    """The page's request as `export_unreal()` arguments, refusing anything unexpected."""
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    kind = data.get("kind")
    if kind not in up.KINDS:
        raise ValueError("Choose what this is: object, creature or equipment piece")
    try:
        size = float(data.get("size_cm"))
    except (TypeError, ValueError):
        raise ValueError("Give the real size in centimetres") from None
    if not 0.1 <= size <= 100000:
        raise ValueError("Give the real size in centimetres, between 0.1 and 100,000")
    measure = data.get("measure", "height")
    if measure not in up.MEASURES:
        raise ValueError("Measure the height or the longest side")
    request = {"source": resolve(output, data.get("file")), "kind": kind, "size_cm": size, "measure": measure,
               "animations": [resolve(output, a) for a in data.get("animations") or []] if kind == "creature" else [],
               "check": bool(data.get("check"))}
    if kind == "equipment":
        request["body"] = resolve(output, data.get("body"))
    name = data.get("name")
    if name:
        request["name"] = up.base_name(str(name))
    return request


def run_job(output: Path, job_id: str, request: dict, export=None, check=None) -> None:
    """The background half: export, then check in Unreal when asked. Never raises."""
    from export_unreal import default_base

    if export is None:
        from export_unreal import export_unreal as export
    job = JOBS[job_id]
    try:
        base = request.get("name") or default_base(request["source"], request["kind"], request.get("animations"))
        out_dir = output / "unreal" / f"{base}__{time.strftime('%Y%m%d-%H%M%S')}"
        arguments = {k: v for k, v in request.items() if k not in ("source", "check")}
        job.update(state="exporting", message="Reshaping the model in Blender…")
        result = export(request["source"], out_dir=out_dir, log=lambda message: None, **arguments)
        job.update(result=result, folder=str(Path(result["out_dir"]).resolve()))
        if request["check"]:
            if check is None:
                from unreal_import_check import check
            job.update(state="checking", message="Importing into Unreal and checking (about two minutes)…")
            job["checked"] = bool(check([Path(result["out_dir"])]))
        job.update(state="done", message="Ready for Unreal." if result["ok"] else "Exported, with problems.")
    except (ValueError, RuntimeError, OSError, SystemExit) as exc:
        job.update(state="error", message=str(exc) or type(exc).__name__)


def start(output: Path, request: dict, runner=None) -> dict:
    job_id = uuid.uuid4().hex
    with LOCK:
        if any(j["state"] in ("queued", "exporting", "checking") for j in JOBS.values()):
            raise RuntimeError("An export is already running; wait for it to finish")
        JOBS[job_id] = {"id": job_id, "state": "queued", "message": "Starting…", "created": time.time()}
    (runner or (lambda: threading.Thread(target=run_job, args=(output, job_id, request), daemon=True).start()))()
    return {"id": job_id}


def files(output: Path, job: dict) -> list[dict]:
    folder = Path(job.get("folder") or "")
    if not job.get("folder") or not folder.is_dir():
        return []
    root = output.resolve()
    return [{"name": p.name, "path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size}
            for p in sorted(folder.iterdir()) if p.is_file()]


def status(output: Path, job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise KeyError(job_id)
    view = {k: job[k] for k in ("id", "state", "message") if k in job}
    view["files"] = files(output, job)
    result = job.get("result") or {}
    view["problems"] = result.get("problems", [])
    folder = Path(job.get("folder") or "")
    report = folder / "unreal_check.json"
    if job.get("folder") and report.is_file():
        view["unreal"] = json.loads(report.read_text(encoding="utf-8"))
    return view


def zip_bytes(output: Path, job_id: str) -> tuple[str, bytes]:
    job = JOBS.get(job_id)
    if job is None or job.get("state") != "done":
        raise KeyError(job_id)
    folder = Path(job["folder"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                archive.write(path, f"{folder.name}/{path.name}")
    return f"{folder.name}.zip", buffer.getvalue()


def local_desktop(handler) -> bool:
    """Opening a folder only makes sense for someone sitting at this computer."""
    host = urlsplit("//" + handler.headers.get("Host", "")).hostname or ""
    if not blender_edit_api.local_address(handler.client_address[0]) or not blender_edit_api.local_address(host):
        return False
    return not any(os.environ.get(key) for key in ("SSH_CONNECTION", "SSH_CLIENT", "RUNPOD_POD_ID"))


def reveal(folder: Path, popen=subprocess.Popen) -> None:
    if sys.platform.startswith("win"):
        os.startfile(folder)  # a folder the server itself wrote, opened on this computer
    else:
        popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])


def capabilities(handler) -> dict:
    editor = up.find_unreal_editor()
    local = local_desktop(handler)
    return {"blender": find_blender() is not None, "unreal": editor is not None and local,
            "unreal_version": editor.parts[-5] if editor is not None else None,
            "can_reveal": local, "kinds": list(up.KINDS), "measures": list(up.MEASURES)}


def _check(handler, mutation: bool) -> None:
    blender_edit_api.check_request(handler, mutation=False)
    if mutation and (not handler.headers.get("Origin") or handler.headers.get("X-AssetFurnace-Request") != HEADER):
        raise PermissionError("Use this Studio page to export models")


def _drain(handler) -> None:
    """Read a small refused body first: closing on a client still sending it aborts the
    connection on Windows, and the page sees a network error instead of the reason."""
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        return
    if 0 < length <= MAX_BODY and not handler.headers.get("Transfer-Encoding"):
        handler.rfile.read(length)


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    path = urlsplit(handler.path).path
    if not path.startswith("/api/unreal-export"):
        return False
    parts = path.strip("/").split("/")
    try:
        _check(handler, mutation)
        if not mutation and parts == ["api", "unreal-export"]:
            handler._send_json(200, capabilities(handler))
        elif mutation and parts == ["api", "unreal-export", "jobs"]:
            if handler.headers.get_content_type() != "application/json":
                raise ValueError("Expected JSON")
            request = parse(output, json.loads(blender_edit_api.body(handler, MAX_BODY)))
            if request["check"] and not capabilities(handler)["unreal"]:
                raise ValueError("Unreal Engine 5 is not installed on this computer; export without the check")
            handler._send_json(202, start(output, request))
        elif not mutation and len(parts) == 4 and parts[2] == "jobs":
            handler._send_json(200, status(output, parts[3]))
        elif not mutation and len(parts) == 5 and parts[2] == "jobs" and parts[4] == "zip":
            name, data = zip_bytes(output, parts[3])
            handler.send_response(200)
            handler.send_header("Content-Type", "application/zip")
            handler.send_header("Content-Disposition", f'attachment; filename="{name}"')
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        elif mutation and len(parts) == 5 and parts[2] == "jobs" and parts[4] == "reveal":
            if not local_desktop(handler):
                raise ValueError("The folder is on the server; download the zip instead")
            job = JOBS.get(parts[3])
            if not job or not job.get("folder"):
                raise KeyError(parts[3])
            reveal(Path(job["folder"]))
            handler._send_json(200, {"opened": True})
        else:
            handler._send_json(404, {"error": "Unknown Unreal export endpoint"})
    except PermissionError as exc:
        _drain(handler)
        handler.close_connection = True
        handler._send_json(403, {"error": str(exc)})
    except KeyError:
        handler._send_json(404, {"error": "No such export; it may be from before the Studio restarted"})
    except RuntimeError as exc:
        handler._send_json(409, {"error": str(exc)})
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        handler.close_connection = True
        handler._send_json(422, {"error": str(exc)})
    return True
