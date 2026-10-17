"""The studio's Characters in pieces page (feature 4).

    GET  /api/slot-parts                 base bodies, sets made so far, the slot catalogue, the run
    POST /api/slot-parts/bodies          {"name", "picture", "model", "rigged", "clips", "labels"}:
                                         a Library character (paths under output/) as a base body
    POST /api/slot-parts/run             {"mode": 2|3, "body", "set", "slots": {slot: options},
                                         "outfit" (mode 3) | "concept" (mode 2, under output/),
                                         "weapon", "shield", "style", "seeds"}
    POST /api/slot-parts/run/cancel

Mode 1 (one piece) is the studio's usual + Create; it needs nothing here. A run is
`scripts/slot_parts.py run` in the background, one at a time, its output in
`output/sets/<set>/<body>/run.log`. Requests are checked like the Batches page: the local
Studio address, same origin, and a custom header on anything that starts or stops work.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import batch_api

from image_to_3dlab import processes
from image_to_3dlab.slot_parts import body as body_mod
from image_to_3dlab.slot_parts import slots

REPO = Path(__file__).resolve().parents[1]
HEADER = "slot-parts"
MAX_BODY = 64 * 1024
_JOB: dict[str, Any] = {}


def _inside(output: Path, relative: str | None) -> Path:
    """A path the page gave, which must name a file under output/."""
    if not relative:
        raise ValueError("a file is missing")
    path = (output / str(relative).lstrip("/\\")).resolve()
    if output.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"{relative} is not a file in the studio's output folder")
    return path


def rel(output: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(output.resolve()).as_posix()


def list_bodies(output: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted((output / "bodies").glob("*/body.json")):
        try:
            record = body_mod.load(path)
        except (OSError, ValueError):
            continue
        image = Path(record["image"])
        found.append({"name": record["name"], "labels": record.get("labels", {}), "clips": sorted(record.get("clips", {})),
                      "picture": rel(output, image) if output.resolve() in image.resolve().parents else None})
    return found


def list_sets(output: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted((output / "sets").glob("*/*/*.parts.json")):
        try:
            parts = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        preview = path.parent / "fit" / "preview.glb"
        checks = parts.get("checks") or {}
        found.append({"set": parts.get("set"), "body": (parts.get("body") or {}).get("name"), "mode": parts.get("mode"),
                      "pieces": sorted(parts.get("pieces") or {}), "created": parts.get("created"),
                      "covered": (parts.get("fit") or {}).get("covered_share"),
                      "statuses": {k: v.get("status") for k, v in checks.items()},
                      "preview": rel(output, preview) if preview.is_file() else None, "record": rel(output, path)})
    found.sort(key=lambda s: s["created"] or "", reverse=True)
    return found


def catalogue() -> list[dict[str, Any]]:
    return [{"name": s.name, "paired": s.paired, "drawn": bool(s.roles), "held": s.name in ("weapon", "shield")}
            for s in slots.CATALOGUE.values()]


def register_body(output: Path, payload: dict) -> dict:
    picture = _inside(output, payload.get("picture"))
    model = _inside(output, payload.get("model"))
    rigged = _inside(output, payload.get("rigged"))
    clips = {str(c["name"]): _inside(output, c["file"]) for c in payload.get("clips") or [] if c.get("file")}
    labels = {str(k): str(v) for k, v in (payload.get("labels") or {}).items() if str(k).strip()}
    path = body_mod.register(str(payload.get("name", "")), picture, rigged, model.with_suffix(".svviews"), None,
                             labels, output / "bodies", clips)
    return {"name": body_mod.load(path)["name"], "record": rel(output, path)}


def run_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """The `slot_parts.py run` command for a page request, and its log file."""
    mode = int(payload.get("mode", 3))
    if mode not in (2, 3):
        raise ValueError("mode 1 (one piece) is + Create; this page makes modes 2 and 3")
    body_json = output / "bodies" / str(payload.get("body", "")) / "body.json"
    if not body_mod.NAME.match(str(payload.get("body", ""))) or not body_json.is_file():
        raise ValueError("choose a registered base body")
    set_name = str(payload.get("set", ""))
    if not body_mod.NAME.match(set_name):
        raise ValueError("give the set a name: letters, digits, '.', '_' or '-'")
    raw = {k: v for k, v in (payload.get("slots") or {}).items() if k in slots.DRAWN}
    chosen = slots.choose(raw) if raw else {}
    seeds = [int(s) for s in str(payload.get("seeds", "3")).split(",") if s.strip()][:5] or [3]
    folder = output / "sets" / set_name / body_mod.load(body_json)["name"]
    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "slots.json"
    spec.write_text(json.dumps(chosen, indent=1) + "\n", encoding="utf-8")
    command = [sys.executable, str(REPO / "scripts" / "slot_parts.py"), "run", "--body", str(body_json),
               "--set", set_name, "--slots-json", str(spec), "--seeds", ",".join(map(str, seeds)),
               "--concept-seeds", str(seeds[0]), "--root", str(output / "sets")]
    if mode == 2:
        command += ["--concept", str(_inside(output, payload.get("concept")))]
    else:
        outfit = str(payload.get("outfit", "")).strip()
        if not outfit:
            raise ValueError("describe the outfit to draw on the body")
        command += ["--outfit", outfit[:600]]
    for held in ("weapon", "shield"):
        words = str(payload.get(held) or "").strip()
        if words:
            command += [f"--{held}", words[:300]]
    if str(payload.get("style") or "").strip():
        command += ["--style", str(payload["style"]).strip()[:300]]
    if not chosen and not any(f"--{h}" in command for h in ("weapon", "shield")):
        raise ValueError("choose at least one slot, a weapon or a shield")
    return command, folder / "run.log"


def job_status(output: Path) -> dict[str, Any]:
    if not _JOB:
        return {"running": False}
    process = _JOB["process"]
    code = process.poll()
    log = Path(_JOB["log"])
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:] if log.is_file() else []
    parts = Path(_JOB["folder"]) / f"{_JOB['set']}.parts.json"
    return {"running": code is None, "returncode": code, "set": _JOB["set"], "body": _JOB["body"],
            "started": _JOB["started"], "log": tail,
            "parts": rel(output, parts) if parts.is_file() and parts.stat().st_mtime >= _JOB["started"] else None}


def start_run(output: Path, payload: dict, popen=subprocess.Popen) -> dict[str, Any]:
    if _JOB and _JOB["process"].poll() is None:
        raise ValueError(f"a set is already being made ({_JOB['set']}); wait or stop it")
    command, log = run_command(output, payload)
    handle = log.open("w", encoding="utf-8")
    # its own process group: Stop must reach the edits, Pixal3D and Blender it starts, not
    # only the script that starts them
    process = popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO), **processes.group_popen_kwargs())
    _JOB.clear()
    _JOB.update(process=process, log=str(log), folder=str(log.parent), set=payload["set"], body=payload["body"],
                started=time.time())
    return job_status(output)


def cancel_run(stop=processes.terminate_group) -> dict[str, Any]:
    """Stop the run and every process under it, then wait for it, so a new run cannot start
    while the old one's Blender or Pixal3D still holds the GPU."""
    if _JOB and _JOB["process"].poll() is None:
        process = _JOB["process"]
        stop(process.pid)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    return {"stopped": True}


def _body(handler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if not 0 < length <= MAX_BODY:
        return {}
    return json.loads(handler.rfile.read(length))


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    """Answer a /api/slot-parts request; False for any other path, so the caller goes on."""
    path = urlsplit(handler.path).path
    if path != "/api/slot-parts" and not path.startswith("/api/slot-parts/"):
        return False
    parts = [unquote(p) for p in path.split("/") if p]
    try:
        batch_api.check_request(handler, mutation=False)  # local address, same origin
        if mutation and (not handler.headers.get("Origin") or handler.headers.get("X-AssetFurnace-Request") != HEADER):
            raise PermissionError("Use this Studio page to make sets")
        if not mutation and parts == ["api", "slot-parts"]:
            handler._send_json(200, {"base": "/output/", "bodies": list_bodies(output), "sets": list_sets(output),
                                     "slots": catalogue(), "job": job_status(output)})
        elif mutation and parts == ["api", "slot-parts", "bodies"]:
            handler._send_json(201, register_body(output, _body(handler)))
        elif mutation and parts == ["api", "slot-parts", "run"]:
            handler._send_json(202, start_run(output, _body(handler)))
        elif mutation and parts == ["api", "slot-parts", "run", "cancel"]:
            handler._send_json(200, cancel_run())
        else:
            handler._send_json(404, {"error": "Unknown endpoint"})
    except PermissionError as exc:
        batch_api._drain(handler)
        handler.close_connection = True
        handler._send_json(403, {"error": str(exc)})
    except (ValueError, KeyError, TypeError, OSError, LookupError) as exc:
        handler.close_connection = True
        handler._send_json(422, {"error": str(exc)})
    return True
