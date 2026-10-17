"""The studio's Characters in pieces page (feature 4).

    GET  /api/slot-parts                 base bodies, sets made so far, the slot catalogue, the run
    POST /api/slot-parts/bodies          {"name", "picture", "model", "rigged", "clips", "labels"}:
                                         a Library character (paths under output/) as a base body
    POST /api/slot-parts/run             {"mode": 2|3, "body", "set", "slots": {slot: options},
                                         "outfit" (mode 3) | "concept" (mode 2, under output/),
                                         "weapon", "shield", "style", "seeds", "choose"}: with
                                         "choose" and several seeds, the run stops once every
                                         piece's pictures are made and scored;
                                         {"resume": true, "body", "set", "picks": {slot: seed}}
                                         goes on from there with the person's picks
                                         ("concept": seed when the run waits on the outfits, asked
                                         with "concept_seeds": "3,4,5" and "choose_concept");
                                         {"action": "fit" | "refit" | "unreal", "body", "set",
                                         "to" (refit), "size_cm" (unreal)}: fit a made set again,
                                         carry it to another body, export it for Unreal
    POST /api/slot-parts/run/cancel
    POST /api/slot-parts/materials       {"set", "body", "pieces": {slot: material | null}}: a
                                         person's material for pieces of a made set (null: back
                                         to the rule's proposal); answers every piece's material

Mode 1 (one piece) is the studio's usual + Create; it needs nothing here. A run is
`scripts/slot_parts.py run` in the background, one at a time, its output in
`output/sets/<set>/<body>/run.log`. Requests are checked like the Batches page: the local
Studio address, same origin, and a custom header on anything that starts or stops work.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import batch_api

from image_to_3dlab import piece_materials, processes
from image_to_3dlab.slot_parts import body as body_mod
from image_to_3dlab.slot_parts import runs

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


# The pure parts are shared with the MCP server: one place builds the commands.
rel = runs.rel
list_bodies = runs.list_bodies
list_sets = runs.list_sets
catalogue = runs.catalogue
waiting_picks = runs.waiting_picks


def register_body(output: Path, payload: dict) -> dict:
    clips = {str(c["name"]): _inside(output, c["file"]) for c in payload.get("clips") or [] if c.get("file")}
    return runs.register_body(output, str(payload.get("name", "")), _inside(output, payload.get("picture")),
                              _inside(output, payload.get("model")), _inside(output, payload.get("rigged")),
                              clips, payload.get("labels") or {})


def _checked(output: Path, payload: dict, action: str = "run") -> dict:
    """The page's request with its concept, if any, found inside the output folder."""
    if action == "run" and int(payload.get("mode", 3)) == 2 and not payload.get("resume"):
        return {**payload, "concept": _inside(output, payload.get("concept"))}
    return payload


def made_set(output: Path, payload: dict) -> Path:
    """A made set's folder, `output/sets/<set>/<body>/`, from the page's names."""
    body, set_name = str(payload.get("body", "")), str(payload.get("set", ""))
    if not body_mod.NAME.match(body) or not body_mod.NAME.match(set_name):
        raise ValueError("name a set and its body: letters, digits, '.', '_' or '-'")
    folder = output / "sets" / set_name / body
    if not any(folder.glob("*.parts.json")):
        raise ValueError(f"{set_name} on {body} is not a finished set")
    return folder


def save_materials(output: Path, payload: dict) -> dict:
    """A person's materials for pieces of a set; every piece's material afterwards."""
    folder = made_set(output, payload)
    pieces = payload.get("pieces")
    if not isinstance(pieces, dict) or not pieces:
        raise ValueError("give the pieces' materials, e.g. {\"helmet\": \"leather\"}")
    piece_materials.save_corrections(folder, {str(k): (None if v in (None, "") else str(v)) for k, v in pieces.items()})
    return {"set": payload["set"], "body": payload["body"], "materials": piece_materials.set_materials(folder)}


def resume_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    return runs.resume_command(output, payload)


def run_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """The `slot_parts.py run` command for a page request, and its log file."""
    if not payload.get("resume") and int(payload.get("mode", 3)) not in (2, 3):
        raise ValueError("mode 1 (one piece) is + Create; this page makes modes 2 and 3")
    return runs.run_command(output, _checked(output, payload))


def job_status(output: Path) -> dict[str, Any]:
    if not _JOB:
        return {"running": False}
    code = _JOB["process"].poll()
    return {**runs.status(output, _JOB["record"], running=code is None), "returncode": code,
            "picks": waiting_picks(output, Path(_JOB["folder"])) if code == 0 else None}


def start_run(output: Path, payload: dict, popen=subprocess.Popen) -> dict[str, Any]:
    if _JOB and _JOB["process"].poll() is None:
        raise ValueError(f"a set is already being made ({_JOB['set']}); wait or stop it")
    action = str(payload.get("action") or ("resume" if payload.get("resume") else "run"))
    if action not in runs.COMMANDS:
        raise ValueError(f"unknown action {action!r}: {', '.join(runs.COMMANDS)}")
    if action == "run" and int(payload.get("mode", 3)) not in (2, 3):
        raise ValueError("mode 1 (one piece) is + Create; this page makes modes 2 and 3")
    process, record = runs.start(output, action, _checked(output, payload, action), popen=popen)
    _JOB.clear()
    _JOB.update(process=process, record=record, log=record["log"], folder=record["folder"], set=payload["set"],
                body=payload["body"], started=record["started"])
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
                                     "slots": catalogue(), "materials": list(piece_materials.MATERIALS),
                                     "job": job_status(output)})
        elif mutation and parts == ["api", "slot-parts", "bodies"]:
            handler._send_json(201, register_body(output, _body(handler)))
        elif mutation and parts == ["api", "slot-parts", "run"]:
            handler._send_json(202, start_run(output, _body(handler)))
        elif mutation and parts == ["api", "slot-parts", "run", "cancel"]:
            handler._send_json(200, cancel_run())
        elif mutation and parts == ["api", "slot-parts", "materials"]:
            handler._send_json(200, save_materials(output, _body(handler)))
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
