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
    POST /api/slot-parts/grip            {"set", "body", "slot", "at": [x, y], "class", "size_cm"}: a
                                         weapon's or shield's grip, corrected (slot_parts/grip.py)
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
import blender_edit_api

from image_to_3dlab import piece_materials, processes
from image_to_3dlab.slot_parts import runs, slots

REPO = Path(__file__).resolve().parents[1]
HEADER = "slot-parts"
MAX_BODY = 64 * 1024


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
                              clips, payload.get("labels") or {}, payload.get("dressing"),
                              str(payload.get("dressing_words") or ""))


def _checked(output: Path, payload: dict, action: str = "run") -> dict:
    """The page's request with its concept, if any, found inside the output folder. Mode 2
    needs one only for pieces drawn on the body: weapons and shields alone have none."""
    drawn = any(slot in slots.DRAWN for slot in payload.get("slots") or {})
    if (action == "run" and int(payload.get("mode", 3)) == 2 and not payload.get("resume")
            and (drawn or payload.get("concept"))):
        return {**payload, "concept": _inside(output, payload.get("concept"))}
    return payload


def save_grip(output: Path, payload: dict) -> dict:
    """A person's grip for a set's weapon or shield (the marker, the class, the length); the same
    function the MCP server's pieces_grip and `slot_parts.py grip` call."""
    return runs.save_grip(output, payload)


def save_materials(output: Path, payload: dict) -> dict:
    """A person's materials for pieces of a set; every piece's material afterwards (the same
    function the MCP server's pieces_materials calls)."""
    return runs.save_materials(output, payload)


def resume_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    return runs.resume_command(output, payload)


def run_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """The `slot_parts.py run` command for a page request, and its log file."""
    if not payload.get("resume") and int(payload.get("mode", 3)) not in (2, 3):
        raise ValueError("mode 1 (one piece) is + Create; this page makes modes 2 and 3")
    return runs.run_command(output, _checked(output, payload))


def job_status(output: Path) -> dict[str, Any]:
    """The last run as `output/sets/.run.json` records it: one started before this server
    did, or by an agent, is seen and can be stopped the same way."""
    return runs.status(output)


def start_run(output: Path, payload: dict, popen=subprocess.Popen) -> dict[str, Any]:
    action = str(payload.get("action") or ("resume" if payload.get("resume") else "run"))
    if action not in runs.COMMANDS:
        raise ValueError(f"unknown action {action!r}: {', '.join(runs.COMMANDS)}")
    if action == "run" and int(payload.get("mode", 3)) not in (2, 3):
        raise ValueError("mode 1 (one piece) is + Create; this page makes modes 2 and 3")
    runs.start(output, action, _checked(output, payload, action), popen=popen)
    return job_status(output)


def cancel_run(output: Path, stop=processes.terminate_group) -> dict[str, Any]:
    """Stop the run and every process under it, then wait for it (up to 30 s), so a new run
    cannot start while the old one's Blender or Pixal3D still holds the GPU."""
    return runs.cancel(output, stop=stop, wait=30.0)


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
        blender_edit_api.check_request(handler, mutation=mutation, header=HEADER,
                                       refusal="Use this Studio page to make sets")
        if not mutation and parts == ["api", "slot-parts"]:
            handler._send_json(200, {"base": "/output/", "bodies": list_bodies(output), "sets": list_sets(output),
                                     "slots": catalogue(), "materials": list(piece_materials.MATERIALS),
                                     "job": job_status(output)})
        elif mutation and parts == ["api", "slot-parts", "bodies"]:
            handler._send_json(201, register_body(output, _body(handler)))
        elif mutation and parts == ["api", "slot-parts", "run"]:
            handler._send_json(202, start_run(output, _body(handler)))
        elif mutation and parts == ["api", "slot-parts", "run", "cancel"]:
            handler._send_json(200, cancel_run(output))
        elif mutation and parts == ["api", "slot-parts", "grip"]:
            handler._send_json(200, save_grip(output, _body(handler)))
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
