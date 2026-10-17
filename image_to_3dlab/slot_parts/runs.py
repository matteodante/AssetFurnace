"""Starting and following `scripts/slot_parts.py` runs, for the studio and for agents.

The studio's Characters in pieces page (`viewer/slot_parts_api.py`) and the MCP server
(`image_to_3dlab/mcp_server.py`) both start the same script with the same arguments, built
here once: a whole set (`run`, modes 2 and 3), going on from the pictures with picks
(`run --resume --pick`), fitting again (`fit`), carrying a set to another body (`refit`)
and exporting it for Unreal (`unreal`). One run at a time on the machine: whoever starts
one records its process in `output/sets/.run.json`, and either route sees it.

Paths given to these functions are already resolved; the studio checks first that they
are inside its output folder.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from image_to_3dlab import processes
from image_to_3dlab.slot_parts import body as body_mod
from image_to_3dlab.slot_parts import candidates as candidates_mod
from image_to_3dlab.slot_parts import slots

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "slot_parts.py"
RUN_FILE = ".run.json"   # in output/sets/
MAX_SEEDS = 5


def rel(output: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(Path(output).resolve()).as_posix()


# ---------------------------------------------------------------------- what exists

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


def register_body(output: Path, name: str, picture: Path, model: Path, rigged: Path,
                  clips: dict[str, Path] | None = None, labels: dict[str, str] | None = None) -> dict:
    """A rigged character as a base body; `model` is the Pixal3D model it was rigged from,
    whose saved camera puts the bones on the picture."""
    labels = {str(k): str(v) for k, v in (labels or {}).items() if str(k).strip()}
    path = body_mod.register(str(name), Path(picture), Path(rigged), Path(model).with_suffix(".svviews"), None,
                             labels, output / "bodies", {str(k): Path(v) for k, v in (clips or {}).items()})
    return {"name": body_mod.load(path)["name"], "record": rel(output, path)}


# ------------------------------------------------------------------------ commands

def _body_json(output: Path, name: Any) -> Path:
    body_json = output / "bodies" / str(name or "") / "body.json"
    if not body_mod.NAME.match(str(name or "")) or not body_json.is_file():
        raise ValueError("choose a registered base body")
    return body_json


def _set_name(name: Any) -> str:
    if not body_mod.NAME.match(str(name or "")):
        raise ValueError("give the set a name: letters, digits, '.', '_' or '-'")
    return str(name)


def set_folder(output: Path, set_name: str, body: str) -> Path:
    return output / "sets" / set_name / body_mod.load(_body_json(output, body))["name"]


def _base(body_json: Path, set_name: str) -> list[str]:
    return [sys.executable, str(SCRIPT), "run", "--body", str(body_json), "--set", set_name]


def run_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """`slot_parts.py run` for a whole set, and its log file.

    payload: mode (2: the outfit already drawn, `concept`; 3: drawn from `outfit` words),
    body, set, slots ({slot: {route, split, prompt}}), weapon, shield, style, seeds ("3,7"),
    choose (stop after the pictures to pick, when there are several seeds). `concept` is a
    path, already checked by the caller.
    """
    if payload.get("resume"):
        return resume_command(output, payload)
    mode = int(payload.get("mode", 3))
    if mode not in (2, 3):
        raise ValueError("mode 1 (one piece) is + Create; this page makes modes 2 and 3")
    body_json = _body_json(output, payload.get("body"))
    set_name = _set_name(payload.get("set"))
    raw = {k: v for k, v in (payload.get("slots") or {}).items() if k in slots.DRAWN}
    chosen = slots.choose(raw) if raw else {}
    seeds = [int(s) for s in str(payload.get("seeds", "3")).split(",") if s.strip()][:MAX_SEEDS] or [3]
    folder = output / "sets" / set_name / body_mod.load(body_json)["name"]
    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "slots.json"
    spec.write_text(json.dumps(chosen, indent=1) + "\n", encoding="utf-8")
    command = _base(body_json, set_name) + [
        "--slots-json", str(spec), "--seeds", ",".join(map(str, seeds)),
        "--concept-seeds", str(seeds[0]), "--root", str(output / "sets")]
    if mode == 2:
        concept = payload.get("concept")
        if not concept or not Path(concept).is_file():
            raise ValueError("mode 2 needs the concept already drawn on the body")
        command += ["--concept", str(concept)]
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
    if payload.get("choose") and len(seeds) > 1 and chosen:
        command.append("--stop-at-pictures")
    return command, folder / "run.log"


def resume_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """The run that goes on from the pictures, with the picks ({slot: seed})."""
    body_json = _body_json(output, payload.get("body"))
    set_name = _set_name(payload.get("set"))
    folder = output / "sets" / set_name / body_mod.load(body_json)["name"]
    if not (folder / "run_state.json").is_file():
        raise ValueError(f"{set_name} has no pictures waiting for picks")
    offered = candidates_mod.load(folder / "pieces")["slots"]
    picks = {}
    for slot, seed in (payload.get("picks") or {}).items():
        if slot not in offered or str(seed) not in {str(s) for s in offered[slot]["candidates"]}:
            raise ValueError(f"{slot}: {seed} is not one of its pictures")
        picks[slot] = int(seed)
    command = _base(body_json, set_name) + ["--slots-json", str(folder / "slots.json"),
                                            "--root", str(output / "sets"), "--resume"]
    if picks:
        command += ["--pick", ",".join(f"{slot}={seed}" for slot, seed in picks.items())]
    return command, folder / "run.log"


def _fitted(output: Path, payload: dict) -> tuple[Path, Path, Path]:
    """A set's folder on a body, that body's record, and the set's fit folder (which must exist)."""
    body_json = _body_json(output, payload.get("body"))
    folder = output / "sets" / _set_name(payload.get("set")) / body_mod.load(body_json)["name"]
    if not (folder / "fit" / "fit.json").is_file():
        raise ValueError(f"{payload.get('set')} has no fitted pieces on {payload.get('body')}")
    return folder, body_json, folder / "fit"


def fit_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """Fit a set's pieces on its body again (after a piece was changed), no new pictures or 3D."""
    body_json = _body_json(output, payload.get("body"))
    folder = output / "sets" / _set_name(payload.get("set")) / body_mod.load(body_json)["name"]
    pieces_json = folder / "pieces" / "pieces.json"
    if not pieces_json.is_file():
        raise ValueError(f"{payload.get('set')} has no pieces on {payload.get('body')}")
    return [sys.executable, str(SCRIPT), "fit", "--pieces", str(pieces_json), "--body", str(body_json),
            "--out", str(folder / "fit")], folder / "fit.log"


def refit_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """Carry a fitted set from its body (`body`) to another registered body (`to`)."""
    folder, body_json, fit = _fitted(output, payload)
    target = _body_json(output, payload.get("to"))
    if target == body_json:
        raise ValueError("choose another body to carry the set to")
    new_folder = folder.parent / body_mod.load(target)["name"]
    return [sys.executable, str(SCRIPT), "refit", "--fit", str(fit), "--from", str(body_json),
            "--body", str(target), "--out", str(new_folder / "fit")], new_folder / "refit.log"


def unreal_command(output: Path, payload: dict) -> tuple[list[str], Path]:
    """Export a fitted set for Unreal 5.8: the masked body and every piece on its Skeleton."""
    folder, body_json, fit = _fitted(output, payload)
    size = float(payload.get("size_cm") or 180.0)
    if not 0 < size <= 100_000:
        raise ValueError("size_cm must be the body's real height in centimetres")
    return [sys.executable, str(SCRIPT), "unreal", "--fit", str(fit), "--body", str(body_json),
            "--size-cm", f"{size:g}", "--out", str(folder / "unreal")], folder / "unreal.log"


COMMANDS = {"run": run_command, "resume": resume_command, "fit": fit_command, "refit": refit_command,
            "unreal": unreal_command}


def waiting_picks(output: Path, folder: Path) -> dict[str, Any] | None:
    """The candidates a stopped run waits on: every slot's pictures and scores."""
    if not (folder / "run_state.json").is_file():
        return None
    out = {}
    for slot, entry in candidates_mod.load(folder / "pieces")["slots"].items():
        out[slot] = {"auto": entry["auto"], "picked": entry["picked"], "doubt": bool(entry.get("doubt")),
                     "candidates": [{"seed": int(seed), "picture": rel(output, Path(c["picture"])),
                                     "score": c["score"], "is_piece": c.get("is_piece"), "shape": c["shape"]["score"]}
                                    for seed, c in sorted(entry["candidates"].items(), key=lambda kv: int(kv[0]))]}
    return out


# ----------------------------------------------------------------- one run at a time

def _identity(pid: int) -> str | None:
    from image_to_3dlab.batch import process_identity

    return process_identity(pid)


def active_run(output: Path) -> dict[str, Any] | None:
    """The run recorded in `output/sets/.run.json`, while its process is alive."""
    try:
        record = json.loads((output / "sets" / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = record.get("pid") if isinstance(record, dict) else None
    if not isinstance(pid, int) or not processes.process_alive(pid):
        return None
    if record.get("identity") and _identity(pid) not in (None, record["identity"]):
        return None   # the number went to another process since
    return record


def start(output: Path, action: str, payload: dict, popen=subprocess.Popen) -> tuple[Any, dict[str, Any]]:
    """Start one action (`COMMANDS`) in the background, in its own process group; refuse
    while another run is alive. Returns the process and its record."""
    running = active_run(output)
    if running:
        raise ValueError(f"a set is already being made ({running.get('set')}); wait or stop it")
    command, log = COMMANDS[action](output, payload)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        # its own group: Stop must reach the edits, Pixal3D and Blender it starts
        process = popen(command, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        cwd=str(REPO), **processes.group_popen_kwargs())
    # waited on, so a finished run does not linger as a zombie that still looks alive
    threading.Thread(target=process.wait, daemon=True).start()
    record = {"pid": process.pid, "identity": _identity(process.pid), "action": action,
              "set": str(payload.get("set")), "body": str(payload.get("body")),
              "to": payload.get("to"), "log": str(log), "folder": str(log.parent), "started": time.time()}
    (output / "sets").mkdir(parents=True, exist_ok=True)
    (output / "sets" / RUN_FILE).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return process, record


def status(output: Path, record: dict[str, Any] | None = None, running: bool | None = None) -> dict[str, Any]:
    """Where the last run stands: running or not, the end of its log, the set's record
    when it was written by this run, and the pictures waiting for a pick."""
    if record is None:
        try:
            record = json.loads((output / "sets" / RUN_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"running": False}
    if running is None:
        running = active_run(output) is not None
    log = Path(record["log"])
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:] if log.is_file() else []
    folder = Path(record["folder"])
    parts = folder / f"{record['set']}.parts.json"
    return {"running": running, "action": record.get("action", "run"), "set": record["set"], "body": record["body"],
            "started": record["started"], "log": tail,
            "parts": rel(output, parts) if parts.is_file() and parts.stat().st_mtime >= record["started"] else None,
            "picks": None if running else waiting_picks(output, folder)}


def cancel(output: Path, stop=processes.terminate_group) -> dict[str, Any]:
    running = active_run(output)
    if running is None:
        return {"stopped": False, "reason": "no set is being made"}
    stop(running["pid"])
    return {"stopped": True, "pid": running["pid"]}
