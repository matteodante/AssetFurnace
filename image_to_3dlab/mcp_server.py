"""An MCP server for batches, so an agent in another repository can drive AssetFurnace.

    python -m image_to_3dlab.mcp_server            # stdio; needs `pip install "mcp>=2.3,<3"`

In the other repository's `.mcp.json`, with this repository's virtual environment:

    {"mcpServers": {"assetfurnace": {
        "command": "C:/path/to/AssetFurnace/.venv/Scripts/python.exe",
        "args": ["-m", "image_to_3dlab.mcp_server"],
        "cwd": "C:/path/to/AssetFurnace"}}}

The tools are the batch runner's own operations (`docs/batches.md`): check a spec, start
it, read its state, list its files, stop it, list the steps. A batch runs for hours, so
`submit_batch` starts `pipeline.py --run-batch` as a background process and answers at
once; the state file is how anyone, agent or person, follows it. Nothing here downloads
weights: a spec that needs a missing install is refused with the command to run.

Each tool is a plain function, testable without MCP; `build_server` only exposes them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from . import batch, step_catalog
from .processes import group_popen_kwargs, terminate_group

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "pipeline.py"
RUN_LOG = "run.log"
INSTALL_HINT = 'The MCP server needs the MCP SDK: pip install "mcp>=2.3,<3"'


def _intent(use_case: str | None, distribution: str | None,
            commercial: bool | None) -> dict[str, Any]:
    return {key: value for key, value in (("use_case", use_case),
                                          ("distribution", distribution),
                                          ("commercial_intent", commercial))
            if value is not None}


def _check(spec_path: str, intent: dict[str, Any], retry_failed: bool | None = None,
           ) -> tuple[batch.BatchSpec | None, list[str]]:
    """Load and preflight a spec: every row, or with `retry_failed` given, only the rows a
    run would execute now (a resume does not need what its done rows needed)."""
    try:
        spec = batch.load_batch(Path(spec_path), intent=intent)
    except batch.BatchError as exc:
        return None, exc.problems
    if retry_failed is None:
        return spec, batch.preflight(spec)
    state = batch.read_json(spec.directory / batch.STATE_FILE)
    return spec, batch.preflight(spec, batch.pending_work(spec, state, retry_failed))


def validate_batch(spec: str, use_case: str | None = None, distribution: str | None = None,
                   commercial: bool | None = None) -> dict[str, Any]:
    """Check a batch spec (JSON or CSV) and everything its steps need, without running
    anything or downloading anything. Returns the problems, if any. use_case,
    distribution and commercial are for a CSV spec; a JSON spec states its own."""
    loaded, problems = _check(spec, _intent(use_case, distribution, commercial))
    result: dict[str, Any] = {"ok": not problems, "problems": problems,
                              "steps": batch.available_steps()}
    if loaded is not None:
        result.update(name=loaded.name, directory=str(loaded.directory),
                      rows=len(loaded.rows), intent=loaded.intent)
    return result


def launch(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    """Start the runner detached from this server, in its own process group."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        process = subprocess.Popen(command, cwd=str(REPO), stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, env=env,
                                   **group_popen_kwargs())
    # Waited on in the background: on POSIX an exited child nobody waits for stays a
    # zombie, which still answers kill(pid, 0) and would hold the batch lock forever.
    threading.Thread(target=process.wait, daemon=True).start()
    return process.pid


def submit_batch(spec: str, retry_failed: bool = False, use_case: str | None = None,
                 distribution: str | None = None, commercial: bool | None = None,
                 ) -> dict[str, Any]:
    """Start a batch in the background and return at once. Running the same spec again
    resumes it; retry_failed also re-runs rows that failed. One batch runs at a time.
    use_case, distribution and commercial are for a CSV spec, as in validate_batch."""
    intent = _intent(use_case, distribution, commercial)
    loaded, problems = _check(spec, intent, retry_failed)
    if loaded is None or problems:
        return {"started": False, "problems": problems}
    directory = loaded.directory
    # Reserve the lock before starting anything: two submissions at once cannot both
    # pass. The runner gets the reservation's token and takes the lock over.
    token = uuid.uuid4().hex
    try:
        batch.acquire_lock(loaded.output_dir, loaded.name, os.getpid(), token, starting=True)
    except batch.BatchError as exc:
        return {"started": False, "problems": exc.problems}
    command = [sys.executable, str(PIPELINE), "--run-batch", str(loaded.path)]
    if retry_failed:
        command.append("--retry-failed")
    if "use_case" in intent:
        command += ["--use-case", intent["use_case"]]
    if "distribution" in intent:
        command += ["--distribution", intent["distribution"]]
    if intent.get("commercial_intent"):
        command.append("--commercial")
    log_path = directory / RUN_LOG
    try:
        pid = launch(command, log_path, env={**os.environ, batch.LOCK_TOKEN_ENV: token})
    except OSError as exc:
        batch.release_lock(loaded.output_dir, token)
        return {"started": False, "problems": [f"the runner could not start: {exc}"]}
    # In the runner's name until it takes the lock over itself, so Stop works at once.
    batch.acquire_lock(loaded.output_dir, loaded.name, pid, token)
    return {"started": True, "pid": pid, "name": loaded.name, "directory": str(directory),
            "log": str(log_path)}


def resume_batch(batch_ref: str, retry_failed: bool = False) -> dict[str, Any]:
    """Start a stopped batch again from where it left off, with the spec and intent it ran
    with. retry_failed also re-runs the rows that failed."""
    try:
        directory = _directory(batch_ref)   # a spec or its batch folder, as elsewhere
    except batch.BatchError as exc:
        return {"started": False, "problems": exc.problems}
    state = batch.read_json(directory / batch.STATE_FILE) if directory.is_dir() else None
    if not state:
        return {"started": False, "problems": [f"{directory.name} never ran: submit its spec"]}
    spec = state["spec"]["path"]
    if Path(spec).suffix.lower() != ".csv":
        return submit_batch(spec, retry_failed=retry_failed)
    intent = state.get("intent", {})
    return submit_batch(spec, retry_failed=retry_failed, use_case=intent.get("use_case"),
                        distribution=intent.get("distribution"),
                        commercial=intent.get("commercial_intent") or None)


def _directory(target: str) -> Path:
    """A batch folder, or the spec that names one."""
    path = Path(target).expanduser().resolve()
    # a folder, or one not made yet (a batch whose runner is just starting); a spec by its
    # .json or .csv file
    if path.is_dir() or (not path.exists() and path.suffix.lower() not in (".json", ".csv")):
        return path
    return batch.spec_directory(path)   # its name and output only: rows may have gone stale


def batch_status(batch_ref: str) -> dict[str, Any]:
    """The state of a batch, by its folder or its spec: counts per status, and per row its
    status, the reason it failed and the status of each step. `active` is the pid of the
    process running it, or null."""
    directory = _directory(batch_ref)
    state = batch.read_json(directory / batch.STATE_FILE) or {}
    active = batch.lock_holder(directory)
    rows = [{"id": row_id, "status": entry.get("status"), "reason": entry.get("reason"),
             "note": entry.get("note"),
             "steps": {name: step.get("status")
                       for name, step in entry.get("steps", {}).items()},
             "finished_at": entry.get("finished_at"),
             **({"waiting": _waiting(directory, entry)} if entry.get("status") == "waiting" else {})}
            for row_id, entry in batch.rows_as_seen(state, active is not None).items()]
    return {"name": state.get("name", directory.name), "directory": str(directory),
            "active": active, "counts": batch.summary(state, active is not None),
            "updated_at": state.get("updated_at"), "rows": rows}


def list_outputs(batch_ref: str, row_id: str | None = None) -> dict[str, Any]:
    """The finished rows of a batch and their files, as absolute paths on this machine:
    the final asset, its .provenance.json, the reference images and every file a step
    wrote. Give row_id for one row."""
    directory = _directory(batch_ref)
    index = batch.read_json(directory / batch.INDEX_FILE) or {"rows": []}
    rows = index["rows"]
    if row_id is not None:
        rows = [row for row in rows if row["id"] == row_id]
        if not rows:
            raise ValueError(f"row {row_id} has no finished outputs in {directory.name}")

    def absolute(path: str | None) -> str | None:
        return None if path is None else str((directory / path).resolve())

    listed = [{**row, "asset": absolute(row.get("asset")),
               "sidecar": absolute(row.get("sidecar")),
               "references": [absolute(path) for path in row.get("references", [])],
               **({"picture": absolute(row["picture"])} if row.get("picture") else {}),
               "files": [absolute(path) for path in row.get("files", [])],
               **({"unreal": {key: absolute(path) for key, path in row["unreal"].items()}}
                  if row.get("unreal") else {}),
               **({"checks": {**row["checks"], "report": absolute(row["checks"]["report"])}}
                  if row.get("checks") else {})} for row in rows]
    return {"name": index.get("batch", directory.name), "directory": str(directory),
            "rows": listed}


def cancel_batch(batch_ref: str) -> dict[str, Any]:
    """Stop a running batch. The row in progress goes back in the queue (batch_status
    shows it queued once the runner is gone): resume_batch carries on from it."""
    directory = _directory(batch_ref)
    holder = batch.lock_holder(directory)
    if holder is None:
        return {"cancelled": False, "reason": "the batch is not running"}
    if (batch.lock_info(directory) or {}).get("starting"):
        # the pid is still the server that is starting it: stopping it would stop the server
        return {"cancelled": False, "reason": "the batch is starting; stop it in a moment"}
    terminate_group(holder)
    return {"cancelled": True, "pid": holder}


def _waiting(directory: Path, entry: dict[str, Any]) -> dict[str, Any] | None:
    """The step a row waits on and its candidates, with absolute paths to look at."""
    for name, step in entry.get("steps", {}).items():
        if step.get("status") == "waiting":
            return {"step": name, "candidates": [
                {"choice": candidate["choice"], "picture": str((directory / candidate["file"]).resolve())}
                for candidate in step.get("candidates", [])]}
    return None


def _offered(entry: dict[str, Any], step: str) -> tuple[list[Any], str | None]:
    """What a row's step offered to pick from, and the key of what they were made from:
    the candidates it waits on, or those a finished pick was made among (another pick
    changes it)."""
    done = entry.get("steps", {}).get(step, {})
    if done.get("status") == "waiting":
        return [candidate["choice"] for candidate in done.get("candidates", [])], done.get("key")
    record = done.get("record") or {}
    # the key on the step, or in its record for a pick made by the first version of picture
    return list(record.get("candidates", [])), done.get("key") or record.get("key")


def _spec_pick(spec: str | None, row_id: str, step: str) -> bool:
    """Whether the spec itself sets this step's pick for the row (it wins over picks.json)."""
    data = batch.read_json(Path(spec)) if spec and Path(spec).suffix.lower() == ".json" else None
    if not isinstance(data, dict):
        return False
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    row = next((row for row in rows if isinstance(row, dict) and row.get("id") == row_id), {})
    for source in ((data.get("defaults") or {}).get(step), row.get(step)):
        if isinstance(source, dict) and "pick" in source:
            return True
    return False


def pick_candidate(batch_ref: str, row_id: str, step: str, choice: int) -> dict[str, Any]:
    """Choose one of the candidates a row's step made (batch_status lists them under
    `waiting`), and go on: the batch is resumed, and the step uses the pick without
    drawing again. A pick on a finished row changes it and reruns what follows. A pick
    written in the spec (`"pick"` in the step's settings) wins over this one."""
    directory = _directory(batch_ref)
    state = batch.read_json(directory / batch.STATE_FILE) or {}
    entry = state.get("rows", {}).get(row_id)
    if entry is None:
        return {"picked": False, "problem": f"{row_id} is not a row of {directory.name}"}
    offered, key = _offered(entry, step)
    if len(offered) < 2:
        return {"picked": False, "problem": f"{row_id}'s {step} has nothing to pick from"}
    if choice not in offered:
        return {"picked": False,
                "problem": f"{choice} is not one of {row_id}'s {step} candidates: "
                           f"{', '.join(map(str, offered))}"}
    spec = (state.get("spec") or {}).get("path")
    if _spec_pick(spec, row_id, step):
        return {"picked": False, "problem": f"the spec sets {step}'s pick for {row_id}; "
                                            f"change it there ({spec})"}
    picks_path = directory / batch.PICKS_FILE
    picks = batch.read_json(picks_path)
    picks = picks if isinstance(picks, dict) else {}
    # with what it was picked among: new words or settings leave it behind
    picks.setdefault(row_id, {})[step] = {"choice": choice, "key": key}
    batch.write_json(picks_path, picks)
    if batch.lock_holder(directory) is not None:
        return {"picked": True, "resumed": None,
                "note": "the batch is running: the pick is used when it runs next "
                        "(resume_batch once it ends)"}
    return {"picked": True, "resumed": resume_batch(str(directory))}


def rerun_step(batch_ref: str, row_id: str, step: str) -> dict[str, Any]:
    """Run one step of a row again, and every step after it, with the same settings (to
    change settings, edit the spec: the row reruns from its first changed step). The
    steps before keep their results. Refused while the batch runs."""
    directory = _directory(batch_ref)
    if batch.lock_holder(directory) is not None:
        return {"started": False, "problems": [f"{directory.name} is running; cancel it or wait"]}
    state_path = directory / batch.STATE_FILE
    state = batch.read_json(state_path) or {}
    entry = state.get("rows", {}).get(row_id)
    if entry is None or step not in entry.get("steps", {}):
        return {"started": False, "problems": [f"{row_id} has no {step} step in {directory.name}"]}
    entry["steps"][step] = {"status": "queued", "note": "rerun asked"}
    entry.update(status="queued", note=f"{step} asked to run again")
    batch.write_json(state_path, state)
    return resume_batch(str(directory))


def describe_steps() -> dict[str, Any]:
    """Every step a batch row can run, in chain order: what it takes and makes, its
    settings (type, default, range or choices), whether it can be skipped or replaced
    with a file of your own, what makes several to pick from, its command line, and
    whether it is installed in this repository yet."""
    available = batch.available_steps()
    steps = [{**described, "available": available.get(described["name"], False)}
             for described in step_catalog.describe()]
    listed = {described["name"] for described in steps}
    steps += [{"name": name, "available": available[name], "described": False}
              for name in batch.STEP_NAMES if name not in listed]
    return {"steps": steps}


AGENT_SPECS = batch.DEFAULT_OUTPUT / ".agent"


def run_steps(name: str, steps: list[str], prompt: str | None = None, images: list[str] | None = None,
              asset: str | None = None, kind: str = "prop", faces: int | None = None,
              size_cm: float | None = None, measure: str | None = None, backend: str | None = None,
              parameters: dict[str, Any] | None = None, options: dict[str, dict[str, Any]] | None = None,
              use_case: str = "showcase", distribution: str = "private", commercial: bool = False,
              allow_conditional: bool = True) -> dict[str, Any]:
    """Run one asset through some steps, from words (`prompt`, with "picture" first),
    pictures (`images`) or a GLB (`asset`). It is a batch of one row named `name`, written
    to output/batches/.agent/<name>.json and started like submit_batch: calling it again
    with other settings reruns from the first changed step. steps: as describe_steps
    lists them, e.g. ["picture", "generate", "finish", "checks"]. options: each step's
    settings under its name, e.g. {"picture": {"count": 4}, "finish": {"paint": false}}.
    Paths are absolute, on this machine."""
    row: dict[str, Any] = {"id": name, "steps": list(steps), "kind": kind}
    for key, value in (("prompt", prompt), ("images", images), ("asset", asset), ("faces", faces),
                       ("size_cm", size_cm), ("measure", measure), ("backend", backend),
                       ("parameters", parameters)):
        if value is not None:
            row[key] = value
    row.update(options or {})
    spec = {"schema_version": 1, "name": name, "use_case": use_case, "distribution": distribution,
            "commercial_intent": commercial, "license_policy": {"allow_conditional": allow_conditional},
            "output": {"directory": str(batch.DEFAULT_OUTPUT)}, "rows": [row]}
    if not batch.ID_PATTERN.fullmatch(name):
        return {"started": False, "problems": [f"name {name!r}: letters, digits, '_', '.', '-'"]}
    AGENT_SPECS.mkdir(parents=True, exist_ok=True)
    path = AGENT_SPECS / f"{name}.json"
    batch.write_json(path, spec)
    return {**submit_batch(str(path)), "spec": str(path)}


def list_steps() -> dict[str, Any]:
    """The step names a row can ask for, and whether each is available yet. A step that
    is not available is skipped with a warning, not an error."""
    return {"steps": batch.available_steps()}


# ------------------------------------------------------------- the review queue (7)
# The same calls as the studio's review sheet and scripts/review_queue.py.

OUTPUT = REPO / "output"


def _scripts() -> None:
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))


def _review_log(log: str | None) -> Path:
    from .checks import review

    return Path(log) if log else review.LOG


def review_queue(log: str | None = None) -> dict[str, Any]:
    """What waits for review (to review first, then sent back, newest first) and the latest
    approvals. Each item: key, state, checks status and the checks that failed or warned,
    the asset's path and its version (asset_sha256), the notes of earlier send-backs."""
    from .checks import review

    return review.overview(_review_log(log))


def review_item(key: str, log: str | None = None) -> dict[str, Any]:
    """One asset in the queue: its state, notes, history and full checks report."""
    from .checks import review

    try:
        return review.item_with_report(_review_log(log), key)
    except KeyError:
        return {"problem": f"{key} is not in the review queue"}


def check_asset(glb: str, key: str | None = None, profile: str | None = None,
                set: dict[str, Any] | None = None, reference: str | None = None,
                height: float | None = None, width: float | None = None, depth: float | None = None,
                log: str | None = None) -> dict[str, Any]:
    """Run the automatic checks on a GLB, write <name>.checks.json beside it and put it in
    the review queue, as scripts/check_asset.py does. profile: prop (default), creature,
    equipment_part. set: thresholds, e.g. {"budget.faces.max": 20000}. reference: a GLB or
    picture for size and silhouette; height/width/depth: wanted size in metres. Returns the
    item as review_item does."""
    _scripts()
    from check_asset import parse_overrides

    from .checks import Asset, profile_names, review

    try:
        if not Path(glb).is_file():
            raise ValueError(f"{glb} does not exist")
        profile = profile or "prop"
        if profile not in profile_names():
            raise ValueError(f"profile must be one of {', '.join(profile_names())}")
        overrides = parse_overrides([f"{name}={json.dumps(value)}" for name, value in (set or {}).items()])
        numbers = {axis: value for axis, value in (("height", height), ("width", width), ("depth", depth))
                   if value is not None}
        asset = Asset(glb=Path(glb), key=key, reference=Path(reference) if reference else None,
                      reference_numbers=numbers or None)
        report, _ = review.check_and_enqueue(asset, log=_review_log(log), profile=profile,
                                             overrides=overrides, key=key)
    except (ValueError, TypeError) as exc:
        return {"problem": str(exc)}
    return review_item(report["asset_key"], log)


def review_decide(key: str, decision: str, asset_sha256: str, note: str = "",
                  log: str | None = None) -> dict[str, Any]:
    """Approve the version waiting for review (decision "approved") or send it back with a
    note saying what to change ("sent_back"). asset_sha256 is the version you looked at,
    as review_queue gives it: a newer one in the meantime refuses the decision. Also
    written as a verdict in verdicts.jsonl, beside the log, as the studio does."""
    _scripts()
    import review_queue as review_cli

    from .checks.review import ReviewConflict

    path = _review_log(log)
    try:
        item, problem = review_cli.decide(key, decision, note, log=path, register=path.parent / "verdicts.jsonl",
                                          asset_sha256=asset_sha256)
    except (ReviewConflict, ValueError, KeyError) as exc:
        return {"decided": False, "problem": str(exc.args[0] if isinstance(exc, KeyError) else exc)}
    return {"decided": True, "item": item, "verdict_problem": problem}


# ----------------------------------------------------- characters in pieces (4)
# scripts/slot_parts.py, started with the arguments the studio's pieces page uses.

PIECES_POPEN = subprocess.Popen


def _pieces(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    from .slot_parts import runs

    try:
        process, record = runs.start(OUTPUT, action, payload, popen=PIECES_POPEN)
    except (ValueError, OSError, LookupError) as exc:
        return {"started": False, "problems": [str(exc)]}
    return {"started": True, "pid": process.pid, "action": action, "log": record["log"],
            "folder": record["folder"]}


def pieces_list() -> dict[str, Any]:
    """The registered base bodies, the sets made so far and the slots a set can have."""
    from .slot_parts import runs

    return {"output": str(OUTPUT), "bodies": runs.list_bodies(OUTPUT), "sets": runs.list_sets(OUTPUT),
            "slots": runs.catalogue()}


def pieces_add_body(name: str, picture: str, model: str, rigged: str, clips: dict[str, str] | None = None,
                    labels: dict[str, str] | None = None) -> dict[str, Any]:
    """Register a rigged humanoid as a base body: its front picture, the Pixal3D model it
    was made from (its saved camera puts the bones on the picture), the rigged GLB, and its
    own animated clips ({"walk": "body_walk.glb"}) and labels ({"race": "elf"})."""
    from .slot_parts import runs

    files = [picture, model, rigged, *(clips or {}).values()]
    missing = next((path for path in files if not Path(path).is_file()), None)
    if missing:
        return {"problem": f"{missing} does not exist"}
    try:
        return runs.register_body(OUTPUT, name, Path(picture), Path(model), Path(rigged),
                                  {k: Path(v) for k, v in (clips or {}).items()}, labels)
    except (ValueError, OSError) as exc:
        return {"problem": str(exc)}


def pieces_run(body: str, set: str, mode: int = 3, slots: list[str] | dict[str, Any] | None = None,
               outfit: str = "", concept: str | None = None, weapon: str = "", shield: str = "",
               style: str = "", seeds: str = "3", choose: bool = False, bible: str | None = None,
               concept_seeds: str = "", choose_concept: bool = False) -> dict[str, Any]:
    """Make a set of pieces on a base body, in the background (one run at a time).
    mode 3 draws the outfit from `outfit` words; mode 2 cuts a `concept` already drawn on
    the body. slots: names (["helmet", "torso", "boots"]) or {slot: {"route": "C"|"B0",
    "split": bool, "prompt": "..."}}. weapon/shield: words. seeds: "3,7,11" makes one
    picture per seed per piece; choose stops after the pictures so pieces_pick can choose.
    bible: a style bible's words for every piece ("-" for the active one), off by default.
    concept_seeds: "3,4,5" draws the mode 3 outfit once per seed; choose_concept stops after
    them so pieces_pick({"concept": 4}) can choose (default: the first is taken)."""
    if isinstance(slots, list):
        slots = {name: {} for name in slots}
    return _pieces("run", {"mode": mode, "body": body, "set": set, "slots": slots or {}, "outfit": outfit,
                           "concept": concept, "weapon": weapon, "shield": shield, "style": style,
                           "seeds": seeds, "choose": choose, "bible": bible, "concept_seeds": concept_seeds,
                           "choose_concept": choose_concept})


def pieces_pick(body: str, set: str, picks: dict[str, int] | None = None) -> dict[str, Any]:
    """Go on from the pictures a stopped run made, with your picks ({"boots": 7}, a weapon
    or shield too, or {"concept": 4} when the run waits on the outfit); slots you leave out
    keep the best-scored picture. pieces_status lists them."""
    return _pieces("resume", {"body": body, "set": set, "picks": picks or {}})


def pieces_fit(body: str, set: str) -> dict[str, Any]:
    """Fit a set's pieces on its body again (no new pictures or 3D)."""
    return _pieces("fit", {"body": body, "set": set})


def pieces_refit(body: str, set: str, to: str) -> dict[str, Any]:
    """Carry a set fitted on `body` over to another registered body `to`."""
    return _pieces("refit", {"body": body, "set": set, "to": to})


def pieces_unreal(body: str, set: str, size_cm: float = 180.0) -> dict[str, Any]:
    """Export a fitted set for Unreal 5.8: the masked body and every piece on its Skeleton.
    size_cm is the body's real height."""
    return _pieces("unreal", {"body": body, "set": set, "size_cm": size_cm})


def pieces_status() -> dict[str, Any]:
    """The last run: running or not, the end of its log, the set's record when done, and
    the pictures waiting for a pick (absolute paths, with their scores)."""
    from .slot_parts import runs

    status = runs.status(OUTPUT)
    if status.get("parts"):
        status["parts"] = str((OUTPUT / status["parts"]).resolve())
    for entry in (status.get("picks") or {}).values():
        for candidate in entry["candidates"]:
            candidate["picture"] = str((OUTPUT / candidate["picture"]).resolve())
    return status


def pieces_cancel() -> dict[str, Any]:
    """Stop the run in progress and everything it started."""
    from .slot_parts import runs

    return runs.cancel(OUTPUT, stop=terminate_group)


TOOLS = (validate_batch, submit_batch, resume_batch, batch_status, list_outputs, cancel_batch,
         list_steps, review_queue, review_item, check_asset, review_decide, pieces_list, pieces_add_body,
         pieces_run, pieces_pick, pieces_fit, pieces_refit, pieces_unreal, pieces_status, pieces_cancel,
         describe_steps, run_steps, pick_candidate, rerun_step)


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise SystemExit(INSTALL_HINT) from exc
    server = MCPServer(
        "assetfurnace-batches",
        instructions="Run AssetFurnace asset batches: validate a spec, submit it, follow "
                     "its state, collect its files. Review the queue: check an asset, "
                     "approve it or send it back. Make characters in pieces: a set on a base "
                     "body, pick its pictures, fit, refit, export for Unreal. Paths are on "
                     "this machine.",
    )
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main() -> int:
    build_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
