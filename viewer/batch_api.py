"""The studio's Batches page, and batch rows in the Library (docs/batches.md).

    GET  /api/batches                   every batch under output/batches, with its counts
    GET  /api/batches/<name>            its rows: status, reason, steps, review state, file
    POST /api/batches                   {"path"} or {"filename", "content"}, plus for a CSV
                                        use_case / distribution / commercial; check_only to
                                        validate without starting
    POST /api/batches/<name>/resume     {"retry_failed": bool}
    POST /api/batches/<name>/cancel
    POST /api/batches/<name>/pick       {"row", "step", "choice"}: pick a waiting row's candidate, resume
    POST /api/batches/<name>/rerun      {"row", "step"}: run a row's step again, and what follows

Starting, resuming and stopping are the MCP server's own functions
(`image_to_3dlab/mcp_server.py`), so the page and an agent do exactly the same thing: the
runner is `pipeline.py --run-batch` in the background, and the state file is what both
read. An uploaded spec is kept in `output/batches/.uploads/`; its paths are resolved from
there, so a spec meant for upload uses absolute paths, or is given by its path on disk.

Requests are checked like the Blender edit endpoints: the local Studio address only, same
origin, and a custom header on anything that starts or stops work.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from blender_edit_api import local_address

from image_to_3dlab import batch, mcp_server

BATCHES = "batches"
UPLOADS = ".uploads"
HEADER = "batch"
UPLOAD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(json|csv)", re.IGNORECASE)
MAX_SPEC = 2 * 1024 * 1024   # a 4,000-row CSV is about 400 KB


def batches_root(output: Path) -> Path:
    return output / BATCHES


def _directory(output: Path, name: str) -> Path | None:
    if not batch.ID_PATTERN.fullmatch(name or ""):
        return None
    directory = batches_root(output) / name
    return directory if (directory / batch.STATE_FILE).is_file() else None


def list_batches(output: Path) -> list[dict[str, Any]]:
    """Every batch that has run under output/batches, the most recently updated first."""
    root = batches_root(output)
    found = []
    # one read of the lock for the whole list (it can run a process to check the pid);
    # a runner just started holds it before it writes its state: list that one as starting
    lock = batch.live_lock(root)
    running = {str(lock["batch"])} if lock else set()
    names = {d.name for d in root.iterdir() if d.is_dir()} if root.is_dir() else set()
    for name in sorted(names | running):
        directory = root / name
        state = batch.read_json(directory / batch.STATE_FILE)
        active = lock["pid"] if lock and lock.get("batch") == name else None
        if name.startswith(".") or not (isinstance(state, dict) or active):
            continue
        state = state if isinstance(state, dict) else {}
        found.append({"name": name, "counts": batch.summary(state, active is not None),
                      "total": len(state.get("rows", {})), "active": active,
                      "updated_at": state.get("updated_at"),
                      "spec": (state.get("spec") or {}).get("path")})
    found.sort(key=lambda item: item["updated_at"] or "", reverse=True)
    return found


def batch_detail(output: Path, name: str) -> dict[str, Any] | None:
    directory = _directory(output, name)
    if directory is None:
        return None
    status = mcp_server.batch_status(str(directory))
    spec = batch.read_json(directory / batch.SPEC_COPY) or {}
    rows_in_spec = {row["id"]: row for row in spec.get("rows", []) if isinstance(row, dict)}
    state_rows = (batch.read_json(directory / batch.STATE_FILE) or {}).get("rows", {})
    reviews = batch.review_index(output / batch.REVIEW_LOG)   # once, not per row
    for row in status["rows"]:
        entry = state_rows.get(row["id"], {})
        asset = entry.get("asset") if row["status"] == "done" else None
        row["kind"] = rows_in_spec.get(row["id"], {}).get("kind")
        row["file"] = _from_output(directory / asset, output) if asset else None
        row["review"] = (batch.review_in(reviews, row["id"], batch.asset_sha(entry, directory))
                         if asset else None)
        for candidate in (row.get("waiting") or {}).get("candidates", []):
            candidate["picture"] = _from_output(Path(candidate["picture"]), output)   # served under /output/
    status["directory"] = _from_output(directory, output)
    return status


def row_action(directory: Path, action: str, body: dict) -> tuple[int, dict[str, Any]]:
    """A pick or a rerun from the page: the MCP server's own pick_candidate and rerun_step,
    so the page and an agent do the same thing."""
    row, step = body.get("row"), body.get("step")
    if not isinstance(row, str) or not isinstance(step, str) or step not in batch.STEP_NAMES:
        raise ValueError("name the row and one of its steps")
    if action == "rerun":
        result = mcp_server.rerun_step(str(directory), row, step)
        return (202 if result.get("started") else 422), result
    choice = body.get("choice")
    if isinstance(choice, bool) or not isinstance(choice, int):
        raise ValueError("the choice is one of the candidates' numbers")
    result = mcp_server.pick_candidate(str(directory), row, step, choice)
    if not result.get("picked"):
        raise ValueError(result.get("problem") or "the pick was not taken")
    resumed = result.get("resumed")
    if isinstance(resumed, dict) and not resumed.get("started"):
        # kept for the next run, but the batch did not go on: say why, not a silent "Pick one"
        problems = "; ".join(resumed.get("problems") or []) or "the batch could not resume"
        return 422, {**result, "error": f"the pick is saved, but the batch did not resume: {problems}"}
    return 200, result


def _from_output(path: Path, output: Path) -> str | None:
    try:
        return path.resolve().relative_to(output.resolve()).as_posix()
    except ValueError:
        return None


def library_nodes(output: Path) -> dict[str, list[dict[str, Any]]]:
    """The finished rows of every batch, as the Library's own kinds of node: the row's
    picture, its generated model and its finished model, linked like the studio's runs
    (`assets_api.list_assets` joins them by sha256), so a rig made later in the studio
    from a batch model lands on the same asset."""
    nodes: dict[str, list[dict[str, Any]]] = {"picture": [], "model": [], "finished": []}
    root = batches_root(output)
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        index = batch.read_json(directory / batch.INDEX_FILE)
        state = batch.read_json(directory / batch.STATE_FILE)
        if directory.name.startswith(".") or not isinstance(index, dict) or not state:
            continue
        for row in index.get("rows", []):
            steps = state.get("rows", {}).get(row.get("id"), {}).get("steps", {})
            references = [directory / ref for ref in row.get("references", [])]
            picture = next((ref for ref in references if ref.is_file()), None)
            model = _made(directory, steps, "generate")
            finished = _made(directory, steps, "finish")
            if model is None and finished is None and row.get("asset"):
                # a row that only checked or exported an existing GLB: show that one, if it
                # is inside output/ where the studio can serve it
                final = (directory / row["asset"]).resolve()
                if final.is_file() and output.resolve() in final.parents:
                    model = final
            # `<picture stem>__...` is how the Library ties a picture to its model, so the
            # batch is part of both stems, as a hash of batch and row: names containing
            # `__` cannot run into each other. The name shown is the row's id.
            identity = hashlib.sha256(f"{directory.name}/{row['id']}".encode()).hexdigest()[:12]
            stem = f"{row['id']}__{identity}"
            # `link` ties a row's nodes together whatever their bytes: two batches with
            # identical files must stay two assets, which a link by sha256 cannot tell
            if picture is not None:
                nodes["picture"].append({"stem": stem, "file": picture, "link": stem})
            if model is not None:
                nodes["model"].append({"stem": f"{stem}__batch", "file": model, "link": stem})
            if finished is not None:
                nodes["finished"].append({"stem": row["id"], "file": finished, "source": None,
                                          "image": None, "link": stem})
    return nodes


def _made(directory: Path, steps: dict[str, Any], step: str) -> Path | None:
    """The asset a step of a row left, if it is still there."""
    relative = steps.get(step, {}).get("asset")
    path = directory / relative if relative else None
    return path if path is not None and path.is_file() else None


def submit(output: Path, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Check or start a spec given by its path on disk or uploaded as text."""
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object")
    if payload.get("content") is not None:
        filename = str(payload.get("filename") or "")
        if not UPLOAD_NAME.fullmatch(filename):
            raise ValueError("Name the spec like lot.json or lot.csv: letters, digits, _ . -")
        content = str(payload["content"])
        if len(content.encode()) > MAX_SPEC:
            raise ValueError("The spec is larger than 2 MB")
        # one folder per content: a batch's state points at its spec, so a later upload
        # under the same name must not change what a resume of the earlier one reads
        uploads = batches_root(output) / UPLOADS / hashlib.sha256(content.encode()).hexdigest()[:16]
        uploads.mkdir(parents=True, exist_ok=True)
        spec = uploads / filename
        spec.write_text(content, encoding="utf-8")
    elif payload.get("path"):
        spec = Path(str(payload["path"])).expanduser()
        if spec.suffix.lower() not in (".json", ".csv") or not spec.is_file():
            raise ValueError(f"No JSON or CSV spec at {spec}")
    else:
        raise ValueError("Give the spec's path on this computer, or choose the file")
    commercial = payload.get("commercial")
    intent = {"use_case": payload.get("use_case") or None,
              "distribution": payload.get("distribution") or None,
              "commercial": True if commercial else None}
    if spec.suffix.lower() != ".csv":
        intent = {"use_case": None, "distribution": None, "commercial": None}
    result = mcp_server.validate_batch(str(spec), **intent)
    if result.get("directory") and Path(result["directory"]).parent != batches_root(output).resolve():
        # the page lists output/batches only: a batch elsewhere could not be followed
        result.update(ok=False, problems=[*result["problems"], (
            f"The studio follows batches in {batches_root(output)}. Remove output.directory "
            "from the spec, or run it with pipeline.py --run-batch")])
    previous = batch.read_json(Path(result["directory"]) / batch.STATE_FILE) if result.get("directory") else None
    ran_from = ((previous or {}).get("spec") or {}).get("path")
    if ran_from and Path(ran_from) != Path(spec).resolve():
        # two files under one batch name would share its folder and state, and resuming
        # the first would run the second
        result.update(ok=False, problems=[*result["problems"], (
            f"A batch named {result['name']} already ran from another file ({ran_from}). "
            "Rename this file, or give the spec its own name")])
    if payload.get("check_only") or not result["ok"]:
        return (200 if result["ok"] else 422), {**result, "started": False}
    result = mcp_server.submit_batch(str(spec), **intent)
    return (202 if result["started"] else 422), result


def check_request(handler, *, mutation: bool) -> None:
    """The local Studio page only: loopback Host on a loopback server (no DNS rebinding),
    same origin, and on mutations a custom header a cross-site form cannot send."""
    host = handler.headers.get("Host", "")
    parsed = urlsplit("//" + host)
    if not parsed.hostname or parsed.username or parsed.path or parsed.query:
        raise PermissionError("Invalid request host")
    if local_address(handler.server.server_address[0]) and not local_address(parsed.hostname):
        raise PermissionError("Use the local Studio address")
    origin = handler.headers.get("Origin")
    if origin:
        source = urlsplit(origin)
        if source.scheme not in ("http", "https") or source.netloc.lower() != host.lower():
            raise PermissionError("Use this Studio page to run batches")
    if handler.headers.get("Sec-Fetch-Site") == "cross-site":
        raise PermissionError("Cross-site batch requests are not allowed")
    if mutation and (not origin or handler.headers.get("X-AssetFurnace-Request") != HEADER):
        raise PermissionError("Use this Studio page to run batches")


def _body(handler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if not 0 < length <= MAX_SPEC * 2:
        return {}
    return json.loads(handler.rfile.read(length))


def _drain(handler) -> None:
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except ValueError:
        return
    if 0 < length <= MAX_SPEC * 2:
        handler.rfile.read(length)


def handle(handler, output: Path, *, mutation: bool = False) -> bool:
    """Answer a /api/batches request; False for any other path, so the caller goes on."""
    path = urlsplit(handler.path).path
    if path != "/api/batches" and not path.startswith("/api/batches/"):
        return False
    parts = [unquote(part) for part in path.split("/") if part]
    try:
        check_request(handler, mutation=mutation)
        if not mutation and parts == ["api", "batches"]:
            handler._send_json(200, {"base": "/output/", "batches": list_batches(output)})
        elif not mutation and len(parts) == 3:
            detail = batch_detail(output, parts[2])
            handler._send_json(200 if detail else 404, detail or {"error": "No such batch"})
        elif mutation and parts == ["api", "batches"]:
            handler._send_json(*submit(output, _body(handler)))
        elif mutation and len(parts) == 4 and parts[3] in ("pick", "rerun"):
            directory = _directory(output, parts[2])
            if directory is None:
                handler._send_json(404, {"error": "No such batch"})
            else:
                handler._send_json(*row_action(directory, parts[3], _body(handler)))
        elif mutation and len(parts) == 4 and parts[3] in ("resume", "cancel"):
            directory = _directory(output, parts[2])
            lock = batch.live_lock(batches_root(output))
            if directory is None and parts[3] == "cancel" and lock and lock.get("batch") == parts[2]:
                # starting: listed (and offered Stop) before its runner wrote any state
                directory = batches_root(output) / parts[2]
            if directory is None:
                handler._send_json(404, {"error": "No such batch"})
            elif parts[3] == "resume":
                retry = bool(_body(handler).get("retry_failed"))
                result = mcp_server.resume_batch(str(directory), retry_failed=retry)
                handler._send_json(202 if result["started"] else 422, result)
            else:
                handler._send_json(200, mcp_server.cancel_batch(str(directory)))
        else:
            handler._send_json(404, {"error": "Unknown batch endpoint"})
    except PermissionError as exc:
        _drain(handler)   # an unread body makes Windows reset the connection, not answer
        handler.close_connection = True
        handler._send_json(403, {"error": str(exc)})
    except (ValueError, KeyError, TypeError, OSError, batch.BatchError) as exc:
        handler.close_connection = True
        handler._send_json(422, {"started": False, "error": str(exc), "problems": [str(exc)]})
    return True
