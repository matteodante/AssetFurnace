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

import subprocess
import sys
from pathlib import Path
from typing import Any

from . import batch
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


def _check(spec_path: str, intent: dict[str, Any]) -> tuple[batch.BatchSpec | None, list[str]]:
    try:
        spec = batch.load_batch(Path(spec_path), intent=intent)
    except batch.BatchError as exc:
        return None, exc.problems
    return spec, batch.preflight(spec)


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


def launch(command: list[str], log_path: Path) -> int:
    """Start the runner detached from this server, in its own process group."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        process = subprocess.Popen(command, cwd=str(REPO), stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   **group_popen_kwargs())
    return process.pid


def submit_batch(spec: str, retry_failed: bool = False, use_case: str | None = None,
                 distribution: str | None = None, commercial: bool | None = None,
                 ) -> dict[str, Any]:
    """Start a batch in the background and return at once. Running the same spec again
    resumes it; retry_failed also re-runs rows that failed. One batch runs at a time.
    use_case, distribution and commercial are for a CSV spec, as in validate_batch."""
    intent = _intent(use_case, distribution, commercial)
    loaded, problems = _check(spec, intent)
    if loaded is None or problems:
        return {"started": False, "problems": problems}
    directory = loaded.directory
    holder = batch.lock_holder(directory)
    if holder is not None:
        problem = f"batch {loaded.name} is already running (pid {holder})"
        return {"started": False, "problems": [problem]}
    others = [path.name for path in batch.running_batches(loaded.output_dir)]
    if others:
        problem = f"another batch is using the GPU: {', '.join(others)}"
        return {"started": False, "problems": [problem]}
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
    pid = launch(command, log_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / batch.LOCK_FILE).write_text(str(pid))
    return {"started": True, "pid": pid, "name": loaded.name, "directory": str(directory),
            "log": str(log_path)}


def resume_batch(batch_ref: str, retry_failed: bool = False) -> dict[str, Any]:
    """Start a stopped batch again from where it left off, with the spec and intent it ran
    with. retry_failed also re-runs the rows that failed."""
    directory = Path(batch_ref).expanduser().resolve()
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
    if path.is_dir():
        return path
    return batch.load_batch(path).directory


def batch_status(batch_ref: str) -> dict[str, Any]:
    """The state of a batch, by its folder or its spec: counts per status, and per row its
    status, the reason it failed and the status of each step. `active` is the pid of the
    process running it, or null."""
    directory = _directory(batch_ref)
    state = batch.read_json(directory / batch.STATE_FILE) or {}
    rows = [{"id": row_id, "status": entry.get("status"), "reason": entry.get("reason"),
             "steps": {name: step.get("status")
                       for name, step in entry.get("steps", {}).items()},
             "finished_at": entry.get("finished_at")}
            for row_id, entry in state.get("rows", {}).items()]
    return {"name": state.get("name", directory.name), "directory": str(directory),
            "active": batch.lock_holder(directory), "counts": batch.summary(state),
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
               "files": [absolute(path) for path in row.get("files", [])]} for row in rows]
    return {"name": index.get("batch", directory.name), "directory": str(directory),
            "rows": listed}


def cancel_batch(batch_ref: str) -> dict[str, Any]:
    """Stop a running batch. The row in progress goes back in the queue: submitting the
    same spec again resumes from it."""
    directory = _directory(batch_ref)
    holder = batch.lock_holder(directory)
    if holder is None:
        return {"cancelled": False, "reason": "the batch is not running"}
    terminate_group(holder)
    return {"cancelled": True, "pid": holder}


def list_steps() -> dict[str, Any]:
    """The step names a row can ask for, and whether each is available yet. A step that
    is not available is skipped with a warning, not an error."""
    return {"steps": batch.available_steps()}


TOOLS = (validate_batch, submit_batch, resume_batch, batch_status, list_outputs, cancel_batch,
         list_steps)


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise SystemExit(INSTALL_HINT) from exc
    server = MCPServer(
        "assetfurnace-batches",
        instructions="Run AssetFurnace asset batches: validate a spec, submit it, follow "
                     "its state, collect its files. Paths are on this machine.",
    )
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main() -> int:
    build_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
