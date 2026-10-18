"""Text to image in the studio: one picture at a time, as a job the page follows.

The engine (the Turbo recipe, the sd-cli command, the licence record) is
`image_to_3dlab/text_image.py`, shared with `scripts/text_image.py` and the batch step
`picture`; its notes on why the Turbo runs with its own schedule are there.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import gpu_lock, text_image
from image_to_3dlab.host import NVIDIA, host_platform
from image_to_3dlab.sdcpp import NO_GPU_HELP

# The engine half lives in the package, so the command line and the batches run the
# same thing; the names are kept here for the studio's own modules.
from image_to_3dlab.text_image import (  # noqa: F401 - re-exported for the studio
    ATTRIBUTION,
    BINARY,
    DEFAULTS,
    HF_HUB,
    LICENSE_NAME,
    LICENSE_URL,
    MAX_PROMPT,
    MAX_SIZE,
    MIN_SIZE,
    MODEL_ID,
    OUTPUT_RIGHTS,
    OUTPUT_ROOT,
    SAMPLERS,
    SIZE_STEP,
    T2I_DEFAULTS,
    TURBO_SIGMAS,
    TURBO_STEPS,
    WEIGHT_FILES,
    MissingWeights,
    build_command,
    clean_settings,
    failure_reason,
    find_weight,
    is_installed,
    parse_progress,
    prepare_prompt,
    provenance,
    resolve_weights,
    run_sdcli,
    slug,
    weight_manifest,
)


@dataclass
class ImageJob:
    id: str
    directory: Path
    output_path: Path
    prompt: str
    settings: dict[str, Any]
    # For steps built on this one (image_to_3dlab/concept): arguments put right after the
    # binary, and fields added to the sidecar (never over its own, such as the licence).
    extra_args: list[str] = field(default_factory=list)
    extra_provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.log_lines: deque[str] = deque(maxlen=400)
        self.condition = threading.Condition()
        self.process: subprocess.Popen[bytes] | None = None
        self.cancel_requested = False
        self.error: str | None = None

    def emit(self, event: dict[str, Any]) -> None:
        event = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def set_status(self, status: str, **extra: Any) -> None:
        self.status = status
        self.emit({"status": status, **extra})

    def describe(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "prompt": self.prompt,
            "settings": self.settings,
            "error": self.error,
            "elapsed_seconds": round(time.monotonic() - self.started, 1),
            "result_url": f"/api/image/{self.id}/result.png"
            if self.status == "done" else None,
            "last_event": self.events[-1] if self.events else None,
            "log_tail": "\n".join(self.log_lines)[-4000:],
            "picture": self.picture_path() if self.status == "done" else None,
        }

    def picture_path(self) -> str:
        """Where the picture sits relative to output/, as the studio's Library names it."""
        return f"images/{self.directory.name}/{self.output_path.name}"


class ImageJobManager:
    """One picture at a time. Two at once would only fight over the same memory."""

    def __init__(self, output_root: Path = OUTPUT_ROOT):
        self.output_root = output_root
        self.jobs: dict[str, ImageJob] = {}
        self.lock = threading.Lock()
        self.active: str | None = None

    def create(self, prompt: str, settings: dict[str, Any], extra_args: list[str] | None = None,
               extra_provenance: dict[str, Any] | None = None, name: str | None = None) -> ImageJob:
        with self.lock:
            if self.active is not None:
                running = self.jobs.get(self.active)
                if running and running.status in {"queued", "running", "cancelling"}:
                    raise RuntimeError("an image is already being generated")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{slug(name or prompt)}__{stamp}"
            directory = self.output_root / name
            directory.mkdir(parents=True, exist_ok=True)
            job = ImageJob(uuid.uuid4().hex, directory, directory / f"{name}.png",
                           prompt, settings, list(extra_args or []), dict(extra_provenance or {}))
            self.jobs[job.id] = job
            self.active = job.id
            return job

    def finish(self, job: ImageJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status not in {"queued", "running"}:
            return False
        job.cancel_requested = True
        job.set_status("cancelling")
        if job.process and job.process.poll() is None:
            job.process.terminate()
        return True


# how often a picture waiting for the GPU looks again (and tells the page it still waits)
GPU_POLL = 10.0


class _Cancelled(Exception):
    pass


def run_job(job: ImageJob, manager: ImageJobManager,
            weights: dict[str, Path] | None = None) -> None:
    """Drive one sd-cli run to completion, emitting progress as it goes. sd-cli runs
    holding the shared GPU lock, as every GPU job here does; while another session holds
    it the job stays queued, says whom it waits for, and can be cancelled."""
    try:
        weights = weights or resolve_weights()

        def waiting(seconds: float) -> None:
            if job.cancel_requested:
                raise _Cancelled
            found = gpu_lock.holder(text_image.GPU_LOCK)
            job.set_status("queued", message=f"Waiting for the GPU: {found[1] if found else 'busy'}")
            time.sleep(seconds)
            if job.cancel_requested:
                raise _Cancelled

        def gpu(session: str, path: Path = text_image.GPU_LOCK):
            """The GPU, waited for the way the page can cancel (the rewriter and sd-cli)."""
            return gpu_lock.gpu(session, path=path, poll=GPU_POLL, sleep=waiting)

        if job.settings.get("rewrite") and not job.extra_args:
            from image_to_3dlab import prompt_rewrite

            gap = prompt_rewrite.missing()
            job.set_status("queued", message="Rewriting the prompt (Qwen-Image's own recipe, on this machine)"
                           if gap is None else f"Drawing without prompt rewriting: {gap} is not installed "
                                               f"({prompt_rewrite.SETUP_HINT})")
        sent, notes = prepare_prompt(job.prompt, job.settings, weights=weights, references=bool(job.extra_args),
                                     log=job.log_lines.append, lock_path=text_image.GPU_LOCK, lock=gpu)
        if job.cancel_requested:
            raise _Cancelled
        command = build_command(sent, job.settings, job.output_path, weights)
        if job.extra_args:
            command = command[:1] + job.extra_args + command[1:]

        def on_line(line: str) -> None:
            job.log_lines.append(line)
            event = parse_progress(line)
            if event:
                job.emit(event)

        with gpu("studio-image"):
            if job.cancel_requested:
                raise _Cancelled
            job.set_status("running", total_steps=job.settings["steps"])
            started = time.monotonic()
            # Only where a GPU is expected from a runtime-loaded backend. The Mac build is
            # Metal and says nothing of the sort.
            code, _, no_gpu = run_sdcli(command, watch_gpu=host_platform() == NVIDIA, on_line=on_line,
                                        on_process=lambda process: setattr(job, "process", process))
        elapsed = time.monotonic() - started
        if job.cancel_requested:
            job.set_status("cancelled")
        elif no_gpu:
            job.error = NO_GPU_HELP
            job.set_status("error", error=job.error,
                           log="\n".join(list(job.log_lines)[-12:]))
        elif code != 0 or not job.output_path.exists():
            reason = failure_reason(job.log_lines)
            job.error = f"sd-cli exited with code {code}"
            if reason:
                job.error += f": {reason}"
            job.set_status("error", error=job.error,
                           log="\n".join(list(job.log_lines)[-12:]))
        else:
            sidecar = job.output_path.with_suffix(".provenance.json")
            record = provenance(job.prompt, job.settings, elapsed, job.output_path, prompt_sent=sent)
            sidecar.write_text(json.dumps({**job.extra_provenance, **notes, **record}, indent=2), encoding="utf-8")
            job.set_status("done", result_url=f"/api/image/{job.id}/result.png",
                           elapsed_seconds=round(elapsed, 1))
    except _Cancelled:
        job.set_status("cancelled")
    except MissingWeights as exc:
        job.error = str(exc)
        job.set_status("error", error=job.error, needs_setup=True)
    except Exception as exc:  # noqa: BLE001 - the page needs the reason, whatever it is
        job.error = f"{type(exc).__name__}: {exc}"
        job.set_status("error", error=job.error)
    finally:
        manager.finish(job)


MANAGER = ImageJobManager()


def with_style(prompt: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The prompt with the style bible's words when the request names one (`bible`: "-"
    for the active one, and the asset's `kind`), and the provenance that says so."""
    from image_to_3dlab.style import apply

    style = apply.resolve(payload.get("bible"), payload.get("kind"))
    return (apply.styled(prompt, style), {"style": apply.record(style)}) if style else (prompt, {})


def t2i_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Text to image as the model card says (size, transparency, rewriting), where the
    request does not say otherwise."""
    return {**{k: T2I_DEFAULTS[k] for k in ("width", "height", "transparent", "rewrite")}, **raw}


def start(prompt: str, raw_settings: dict[str, Any], extra_provenance: dict[str, Any] | None = None) -> ImageJob:
    settings = clean_settings(t2i_settings(raw_settings))
    job = MANAGER.create(prompt, settings, extra_provenance=extra_provenance)
    threading.Thread(target=run_job, args=(job, MANAGER), daemon=True).start()
    return job
