"""Text to image, one step upstream of everything else in this repo.

Every other tool here starts from a picture. This one makes the picture, for people who
do not have one yet. It drives `stable-diffusion.cpp` (a prebuilt Metal binary in
`vendor/sdcpp/`) against Qwen-Image 2.1 Turbo in GGUF form.

Two things are deliberate:

**The model is the Turbo, run with its own schedule.** Qwen-Image 2.1 comes in two
checkpoints. The base one is meant for about 40 steps with guidance (`--cfg-scale 6.0` in
the sd.cpp guide); the Turbo is distilled for 8 steps at cfg 1.0 with its own sigmas
(`--sigmas qwen-image-2.1-turbo`). Until 2026-10-16 this file ran the base checkpoint with
the Turbo's numbers (8 steps, cfg 1.0, the default schedule): the sampler stopped half
way, and pictures came out soft, with frayed edges and faint "ghost" shapes, worst when
reference pictures were given. Compared on six subjects with the same prompts and seeds
(knight, knight with three references, a helmet on its own, wolf, well, sword; RTX 5070
Ti): base at 8 steps was blurred every time; base at 40 steps and cfg 6 was clean but took
80 s, and 8 minutes with three references; the Turbo was as clean or cleaner in 15 s, 48 s
with three references. At 1536 px the Turbo added things nobody asked for around the
subject (pauldrons around a sword, a mask for a helmet), so the default is 1024 px; the
3D generators downstream read about 518 px anyway. The step count is the length of the
sigma list, so `steps` is recorded but not taken from the form.

**The licence travels with the image.** The pictures Qwen-Image makes are yours, per Qwen's
own statement. Every run writes a sidecar saying so, with the licence, because
a PNG in a folder six months from now remembers nothing on its own.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab.host import NVIDIA, executable, host_platform
from image_to_3dlab.provenance import LICENSES, QWEN_OUTPUT_RIGHTS
from image_to_3dlab.sdcpp import NO_GPU_HELP, BackendWatch

HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
BINARY = executable(REPO / "vendor" / "sdcpp", "sd-cli")
# Every picture lands in output/images/. Its licence (Qwen's research licence, today) is
# recorded in the provenance written beside it, which is what travels with the file; a
# folder name did not, and sorting by it only made the pictures harder to find.
OUTPUT_ROOT = REPO / "output" / "images"

MODEL_ID = "qwen-image-2.1-turbo"
LICENSE_NAME = "Qwen Research License"
LICENSE_URL = "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE"
ATTRIBUTION = "Built with Qwen"
OUTPUT_RIGHTS = QWEN_OUTPUT_RIGHTS

# Each weight file, as (hugging face cache directory, filename glob).
WEIGHT_FILES = {
    "diffusion_model": ("models--unsloth--Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q8_0.gguf"),
    "llm": ("models--Qwen--Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
    "vae": ("models--Comfy-Org--Qwen-Image-2.1", "qwen_image_2.1_vae_bf16.safetensors"),
}

# The Turbo's own schedule (sd.cpp preset): eight sigmas, so eight steps.
TURBO_SIGMAS = "qwen-image-2.1-turbo"
TURBO_STEPS = 8

DEFAULTS: dict[str, Any] = {
    "width": 1024,
    "height": 1024,
    "steps": TURBO_STEPS,
    "cfg_scale": 1.0,
    "sampler": "euler",
    "seed": 42,
    "negative_prompt": "",
}

SAMPLERS = ("euler", "euler_a", "heun", "dpm2", "dpmpp2m", "lcm")
# sd.cpp requires both dimensions to be a multiple of 32; anything else fails deep in the
# run rather than at the door.
SIZE_STEP = 32
MIN_SIZE, MAX_SIZE = 256, 1536
MAX_PROMPT = 2000

# "  |=====>    | 7/10 - 13.38s/it"
PROGRESS = re.compile(r"\|\s*(\d+)/(\d+)\s*-\s*([\d.]+)s/it")
COMPLETED = re.compile(r"generate_image completed in ([\d.]+)s")
DECODE = re.compile(r"decode_first_stage completed, taking ([\d.]+)s")


class MissingWeights(RuntimeError):
    """Raised with the names of what is absent, so the page can say what to fetch."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            "Qwen-Image is not set up yet. Missing: " + ", ".join(missing)
            + ". Open Setup & Status to download it."
        )


def find_weight(directory: str, filename: str) -> Path | None:
    """Locate one cached file, whatever snapshot hash it landed under."""
    root = HF_HUB / directory
    if not root.exists():
        return None
    matches = sorted(root.glob(f"snapshots/*/**/{filename}"))
    return matches[0] if matches else None


def resolve_weights() -> dict[str, Path]:
    found: dict[str, Path] = {}
    missing: list[str] = []
    for key, (directory, filename) in WEIGHT_FILES.items():
        path = find_weight(directory, filename)
        if path is None:
            missing.append(filename)
        else:
            found[key] = path
    if missing:
        raise MissingWeights(missing)
    return found


def is_installed() -> bool:
    """Both halves, build and weights, the way the rest of the viewer separates them."""
    if not BINARY.exists():
        return False
    try:
        resolve_weights()
    except MissingWeights:
        return False
    return True


def clean_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce whatever the form sent into something sd-cli will accept.

    Sizes are rounded to a multiple of 32 rather than rejected: a user typing 700 means
    "about this big", and failing them for it would be pedantry. Everything else is
    clamped to a range the binary tolerates.
    """
    settings = dict(DEFAULTS)
    for key in ("width", "height"):
        value = raw.get(key, DEFAULTS[key])
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = DEFAULTS[key]
        number = max(MIN_SIZE, min(MAX_SIZE, number))
        settings[key] = max(SIZE_STEP, round(number / SIZE_STEP) * SIZE_STEP)
    # The Turbo's schedule fixes the step count; a number from the form would only be a
    # label that does not match what ran.
    settings["steps"] = TURBO_STEPS
    try:
        settings["cfg_scale"] = max(0.0, min(20.0, float(raw.get("cfg_scale", DEFAULTS["cfg_scale"]))))
    except (TypeError, ValueError):
        settings["cfg_scale"] = DEFAULTS["cfg_scale"]
    try:
        settings["seed"] = int(raw.get("seed", DEFAULTS["seed"]))
    except (TypeError, ValueError):
        settings["seed"] = DEFAULTS["seed"]
    sampler = str(raw.get("sampler", DEFAULTS["sampler"]))
    settings["sampler"] = sampler if sampler in SAMPLERS else DEFAULTS["sampler"]
    settings["negative_prompt"] = str(raw.get("negative_prompt", ""))[:MAX_PROMPT]
    return settings


def build_command(prompt: str, settings: dict[str, Any], output_path: Path,
                  weights: dict[str, Path], binary: Path = BINARY) -> list[str]:
    """The full argument list, as a pure function so a test can read it without a GPU.

    `--offload-to-cpu` is deliberately absent. The upstream recipe includes it, but it
    shuttles weights between RAM and VRAM to save video memory a unified-memory Mac does
    not have separately; measured, it was both slower and used MORE peak RAM.
    """
    command = [
        str(binary),
        "--diffusion-model", str(weights["diffusion_model"]),
        "--vae", str(weights["vae"]),
        "--llm", str(weights["llm"]),
        "-p", prompt,
        "--cfg-scale", str(settings["cfg_scale"]),
        "--sampling-method", settings["sampler"],
        "--sigmas", TURBO_SIGMAS,
        "-W", str(settings["width"]),
        "-H", str(settings["height"]),
        "--seed", str(settings["seed"]),
        "--diffusion-fa",
        "-v",
        "-o", str(output_path),
    ]
    if settings.get("negative_prompt"):
        command[command.index("-p") + 2:command.index("-p") + 2] = [
            "--negative-prompt", settings["negative_prompt"]
        ]
    return command


def parse_progress(line: str) -> dict[str, Any] | None:
    """Turn one line of sd-cli chatter into an event, or None if it says nothing useful.

    sd-cli redraws its progress bar with carriage returns, so a single read can carry
    several updates; callers split on both \\r and \\n before calling this.
    """
    match = PROGRESS.search(line)
    if match:
        current, total, seconds = int(match[1]), int(match[2]), float(match[3])
        remaining = max(0, total - current) * seconds
        return {
            "phase": "sampling",
            "step": current,
            "total_steps": total,
            "seconds_per_step": seconds,
            "percent": round(100.0 * current / total, 1) if total else 0.0,
            "eta_seconds": round(remaining, 1),
        }
    match = DECODE.search(line)
    if match:
        return {"phase": "decoding", "decode_seconds": float(match[1]), "percent": 100.0}
    match = COMPLETED.search(line)
    if match:
        return {"phase": "finished", "generate_seconds": float(match[1]), "percent": 100.0}
    return None


def failure_reason(log_lines) -> str | None:
    """sd-cli's last meaningful words, or None if it said nothing but progress.

    A failed run used to surface as "sd-cli exited with code 2" while the actual
    explanation -- a rejected flag, a missing file -- sat unread in the log tail, and
    the image tab has no log panel to find it in. So the one line that matters goes
    into the error itself; it is shown as plain text, hence one line only.
    """
    for line in reversed(list(log_lines)):
        text = line.strip()
        if not text or PROGRESS.search(text):
            continue
        return text
    return None


def slug(prompt: str, limit: int = 40) -> str:
    """A short folder-safe name taken from the prompt's first few words."""
    words = re.sub(r"[^a-z0-9\s-]", "", prompt.lower()).split()
    name = "-".join(words)[:limit].strip("-")
    return name or "image"


def weight_manifest() -> dict[str, dict[str, str]]:
    """Which repository and file each part of the model came from.

    Named fields rather than the raw WEIGHT_FILES tuples: a sidecar that records
    "('models--leejet--Qwen-Image-2.1-GGUF', 'qwen_image_2.1-Q8_0.gguf')" as a single
    string is a tuple's repr leaking into a record meant to be read by people.
    """
    return {
        key: {"cache_dir": directory, "file": filename}
        for key, (directory, filename) in WEIGHT_FILES.items()
    }


def provenance(prompt: str, settings: dict[str, Any], seconds: float,
               output_path: Path) -> dict[str, Any]:
    """What this picture is, and what the licence lets you do with it."""
    return {
        "schema_version": 1,
        "kind": "generated-image",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "id": MODEL_ID,
            "runtime": "stable-diffusion.cpp (Metal)",
            "weights": weight_manifest(),
        },
        "license": {
            "name": LICENSE_NAME,
            "url": LICENSE_URL,
            "classification": LICENSES["qwen-image-2.1"].classification,
            "attribution": ATTRIBUTION,
            # Key kept for sidecar compatibility; it now says what derivatives do NOT inherit.
            "inherited_by_derivatives": OUTPUT_RIGHTS,
        },
        "prompt": prompt,
        "settings": settings,
        "elapsed_seconds": round(seconds, 1),
        "output": output_path.name,
    }


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


def run_job(job: ImageJob, manager: ImageJobManager,
            weights: dict[str, Path] | None = None) -> None:
    """Drive one sd-cli run to completion, emitting progress as it goes."""
    try:
        weights = weights or resolve_weights()
        command = build_command(job.prompt, job.settings, job.output_path, weights)
        if job.extra_args:
            command = command[:1] + job.extra_args + command[1:]
        job.set_status("running", total_steps=job.settings["steps"])
        started = time.monotonic()
        job.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO),
        )
        assert job.process.stdout is not None
        # Only where a GPU is expected from a runtime-loaded backend. The Mac build is
        # Metal and says nothing of the sort.
        watch = BackendWatch() if host_platform() == NVIDIA else None
        no_gpu = False
        buffer = b""
        while True:
            # read1, not read: read(256) waits for 256 bytes, so a short line sat unseen
            # until more output came, which is useless for stopping a CPU-only run.
            chunk = job.process.stdout.read1(256)
            if not chunk:
                break
            buffer += chunk
            # sd-cli redraws its bar with \r, so split on both or the whole run arrives
            # as one unreadable line at the end.
            parts = re.split(rb"[\r\n]", buffer)
            buffer = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                job.log_lines.append(line)
                if watch is not None and not no_gpu and watch.feed(line):
                    no_gpu = True
                    job.process.terminate()
                event = parse_progress(line)
                if event:
                    job.emit(event)
        code = job.process.wait()
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
            record = provenance(job.prompt, job.settings, elapsed, job.output_path)
            sidecar.write_text(json.dumps({**job.extra_provenance, **record}, indent=2))
            job.set_status("done", result_url=f"/api/image/{job.id}/result.png",
                           elapsed_seconds=round(elapsed, 1))
    except MissingWeights as exc:
        job.error = str(exc)
        job.set_status("error", error=job.error, needs_setup=True)
    except Exception as exc:  # noqa: BLE001 - the page needs the reason, whatever it is
        job.error = f"{type(exc).__name__}: {exc}"
        job.set_status("error", error=job.error)
    finally:
        manager.finish(job)


MANAGER = ImageJobManager()


def start(prompt: str, raw_settings: dict[str, Any]) -> ImageJob:
    settings = clean_settings(raw_settings)
    job = MANAGER.create(prompt, settings)
    threading.Thread(target=run_job, args=(job, MANAGER), daemon=True).start()
    return job
