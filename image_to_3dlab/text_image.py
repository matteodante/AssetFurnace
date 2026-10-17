"""Text to image, one step upstream of everything else in this repo.

The engine half: settings, the sd-cli command, its output, the licence record, and
`generate()`, one picture start to end. The studio's jobs (`viewer/image_api.py`), the
command line (`scripts/text_image.py`) and the batch step `picture` all run this.

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

**The licence travels with the image.** Qwen-Image's licence allows it for non-commercial
use only, research or evaluation (Qwen Research License 1(i), 2(a)). Every run writes a
sidecar saying so, with the licence, because a PNG in a folder six months from now
remembers nothing on its own.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_to_3dlab import gpu_lock
from image_to_3dlab.host import NVIDIA, executable, host_platform
from image_to_3dlab.provenance import LICENSES, QWEN_OUTPUT_RIGHTS
from image_to_3dlab.sdcpp import NO_GPU_HELP, BackendWatch

REPO = Path(__file__).resolve().parents[1]
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




def run_sdcli(command: list[str], *, watch_gpu: bool = False,
              on_line: Callable[[str], None] = lambda line: None,
              on_process: Callable[[subprocess.Popen], None] = lambda process: None,
              ) -> tuple[int, list[str], bool]:
    """Run sd-cli to the end, line by line: (exit code, its last lines, stopped for want
    of a GPU). With `watch_gpu`, a run that loads only the CPU backend is stopped at once
    (on an NVIDIA machine that is a broken install, not a slow picture)."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(REPO))
    on_process(process)
    assert process.stdout is not None
    watch = BackendWatch() if watch_gpu else None
    no_gpu = False
    lines: list[str] = []
    buffer = b""
    while True:
        # read1, not read: read(256) waits for 256 bytes, so a short line sat unseen
        # until more output came, which is useless for stopping a CPU-only run.
        chunk = process.stdout.read1(256)
        if not chunk:
            break
        buffer += chunk
        # sd-cli redraws its bar with \r, so split on both or the whole run arrives as
        # one unreadable line at the end.
        parts = re.split(rb"[\r\n]", buffer)
        buffer = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            del lines[:-400]
            if watch is not None and not no_gpu and watch.feed(line):
                no_gpu = True
                process.terminate()
            on_line(line)
    return process.wait(), lines, no_gpu


# The GPU lock file; a test points it elsewhere.
GPU_LOCK = gpu_lock.LOCK


def generate(prompt: str, raw_settings: dict[str, Any], output_path: Path, *,
             weights: dict[str, Path] | None = None, extra_args: list[str] | None = None,
             extra_provenance: dict[str, Any] | None = None,
             log: Callable[[str], None] = lambda line: None, session: str = "text_image") -> Path:
    """One picture, start to end, holding the shared GPU lock while sd-cli runs; the
    picture and its `.provenance.json` beside it. RuntimeError with sd-cli's own reason
    when it fails; MissingWeights when the model is not installed (nothing downloads)."""
    if not prompt.strip():
        raise ValueError("the prompt is empty")
    if len(prompt) > MAX_PROMPT:
        raise ValueError(f"the prompt is longer than {MAX_PROMPT} characters")
    settings = clean_settings(raw_settings)
    weights = weights or resolve_weights()
    command = build_command(prompt, settings, output_path, weights)
    if extra_args:
        command = command[:1] + list(extra_args) + command[1:]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(line: str) -> None:
        event = parse_progress(line)
        if event and event.get("phase") == "sampling":
            log(f"step {event['step']}/{event['total_steps']}")

    started = time.monotonic()
    with gpu_lock.gpu(session, path=GPU_LOCK):
        code, lines, no_gpu = run_sdcli(command, watch_gpu=host_platform() == NVIDIA, on_line=progress)
    elapsed = time.monotonic() - started
    if no_gpu:
        raise RuntimeError(NO_GPU_HELP)
    if code != 0 or not output_path.is_file():
        reason = failure_reason(lines)
        raise RuntimeError(f"sd-cli exited with code {code}" + (f": {reason}" if reason else ""))
    record = provenance(prompt, settings, elapsed, output_path)
    output_path.with_suffix(".provenance.json").write_text(
        json.dumps({**(extra_provenance or {}), **record}, indent=2), encoding="utf-8")
    return output_path
