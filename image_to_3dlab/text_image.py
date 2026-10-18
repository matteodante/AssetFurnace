"""Text to image, one step upstream of everything else in this repo.

The engine half: settings, the sd-cli command, its output, the licence record, and
`generate()`, one picture start to end. The studio's jobs (`viewer/image_api.py`), the
command line (`scripts/text_image.py`) and the batch step `picture` all run this.

Every other tool here starts from a picture. This one makes the picture, for people who
do not have one yet. It drives `stable-diffusion.cpp` (a prebuilt binary in `vendor/sdcpp/`:
Metal on a Mac, CUDA on Windows, Vulkan or ROCm on Linux) against Qwen-Image 2.1 Turbo in
GGUF form.

Two things are deliberate:

**The model is the Turbo, run with its own schedule.** Qwen-Image 2.1 comes in two
checkpoints. The base one is meant for about 40 steps with guidance (`--cfg-scale 6.0` in
the sd.cpp guide); the Turbo is distilled for 8 steps at cfg 1.0 with its own sigmas
(`--sigmas qwen-image-2.1-turbo`). The base checkpoint at the Turbo's numbers stops half way:
pictures come out soft, with frayed edges and faint "ghost" shapes, worst with reference
pictures. Measured on six subjects with the same prompts and seeds (RTX 5070 Ti): the base
at 8 steps was blurred every time; at 40 steps and cfg 6 it was clean but took 80 s, and 8
minutes with three references; the Turbo was as clean or cleaner in 15 s, 48 s with three
references. The step count is the length of the sigma list, so `steps` is recorded but not
taken from the form.

**Text to image runs as the model card says (user decision, 2026-10-17).** The Turbo's card
and the Qwen-Image-2.1 README give the resolution presets (2048x2048 and six other ratios),
a prompt template for a transparent background, and an official prompt rewriting step;
`T2I_DEFAULTS` and `prepare_prompt()` apply all three to text to image (the studio's
Create and `scripts/text_image.py`). Measured on an RTX 5070 Ti (16 GB, Q8_0, sd-cli
master-961): 2048x2048 runs in 72 s against 14 s at 1024 (peak 14.7 GB; sd-cli retries the
VAE decode tiled), and the template gives a real alpha (62 to 75% transparent) at no extra
time. With reference pictures 2048 does not fit (the vision encoder fails), so runs that
draw from references (concepts, the pieces of a character) keep `DEFAULTS`, 1024, and their
prompt as written. `DEFAULTS` is also what the batches start from, through the step
catalogue.

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

# Qwen-Image-2.1's resolution presets: the Turbo's model card says "Use the same resolution
# presets as Qwen-Image-2.1" and lists these (HF card and QwenLM/Qwen-Image-2.1 README,
# read 2026-10-17).
CARD_SIZES = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
    "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752),
}
# 2048 with reference pictures does not fit 16 GB (2026-10-17: the vision encoder fails)
REFERENCE_SIZE = 1024

DEFAULTS: dict[str, Any] = {
    "width": REFERENCE_SIZE,
    "height": REFERENCE_SIZE,
    "steps": TURBO_STEPS,
    "cfg_scale": 1.0,
    "sampler": "euler",
    "seed": 42,
    "negative_prompt": "",
    "transparent": False,
    "rewrite": False,
}
# Text to image, as the card says: its default size, a transparent background, and the
# official prompt rewriting (see the module docstring and prepare_prompt).
T2I_DEFAULTS: dict[str, Any] = {**DEFAULTS, "width": CARD_SIZES["1:1"][0], "height": CARD_SIZES["1:1"][1],
                                "transparent": True, "rewrite": True}

SAMPLERS = ("euler", "euler_a", "heun", "dpm2", "dpmpp2m", "lcm")
# sd.cpp requires both dimensions to be a multiple of 32; anything else fails deep in the
# run rather than at the door.
SIZE_STEP = 32
MIN_SIZE, MAX_SIZE = 256, max(max(size) for size in CARD_SIZES.values())
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
    for key in ("transparent", "rewrite"):
        settings[key] = _yes(raw.get(key, DEFAULTS[key]))
    return settings


def _yes(value: Any) -> bool:
    """A switch from a form or a JSON body: only a clear yes turns it on."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# QwenLM/Qwen-Image-2.1 README, "Transparent image generation (RGBA)", and the same template
# in stable-diffusion.cpp's docs/qwen_image_2.1.md ("Alpha channel"), read 2026-10-17.
RGBA_HEAD = "This is an RGBA image with transparency."
RGBA_TAIL = "The image has alpha channel and the background is transparent."


def rgba_prompt(words: str) -> str:
    """`words` in the official transparent-background template (once)."""
    words = words.strip()
    if words.startswith(RGBA_HEAD) and words.endswith(RGBA_TAIL):
        return words
    return f"{RGBA_HEAD} {words.rstrip('.').strip()}. {RGBA_TAIL}"


def prepare_prompt(prompt: str, settings: dict[str, Any], *, weights: dict[str, Path] | None = None,
                   references: bool = False, log: Callable[[str], None] = lambda line: None,
                   lock_path: Path | None = None, lock: Callable | None = None) -> tuple[str, dict[str, Any]]:
    """The prompt sd-cli gets, and what the record should say about it.

    Text to image with `rewrite`: QwenLM's rewriting (image_to_3dlab/prompt_rewrite.py) on a
    brief that already asks for transparency when `transparent` is on, since the rewriter
    keeps what a brief fixes and invents what it leaves open, a background included. Then
    the transparent-background template. A run from reference pictures keeps its prompt.
    `lock` takes the GPU for the rewriter (gpu_lock.gpu, or the studio's cancellable wait)."""
    if references:
        return prompt, {}
    notes: dict[str, Any] = {}
    sent = prompt
    if settings.get("rewrite"):
        from image_to_3dlab import prompt_rewrite

        brief = rgba_prompt(prompt) if settings.get("transparent") else prompt
        model = (weights or resolve_weights())["llm"]
        found = prompt_rewrite.rewrite(brief, model=model, log=log, lock_path=lock_path,
                                       **({"lock": lock} if lock else {}))
        if found is not None:
            sent = found.pop("prompt")
            notes["prompt_rewrite"] = {"applied": True, **found}
        else:
            gap = prompt_rewrite.missing()
            notes["prompt_rewrite"] = {"applied": False, "reason": f"{gap} is not installed ({prompt_rewrite.SETUP_HINT})"
                                       if gap else "the rewriter gave no usable answer"}
    if settings.get("transparent"):
        sent = rgba_prompt(sent)
    return sent, notes


# stable-diffusion.cpp computes the negative branch only when the guidance is not 1
# (resolve_guidance in src/pipeline/request.cpp, master f0aac52, read 2026-10-17); the
# Turbo's card says "CFG=1 by default". At 1 a negative prompt changes nothing.
NEGATIVE_AT_CFG_1 = ("not applied: at cfg_scale 1 stable-diffusion.cpp does not compute the "
                     "negative branch; raise cfg_scale above 1, or put what to avoid in the prompt")


def negative_applies(settings: dict[str, Any]) -> bool:
    """Whether the run will use the negative prompt at all."""
    return bool(settings.get("negative_prompt")) and float(settings.get("cfg_scale", 1.0)) > 1.0


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
    if negative_applies(settings):
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


# The GPU backend of the sd-cli build scripts/bootstrap_qwen_image.py installs on each machine
# (its BUILDS; a test holds the two together).
RUNTIMES = {"macos-arm64": "Metal", "windows-nvidia": "CUDA 12", "linux-nvidia": "Vulkan", "linux-amd": "ROCm"}


def runtime(target: str | None = None) -> str:
    """What drew the picture on this machine, as its record says it."""
    from image_to_3dlab.host import build_target

    backend = RUNTIMES.get(target or build_target() or "")
    return f"stable-diffusion.cpp ({backend})" if backend else "stable-diffusion.cpp"


def provenance(prompt: str, settings: dict[str, Any], seconds: float,
               output_path: Path, prompt_sent: str | None = None) -> dict[str, Any]:
    """What this picture is, and what the licence lets you do with it. A negative prompt
    the run could not use is recorded as such, not among the settings that made it."""
    settings = dict(settings)
    ignored = None
    if settings.get("negative_prompt") and not negative_applies(settings):
        ignored = {"text": settings.pop("negative_prompt"), "reason": NEGATIVE_AT_CFG_1}
    record = {
        "schema_version": 1,
        "kind": "generated-image",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "id": MODEL_ID,
            "runtime": runtime(),
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
    if ignored:
        record["negative_prompt_not_applied"] = ignored
    if prompt_sent is not None and prompt_sent != prompt:
        record["prompt_sent"] = prompt_sent      # rewritten, or in the transparency template
    if settings.get("transparent") or settings.get("rewrite") or (
            settings.get("width"), settings.get("height")) in CARD_SIZES.values():
        record["settings_source"] = ("Qwen-Image-2.1-Turbo model card and QwenLM/Qwen-Image-2.1 README "
                                     "(resolution presets, RGBA template, prompt rewriting), read 2026-10-17")
    return record




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
    output_path = Path(output_path).resolve()   # sd-cli runs from the repository, not the caller's folder
    settings = clean_settings(raw_settings)
    if settings["negative_prompt"] and not negative_applies(settings):
        log(f"negative prompt {NEGATIVE_AT_CFG_1}")
    weights = weights or resolve_weights()
    sent, notes = prepare_prompt(prompt, settings, weights=weights, references=bool(extra_args), log=log,
                                 lock_path=GPU_LOCK)
    command = build_command(sent, settings, output_path, weights)
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
    record = provenance(prompt, settings, elapsed, output_path, prompt_sent=sent)
    output_path.with_suffix(".provenance.json").write_text(
        json.dumps({**(extra_provenance or {}), **notes, **record}, indent=2), encoding="utf-8")
    return output_path
