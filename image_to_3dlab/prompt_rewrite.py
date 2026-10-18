"""Qwen-Image 2.1's official prompt rewriting, run on this machine.

The README of QwenLM/Qwen-Image-2.1 (read 2026-10-17): "we recommend using the official
prompt rewriting models to expand short prompts". Those are two fine-tuned Qwen3.5-VL 9B
checkpoints (Qwen-Image-2.1-PE-T2I / -PE-I2I) that are not installed here. What is: the
text encoder Qwen-Image already uses, Qwen3-VL-8B-Instruct (Q4_K_M GGUF), and llama.cpp
(`vendor/llama-cpp`, the style judge's build). So this runs QwenLM's own t2i system prompt
(`prompt_rewrite/prompts/system_prompt_t2i.txt`, pinned to a commit and a SHA-256) with
their production t2i sampling (`prompt_rewrite/pe_core.py`: temperature 1.0, top-p 0.95,
top-k 20, min-p 0, presence penalty 1.5) on that model. It is the official recipe on a
base model, not the fine-tuned rewriter; the record says which.

Nothing here calls a remote API. Qwen-Image's setup installs both the system prompt and
llama.cpp. When either is missing (an install from before llama.cpp came with it), the
picture is drawn from the prompt as written, and the studio and the run say what to install.

Measured 2026-10-17 on an RTX 5070 Ti: 6 to 18 s a prompt, model load included. A brief
that leaves the background open gets one invented (a castle courtyard behind a helmet),
which is why text to image asks for transparency in the brief itself: the rewriter keeps
what the brief fixes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from image_to_3dlab import gpu_lock

REPO = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT_COMMIT = "ab96caa944157ca28a4f36b695c3047549f48f98"   # QwenLM/Qwen-Image-2.1, 2026-10-14
SYSTEM_PROMPT_URL = ("https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/"
                     f"{SYSTEM_PROMPT_COMMIT}/prompt_rewrite/prompts/system_prompt_t2i.txt")
SYSTEM_PROMPT_SHA256 = "a77c9a06c59b120741141d9514b95682bb8761d02bec49ca61def7b2b3d9fb99"
SYSTEM_PROMPT_BYTES = 10_045
SYSTEM_PROMPT_LICENSE = "Qwen Research License Agreement (non-commercial: research or evaluation)"
SYSTEM_PROMPT_FILE = REPO / "vendor" / "qwen-image-pe" / "system_prompt_t2i.txt"

# pe_core.py's PROFILES["t2i"]: "a wrong penalty does not fail, it quietly changes the
# distribution you sample from"
SAMPLING = {"--temp": "1.0", "--top-p": "0.95", "--top-k": "20", "--min-p": "0.0",
            "--presence-penalty": "1.5"}
MAX_TOKENS = 3000          # the answer is one paragraph; the profile's 16256 is for thinking
CONTEXT = 8192             # the system prompt is about 2,300 tokens
SEED = 42
TIMEOUT = 600
MODEL = "Qwen3-VL-8B-Instruct Q4_K_M (Qwen-Image's own text encoder), llama.cpp, local"


def system_prompt_path() -> Path | None:
    return SYSTEM_PROMPT_FILE if SYSTEM_PROMPT_FILE.is_file() else None


def llama_completion(repo: Path = REPO) -> Path | None:
    """llama.cpp's one-shot tool, wherever its archive put it under vendor/llama-cpp/."""
    root = repo / "vendor" / "llama-cpp"
    if not root.is_dir():
        return None
    for name in ("llama-completion.exe", "llama-completion"):
        found = sorted(root.rglob(name))
        if found:
            return found[0]
    return None


SETUP_HINT = "Setup & Status, Qwen-Image, installs it"


def missing() -> str | None:
    """What the rewriting still needs, in words, or None when it can run."""
    if system_prompt_path() is None:
        return "the rewriting prompt"
    if llama_completion() is None:
        return "llama.cpp"
    return None


def state() -> dict[str, Any]:
    """Whether the rewriting can run here, for the studio: {"ready", "missing", "setup"}."""
    gap = missing()
    return {"ready": gap is None, "missing": gap, "setup": SETUP_HINT if gap else None}


def install_system_prompt(fetch: Callable[[str], bytes] | None = None) -> Path:
    """Fetch the pinned system prompt (10 KB) into vendor/; refuse one with another checksum."""
    def default(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    data = (fetch or default)(SYSTEM_PROMPT_URL)
    got = hashlib.sha256(data).hexdigest()
    if got != SYSTEM_PROMPT_SHA256:
        raise SystemExit(f"the rewriting prompt does not match its pinned checksum (expected "
                         f"{SYSTEM_PROMPT_SHA256}, got {got}); nothing was written")
    SYSTEM_PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT_FILE.write_bytes(data)
    return SYSTEM_PROMPT_FILE


def _watched(command: list[str], on_process: Callable, popen: Callable) -> subprocess.CompletedProcess:
    """`command` run as `subprocess.run` would, its process handed to `on_process` first."""
    process = popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                    text=True, encoding="utf-8", errors="replace")
    on_process(process)
    try:
        out, err = process.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, out, err)


def parse_answer(text: str) -> dict[str, Any] | None:
    """The last JSON object in the answer that has a `rewritten_prompt` (the official answer
    contract: {"rewritten_prompt": ..., "wh_ratio": ...} at the end)."""
    found = None
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in starts:
        depth, quoted, escaped = 0, False, False
        for end in range(start, len(text)):
            c = text[end]
            if quoted:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    quoted = False
                continue
            if c == '"':
                quoted = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:end + 1])
                    except ValueError:
                        break
                    if isinstance(value, dict) and isinstance(value.get("rewritten_prompt"), str) \
                            and value["rewritten_prompt"].strip():
                        found = value
                    break
    return found


def rewrite(brief: str, *, model: Path, log: Callable[[str], None] = lambda line: None,
            run: Callable = subprocess.run, lock: Callable = gpu_lock.gpu,
            lock_path: Path | None = None, on_process: Callable | None = None,
            popen: Callable = subprocess.Popen) -> dict[str, Any] | None:
    """The rewritten prompt and the ratio the rewriter chose, or None (and a line in `log`)
    when it is not installed or gave nothing usable. Holds the shared GPU lock while it runs.
    `on_process` gets llama-completion's process as it starts, so a cancelled picture can stop it."""
    system, binary = system_prompt_path(), llama_completion()
    if system is None or binary is None:
        gap = "the rewriting prompt" if system is None else "llama.cpp"
        log(f"prompt rewriting: {gap} is not installed ({SETUP_HINT}); drawing without rewriting")
        return None
    command = [str(binary), "-m", str(model), "-sysf", str(system), "-p", brief, "-cnv", "-st", "--jinja",
               "-ngl", "99", "-n", str(MAX_TOKENS), "-c", str(CONTEXT), "--seed", str(SEED),
               "--no-display-prompt"]
    for flag, value in SAMPLING.items():
        command += [flag, value]
    started = time.monotonic()
    try:
        with lock("prompt-rewrite", **({"path": lock_path} if lock_path else {})):
            if on_process is None:
                done = run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           stdin=subprocess.DEVNULL, timeout=TIMEOUT, check=False)
            else:
                done = _watched(command, on_process, popen)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"prompt rewriting failed ({exc}); drawing without rewriting")
        return None
    answer = parse_answer(done.stdout or "") if done.returncode == 0 else None
    if answer is None:
        why = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or ["no answer"]
        log(f"prompt rewriting gave nothing usable ({why[0][:200]}); drawing without rewriting")
        return None
    return {"prompt": answer["rewritten_prompt"].strip(), "wh_ratio": str(answer.get("wh_ratio") or ""),
            "seconds": round(time.monotonic() - started, 1), "model": MODEL,
            "system_prompt": {"url": SYSTEM_PROMPT_URL, "sha256": SYSTEM_PROMPT_SHA256,
                              "license": SYSTEM_PROMPT_LICENSE},
            "sampling": {k.lstrip("-"): v for k, v in SAMPLING.items()}}
