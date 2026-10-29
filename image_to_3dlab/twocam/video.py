"""Film a two-camera clip on this machine: MiniMax H3, image-to-video, run by stable-diffusion.cpp.

AnyMove filmed its clips on fal; this keeps them local. stable-diffusion.cpp (the engine
`scripts/bootstrap_qwen_image.py` installs for pictures) has run MiniMax H3 since August
2026. The route that fits a 16 GB card and 32 GB of RAM: the pruned video model at Q4_K_M,
its Qwen3-VL text encoder at Q2_K_M, and the fp16 video VAE, about 30 GB in all. Measured
on an RTX 5070 Ti: 864 x 480 x 39 frames in 2.5 min (14.4 GB of video memory at the peak),
1024 x 576 x 73 frames in about 6.

Nothing is downloaded without a yes: `weights_plan()` says what and how big, and
`download()` fetches only when told to. MiniMax H3's weights are under the MiniMax H3
Community License; its page asks users in the USA, the EU, the UK and South Korea to apply
with a form first. Free to run, so a creature can be filmed in several takes and the best
one kept (`scripts/twocam_mocap.py film --takes`).
"""

from __future__ import annotations

import glob
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ENGINE = "MiniMax H3 (stable-diffusion.cpp, local)"
LICENSE = "MiniMax H3 Community License Agreement"
LICENSE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
LICENSE_NOTE = ("MiniMax asks users in the USA, the EU, the UK and South Korea to apply with the form on "
                "https://huggingface.co/MiniMaxAI/MiniMax-H3 before using the weights")
FPS = 24  # MiniMax H3 always runs at 24 fps


@dataclass(frozen=True)
class Weight:
    role: str
    repo: str
    filename: str
    gigabytes: float


WEIGHTS = (
    Weight("diffusion_model", "leejet/MiniMax-H3-GGUF", "minimax_h3_fl2va_pruned-Q4_K_M.gguf", 11.42),
    Weight("llm", "leejet/MiniMax-H3-GGUF", "qwen3vl_32b_minimax_h3-Q2_K_M.gguf", 13.10),
    Weight("vae", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", 5.21),
)


class VideoError(RuntimeError):
    pass


def hub_root(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    return Path(env.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def find_weight(weight: Weight, root: Path | None = None) -> Path | None:
    """The cached file, resolved to the real file: the hub cache links snapshots to blobs,
    and sd-cli on Windows cannot open such a link ("cannot inspect model source")."""
    root = root or hub_root()
    folder = root / ("models--" + weight.repo.replace("/", "--"))
    for match in sorted(glob.glob(str(folder / "snapshots" / "*" / weight.filename))):
        path = Path(match).resolve()
        if path.is_file():
            return path
    return None


def weights_plan(root: Path | None = None) -> dict:
    """What filming needs, what is here, and how much a download would fetch."""
    files = {w.role: find_weight(w, root) for w in WEIGHTS}
    missing = [w for w in WEIGHTS if files[w.role] is None]
    return {"engine": ENGINE, "license": LICENSE, "license_url": LICENSE_URL, "license_note": LICENSE_NOTE,
            "ready": not missing, "missing": [f"{w.repo}/{w.filename} ({w.gigabytes:.1f} GB)" for w in missing],
            "download_gb": round(sum(w.gigabytes for w in missing), 1),
            "files": {k: (str(v) if v else None) for k, v in files.items()}}


def download(log: Callable[[str], None] | None = None) -> dict[str, Path]:
    """Fetch the missing weights. Call only after the user said yes to `weights_plan()`."""
    from huggingface_hub import hf_hub_download

    say = log or (lambda _msg: None)
    out = {}
    for w in WEIGHTS:
        found = find_weight(w)
        if found is None:
            say(f"downloading {w.repo}/{w.filename} ({w.gigabytes:.1f} GB)")
            found = Path(hf_hub_download(w.repo, w.filename)).resolve()
        out[w.role] = found
    return out


def frames_for(seconds: float) -> int:
    """H3 makes 17k + 5 frames; the fewest that last at least `seconds` at 24 fps."""
    want = max(5, round(seconds * FPS))
    return 17 * max(0, -(-(want - 5) // 17)) + 5


def size_for(width: int, height: int) -> tuple[int, int]:
    """Both sides rounded to a multiple of 32, as the model needs."""
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


def film_command(binary: Path, weights: dict[str, Path], prompt: str, start: Path, out: Path, *,
                 width: int = 1024, height: int = 576, seconds: float = 3.0, seed: int = 1,
                 steps: int = 20) -> list[str]:
    """sd-cli's arguments for one take: the reference frame as the first frame, no sound.
    The sampler settings are leejet's published workflow (res_multistep, simple, 20 steps,
    no CFG); weights stay in RAM and move to the card as needed (--offload-to-cpu)."""
    if Path(out).suffix.lower() not in (".webm", ".avi", ".webp"):
        raise VideoError("sd-cli writes .webm, .avi or animated .webp, not " + (Path(out).suffix or "no suffix"))
    w, h = size_for(width, height)
    return [str(binary), "-M", "vid_gen",
            "--diffusion-model", str(weights["diffusion_model"]), "--vae", str(weights["vae"]),
            "--llm", str(weights["llm"]), "-p", prompt, "-i", str(start),
            "--cfg-scale", "1.0", "--sampling-method", "res_multistep", "--scheduler", "simple",
            "--steps", str(int(steps)), "-W", str(w), "-H", str(h), "--fps", str(FPS),
            "--video-frames", str(frames_for(seconds)), "--offload-to-cpu", "--flash-attn", "on",
            "--vae-tiling", "--rng", "cpu", "-s", str(int(seed)), "-o", str(out)]


def film(binary: Path, prompt: str, start: Path, out: Path, *, log: Callable[[str], None] | None = None,
         **settings) -> Path:
    """Film one take; returns the clip. Raises VideoError with sd-cli's own last words."""
    plan = weights_plan()
    if not plan["ready"]:
        raise VideoError("the video model is not downloaded: " + ", ".join(plan["missing"])
                         + f" ({plan['download_gb']} GB). `scripts/twocam_mocap.py film --download` fetches it")
    if not Path(binary).is_file():
        raise VideoError(f"stable-diffusion.cpp is not installed at {binary}: scripts/bootstrap_qwen_image.py sets it up")
    weights = {k: Path(v) for k, v in plan["files"].items()}
    cmd = film_command(binary, weights, prompt, start, out, **settings)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0 or not Path(out).is_file():
        lines = [x for x in (proc.stdout + proc.stderr).replace("\r", "\n").splitlines() if "[E]" in x or "error" in x]
        raise VideoError(f"sd-cli failed (exit {proc.returncode}): " + " | ".join(lines[-5:]))
    if log:
        timing = [x for x in proc.stdout.splitlines() if "generate_video completed" in x]
        log(timing[-1].split("---")[0].strip() if timing else f"filmed {out}")
    return Path(out)
