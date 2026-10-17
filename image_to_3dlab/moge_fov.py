"""The camera's field of view for Pixal3D, estimated from the picture with MoGe-2, as the
official recipe does (TencentARC/Pixal3D `inference.py`, master, read 2026-10-17:
`get_camera_params_wild_moge`, `camera_angle_x = 2 * atan(width / (2 * fx))`).

MoGe-2 (Ruicheng/moge-2-vitl, MIT) runs in its own environment, `vendor/moge/.venv`, which
`scripts/bootstrap_pixal3d.py --moge` makes: PyTorch for the machine, the MoGe package at the last
MoGe-2 commit, and the weights at a pinned revision. `scripts/moge_worker.py` runs there and
answers the focal length; this module turns it into the field of view. Nothing is fetched
here.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path

from image_to_3dlab import gpu_lock

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "Ruicheng/moge-2-vitl"
REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
WEIGHTS_FILE = "model.pt"
WEIGHTS_BYTES = 1_305_030_700
LICENSE = "MIT"
# MoGe 2.0.0's last commit (2025-11-02): the next one is MoGe-3, which needs FlexGEMM built with CUDA.
CODE_COMMIT = "07444410f1e33f402353b99d6ccd26bd31e469e8"
CODE_URL = f"git+https://github.com/microsoft/MoGe.git@{CODE_COMMIT}"
WORKER = REPO / "scripts" / "moge_worker.py"
# PyTorch for MoGe-2's environment, pinned against the pinned MoGe commit.
TORCH_VERSION, TORCHVISION_VERSION = "2.11.0", "0.26.0"
TIMEOUT = 900   # seconds; a hung estimate must not keep the GPU
FOV_RANGE = (0.05, 2.0)   # radians: what trellis-cli is ever given


def env_dir(repo: Path = REPO) -> Path:
    return repo / "vendor" / "moge" / ".venv"


def python(repo: Path = REPO) -> Path:
    """The environment's interpreter, named the way this OS names it."""
    import os

    folder = env_dir(repo)
    return folder / "Scripts" / "python.exe" if os.name == "nt" else folder / "bin" / "python"


def weights_path() -> Path | None:
    """The pinned model.pt in the Hugging Face cache, or None."""
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(REPO_ID, WEIGHTS_FILE, revision=REVISION)
    except Exception:  # noqa: BLE001 - a broken or missing hub library means: not installed
        return None
    return Path(found) if isinstance(found, str) else None


def installed(repo: Path = REPO) -> bool:
    return python(repo).is_file() and weights_path() is not None


def preprocessed(image: Path, out: Path) -> Path:
    """The picture as the official recipe hands it to MoGe-2 (`Pixal3DImageTo3DPipeline.
    preprocess_image`): at most 1024 px, cropped to a square 1.1 times the bounding box of
    the alpha above 0.8, on black. The field of view is the crop's, which is what trellis-cli
    gets after cropping the same way."""
    import numpy as np
    from PIL import Image

    with Image.open(image) as opened:
        picture = opened.convert("RGBA")
    scale = min(1.0, 1024 / max(picture.size))
    if scale < 1:
        picture = picture.resize((int(picture.width * scale), int(picture.height * scale)), Image.Resampling.LANCZOS)
    rgba = np.asarray(picture)
    found = np.argwhere(rgba[:, :, 3] > 0.8 * 255)
    if len(found):
        x0, y0, x1, y1 = found[:, 1].min(), found[:, 0].min(), found[:, 1].max(), found[:, 0].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = int(max(x1 - x0, y1 - y0) * 1.1)
        picture = picture.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))
    values = np.asarray(picture).astype(np.float32) / 255
    rgb = values[:, :, :3] * values[:, :, 3:4]   # on black
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(out)
    return out


def camera_angle_x(fx_normalized: float) -> float:
    """The horizontal field of view, radians, from MoGe's focal length as a share of the width."""
    return 2 * math.atan(1.0 / (2 * float(fx_normalized)))


def on_gpu() -> bool:
    """Whether the environment's PyTorch computes on a GPU here (CUDA on NVIDIA, Metal on a Mac)."""
    from image_to_3dlab import host

    return host.host_platform() in (host.NVIDIA, host.APPLE)


def estimate(image: Path, run: Callable = subprocess.run, lock=gpu_lock.gpu) -> float | None:
    """The field of view of `image` (radians), or None when MoGe-2 is not set up, fails, takes
    too long or answers something trellis-cli could not take."""
    import contextlib
    import tempfile

    if not installed():
        return None
    with tempfile.TemporaryDirectory(prefix="af-moge-") as folder:
        crop = preprocessed(image, Path(folder) / "crop.png")
        command = [str(python()), str(WORKER), str(crop), str(weights_path())]
        try:
            with lock("moge-2 fov") if on_gpu() else contextlib.nullcontext():
                done = run(command, capture_output=True, text=True, check=False, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
    if done.returncode != 0:
        return None
    try:
        fov = camera_angle_x(json.loads(done.stdout.strip().splitlines()[-1])["fx_normalized"])
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        return None
    return fov if FOV_RANGE[0] <= fov <= FOV_RANGE[1] else None
