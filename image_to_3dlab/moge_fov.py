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


def camera_angle_x(fx_normalized: float) -> float:
    """The horizontal field of view, radians, from MoGe's focal length as a share of the width."""
    return 2 * math.atan(1.0 / (2 * float(fx_normalized)))


def estimate(image: Path, run: Callable = subprocess.run, lock=gpu_lock.gpu) -> float | None:
    """The field of view of `image` (radians), or None when MoGe-2 is not set up or fails."""
    if not installed():
        return None
    command = [str(python()), str(WORKER), str(image), str(weights_path())]
    with lock("moge-2 fov"):   # it uses the GPU when its PyTorch has one
        done = run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return None
    try:
        return camera_angle_x(json.loads(done.stdout.strip().splitlines()[-1])["fx_normalized"])
    except (ValueError, KeyError, IndexError):
        return None
