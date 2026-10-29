#!/usr/bin/env python3
"""Install UniRig (VAST-AI, MIT), which draws the skeleton of any creature for the creature rig.

    python scripts/bootstrap_unirig.py          # says what it will fetch, then asks
    python scripts/bootstrap_unirig.py --yes    # non-interactive (the viewer passes this)
    python scripts/bootstrap_unirig.py --check  # exit 0 if installed, 1 if not

What it does, in order:
1. Clones UniRig into `vendor/UniRig` at a pinned commit (git-ignored, a few MB).
2. Applies `patch_unirig_portable.py`, so its skeleton model runs without flash-attn.
3. Makes a Python 3.11 venv there with torch 2.11 for CUDA 12.8 and upstream's
   requirements (flash-attn left out), about 6 GB. The pins are not taste: torch_scatter and
   torch_cluster load only against the torch they were built for, and the newest pair with
   Windows wheels is for torch 2.11 + cu128; spconv's newest build is cu126, which needs no
   match since it does not link against torch. Measured on an RTX 5070 Ti (sm_120).
4. Fetches the skeleton checkpoint (1.4 GB) at a pinned revision into the shared Hugging
   Face cache, where upstream looks, and the OPT-350m config it reads (no weights).

NVIDIA only: the skeleton model's point sampling runs a CUDA kernel. Without UniRig the
creature rig falls back to a skeleton traced from the mesh's centre lines.

Nothing is fetched without an explicit yes: AGENTS.md forbids weight downloads the user has
not chosen. The checkpoint is a pickle, which can run code when loaded, so its revision is
pinned to the one tested.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import patch_unirig_portable as portable

from image_to_3dlab import host

UPSTREAM = "https://github.com/VAST-AI-Research/UniRig.git"
COMMIT = "6793c6640ff01c8fb389f3993434124bb43d2933"
VENDOR = REPO / "vendor" / "UniRig"
PYTHON_VERSION = "3.11"

TORCH = ("torch==2.11.0", "torchvision==0.26.0")
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PYG_FIND_LINKS = "https://data.pyg.org/whl/torch-2.11.0+cu128.html"
PYG = ("torch_scatter", "torch_cluster")
SPCONV = "spconv-cu126"
MIN_CUDA = (12, 8)       # what the driver must run for the cu128 build
SKIPPED_REQUIREMENTS = ("flash_attn", "flash-attn")
EXTRA_PACKAGES = ("numpy==1.26.4",)

HF_REPO = "VAST-AI/UniRig"
HF_REVISION = "36842e2b5947e9e60f89275b83208c8e74071c63"
CHECKPOINT = "skeleton/articulation-xl_quantization_256/model.ckpt"
CHECKPOINT_BYTES = 1_439_617_174
OPT_REPO = "facebook/opt-350m"


def venv_python(root: Path = VENDOR) -> Path:
    return host.venv_python(root)


def announcement(build: bool = True, weights: bool = True) -> str:
    lines = ["Creature rig skeletons: UniRig by VAST-AI (MIT licence), NVIDIA only"]
    if build:
        lines.append(f"  code     {UPSTREAM} @ {COMMIT[:7]} -> {VENDOR.relative_to(REPO)}/")
        lines.append(f"  packages a Python {PYTHON_VERSION} venv there, about 6 GB "
                     "(torch 2.11 for CUDA 12.8, spconv, PyTorch Geometric)")
    if weights:
        lines.append(f"  weights  {HF_REPO} @ {HF_REVISION[:7]}, skeleton model "
                     f"{CHECKPOINT_BYTES / 1e9:.1f} GB, into the Hugging Face cache")
        lines.append(f"           plus {OPT_REPO}'s config only (a few KB)")
    return "\n".join(lines)


def build_present(root: Path = VENDOR) -> bool:
    return venv_python(root).exists() and (root / "run.py").exists()


def weights_present() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    found = try_to_load_from_cache(repo_id=HF_REPO, filename=CHECKPOINT, revision=HF_REVISION)
    return isinstance(found, str) and Path(found).is_file()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_pinned(root: Path = VENDOR, runner=run) -> None:
    """Clone at COMMIT, or move an existing clone there (`git checkout` keeps our patch)."""
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        runner(["git", "clone", UPSTREAM, str(root)])
    runner(["git", "-C", str(root), "checkout", "--quiet", COMMIT])


def check_machine(nvidia: bool, driver: tuple[int, int] | None) -> None:
    """Refuse before anything is fetched, in words a person can act on."""
    if not nvidia:
        raise SystemExit("UniRig needs an NVIDIA card (its skeleton model runs a CUDA kernel). "
                         "Without it the creature rig traces a skeleton from the mesh instead.")
    if driver is None or driver < MIN_CUDA:
        found = f"CUDA {driver[0]}.{driver[1]}" if driver else "no CUDA version"
        raise SystemExit(f"Your NVIDIA driver reports {found}; UniRig's torch needs CUDA "
                         f"{MIN_CUDA[0]}.{MIN_CUDA[1]} or newer. Update the driver, then run setup again.")


def requirements_without_flash_attn(text: str) -> str:
    """Upstream's requirements minus flash-attn, which only its skin model needs."""
    keep = [line for line in text.splitlines()
            if not line.strip().lower().replace("-", "_").startswith(SKIPPED_REQUIREMENTS[0])]
    return "\n".join(keep) + "\n"


def pip_commands(uv: str, python: Path, requirements: Path) -> list[list[str]]:
    """In order: torch from its CUDA index, upstream's requirements (flash-attn out), the
    PyTorch Geometric wheels built for exactly that torch, spconv. Pure, so it can be tested."""
    pip = [uv, "pip", "install", "--python", str(python)]
    return [[*pip, *TORCH, "--index-url", TORCH_INDEX],
            [*pip, "-r", str(requirements), *EXTRA_PACKAGES],
            [*pip, *PYG, "-f", PYG_FIND_LINKS, "--no-index", "--no-cache"],
            [*pip, SPCONV]]


def venv_needs_rebuild(root: Path = VENDOR) -> bool:
    cfg = root / ".venv" / "pyvenv.cfg"
    return not cfg.exists() or f"version_info = {PYTHON_VERSION}" not in cfg.read_text()


def install_packages(root: Path = VENDOR, runner=run) -> None:
    uv = host.find_uv()
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    if venv_needs_rebuild(root):
        shutil.rmtree(root / ".venv", ignore_errors=True)
        runner([uv, "venv", str(root / ".venv"), "--python", PYTHON_VERSION])
    requirements = root / ".venv" / "requirements-no-flash-attn.txt"
    requirements.write_text(requirements_without_flash_attn((root / "requirements.txt").read_text(encoding="utf-8")),
                            encoding="utf-8")
    for command in pip_commands(uv, venv_python(root), requirements):
        runner(command)


def install_weights() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as missing:
        raise SystemExit("huggingface_hub is not installed. pip install -r requirements-dev.txt") from missing
    print(f"\nFetching {CHECKPOINT} ({CHECKPOINT_BYTES / 1e9:.2f} GB)...", flush=True)
    hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT, revision=HF_REVISION)
    print(f"Fetching {OPT_REPO}'s config...", flush=True)
    hf_hub_download(repo_id=OPT_REPO, filename="config.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation. For the viewer and for scripts.")
    parser.add_argument("--build-only", action="store_true", help="Clone, patch and install packages; leave the weights.")
    parser.add_argument("--weights-only", action="store_true", help="Fetch the weights only.")
    parser.add_argument("--check", action="store_true", help="Exit 0 if code and weights are installed, 1 if not.")
    args = parser.parse_args(argv)

    if args.check:
        ok = build_present() and weights_present()
        print("UniRig is installed." if ok else "UniRig is not installed.")
        return 0 if ok else 1

    build, weights = not args.weights_only, not args.build_only
    if build:
        check_machine(host.has_nvidia_gpu(), host.driver_cuda_version())
    print(announcement(build=build, weights=weights))
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    if build:
        clone_pinned()
        portable.patch(VENDOR)
        install_packages()
    if weights:
        install_weights()
    print("\nDone. A creature's Rig step now draws its skeleton with UniRig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
