#!/usr/bin/env python3
"""Install the game-material models and their environment, after saying what, how big and under which licence.

    python scripts/bootstrap_materials.py --only env,marigold-lighting   # what baked-light removal needs
    python scripts/bootstrap_materials.py                                # every part, asks first
    python scripts/bootstrap_materials.py --yes                          # non-interactive (viewer, agents)

Parts:
- `env`: a Python 3.11 environment in `vendor/materials/.venv` (or `I2L_MATERIALS_ENV`) with
  PyTorch, diffusers and transformers. Its own, because diffusers still needs huggingface-hub
  below 2 and the viewer's environment is past it.
- `marigold-lighting`, `marigold-appearance`: Marigold-IID v1.1 (prs-eth), fp16, CreativeML
  Open RAIL++-M with use restrictions.
- `qwen35`: Qwen3.5-4B (Apache-2.0), which names the materials it sees.
- `sam3`: SAM 3 (Meta, SAM License). Gated: its owner must request access on Hugging Face
  with a form asking for personal data. This script never does that; it checks for access and
  skips SAM 3 when the account has none.

Hugging Face files are fetched at a pinned revision into the usual cache. Nothing is fetched
without an explicit yes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import host  # noqa: E402
from image_to_3dlab import material_models as mm  # noqa: E402

PARTS = ("env", *mm.MODELS)
ENV_BYTES = int(3.2 * 1024 ** 3)          # measured: torch cu130 + diffusers + transformers


def parts(only: str | None) -> list[str]:
    if not only:
        return list(PARTS)
    chosen = [p.strip() for p in only.split(",") if p.strip()]
    unknown = [p for p in chosen if p not in PARTS]
    if unknown:
        raise SystemExit(f"unknown part {', '.join(unknown)}; choose from {', '.join(PARTS)}")
    return chosen


def _size(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} GB" if value >= 1024 ** 3 else f"{value / 1024 ** 2:.1f} MB"


def announcement(chosen: list[str], env_present: bool = False) -> str:
    lines, total = ["Game materials: this will download"], 0
    for key in chosen:
        if key == "env":
            if env_present:
                lines.append(f"  environment: already in {mm.env_dir()}")
                continue
            lines.append(f"  Python {mm.PYTHON_VERSION} environment with {', '.join(mm.TORCH + mm.ENV_PACKAGES)}"
                         f"  ~{_size(ENV_BYTES)}\n      to   {mm.env_dir()}")
            total += ENV_BYTES
            continue
        model = mm.MODELS[key]
        lines.append(f"  {model.label} ({model.license})  {_size(model.size)}\n"
                     f"      from https://huggingface.co/{model.repo} at {model.revision[:12]}\n"
                     f"      {model.note}")
        if key == "marigold-lighting" or key == "marigold-appearance":
            lines.append(f"      licence: {mm.OPENRAIL_URL}")
        total += model.size
    lines.append(f"  total {_size(total)}")
    return "\n".join(lines)


def env_commands(uv: str, env: Path, python: Path, index: str | None) -> list[list[str]]:
    """Make the environment, then torch from its own index, then the rest. Pure, for tests."""
    pip = [uv, "pip", "install", "--python", str(python)]
    return [[uv, "venv", "--allow-existing", "--python", mm.PYTHON_VERSION, str(env)],
            [*pip, *mm.TORCH, *(["--index-url", index] if index else [])],
            [*pip, *mm.ENV_PACKAGES]]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def make_env(runner=run) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    index = host.torch_cuda_index(host.driver_cuda_version()) if host.has_nvidia_gpu() else None
    for command in env_commands(uv, mm.env_dir(), mm.env_python(), index):
        runner(command)


def has_access(repo: str) -> bool:
    from huggingface_hub import auth_check
    from huggingface_hub.errors import HfHubHTTPError

    try:
        auth_check(repo)
    except (HfHubHTTPError, OSError):
        return False
    return True


def fetch(model) -> None:
    from huggingface_hub import hf_hub_download

    for name in model.files:
        hf_hub_download(model.repo, name, revision=model.revision)


def main(argv: list[str] | None = None, access=has_access, fetcher=fetch, env_maker=make_env) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help=f"comma-separated parts: {', '.join(PARTS)}")
    parser.add_argument("--yes", action="store_true", help="do not ask")
    args = parser.parse_args(argv)
    chosen = parts(args.only)
    for key in [k for k in chosen if k in mm.GATED]:
        if not access(mm.MODELS[key].repo):
            print(f"Skipping {mm.MODELS[key].label}: this Hugging Face account has no access yet. "
                  f"Its owner requests it at https://huggingface.co/{mm.MODELS[key].repo} "
                  "(a form with personal data, so nobody can do it for them).", flush=True)
            chosen.remove(key)
    if not chosen:
        return 1
    print(announcement(chosen, env_present=mm.env_python().is_file()), flush=True)
    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    for key in chosen:
        if key == "env":
            if not mm.env_python().is_file():
                print("Making the materials environment...", flush=True)
                env_maker()
            continue
        print(f"Downloading {mm.MODELS[key].label}...", flush=True)
        fetcher(mm.MODELS[key])
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
