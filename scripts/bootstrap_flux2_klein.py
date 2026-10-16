#!/usr/bin/env python3
"""Fetch FLUX.2 klein 4B, an Apache-2.0 image editor, for the sd-cli that is already here.

It runs in the same stable-diffusion.cpp binary as Qwen-Image (`vendor/sdcpp/sd-cli`),
so this fetches only the weights: three files, about 12.3 GB, every one Apache-2.0. The
VAE is BFL's separate Apache-2.0 decoder, not the one inside FLUX.2-dev, which is under a
non-commercial licence.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use.

    python scripts/bootstrap_flux2_klein.py           # says what it wants, then asks
    python scripts/bootstrap_flux2_klein.py --yes     # for the viewer and for scripts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host

BINARY = host.executable(REPO / "vendor" / "sdcpp", "sd-cli")

# (repo, filename, exact bytes as Hugging Face reports them). The image model unquantised,
# because at 4B it fits a 16 GB card; the text encoder at Q8_0, which the upstream guide
# offers and which leaves room for the model beside it.
WEIGHTS = [
    ("black-forest-labs/FLUX.2-klein-4B", "flux-2-klein-4b.safetensors", 7_751_105_712),
    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", 4_280_405_792),
    ("black-forest-labs/FLUX.2-small-decoder", "full_encoder_small_decoder.safetensors",
     249_519_092),
]

LICENCE = (
    "Apache-2.0 for all three files (FLUX.2 klein 4B, Qwen3-4B, FLUX.2 small decoder).\n"
    "  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
)


def total_gb() -> float:
    return sum(size for _, _, size in WEIGHTS) / 1e9


def announcement() -> str:
    """Exactly what is about to be fetched, before anything is."""
    lines = ["", "About to install:", "", "  backend: FLUX.2 klein 4B (image editing)",
             "  route:   safetensors and GGUF via stable-diffusion.cpp (the sd-cli already installed)",
             f"  weights: {total_gb():.1f} GB total ->  Hugging Face cache"]
    for repo, filename, size in WEIGHTS:
        lines.append(f"             {size / 1e9:>5.2f} GB  {filename}  ({repo})")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


def binary_present() -> bool:
    return BINARY.exists()


def install_weights() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. pip install -r requirements-dev.txt"
        ) from exc
    for repo, filename, size in WEIGHTS:
        print(f"\nFetching {filename} ({size / 1e9:.2f} GB) from {repo}...", flush=True)
        print(f"  {hf_hub_download(repo_id=repo, filename=filename)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for scripts.")
    args = parser.parse_args(argv)

    print(announcement())
    if not binary_present():
        # 12 GB of weights with nothing to run them is the worst way to find out.
        print(f"sd-cli is not installed at {BINARY}. Install the Qwen-Image route first "
              "(python scripts/bootstrap_qwen_image.py --build-only). Nothing downloaded.")
        return 1
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    install_weights()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
