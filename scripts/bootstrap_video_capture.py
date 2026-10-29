#!/usr/bin/env python3
"""Fetch what Animate from video needs: MiniMax H3 to film, DINOv2 to read the joints.

    python scripts/bootstrap_video_capture.py          # says what it will fetch, then asks
    python scripts/bootstrap_video_capture.py --yes    # non-interactive (the viewer passes this)
    python scripts/bootstrap_video_capture.py --check  # exit 0 if everything is here, 1 if not

1. MiniMax H3, pruned and quantised for stable-diffusion.cpp: the model (Q4_K_M), its text
   encoder (Q2_K_M) and video VAE, about 30 GB, under the MiniMax H3 Community License, whose
   terms ask users in the USA, the EU, the UK and South Korea to apply with the form on the
   model page first. It runs through the same `sd-cli` the picture step installs.
2. DINOv2 with registers (Meta, Apache-2.0, 346 MB), the backbone the joint reader is taught
   on for each creature.

Both land in the shared Hugging Face cache, where `scripts/twocam_mocap.py` and the studio's
Animate step look. Nothing is fetched without an explicit yes (AGENTS.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host
from image_to_3dlab.twocam import reader, video

SD_CLI = host.executable(REPO / "vendor" / "sdcpp", "sd-cli")


def announcement(plan: dict, dino_here: bool) -> str:
    lines = ["Animate from video: what it films and reads with"]
    if plan["ready"]:
        lines.append(f"  video    {video.ENGINE}: already here")
    else:
        lines.append(f"  video    {video.ENGINE}, {plan['download_gb']:.1f} GB into the Hugging Face cache")
        lines += [f"           {item}" for item in plan["missing"]]
    lines.append(f"           licence: {video.LICENSE} ({video.LICENSE_URL})")
    lines.append(f"           {video.LICENSE_NOTE}")
    lines.append(f"  reader   {reader.BACKBONE_ID} ({reader.BACKBONE_LICENSE}), "
                 + ("already here" if dino_here else f"{reader.BACKBONE_DOWNLOAD_MB} MB"))
    if not SD_CLI.exists():
        lines.append("  note     filming also needs stable-diffusion.cpp, which the picture step "
                     "(Qwen-Image) installs")
    return "\n".join(lines)


def fetch_dino() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(reader.BACKBONE_ID, allow_patterns=["*.json", "model.safetensors"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation. For the viewer and for scripts.")
    parser.add_argument("--check", action="store_true", help="Exit 0 if everything is here, 1 if not.")
    args = parser.parse_args(argv)

    plan, dino_here = video.weights_plan(), video.dino_cached()
    if args.check:
        ok = plan["ready"] and dino_here
        print("Animate from video is ready." if ok else "Animate from video is not installed.")
        return 0 if ok else 1
    print(announcement(plan, dino_here))
    if plan["ready"] and dino_here:
        print("Nothing to fetch.")
        return 0
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    if not plan["ready"]:
        video.download(log=lambda msg: print(msg, flush=True))
    if not dino_here:
        print(f"downloading {reader.BACKBONE_ID} ({reader.BACKBONE_DOWNLOAD_MB} MB)", flush=True)
        fetch_dino()
    print("\nDone. A creature's Animate step can film and read now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
