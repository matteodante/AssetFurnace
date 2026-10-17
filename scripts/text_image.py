#!/usr/bin/env python3
"""Make a picture from words, as the studio's Create does, from the command line.

    python scripts/text_image.py "a wooden barrel with iron bands, game asset, plain background"
    python scripts/text_image.py "a wolf, side view" --seed 7 --count 4      # four to pick from
    python scripts/text_image.py "a crate" --width 768 --height 768 --out output/images/crates

Qwen-Image 2.1 Turbo through `vendor/sdcpp/sd-cli` (`image_to_3dlab/text_image.py`), the
same engine and settings as the studio. Each picture gets a `.provenance.json` beside it
with the licence. `--count N` makes N pictures with seeds `--seed`, `--seed`+1, ... to pick
from; the batch step `picture` does the same with a pick. Nothing downloads: missing
weights are named, and the studio's Setup page installs them after asking.

Holds the shared GPU lock (`%TEMP%/assetfurnace-gpu.lock`) while sd-cli runs. Prints one
line per picture, or JSON with `--json`. Exit code 0 when every picture was made.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import text_image  # noqa: E402

MAX_COUNT = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", help="what to draw")
    parser.add_argument("--out", type=Path,
                        help="folder for the pictures (default output/images/<words>__<time>/)")
    parser.add_argument("--width", type=int, default=text_image.DEFAULTS["width"],
                        help="pixels, 256 to 1536, rounded to a multiple of 32")
    parser.add_argument("--height", type=int, default=text_image.DEFAULTS["height"])
    parser.add_argument("--seed", type=int, default=text_image.DEFAULTS["seed"], help="same seed, same picture")
    parser.add_argument("--count", type=int, default=1, help=f"pictures to pick from, 1 to {MAX_COUNT}, "
                                                            "with consecutive seeds")
    parser.add_argument("--cfg-scale", type=float, default=text_image.DEFAULTS["cfg_scale"])
    parser.add_argument("--sampler", choices=text_image.SAMPLERS, default=text_image.DEFAULTS["sampler"])
    parser.add_argument("--negative-prompt", default="", help="what to keep out")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    return parser


def make(prompt: str, out: Path | None, settings: dict, count: int = 1, log=print) -> list[Path]:
    """The pictures, one per seed from settings["seed"]; each with its provenance."""
    if not 1 <= count <= MAX_COUNT:
        raise ValueError(f"--count must be from 1 to {MAX_COUNT}")
    name = f"{text_image.slug(prompt)}__{time.strftime('%Y%m%d-%H%M%S')}"
    folder = out or text_image.OUTPUT_ROOT / name
    made = []
    for offset in range(count):
        seed = int(settings.get("seed", text_image.DEFAULTS["seed"])) + offset
        path = folder / (f"{name}.png" if count == 1 else f"{name}_s{seed}.png")
        log(f"text_image: seed {seed} -> {path}")
        made.append(text_image.generate(prompt, {**settings, "seed": seed}, path, log=log))
    return made


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = {"width": args.width, "height": args.height, "seed": args.seed, "cfg_scale": args.cfg_scale,
                "sampler": args.sampler, "negative_prompt": args.negative_prompt}
    quiet = args.json
    try:
        made = make(args.prompt, args.out, settings, args.count,
                    log=(lambda line: None) if quiet else print)
    except (ValueError, RuntimeError) as exc:   # MissingWeights is a RuntimeError
        print(f"text_image: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"pictures": [str(path) for path in made]}, indent=2))
    else:
        for path in made:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
