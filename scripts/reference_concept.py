#!/usr/bin/env python3
"""Draw a new concept from a reference: prepare a map, then redraw, dress, add a piece or turn it.

    python scripts/reference_concept.py prepare wolf/wolf.glb depth --view side
    python scripts/reference_concept.py make redraw --shape concept/maps/<hash>/front_depth_black.png \\
        --subject werewolf --colours "grey fur, a brown mane and pale claws"
    python scripts/reference_concept.py make other_view --view back --depth \\
        --shape images/wolf-front.png --shape concept/maps/<hash>/back_depth_black.png --attempts 3

Paths are relative to the output folder (``--output``, default ``output/``), as in the
studio. ``make`` runs Qwen-Image 2.1 in edit mode through the viewer's image job (same
recipe, provenance and ``<image>.concept.json`` record) and prints each picture with its
silhouette IoU against the shape picture, when that picture has a clean outline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "viewer"))

import concept_api  # noqa: E402

from image_to_3dlab.concept import maps, prompts, sidecar  # noqa: E402

FIELDS = ("subject", "colours", "parts", "structure", "outfit", "guide", "piece", "view", "style")


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    top.add_argument("--output", type=Path, default=REPO / "output")
    sub = top.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="a map from a model (depth, colour, pose) or a picture (lineart)")
    prepare.add_argument("source")
    prepare.add_argument("kind", choices=concept_api.MODEL_KINDS + concept_api.PICTURE_KINDS)
    prepare.add_argument("--view", default="front", choices=maps.VIEWS)
    make = sub.add_parser("make", help="one of the four ways")
    make.add_argument("mode", choices=prompts.MODES)
    make.add_argument("--shape", action="append", default=[], help="shape picture, in order (repeat)")
    make.add_argument("--style-ref", action="append", default=[], help="style picture, after the shape ones")
    for field in FIELDS:
        make.add_argument(f"--{field}")
    make.add_argument("--depth", action="store_true", help="other_view: the second shape picture is that side's depth")
    make.add_argument("--seed", type=int)
    make.add_argument("--attempts", type=int, default=1, help=f"1 to {concept_api.ATTEMPTS['most']}, one seed each")
    make.add_argument("--bible", help="a style bible's words and pictures: - for the active one, or its name")
    make.add_argument("--kind", help="the bible's words for this kind too (prop, creature, character...)")
    return top


def request(args: argparse.Namespace, seed: int) -> dict:
    out = {"mode": args.mode, "shape": args.shape, "style_refs": args.style_ref, "depth": args.depth, "seed": seed}
    out.update({f: getattr(args, f) for f in FIELDS if getattr(args, f) is not None})
    out.update({key: getattr(args, key) for key in ("bible", "kind") if getattr(args, key, None)})
    return out


def seeds(first: int | None, attempts: int) -> list[int]:
    if not 1 <= attempts <= concept_api.ATTEMPTS["most"]:
        raise SystemExit(f"--attempts is between 1 and {concept_api.ATTEMPTS['most']}")
    start = random.randrange(2 ** 31 - attempts) if first is None else first
    return [start + i for i in range(attempts)]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(concept_api.prepare(args.output, args.source, args.kind, args.view), indent=2))
        return 0
    for seed in seeds(args.seed, args.attempts):
        job = concept_api.start(args.output, request(args, seed))
        concept_api.wait(job)
        if job.status != "done":
            print(f"seed {seed}: {job.error or job.status}", file=sys.stderr)
            return 1
        made = sidecar.read(job.output_path) or {}
        score = "" if made.get("iou") is None else f"  IoU {made['iou']:.2f}"
        print(f"{job.output_path}  seed {seed}{score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
