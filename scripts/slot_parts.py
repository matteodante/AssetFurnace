#!/usr/bin/env python3
"""Characters in pieces (feature 4): register a base body, then cut a concept into slots.

    python scripts/slot_parts.py body knight --image body.png --rigged body_rigged.glb \
        --views body.svviews [--label race=human --label sex=male]
    python scripts/slot_parts.py cuts --body output/bodies/knight/body.json --concept concept.png \
        --slots helmet,torso,shoulders:split,gloves,boots --out output/sets/red-knight/knight/cuts

`body` writes `output/bodies/<name>/body.json`: picture, rigged GLB, Pixal3D camera and bone
roles. The body must be a humanoid made from one front picture with Pixal3D (its camera
puts the bones on the picture) and rigged; the cut-out of the picture is Pixal3D's own
`<picture>__matted.png` unless `--matte` names another.

`cuts` takes a concept drawn on that body (an edit of the body picture that keeps pose and
outline) and writes, per chosen slot: `cut_<slot>.png` (the piece, transparent around it),
`<slot>.png` (the piece flattened on grey, ready for the "complete this piece" edit), plus
`slots.png` (the slot map), `overlay.png` (the map over the concept, to check by eye) and
`cuts.json` (boxes and the way back from each canvas to the concept). The concept's
subject comes from its alpha when it is matted, from `--matte`, or from BiRefNet.

Exit code 0 when written, 1 when refused (the reason is printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from image_to_3dlab.slot_parts import body as body_mod
from image_to_3dlab.slot_parts import cuts, masks, slots


def parse_slots(text: str) -> dict[str, dict]:
    """"helmet,shoulders:split,gloves" -> slots.choose input."""
    raw = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        name, _, flag = item.partition(":")
        raw[name] = {"split": True} if flag == "split" else {"route": flag} if flag else {}
    return raw


def subject_mask(concept: Path, matte: Path | None = None, cut_out=None) -> tuple[np.ndarray, str]:
    """(subject mask, where it came from): the concept's own alpha, a given cut-out, or BiRefNet."""
    from image_to_3dlab.matte import is_matted

    if matte is not None:
        return np.asarray(Image.open(matte).convert("RGBA"))[..., 3] > 127, str(matte)
    with Image.open(concept) as picture:
        if is_matted(picture):
            return np.asarray(picture.convert("RGBA"))[..., 3] > 127, "alpha"
        if cut_out is None:
            from image_to_3dlab.matte import cut_out
        rgba, model = cut_out(picture)
    return np.asarray(rgba)[..., 3] > 127, model


def make_cuts(body_json: Path, concept: Path, chosen: dict[str, dict], out: Path, size: int = 1024,
              matte: Path | None = None, cut_out=None) -> dict:
    record = body_mod.load(body_json)
    picture = Image.open(concept).convert("RGB")
    rgb = np.asarray(picture)
    bw, bh = record["image_size"]
    if abs(picture.width / picture.height - bw / bh) > 0.01:
        raise ValueError(f"the concept is {picture.width}x{picture.height} and the body picture {bw}x{bh}: "
                         "a concept drawn on the body keeps its framing")
    fg, source = subject_mask(concept, matte, cut_out)
    if fg.shape != rgb.shape[:2]:
        raise ValueError("the cut-out and the concept differ in size")
    drawn = [name for name in chosen if name in slots.DRAWN]
    labels = masks.slot_labels(fg, body_mod.segments_px(record, picture.size), chosen=set(drawn))
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(out / "slots.png")
    Image.fromarray(masks.overlay(rgb, labels)).save(out / "overlay.png")
    result = {"body": str(Path(body_json).resolve()), "concept": str(Path(concept).resolve()),
              "subject_from": source, "canvas": size, "slots": {}, "skipped": {}}
    for name, options in chosen.items():
        if name not in slots.DRAWN:
            result["skipped"][name] = "made on its own, from a prompt, not cut from the concept"
            continue
        try:
            piece, box = cuts.cut_slot(rgb, labels, name)
        except ValueError as exc:
            result["skipped"][name] = str(exc)
            continue
        canvas, place = cuts.flatten(piece, box, size=size)
        Image.fromarray(piece, "RGBA").save(out / f"cut_{name}.png")
        Image.fromarray(canvas).save(out / f"{name}.png")
        result["slots"][name] = {**options, "box": list(box), "placement": place.to_dict(),
                                 "cut": f"cut_{name}.png", "picture": f"{name}.png",
                                 "pixels": int((labels == slots.label(name)).sum())}
    (out / "cuts.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("body", help="register a rigged base body")
    b.add_argument("name")
    b.add_argument("--image", type=Path, required=True, help="the front T-pose picture the body was made from")
    b.add_argument("--rigged", type=Path, required=True, help="the rigged GLB")
    b.add_argument("--views", type=Path, required=True, help="the Pixal3D <model>.svviews folder")
    b.add_argument("--matte", type=Path, help="cut-out of the picture (default <picture>__matted.png)")
    b.add_argument("--label", action="append", default=[], help="key=value, e.g. race=human (repeatable)")
    b.add_argument("--root", type=Path, default=Path("output/bodies"))
    c = sub.add_parser("cuts", help="cut a concept drawn on a body into one picture per slot")
    c.add_argument("--body", type=Path, required=True, help="body.json from the body command")
    c.add_argument("--concept", type=Path, required=True)
    c.add_argument("--slots", required=True, help="comma list; shoulders:split for two shoulder pieces, "
                                                  "slot:B0 to cut that slot from the whole model instead")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--size", type=int, default=1024, help="canvas side of each flattened piece")
    c.add_argument("--matte", type=Path, help="cut-out of the concept, when it is not matted itself")
    args = p.parse_args(argv)
    try:
        if args.command == "body":
            labels = dict(item.split("=", 1) for item in args.label)
            path = body_mod.register(args.name, args.image, args.rigged, args.views, args.matte, labels, args.root)
            print(f"wrote {path}")
        else:
            result = make_cuts(args.body, args.concept, slots.choose(parse_slots(args.slots)), args.out,
                               args.size, args.matte)
            print(f"wrote {len(result['slots'])} cut(s) to {args.out}: {', '.join(result['slots'])}")
            for name, why in result["skipped"].items():
                print(f"  skipped {name}: {why}")
    except ValueError as exc:
        print(f"refused: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
