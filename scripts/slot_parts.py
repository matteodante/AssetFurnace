#!/usr/bin/env python3
"""Characters in pieces (feature 4): register a base body, then cut a concept into slots.

    python scripts/slot_parts.py body knight --image body.png --rigged body_rigged.glb \
        --views body.svviews [--label race=human --label sex=male]
    python scripts/slot_parts.py cuts --body output/bodies/knight/body.json --concept concept.png \
        --slots helmet,torso,shoulders:split,gloves,boots --out output/sets/red-knight/knight/cuts
    python scripts/slot_parts.py pieces --cuts output/sets/red-knight/knight/cuts/cuts.json \
        --seeds 3,7,11 [--images-only | --models-only --pick helmet=7,boots=3]
    python scripts/slot_parts.py fit --pieces output/sets/red-knight/knight/pieces/pieces.json \
        --body output/bodies/knight/body.json [--out output/sets/red-knight/knight/fit]

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

`pieces` is route C: each flattened cut goes to a Qwen-Image edit ("complete this piece")
once per seed (`pieces/<slot>/s<seed>.png`, with a `.concept.json` record), then the picked
picture (`--pick slot=seed`, else the first seed) goes through Pixal3D, the finish step and
`fix_winding.py`. `pieces.json` says, per slot, which picture, which GLB and where the piece
sits on the concept. `--images-only` stops before 3D so a person can pick; `--models-only`
reuses the pictures already made.

`fit` writes a Blender job and runs `blender_fit_parts.py` headless: each piece placed on
the body from its box, paired pieces per side, a cage fit, skin weights from the body (one
bone for the helmet), and the `covered` mask on the body. It writes `set.glb`, one GLB per
piece, `body_masked.glb` and `fit.json` into `--out` (default `fit/` beside `pieces/`).

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
from image_to_3dlab.slot_parts import cuts, masks, pieces, slots

REPO = Path(__file__).resolve().parents[1]


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


def edit_command(prompt: str, canvas: Path, picture: Path, seed: int) -> list[str]:
    """sd-cli's edit of the day: the studio's text-to-image recipe plus the vision file and `-r`."""
    from image_to_3dlab.concept.edit import find_vision, with_references
    from viewer import image_api

    settings = image_api.clean_settings({"seed": seed})
    command = image_api.build_command(prompt, settings, picture, image_api.resolve_weights())
    vision = find_vision(image_api.HF_HUB)
    if vision is None:
        raise ValueError("Qwen3-VL's vision file is missing: download Qwen-Image in Setup & Status")
    return with_references(command, [canvas], vision)


def _run(command: list[str]) -> None:
    import subprocess

    done = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if done.returncode != 0:
        tool = Path(command[1] if len(command) > 1 and command[1].endswith(".py") else command[0]).name
        raise ValueError(f"{tool} failed ({done.returncode}): {(done.stdout + done.stderr)[-600:]}")


def make_pieces(cuts_json: Path, seeds: list[int], picks: dict[str, int] | None = None, images: bool = True,
                models: bool = True, style: str = "", run=_run, edit=edit_command) -> dict:
    """Route C for every slot of a cuts.json: pictures, then the picked ones in 3D."""
    from image_to_3dlab.concept import sidecar

    cuts_json = Path(cuts_json)
    record = json.loads(cuts_json.read_text())
    folder = cuts_json.parent.parent / "pieces"
    jobs = pieces.picture_jobs(record, cuts_json.parent, folder, seeds, style)
    picks = dict(picks or {})
    unknown = sorted(set(picks) - {job.slot for job in jobs})
    if unknown:
        raise ValueError(f"picked slot(s) not among the cut route C slots: {', '.join(unknown)}")
    if images:
        for job in jobs:
            job.picture.parent.mkdir(parents=True, exist_ok=True)
            command = edit(job.prompt, job.canvas, job.picture, job.seed)
            run(command)
            sidecar.write(job.picture, mode="complete_piece", prompt=job.prompt, seed=job.seed, command=command,
                          references=[(job.canvas, "shape")], inputs={"slot": job.slot})
    result = {"cuts": str(cuts_json.resolve()), "seeds": list(seeds), "slots": {}}
    if models:
        for slot in dict.fromkeys(job.slot for job in jobs):
            seed = picks.get(slot, seeds[0])
            picture = pieces.picture_path(folder, slot, seed)
            if not picture.is_file():
                raise ValueError(f"{slot}: no picture for seed {seed} at {picture}; run without --models-only")
            for command in pieces.model_commands(picture, folder, slot, sys.executable, REPO / "scripts"):
                run(command)
            matte = picture.with_name(f"{picture.stem}__matted.png")
            alpha = np.asarray(Image.open(matte).convert("RGBA"))[..., 3] > 127
            place = cuts.Placement.from_dict(record["slots"][slot]["placement"])
            result["slots"][slot] = {
                "seed": seed, "picture": str(picture), "glb": str(pieces.finished_path(folder, slot)),
                "views": str(pieces.model_path(folder, slot).with_suffix(".svviews")),
                "concept_box": pieces.concept_box(alpha, place)}
        pieces.write_record(folder / "pieces.json", result)
    return result


def fit_job(pieces_json: Path, body_json: Path, out: Path, options: dict | None = None) -> dict:
    """The job `blender_fit_parts.py` reads: the body, the concept's size and each piece's box."""
    record = json.loads(Path(pieces_json).read_text())
    cut_record = json.loads(Path(record["cuts"]).read_text())
    with Image.open(cut_record["concept"]) as concept:
        size = list(concept.size)
    if not record["slots"]:
        raise ValueError(f"{pieces_json} has no pieces: run the pieces command first")
    entries = [{"slot": slot, "glb": entry["glb"], "concept_box": entry["concept_box"]}
               for slot, entry in record["slots"].items()]
    return {"body": str(Path(body_json).resolve()), "concept_size": size, "out": str(Path(out).resolve()),
            "pieces": entries, **(options or {})}


def run_fit(job: dict, blender: Path | None = None, run=None) -> dict:
    """Write job.json and run Blender on it; returns fit.json."""
    from image_to_3dlab.blender import find_blender, missing_help

    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "job.json"
    path.write_text(json.dumps(job, indent=1) + "\n")
    blender = blender or find_blender()
    if blender is None:
        raise ValueError(missing_help())
    (run or _run)([str(blender), "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                   str(REPO / "scripts" / "blender_fit_parts.py"), "--", str(path)])
    return json.loads((out / "fit.json").read_text())


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
    q = sub.add_parser("pieces", help="route C: one clean picture per piece, then the piece in 3D")
    q.add_argument("--cuts", type=Path, required=True, help="cuts.json from the cuts command")
    q.add_argument("--seeds", default="3", help="comma list of seeds, one picture each per slot")
    q.add_argument("--pick", default="", help="slot=seed for the picture to make in 3D (default the first seed)")
    q.add_argument("--style", default="", help="style words added to every piece's prompt")
    stage = q.add_mutually_exclusive_group()
    stage.add_argument("--images-only", action="store_true", help="stop after the pictures")
    stage.add_argument("--models-only", action="store_true", help="reuse the pictures, make the 3D")
    f = sub.add_parser("fit", help="fit the pieces onto the body in Blender, skin them, mask the body")
    f.add_argument("--pieces", type=Path, required=True, help="pieces.json from the pieces command")
    f.add_argument("--body", type=Path, required=True, help="body.json of the body the concept was drawn on")
    f.add_argument("--out", type=Path, help="default: fit/ beside pieces/")
    f.add_argument("--blender", type=Path, help="Blender executable (found automatically when omitted)")
    args = p.parse_args(argv)
    try:
        if args.command == "fit":
            out = args.out or args.pieces.parent.parent / "fit"
            report = run_fit(fit_job(args.pieces, args.body, out), args.blender)
            print(f"fitted {', '.join(report['pieces'])}; body covered {report['covered_share']:.0%}; wrote {out}")
        elif args.command == "pieces":
            seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
            result = make_pieces(args.cuts, seeds, pieces.parse_picks(args.pick), images=not args.models_only,
                                 models=not args.images_only, style=args.style)
            print(f"pieces: {', '.join(result['slots']) or 'pictures only'}")
        elif args.command == "body":
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
