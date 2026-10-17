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
    python scripts/slot_parts.py refit --fit output/sets/red-knight/knight/fit \
        --from output/bodies/knight/body.json --body output/bodies/elf-female/body.json
    python scripts/slot_parts.py unreal --fit output/sets/red-knight/knight/fit \
        --body output/bodies/knight/body.json [--size-cm 180]
    python scripts/slot_parts.py run --body output/bodies/knight/body.json --set red-knight \
        --outfit "crimson and gold plate armour" --slots helmet,torso,shoulders:split,gloves,boots \
        [--weapon "a longsword"] [--shield "a kite shield"] [--concept concept.png]

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
reuses the pictures already made. A slot set to `B0` in `cuts` (`--slots torso:B0`) is the
fallback: the whole concept goes through Pixal3D once and `blender_segment_parts.py` cuts
those slots out of it.

`body --clip walk=body_walk.glb` records the body's own animated clips (Kimodo, retargeted
onto that rig); the checks pose the pieces with them.

`run` does it all for one set on one body, taking the first seed at every choice:
the concept on the body (Qwen edit with the body picture, `--outfit` words) unless
`--concept` gives one, the cuts, the pieces, the weapon and shield made on their own
(`--weapon`, `--shield`), the fit, the checks (`fit.clipping`, `fit.holes` and the rest of
the `equipment_part` profile; every piece goes to the review queue) and
`<set>.parts.json`. Everything lands in `output/sets/<set>/<body>/`.

`refit` carries a fitted set over to another registered body: each piece moves from its
slot's region on the first body to the same region on the second (box onto box), then the
same cage fit, weights and mask run on the new body; output in `<set>/<new body>/fit/`.
Images and 3D are not made again.

`unreal` exports a fitted set for Unreal 5.8 with `export_unreal.py`: the masked body as a
creature, every piece as equipment on the body's Skeleton, and `<body>.unreal-set.json` with
how to assemble them (Leader Pose, the body's opacity mask from its vertex alpha).

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
from image_to_3dlab.slot_parts import candidates, cuts, masks, pieces, slots

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
              "cuts_dir": str(out.resolve()), "subject_from": source, "canvas": size, "slots": {}, "skipped": {}}
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


def score_candidates(record: dict, cuts_dir: Path, folder: Path, seeds: list[int]) -> tuple[dict, str]:
    """Every slot's pictures scored, with Qwen3-VL when it is there; (scores, judge used)."""
    from image_to_3dlab.style import judge, models

    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        server = judge.LlamaServer(log=Path(folder) / "judge.log")
    except models.MissingModel as missing:
        return candidates.score_all(record, cuts_dir, folder, seeds), f"shape only ({missing})"
    with server:
        return candidates.score_all(record, cuts_dir, folder, seeds, ask=candidates.make_asker(server.post)),             "qwen3-vl-8b"


def make_pieces(cuts_json: Path, seeds: list[int], picks: dict[str, int] | None = None, images: bool = True,
                models: bool = True, style: str = "", run=_run, edit=edit_command, score=score_candidates) -> dict:
    """Route C for every slot of a cuts.json: pictures, then the picked ones in 3D.

    With more than one seed, every picture is scored and the best of each slot picked
    (`candidates.json`, which the Pieces page lets the person change); `picks` overrides.
    """
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
        if len(seeds) > 1 and jobs:  # every piece cut from the whole (B0): no pictures to score
            scored, judge_used = score(record, cuts_json.parent, folder, list(seeds))
            candidates.write(folder, scored, judge_used)
        elif candidates.path(folder).is_file():
            candidates.path(folder).unlink()  # an earlier run's picks are not this run's pictures
    # only picks among this run's seeds: an older candidates.json must not choose an old picture
    chosen = {**{slot: seed for slot, seed in candidates.picks(folder).items() if seed in seeds}, **picks}
    result = {"cuts": str(cuts_json.resolve()), "seeds": list(seeds), "slots": {}}
    if models:
        for slot in dict.fromkeys(job.slot for job in jobs):
            seed = chosen.get(slot, seeds[0])
            picture = pieces.picture_path(folder, slot, seed)
            if not picture.is_file():
                raise ValueError(f"{slot}: no picture for seed {seed} at {picture}; run without --models-only")
            for command in pieces.model_commands(picture, folder, slot, sys.executable, REPO / "scripts"):
                run(command)
            from image_to_3dlab.slot_parts import camera

            alpha = camera.subject_alpha(picture)
            place = cuts.Placement.from_dict(record["slots"][slot]["placement"])
            result["slots"][slot] = {
                "seed": seed, "picked_by": "person" if slot in picks else
                candidates.load(folder)["slots"].get(slot, {}).get("by", "first seed"),
                "picture": str(picture), "glb": str(pieces.finished_path(folder, slot)),
                "views": str(pieces.model_path(folder, slot).with_suffix(".svviews")),
                "concept_box": pieces.concept_box(alpha, place)}
        b0 = [slot for slot, entry in record["slots"].items() if entry.get("route") == "B0"]
        if b0:
            result["slots"].update(make_b0(record, b0, folder, run))
        pieces.write_record(folder / "pieces.json", result)
    return result


def make_b0(cut_record: dict, chosen: list[str], folder: Path, run=_run, blender: Path | None = None) -> dict:
    """Route B0 for the given slots: the whole concept in 3D, split by slot in Blender."""
    from image_to_3dlab.blender import find_blender, missing_help
    from image_to_3dlab.slot_parts import camera

    concept = Path(cut_record["concept"])
    for command in pieces.b0_commands(concept, folder, sys.executable, REPO / "scripts"):
        run(command)
    model = folder / "b0" / "model" / "concept_finished.glb"
    views = model.with_name("concept.svviews")
    alpha = camera.subject_alpha(concept)
    cam = camera.from_views(views, alpha)
    slot_map = folder / "b0" / "slots.npy"
    slot_map.parent.mkdir(parents=True, exist_ok=True)
    np.save(slot_map, np.asarray(Image.open(Path(cut_record.get("cuts_dir", folder.parent / "cuts")) / "slots.png")))
    with Image.open(concept) as picture:
        size = list(picture.size)
    job = {"model": str(model), "camera": cam.to_dict(), "slot_map": str(slot_map), "body": cut_record["body"],
           "concept_size": size, "chosen": chosen, "out": str(folder / "b0")}
    path = folder / "b0" / "job.json"
    path.write_text(json.dumps(job, indent=1) + "\n")
    blender = blender or find_blender()
    if blender is None:
        raise ValueError(missing_help())
    run([str(blender), "-b", "--factory-startup", "--python-exit-code", "1", "--python",
         str(REPO / "scripts" / "blender_segment_parts.py"), "--", str(path)])
    seg = json.loads((folder / "b0" / "segment.json").read_text())
    box = list(camera.alpha_box(alpha))
    return {slot: {"route": "B0", "glb": seg["files"][slot], "concept_box": box, "frame": seg["frame"]}
            for slot in chosen if slot in seg["files"]}


def text_command(prompt: str, picture: Path, seed: int) -> list[str]:
    """sd-cli's text to image of the day, for a weapon or shield made on its own."""
    from viewer import image_api

    settings = image_api.clean_settings({"seed": seed})
    return image_api.build_command(prompt, settings, picture, image_api.resolve_weights())


def make_held(slot: str, words: str, folder: Path, seeds: list[int], style: str = "", run=_run,
              generate=text_command) -> dict:
    """A weapon or shield made on its own: pictures, then the first seed's in 3D."""
    prompt = pieces.held_prompt(slot, words, style)
    for seed in seeds:
        picture = pieces.picture_path(folder, slot, seed)
        picture.parent.mkdir(parents=True, exist_ok=True)
        run(generate(prompt, picture, seed))
    picture = pieces.picture_path(folder, slot, seeds[0])
    for command in pieces.model_commands(picture, folder, slot, sys.executable, REPO / "scripts"):
        run(command)
    return {"seed": seeds[0], "picture": str(picture), "glb": str(pieces.finished_path(folder, slot)),
            "prompt": prompt, "held": True}


def check_pieces(fit_dir: Path, body_json: Path, set_name: str) -> dict:
    """The equipment_part checks on every fitted piece, each queued for review."""
    from image_to_3dlab.checks import Asset, check_and_enqueue

    record = body_mod.load(body_json)
    report = json.loads((Path(fit_dir) / "fit.json").read_text())
    body_glb = Path(fit_dir) / "body_masked.glb"
    out = {}
    for name in report["files"]:
        if name in ("set.glb", "preview.glb", "body_masked.glb", "body_trimmed.glb"):
            continue
        asset = Asset(glb=Path(fit_dir) / name, key=f"{set_name}-{record['name']}-{Path(name).stem}",
                      body=body_glb, body_clips=tuple(record.get("clips", {}).values()))
        checked, _ = check_and_enqueue(asset, profile="equipment_part")
        out[Path(name).stem] = {"status": checked["status"],
                                "fit": {r["check"]: r["value"] for r in checked["results"] if r["check"].startswith("fit.")}}
    return out


def make_concept(body_json: Path, outfit: str, folder: Path, seeds: list[int], style: str = "", run=_run,
                 edit=edit_command) -> tuple[Path, dict]:
    """The outfit drawn on the base body (an edit of its picture); the first seed is taken."""
    from image_to_3dlab.concept import prompts, sidecar

    record = body_mod.load(body_json)
    # add, not dress: the chosen pieces drawn on the bodysuit, the rest of the body left as it is
    prompt = prompts.add_piece(outfit, style=style)
    folder.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        picture = folder / f"s{seed}.png"
        command = edit(prompt, Path(record["image"]), picture, seed)
        run(command)
        sidecar.write(picture, mode="add_piece", prompt=prompt, seed=seed, command=command,
                      references=[(Path(record["image"]), "shape")], inputs={"body": record["name"]})
    return folder / f"s{seeds[0]}.png", {"prompt": prompt, "seed": seeds[0], "seeds": list(seeds)}


def run_set(body_json: Path, set_name: str, slot_text: str | dict, outfit: str = "", concept: Path | None = None,
            weapon: str | None = None, shield: str | None = None, concept_seeds=(3,), seeds=(3,), style: str = "",
            root: Path = Path("output/sets"), run=_run, edit=edit_command, generate=text_command,
            fitter=None, checker=None, stop_at_pictures: bool = False, resume: bool = False,
            picks: dict[str, int] | None = None, score=score_candidates) -> Path:
    """One set on one body, start to finish; returns `<set>.parts.json`.

    `stop_at_pictures` ends after every piece's pictures are made and scored, returning
    `candidates.json` so the person can change the picks; `resume` goes on from there with
    the picks (`picks` overrides them) and makes the 3D, the fit and the checks.
    """
    record = body_mod.load(body_json)
    if not body_mod.NAME.match(set_name):
        raise ValueError(f"set name {set_name!r}: letters, digits, '.', '_' or '-'")
    raw = dict(slot_text) if isinstance(slot_text, dict) else parse_slots(slot_text) if slot_text else {}
    for slot, words in (("weapon", weapon), ("shield", shield)):
        if words is not None:
            raw[slot] = {**raw.get(slot, {}), "prompt": words}
    folder = Path(root) / set_name / record["name"]
    state_path = folder / "run_state.json"
    state = None
    if resume:
        if not state_path.is_file():
            raise ValueError(f"nothing to resume for {set_name} on {record['name']}: no pictures waiting")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        # the run as it was asked for: a weapon, shield, style or seeds given apart from the slots
        for slot, options in state.get("held", {}).items():
            raw[slot] = {**dict(options), **raw.get(slot, {})}   # a material asked beside the words
        style = style or state.get("style", "")
        seeds = state.get("seeds") or seeds
    chosen = slots.choose(raw)
    drawn = {k: v for k, v in chosen.items() if k in slots.DRAWN}
    if resume:
        concept, concept_info, mode = Path(state["concept"]), state["concept_info"], state["mode"]
        cut = {"subject_from": state.get("subject_from")}
        if picks:
            candidates.choose(folder / "pieces", picks)
        made = make_pieces(folder / "cuts" / "cuts.json", list(seeds), images=False, style=style, run=run,
                           edit=edit)["slots"] if drawn else {}
        path = _finish_set(body_json, record, set_name, folder, chosen, drawn, made, concept, concept_info, mode,
                           cut, seeds, style, run, generate, fitter, checker)
        state_path.unlink()  # only now: a fit that fails or is stopped can be resumed again
        return path
    if concept is None:
        if not outfit.strip():
            raise ValueError("give the outfit in words (--outfit) or a concept drawn on the body (--concept)")
        concept, concept_info = make_concept(body_json, outfit, folder / "concept", list(concept_seeds), style, run, edit)
        mode = "body_and_equipment"
    else:
        concept_info = {"given": True}
        mode = "dressed_split"
    cut = make_cuts(body_json, Path(concept), drawn, folder / "cuts") if drawn else {"slots": {}, "subject_from": None}
    if stop_at_pictures and drawn:
        make_pieces(folder / "cuts" / "cuts.json", list(seeds), models=False, style=style, run=run, edit=edit,
                    score=score)
        held = {slot: {"prompt": chosen[slot]["prompt"]} for slot in ("weapon", "shield") if slot in chosen}
        state_path.write_text(json.dumps({"waiting": "picks", "concept": str(concept), "concept_info": concept_info,
                                          "mode": mode, "subject_from": cut.get("subject_from"), "held": held,
                                          "style": style, "seeds": list(seeds)}, indent=1) + "\n",
                              encoding="utf-8")
        return candidates.path(folder / "pieces")
    made = make_pieces(folder / "cuts" / "cuts.json", list(seeds), style=style, run=run, edit=edit,
                       score=score)["slots"] if drawn else {}
    return _finish_set(body_json, record, set_name, folder, chosen, drawn, made, concept, concept_info, mode, cut,
                       seeds, style, run, generate, fitter, checker)


def _finish_set(body_json, record, set_name, folder, chosen, drawn, made, concept, concept_info, mode, cut, seeds,
                style, run, generate, fitter, checker) -> Path:
    """Held pieces, the fit, the checks and `<set>.parts.json`, once the drawn pieces are made."""
    from image_to_3dlab.slot_parts import sidecar

    for slot in ("weapon", "shield"):
        if slot in chosen:
            made[slot] = make_held(slot, chosen[slot]["prompt"], folder / "pieces", list(seeds), style, run, generate)
    (folder / "pieces").mkdir(parents=True, exist_ok=True)
    pieces_json = folder / "pieces" / "pieces.json"
    pieces.write_record(pieces_json, {"cuts": str(folder / "cuts" / "cuts.json"), "seeds": list(seeds), "slots": made})
    job = fit_job(pieces_json, body_json, folder / "fit") if drawn else {
        "body": str(Path(body_json).resolve()), "concept_size": [1, 1], "out": str((folder / "fit").resolve()),
        "pieces": [{"slot": s, "glb": e["glb"]} for s, e in made.items()]}
    for slot in ("weapon", "shield"):
        if slot in made and drawn:
            job["pieces"] = [p for p in job["pieces"] if p["slot"] != slot] + [{"slot": slot, "glb": made[slot]["glb"]}]
    fit_report = (fitter or run_fit)(job)
    checks = (checker or check_pieces)(folder / "fit", body_json, set_name)
    parts = sidecar.parts_record(mode=mode, set_name=set_name, body=record, body_json=body_json, concept=Path(concept),
                                 concept_info=concept_info, chosen=chosen, pieces=made, fit=fit_report, checks=checks,
                                 matte=cut.get("subject_from"))
    return sidecar.write(folder, set_name, parts)


def refit_job(fit_dir: Path, source_body: Path, body_json: Path, out: Path) -> dict:
    """The job that moves a fitted set onto another body: every fitted piece, carried over."""
    report = json.loads((Path(fit_dir) / "fit.json").read_text())
    pieces_out = []
    for name in report["files"]:
        if name in ("set.glb", "preview.glb", "body_masked.glb", "body_trimmed.glb"):
            continue
        stem = Path(name).stem
        slot = stem.removesuffix("_L").removesuffix("_R")
        # one shoulders file was fitted unsplit; two (_L and _R) were split on purpose
        pieces_out.append({"slot": slot, "glb": str((Path(fit_dir) / name).resolve()), "transfer": True,
                           "split": stem != slot})
    if not pieces_out:
        raise ValueError(f"{fit_dir} holds no fitted pieces")
    return {"body": str(Path(body_json).resolve()), "source_body": str(Path(source_body).resolve()),
            "concept_size": [1, 1], "out": str(Path(out).resolve()), "pieces": pieces_out,
            "clips": list(body_mod.load(body_json).get("clips", {}).values())}


def unreal_set(fit_dir: Path, body_json: Path, out: Path, size_cm: float = 180.0, exporter=None) -> Path:
    """The body and every piece of a fitted set for Unreal 5.8, sharing one Skeleton, and a
    `<body>.unreal-set.json` saying how to put them together."""
    if exporter is None:
        sys.path.insert(0, str(REPO / "scripts"))
        from export_unreal import export_unreal as exporter
    from image_to_3dlab import piece_materials

    record = body_mod.load(body_json)
    report = json.loads((Path(fit_dir) / "fit.json").read_text())
    try:
        chosen = piece_materials.set_materials(Path(fit_dir).parent)
    except FileNotFoundError:            # a set fitted before its record: the slots' defaults
        chosen = {}
    materials = {}
    base = "".join(part.capitalize() for part in record["name"].replace("_", "-").split("-"))
    body_glb = Path(fit_dir) / "body_masked.glb"
    exported = {"body": exporter(body_glb, kind="creature", size_cm=size_cm, name=base, out_dir=Path(out) / base)}
    pieces_out = {}
    for name in report["files"]:
        if name in ("set.glb", "preview.glb", "body_masked.glb", "body_trimmed.glb"):
            continue
        piece = "".join(part.capitalize() for part in Path(name).stem.split("_"))
        slot = piece_materials.piece_of(name)
        if slot is not None:
            materials[slot] = chosen.get(slot) or piece_materials.propose(slot=slot)
        pieces_out[Path(name).stem] = exporter(Path(fit_dir) / name, kind="equipment", size_cm=size_cm,
                                               name=f"{base}{piece}", body=body_glb, body_name=base,
                                               out_dir=Path(out) / f"{base}{piece}",
                                               material=materials[slot]["material"] if slot else None)
    summary = {
        "body": record["name"], "size_cm": size_cm, "skeleton": f"SK_{base}_Skeleton",
        "meshes": {"body": exported["body"]["glb"], **{k: v["glb"] for k, v in pieces_out.items()}},
        "problems": {k: v.get("problems", []) for k, v in {"body": exported["body"], **pieces_out}.items()},
        "materials": materials,
        "how": [
            "Import the body first, then each piece on the body's Skeleton (each .unreal.json says so).",
            ("Put every piece on a Skeletal Mesh Component under the body's and call SetLeaderPoseComponent(body) "
             "on it, so the pieces follow the body's animation with no cost of their own."),
            ("The body carries a vertex colour: alpha 0 where a piece covers the skin, and its glTF material is "
             "MASK. Unreal's glTF import ignores vertex alpha, so the body needs a Masked material with Vertex "
             "Color alpha in Opacity Mask; unreal_import_check.py makes it (M_<Body>_Masked, from the body's "
             "textures, which stay parameters) and checks it."),
            "The weapon and shield are skinned to one bone (the hand, the forearm) at 100%: no socket needed.",
        ],
    }
    path = Path(out) / f"{record['name']}.unreal-set.json"
    Path(out).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1) + "\n")
    return path


def fit_job(pieces_json: Path, body_json: Path, out: Path, options: dict | None = None) -> dict:
    """The job `blender_fit_parts.py` reads: the body, the concept's size and each piece's box."""
    record = json.loads(Path(pieces_json).read_text())
    cut_record = json.loads(Path(record["cuts"]).read_text())
    with Image.open(cut_record["concept"]) as concept:
        size = list(concept.size)
    if not record["slots"]:
        raise ValueError(f"{pieces_json} has no pieces: run the pieces command first")
    entries = [{"slot": slot, "glb": entry["glb"],
                "split": bool(cut_record["slots"].get(slot, {}).get("split", False)),
                **{k: entry[k] for k in ("concept_box", "frame") if k in entry}}
               for slot, entry in record["slots"].items()]
    clips = list(body_mod.load(body_json).get("clips", {}).values())
    return {"body": str(Path(body_json).resolve()), "concept_size": size, "out": str(Path(out).resolve()),
            "pieces": entries, "clips": clips, **(options or {})}


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
    report = json.loads((out / "fit.json").read_text())
    from image_to_3dlab.slot_parts.sidecar import fit_provenance

    fit_provenance(out, job, report)
    return report


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
    b.add_argument("--clip", action="append", default=[], help="name=animated.glb, the body's own clip (repeatable)")
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
    q.add_argument("--pick", default="", help="slot=seed for the picture to make in 3D (default the best scored)")
    q.add_argument("--style", default="", help="style words added to every piece's prompt")
    stage = q.add_mutually_exclusive_group()
    stage.add_argument("--images-only", action="store_true", help="stop after the pictures")
    stage.add_argument("--models-only", action="store_true", help="reuse the pictures, make the 3D")
    f = sub.add_parser("fit", help="fit the pieces onto the body in Blender, skin them, mask the body")
    f.add_argument("--pieces", type=Path, required=True, help="pieces.json from the pieces command")
    f.add_argument("--body", type=Path, required=True, help="body.json of the body the concept was drawn on")
    f.add_argument("--out", type=Path, help="default: fit/ beside pieces/")
    f.add_argument("--blender", type=Path, help="Blender executable (found automatically when omitted)")
    r = sub.add_parser("run", help="one set on one body, start to finish (the best-scored picture of each piece)")
    r.add_argument("--body", type=Path, required=True)
    r.add_argument("--set", required=True, dest="set_name", help="the set's name, a folder name")
    r.add_argument("--slots", default="", help="as for cuts; empty for a weapon or shield only")
    r.add_argument("--slots-json", type=Path, help="the slots as JSON, {slot: {route, split, prompt}} (the studio's way)")
    r.add_argument("--outfit", default="", help="the outfit in words, drawn on the body")
    r.add_argument("--concept", type=Path, help="a concept already drawn on the body (mode 2)")
    r.add_argument("--weapon", help="a weapon made on its own, in words")
    r.add_argument("--shield", help="a shield made on its own, in words")
    r.add_argument("--concept-seeds", default="3")
    r.add_argument("--seeds", default="3")
    r.add_argument("--style", default="")
    r.add_argument("--root", type=Path, default=Path("output/sets"))
    r.add_argument("--stop-at-pictures", action="store_true",
                   help="stop once every piece's pictures are made and scored, for the person to pick")
    r.add_argument("--resume", action="store_true", help="go on from the pictures with the picks (candidates.json)")
    r.add_argument("--pick", default="", help="slot=seed picks for --resume, over the automatic ones")
    rf = sub.add_parser("refit", help="carry a fitted set over to another base body")
    rf.add_argument("--fit", type=Path, required=True, help="the fit/ folder of the set on its first body")
    rf.add_argument("--from", type=Path, required=True, dest="source", help="body.json of that first body")
    rf.add_argument("--body", type=Path, required=True, help="body.json of the body to carry it to")
    rf.add_argument("--out", type=Path, help="default: <set>/<new body>/fit")
    rf.add_argument("--blender", type=Path)
    un = sub.add_parser("unreal", help="export a fitted set for Unreal 5.8, one Skeleton for all")
    un.add_argument("--fit", type=Path, required=True)
    un.add_argument("--body", type=Path, required=True)
    un.add_argument("--size-cm", type=float, default=180.0, help="the body's real height")
    un.add_argument("--out", type=Path, help="default: unreal/ beside fit/")
    mt = sub.add_parser("materials", help="show or correct the one material of each piece of a set")
    mt.add_argument("set_dir", type=Path, help="output/sets/<set>/<body>/")
    mt.add_argument("--set", action="append", default=[], metavar="SLOT=MATERIAL",
                    help="a person's material for a piece (repeatable)")
    mt.add_argument("--clear", action="append", default=[], metavar="SLOT", help="go back to the rule's proposal")
    args = p.parse_args(argv)
    try:
        if args.command == "materials":
            from image_to_3dlab import piece_materials

            changes = {**{slot: None for slot in args.clear},
                       **dict(item.split("=", 1) for item in args.set)}
            if changes:
                piece_materials.save_corrections(args.set_dir, changes)
            print(json.dumps(piece_materials.set_materials(args.set_dir), indent=1))
        elif args.command == "refit":
            name = body_mod.load(args.body)["name"]
            out = args.out or args.fit.parent.parent / name / "fit"
            report = run_fit(refit_job(args.fit, args.source, args.body, out), args.blender)
            print(f"carried {', '.join(report['pieces'])} over to {name}; body covered {report['covered_share']:.0%}")
        elif args.command == "unreal":
            path = unreal_set(args.fit, args.body, args.out or args.fit.parent / "unreal", args.size_cm)
            print(f"wrote {path}")
        elif args.command == "run":
            slot_text = json.loads(args.slots_json.read_text()) if args.slots_json else args.slots
            path = run_set(args.body, args.set_name, slot_text, args.outfit, args.concept, args.weapon, args.shield,
                           [int(s) for s in args.concept_seeds.split(",") if s.strip()],
                           [int(s) for s in args.seeds.split(",") if s.strip()], args.style, args.root,
                           stop_at_pictures=args.stop_at_pictures, resume=args.resume,
                           picks=pieces.parse_picks(args.pick))
            print(f"{'waiting for picks: ' if args.stop_at_pictures else 'wrote '}{path}")
        elif args.command == "fit":
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
            clips = dict(item.split("=", 1) for item in args.clip)
            path = body_mod.register(args.name, args.image, args.rigged, args.views, args.matte, labels, args.root,
                                     clips)
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
