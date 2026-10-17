"""What the steps that make several pictures to pick from share (`picture`, `concept`).

The candidates live in `rows/<id>/<step>.candidates/`, a folder the runner never empties,
with an index of the key they were made for (`StepContext.candidate_key`: the step's
fingerprint without the pick). The same key keeps them, one by one, so a run stopped after
two of four draws only the other two, and a pick, or another pick later, draws nothing.
Another key clears the folder first, so it never mixes two requests.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from image_to_3dlab.batch_steps import StepContext, StepResult

INDEX = "candidates.json"
# what is kept beside a picture and travels with the chosen one
SIDECARS = (".provenance.json", ".png.concept.json")


def folder(ctx: StepContext, step: str) -> Path:
    return ctx.row_dir / f"{step}.candidates"


def draw_all(ctx: StepContext, step: str, seeds: list[int], draw_one: Callable[[int, Path], None],
             legacy: str | None = None) -> dict[int, Path]:
    """One picture per seed, `s<seed>.png`, reusing those already made for this key (or
    for `legacy`, the key an earlier version of the step gave the same request)."""
    where = folder(ctx, step)
    index = where / INDEX
    try:
        kept = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        kept = {}
    if legacy is not None and kept.get("key") == legacy:
        kept = {"key": ctx.candidate_key, "done": kept.get("done") or kept.get("seeds") or []}
    if kept.get("key") != ctx.candidate_key:
        if where.exists():
            shutil.rmtree(where)
        kept = {"key": ctx.candidate_key, "done": []}
    where.mkdir(parents=True, exist_ok=True)
    pictures = {seed: where / f"s{seed}.png" for seed in seeds}
    for seed, path in pictures.items():
        if seed in kept["done"] and path.is_file():
            continue
        ctx.log(f"{step}: seed {seed}")
        draw_one(seed, path)
        kept["done"] = sorted({*kept["done"], seed})
        index.write_text(json.dumps(kept) + "\n", encoding="utf-8")
    return pictures


def choose(ctx: StepContext, pictures: dict[int, Path], record: dict | None = None) -> StepResult:
    """The pick among the pictures: the row waits while there is none (several and no
    `pick`), fails on a pick that is not one of them, else the chosen picture becomes
    `<step dir>/<id>.png`, with its records, and the row's picture."""
    seeds = list(pictures)
    pick = ctx.options.get("pick", seeds[0] if len(seeds) == 1 else None)
    if pick is None:
        return StepResult(waiting={"candidates": [{"choice": seed, "file": pictures[seed]} for seed in seeds]})
    if pick not in pictures:
        return StepResult(ok=False, reason=f"pick {pick} is not one of the pictures made "
                                           f"(seeds {', '.join(map(str, seeds))})")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    chosen = ctx.step_dir / f"{ctx.row.id}.png"
    shutil.copyfile(pictures[pick], chosen)
    files = [chosen]
    for suffix in SIDECARS:
        source = pictures[pick].with_name(pictures[pick].stem + suffix)
        if not source.is_file():
            continue
        target = chosen.with_name(chosen.stem + suffix)
        data = json.loads(source.read_text(encoding="utf-8"))
        if suffix == ".provenance.json":
            data["picked"] = {"seed": pick, "from": seeds}
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        files.append(target)
    return StepResult(picture=chosen, files=files, record={"seed": pick, "candidates": seeds, **(record or {})})
