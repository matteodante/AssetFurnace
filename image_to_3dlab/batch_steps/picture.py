"""`picture`: the row's picture drawn from its `prompt`, for rows that start from words.

The engine and settings of the studio's Create and `scripts/text_image.py`
(`image_to_3dlab/text_image.py`: Qwen-Image 2.1 Turbo through sd-cli), with the shared GPU
lock held while it draws. Options (`"picture": {...}`, described in `step_catalog`): seed
(default 42), count, pick, width, height, cfg_scale, sampler, negative_prompt.

**Several to pick from.** With `count` above 1 the step draws one picture per seed
(`seed`, `seed`+1, ...) and the row waits for a pick: `"pick": <seed>` in the spec, or the
MCP tool `pick_candidate`. The candidates are kept in `rows/<id>/picture.candidates/`, a
folder the runner never empties, so a pick (or another pick later) copies one of them and
draws nothing again; only new words or settings draw again. The chosen picture is
`picture/<id>.png` with its `.provenance.json`, and the next step (`generate`) reads it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from image_to_3dlab import step_catalog, text_image
from image_to_3dlab.batch_steps import StepContext, StepResult

CANDIDATES = "picture.candidates"
INDEX = "candidates.json"


def readiness() -> list[str]:
    """What is missing to draw, named, never fetched."""
    problems = []
    if not text_image.BINARY.exists():
        problems.append(f"the picture step needs sd-cli at {text_image.BINARY}; the studio's "
                        "Setup page installs it")
    try:
        text_image.resolve_weights()
    except text_image.MissingWeights as exc:
        problems.append(f"the picture step needs Qwen-Image 2.1 Turbo (missing: "
                        f"{', '.join(exc.missing)}); the studio's Setup page downloads it after asking")
    return problems


def preflight(rows) -> list[str]:
    problems = [f"row {row.id}: {problem}" for row in rows
                for problem in step_catalog.check("picture", row.options.get("picture", {}))]
    return problems + (readiness() if rows else [])


def settings_of(options: dict) -> dict:
    """The drawing settings, without the pick and the count."""
    return {key: value for key, value in options.items() if key not in ("pick", "count")}


def seeds_of(options: dict) -> list[int]:
    first = int(options.get("seed", text_image.DEFAULTS["seed"]))
    return [first + offset for offset in range(int(options.get("count", 1)))]


def _key(prompt: str, options: dict) -> str:
    """What the candidates were drawn from: same key, same pictures."""
    payload = {"prompt": prompt, "settings": settings_of(options), "seeds": seeds_of(options)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def candidates(ctx: StepContext) -> dict[int, Path]:
    """The pictures for this prompt and these settings: kept from an earlier run when they
    match, else drawn now (the folder is cleared first, so it never mixes two prompts)."""
    folder = ctx.row_dir / CANDIDATES
    key = _key(ctx.row.prompt or "", ctx.options)
    index = folder / INDEX
    try:
        kept = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        kept = {}
    seeds = seeds_of(ctx.options)
    pictures = {seed: folder / f"s{seed}.png" for seed in seeds}
    if kept.get("key") == key and all(path.is_file() for path in pictures.values()):
        return pictures
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for seed, path in pictures.items():
        ctx.log(f"picture: seed {seed}")
        text_image.generate(ctx.row.prompt or "", {**settings_of(ctx.options), "seed": seed}, path,
                            log=ctx.log, session=f"batch picture {ctx.row.id}")
    index.write_text(json.dumps({"key": key, "seeds": seeds}) + "\n", encoding="utf-8")
    return pictures


def run(ctx: StepContext) -> StepResult:
    pictures = candidates(ctx)
    seeds = list(pictures)
    pick = ctx.options.get("pick", seeds[0] if len(seeds) == 1 else None)
    if pick is None:
        return StepResult(waiting={"candidates": [{"choice": seed, "file": pictures[seed]} for seed in seeds]})
    if pick not in pictures:
        return StepResult(ok=False, reason=f"pick {pick} is not one of the pictures drawn "
                                           f"(seeds {', '.join(map(str, seeds))})")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    chosen = ctx.step_dir / f"{ctx.row.id}.png"
    shutil.copyfile(pictures[pick], chosen)
    record = json.loads(pictures[pick].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    record["picked"] = {"seed": pick, "from": seeds}
    sidecar = chosen.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return StepResult(picture=chosen, files=[chosen, sidecar], record={"seed": pick, "candidates": seeds})
