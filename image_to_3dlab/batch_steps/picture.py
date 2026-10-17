"""`picture`: the row's picture drawn from its `prompt`, for rows that start from words.

The engine and settings of the studio's Create and `scripts/text_image.py`
(`image_to_3dlab/text_image.py`: Qwen-Image 2.1 Turbo through sd-cli), with the shared GPU
lock held while it draws. Options (`"picture": {...}`, described in `step_catalog`): seed
(default 42), count, pick, width, height, cfg_scale, sampler, negative_prompt.

**Several to pick from.** With `count` above 1 the step draws one picture per seed
(`seed`, `seed`+1, ...) and the row waits for a pick: `"pick": <seed>` in the spec, or the
MCP tool `pick_candidate`. The candidates are kept in `rows/<id>/picture.candidates/`, a
folder the runner never empties, so a pick (or another pick later) copies one of them and
draws nothing again; only new words or settings draw again. A pick made through the MCP
server holds for the candidates it was made among (their key, `_candidates`): new words
or settings make the row wait for a new one. The chosen picture is
`picture/<id>.png` with its `.provenance.json`, and the next step (`generate`) reads it.
"""

from __future__ import annotations

from image_to_3dlab import step_catalog, text_image
from image_to_3dlab.batch_steps import StepContext, StepResult, _candidates


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


def run(ctx: StepContext) -> StepResult:
    def draw(seed, path):
        text_image.generate(ctx.row.prompt or "", {**settings_of(ctx.options), "seed": seed}, path,
                            log=ctx.log, session=f"batch picture {ctx.row.id}")

    pictures = _candidates.draw_all(ctx, "picture", seeds_of(ctx.options), draw)
    return _candidates.choose(ctx, pictures)
