"""`picture`: the row's picture drawn from its `prompt`, for rows that start from words.

The engine and settings of the studio's Create and `scripts/text_image.py`
(`image_to_3dlab/text_image.py`: Qwen-Image 2.1 Turbo through sd-cli), with the shared GPU
lock held while it draws. Options (`"picture": {...}`, described in `step_catalog`): seed
(default 42), count, pick, width, height, cfg_scale, sampler, negative_prompt, bible (a
style bible's words after the prompt: "-" for the active one, or its name).

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
    from image_to_3dlab.style import apply

    problems = [f"row {row.id}: {problem}" for row in rows
                for problem in step_catalog.check("picture", row.options.get("picture", {}))]
    for row in rows:
        try:
            apply.resolve(row.options.get("picture", {}).get("bible"), row.kind)
        except ValueError as exc:
            problems.append(f"row {row.id}: picture: {exc}")
    return problems + (readiness() if rows else [])


def settings_of(options: dict) -> dict:
    """The drawing settings, without the pick, the count and the bible (its words go in the
    prompt)."""
    return {key: value for key, value in options.items() if key not in ("pick", "count", "bible")}


def seeds_of(options: dict) -> list[int]:
    first = int(options.get("seed", text_image.DEFAULTS["seed"]))
    return [first + offset for offset in range(int(options.get("count", 1)))]


def legacy_key(prompt: str | None, options: dict) -> str:
    """The candidates' key as the first version of this step made it (the words, the
    settings, the seeds): candidates and picks from then still count."""
    import hashlib
    import json

    payload = {"prompt": prompt or "", "settings": settings_of(options), "seeds": seeds_of(options)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run(ctx: StepContext) -> StepResult:
    from image_to_3dlab.style import apply

    style = apply.resolve(ctx.options.get("bible"), ctx.row.kind)
    extra = {"style": apply.record(style)} if style else None

    def draw(seed, path):
        text_image.generate(apply.styled(ctx.row.prompt or "", style), {**settings_of(ctx.options), "seed": seed},
                            path, log=ctx.log, session=f"batch picture {ctx.row.id}", extra_provenance=extra)

    pictures = _candidates.draw_all(ctx, "picture", seeds_of(ctx.options), draw,
                                    legacy=legacy_key(ctx.row.prompt, ctx.options))
    return _candidates.choose(ctx, pictures)
