"""`concept`: a new concept drawn from the row's picture, as the studio's "Draw from a
reference" panel and `scripts/reference_concept.py make` draw it (feature 2,
`image_to_3dlab/concept/requests.py`: Qwen-Image 2.1 in edit mode, the GPU lock held while
it draws, the `.concept.json` record with the silhouette IoU).

Options (`"concept": {...}`, described in `step_catalog`): mode (required: redraw, dress,
add_piece, other_view, complete_piece), shape and style_refs (files; the shape default is
the row's picture: its first image, or the one a `picture` step drew), subject, colours,
parts, structure, outfit, guide, piece, view, depth, style, seed (default 42), count and
pick. With `count` above 1 the row waits for a pick, like `picture`; the candidates are kept
in `rows/<id>/concept.candidates/`. The chosen concept becomes the row's picture, which
`generate` reads, the row's own images following it as references.
"""

from __future__ import annotations

from pathlib import Path

from image_to_3dlab import step_catalog, text_image
from image_to_3dlab.batch_steps import StepContext, StepResult, _candidates
from image_to_3dlab.batch_steps.picture import seeds_of
from image_to_3dlab.concept import edit, requests


def readiness() -> list[str]:
    """What is missing to draw, named, never fetched."""
    problems = []
    if not text_image.BINARY.exists():
        problems.append(f"the concept step needs sd-cli at {text_image.BINARY}; the studio's Setup page installs it")
    try:
        text_image.resolve_weights()
    except text_image.MissingWeights as exc:
        problems.append(f"the concept step needs Qwen-Image 2.1 Turbo (missing: {', '.join(exc.missing)}); "
                        "the studio's Setup page downloads it after asking")
    if edit.find_vision(text_image.HF_HUB) is None:
        problems.append(f"the concept step needs the vision file {edit.VISION_FILE}; the studio's Setup page "
                        "downloads it after asking")
    return problems


def preflight(rows) -> list[str]:
    problems = []
    for row in rows:
        options = row.options.get("concept", {})
        problems += [f"row {row.id}: concept {problem}" for problem in step_catalog.check("concept", options)]
        if not options.get("mode"):
            problems.append(f'row {row.id}: concept needs a mode, e.g. "concept": {{"mode": "redraw"}}')
            continue
        try:
            from image_to_3dlab.style import apply

            apply.resolve(options.get("bible"), row.kind)
        except ValueError as exc:
            problems.append(f"row {row.id}: concept: {exc}")
        # the words each way needs, checked now on a stand-in picture count
        shape = options.get("shape") or [Path(f"shape{i}.png") for i in range(4)]
        try:
            from image_to_3dlab.concept import prompts

            needed = prompts.shape_pictures(options["mode"], depth=bool(options.get("depth")))
            requests.plan(options["mode"], [Path(p) for p in shape][:needed],
                          [Path(p) for p in options.get("style_refs") or []], options)
        except ValueError as exc:
            problems.append(f"row {row.id}: concept: {exc}")
    return problems + (readiness() if rows else [])


def shape_of(ctx: StepContext) -> list[Path]:
    given = ctx.options.get("shape")
    if given:
        return [Path(p) for p in given]
    return [ctx.picture] if ctx.picture else []


def run(ctx: StepContext) -> StepResult:
    from image_to_3dlab.style import apply

    options = ctx.options
    try:
        style = apply.resolve(options.get("bible"), ctx.row.kind, references=3)
        plan = requests.plan(options.get("mode"), shape_of(ctx),
                             [Path(p) for p in options.get("style_refs") or []], options, bible=style)
    except ValueError as exc:   # a request that does not hold, or an unknown bible
        return StepResult(ok=False, reason=str(exc))

    def draw(seed: int, path: Path) -> None:
        requests.draw(plan, seed, path, log=ctx.log, session=f"batch concept {ctx.row.id}")

    pictures = _candidates.draw_all(ctx, "concept", seeds_of(options), draw)
    return _candidates.choose(ctx, pictures, record={"mode": plan["mode"]})
