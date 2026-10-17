"""`materials`: the row's one material, as the user closed function 8 (one material per piece
or asset, `image_to_3dlab/piece_materials.py`; no automatic zones).

Options (`"materials": {...}`, described in `step_catalog`): material, the person's choice
(metal, leather, cloth...). Without it the rule proposes one: the first material the words
the model was made from name (its prompt, from its provenance), else the default of the
row's kind. The choice and why are written to `material.json` in the step's folder, and
`export_unreal` takes the material for the ORM (its own `material` option wins). The row's
asset is left as it is. Left out of the row, the export keeps the model's own maps, as
`export_unreal.py` without `--material`.
"""

from __future__ import annotations

import json

from image_to_3dlab import piece_materials, step_catalog
from image_to_3dlab.batch_steps import StepContext, StepResult


def preflight(rows) -> list[str]:
    return [f"row {row.id}: materials {problem}" for row in rows
            for problem in step_catalog.check("materials", row.options.get("materials", {}))]


def choose(ctx: StepContext) -> dict:
    """The person's material, else the rule's from the model's words and the row's kind."""
    asked = ctx.options.get("material")
    if asked:
        return {"material": piece_materials.check(asked), "by": "person", "reason": "chosen in the batch spec"}
    # the current model's words, else the row's own model's: a rig's record has no path back
    words = ((piece_materials.prompt_for(ctx.asset) if ctx.asset else "")
             or (piece_materials.prompt_for(ctx.row.asset) if ctx.row.asset else ""))
    kind = ctx.row.kind if ctx.row.kind in piece_materials.KIND_DEFAULTS else None
    return piece_materials.propose(kind=kind, words=words)


def run(ctx: StepContext) -> StepResult:
    try:
        chosen = choose(ctx)
    except ValueError as exc:
        return StepResult(ok=False, reason=str(exc))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    record = ctx.step_dir / "material.json"
    record.write_text(json.dumps({"schema": piece_materials.SCHEMA, **chosen}, indent=1) + "\n", encoding="utf-8")
    ctx.log(f"materials: {chosen['material']} ({chosen['reason']})")
    return StepResult(files=[record], material=chosen["material"], record=chosen)
