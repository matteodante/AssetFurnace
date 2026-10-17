"""`rig`: a skeleton and skin weights for the row's model, as the studio's Animate tab
gives them (SkinTokens, then the skin-weight clean-up; `image_to_3dlab/rigging.py`).

The rigged GLB becomes the row's asset, with a `.provenance.json` carrying the source's
licence and SkinTokens' beside it. SkinTokens holds the shared GPU lock while it runs. The
creature family's test clips are not made here: that is the `clips` step.
"""

from __future__ import annotations

from image_to_3dlab import rigging
from image_to_3dlab.batch_steps import StepContext, StepResult


def preflight(rows) -> list[str]:
    return [] if not rows or rigging.installed() else [rigging.MISSING]


def run(ctx: StepContext) -> StepResult:
    if ctx.asset is None:
        return StepResult(ok=False, reason="no model to rig")
    try:
        rigged, sidecar = rigging.rig_model(ctx.asset, ctx.step_dir, ctx.row.id, origin=f"batch row {ctx.row.id}",
                                            tool="batch step rig: SkinTokens auto-rig", log=ctx.log,
                                            session=f"batch rig {ctx.row.id}")
    except (RuntimeError, ValueError, OSError) as exc:
        return StepResult(ok=False, reason=str(exc))
    return StepResult(asset=rigged, sidecar=sidecar,
                      files=[rigged, sidecar, ctx.step_dir / "input" / "source.glb", ctx.step_dir / "rig.log"])
