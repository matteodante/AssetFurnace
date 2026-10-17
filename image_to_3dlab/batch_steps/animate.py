"""`animate`: preset clips played on the row's rigged model, as the studio's Animate tab
plays them (`scripts/kimodo_retarget.py` on Kimodo clips shipped in
`image_to_3dlab/motions/`; `image_to_3dlab/rigging.py`). CPU only.

Options (`"animate": {...}`, described in `step_catalog`): presets (required, e.g.
`["idle", "walk"]`), speed (default each preset's own), arm_spread (degrees, default 0).
The clips are `<id>_<preset>.glb` with their provenance; the row's asset stays the rigged
model, and `export_unreal` takes the clips as the creature's animations.
"""

from __future__ import annotations

from image_to_3dlab import rigging, step_catalog
from image_to_3dlab.batch_steps import StepContext, StepResult


def preflight(rows) -> list[str]:
    problems = []
    for row in rows:
        options = row.options.get("animate", {})
        problems += [f"row {row.id}: animate {problem}" for problem in step_catalog.check("animate", options)]
        if not options.get("presets"):
            problems.append(f'row {row.id}: animate needs presets, e.g. "animate": {{"presets": ["idle"]}}')
    if rows and not rigging.installed():
        problems.append(rigging.MISSING)
    return problems


def run(ctx: StepContext) -> StepResult:
    if ctx.asset is None:
        return StepResult(ok=False, reason="no rigged model to animate")
    try:
        clips = rigging.animate_model(ctx.asset, list(ctx.options.get("presets") or []), ctx.step_dir,
                                      speed=ctx.options.get("speed"), arm_spread=ctx.options.get("arm_spread", 0.0),
                                      tool="batch step animate: scripts/kimodo_retarget.py", log=ctx.log)
    except (RuntimeError, ValueError, OSError) as exc:
        return StepResult(ok=False, reason=str(exc))
    files = [path for clip in clips for path in (clip, clip.with_suffix(".provenance.json"))]
    return StepResult(files=[*files, ctx.step_dir / "animate.log"], clips=clips,
                      record={"clips": [clip.name for clip in clips]})
