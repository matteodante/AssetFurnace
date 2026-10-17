"""`clips`: a rigged creature's family and its test clips, as the studio makes them after a
rig (feature 6, `scripts/creature_clips.py`): the CC0 library's clips carried onto the rig
by chains, the procedural ones for the rest, and the library's loops also as longer
varied takes (GenMM).

Options (`"clips": {...}`, described in `step_catalog`): family (default recognised from
the skeleton), library (default true), variations (default true), variation_factor and
variation_seed (default the library's 2x and seed 0).

The rig is copied into the step's folder and the clips are written beside the copy, with
the family file and a provenance record per clip; the copy becomes the row's asset, so
`export_unreal` finds the family beside it (for a creature's facing) and takes the clips
as its animations. Runs in SkinTokens' environment (it carries Blender's bpy), CPU only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from image_to_3dlab import creature_family, rigging, step_catalog
from image_to_3dlab.batch_steps import StepContext, StepResult


def preflight(rows) -> list[str]:
    problems = [f"row {row.id}: clips {problem}" for row in rows
                for problem in step_catalog.check("clips", row.options.get("clips", {}))]
    for row in rows:
        if "export_unreal" in row.steps and row.kind != "creature":
            # an object is exported without animations or facing: its clips would be dropped unsaid
            problems.append(f'row {row.id}: the family clips reach the Unreal export only for "kind": "creature" '
                            f"(it is {row.kind})")
    if rows and not rigging.installed():
        problems.append(rigging.MISSING)
    return problems


def run(ctx: StepContext) -> StepResult:
    if ctx.asset is None or not rigging.has_skin(ctx.asset):
        return StepResult(ok=False, reason="the family clips need a rigged model: put rig before clips")
    options = ctx.options
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    rigged = ctx.step_dir / ctx.asset.name
    shutil.copyfile(ctx.asset, rigged)
    record = json.loads(Path(ctx.sidecar).read_text(encoding="utf-8")) if ctx.sidecar and Path(ctx.sidecar).is_file() else None
    sidecar = rigged.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(record or {"license": None}, indent=2), encoding="utf-8")
    command = rigging.clip_command(rigging.venv_python(), rigged, options.get("family", "auto"),
                                   clips=options.get("clips", True),
                                   library=options.get("library", True), variations=options.get("variations", True),
                                   variation_factor=options.get("variation_factor"),
                                   variation_seed=options.get("variation_seed"))
    log = ctx.step_dir / "clips.log"
    ctx.log("clips: the family and its test clips")
    if rigging.run_commands([command], log) != 0:
        return StepResult(ok=False, reason=f"the clip step failed:\n{rigging.log_tail(log)}")
    found = creature_family.read_family(rigged) or {}
    clips = rigging.clip_files(rigged)
    records = rigging.record_clips(rigged, record or {}, tool="batch step clips: scripts/creature_clips.py")
    if found.get("clips_error") and not clips:
        return StepResult(ok=False, reason=found["clips_error"])
    family = creature_family.family_path(rigged)
    return StepResult(asset=rigged, sidecar=sidecar, clips=clips,
                      files=[rigged, sidecar, *([family] if family.is_file() else []), *clips, *records, log],
                      record={"family": found.get("family"), "clips": [clip.name for clip in clips],
                              **({"problems": found["clips_error"]} if found.get("clips_error") else {})})
