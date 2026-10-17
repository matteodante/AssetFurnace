"""`materials`: the row's model with its own maps packed as a game ORM, and tint zones
when asked, as `scripts/game_materials.py` writes it (feature 8, `image_to_3dlab/materials.py`).

Options (`"materials": {...}`, described in `step_catalog`):

- zones: off by default; `sam31`, `sam3` or `sam2` find the tint zones (Qwen3.5-4B names the
  materials unless `materials` lists them; the segmenter works under the GPU lock);
- zones_file and zones_mask: zones of your own instead, a JSON `{zone: material}` with a PNG
  of zone indices in UV space, or an earlier run's `.materials.json` corrected by hand with
  its `_Tint.png` (`material_zones.zones_from_file`);
- texture_prefix and ao.

The row's profile decides how many zones a creature or a part needs. The packed model, its
`_ORM.png` (and `_Tint.png`) and `.materials.json` are written into the step's folder and
become the row's asset, so `export_unreal` finds the mask and its record beside it. Leaving
the step out ships the model's maps as they are.
"""

from __future__ import annotations

from pathlib import Path

from image_to_3dlab import step_catalog
from image_to_3dlab.batch_steps import StepContext, StepResult


def preflight(rows) -> list[str]:
    from image_to_3dlab import material_zones as mz

    problems = []
    for row in rows:
        options = row.options.get("materials", {})
        problems += [f"row {row.id}: materials {problem}" for problem in step_catalog.check("materials", options)]
        own = options.get("zones_file") is not None
        if own and options.get("zones") not in (None, "off"):
            problems.append(f"row {row.id}: materials takes one source of zones: zones or zones_file, not both")
        if own and not options.get("zones_mask"):
            problems.append(f"row {row.id}: materials zones_file needs zones_mask, the PNG of its zones in UV space")
        if options.get("zones") not in (None, "off"):
            problems += [f"row {row.id}: materials zones: {problem}"
                         for problem in mz.readiness(options["zones"], ask=not options.get("materials"))]
    return list(dict.fromkeys(problems))


def run(ctx: StepContext) -> StepResult:
    if ctx.asset is None:
        return StepResult(ok=False, reason="no model to pack the materials of")
    from image_to_3dlab import material_zones as mz
    from image_to_3dlab.materials import make_game_materials

    options = ctx.options
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    zones = record = None
    try:
        if options.get("zones_file"):
            zones, record = mz.zones_from_file(ctx.asset, Path(options["zones_file"]), Path(options["zones_mask"]))
        elif options.get("zones") not in (None, "off"):
            ctx.log(f"materials: finding the tint zones ({options['zones']})")
            zones, record = mz.find_zones(ctx.asset, ctx.step_dir / "work", options["zones"],
                                          list(options["materials"]) if options.get("materials") else None)
        result = make_game_materials(ctx.asset, ctx.step_dir, ao=Path(options["ao"]) if options.get("ao") else None,
                                     texture_prefix=options.get("texture_prefix"), zones=zones, zone_record=record,
                                     profile=ctx.row.profile)
    except (ValueError, RuntimeError) as exc:   # MissingModel is a RuntimeError
        return StepResult(ok=False, reason=str(exc))
    provenance = result.glb_path.with_suffix(".provenance.json")
    written = sorted(p for p in ctx.step_dir.iterdir() if p.is_file())   # every material's maps, not the work
    return StepResult(asset=result.glb_path, sidecar=provenance, files=written,
                      record={"status": result.status, "zones": [z for z in (record or {}).get("vocabulary", [])],
                              **({"zones_from": "file"} if options.get("zones_file") else {})})
