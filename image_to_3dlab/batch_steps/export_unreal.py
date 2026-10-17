"""`export_unreal`: the row's asset as files Unreal Engine 5.8 imports with no touch-ups.

The same `export_unreal()` that `scripts/export_unreal.py` and the studio's Unreal export
run (feature 5, `docs/plans/2026-10-15-unreal-profile-design.md`), with the row's `kind`,
`size_cm` and `measure`. It writes into the step's folder and leaves the row's asset as it
was: the export is an extra output, listed under `unreal` in `outputs.json`. A creature
takes the clips the steps before made (`animate`) as its animations, unless the row names
its own.

Options (`"export_unreal": {...}` in the spec), described in `step_catalog`: name (default
the row's id), collision, lods, tint, forward, body, body_name, animations, material (default
the one a `materials` step before chose; none: the model's own maps). Files are relative to
the spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

from image_to_3dlab import step_catalog
from image_to_3dlab.batch_steps import StepContext, StepResult
from image_to_3dlab.blender import find_blender, missing_help

REPO = Path(__file__).resolve().parents[2]


def export_unreal(source: Path, **settings):
    """`scripts/export_unreal.export_unreal`, imported when a row runs, not at import."""
    sys.path.insert(0, str(REPO / "scripts"))
    from export_unreal import export_unreal as export

    return export(source, **settings)


def read_grip(path: Path) -> dict:
    sys.path.insert(0, str(REPO / "scripts"))
    from export_unreal import read_grip as read

    return read(path)


def preflight(rows) -> list[str]:
    problems = []
    for row in rows:
        # the files it names, now that it will run (the spec's check left them out)
        problems += [f"row {row.id}: export_unreal {problem}" for problem in
                     step_catalog.check("export_unreal", row.options.get("export_unreal", {}))]
        if row.size_cm is None:
            problems.append(f"row {row.id}: export_unreal needs size_cm, the real size in "
                            "centimetres: a generated model has none of its own")
        if row.kind == "equipment" and not row.options.get("export_unreal", {}).get("body"):
            problems.append(f"row {row.id}: equipment needs the rigged body it belongs to: "
                            '"export_unreal": {"body": "body.glb"}')
    if rows and find_blender() is None:
        problems.append(missing_help())
    return problems


def run(ctx: StepContext) -> StepResult:
    if ctx.asset is None:
        return StepResult(ok=False, reason="no asset to export")
    options = ctx.options
    paths = {key: [Path(item) for item in options[key]] if options.get(key) is not None else None
             for key in ("lods", "animations")}
    if paths["animations"] is None and ctx.row.kind == "creature" and ctx.clips:
        paths["animations"] = list(ctx.clips)   # the clips the steps before made
    forward = options.get("forward")
    try:
        result = export_unreal(
            ctx.asset, kind=ctx.row.kind, size_cm=ctx.row.size_cm,
            measure=ctx.row.measure or "height", name=options.get("name") or ctx.row.id,
            out_dir=ctx.step_dir, lods=paths["lods"], collision=options.get("collision"),
            body=Path(options["body"]) if options.get("body") else None,
            animations=paths["animations"],
            tint=False if options.get("tint") is False else Path(options["tint"]) if options.get("tint") else None,
            forward=tuple(float(value) for value in forward) if forward else None,
            body_name=options.get("body_name"), material=options.get("material") or ctx.material, log=ctx.log,
            **({"grip": read_grip(Path(options["grip"]))} if options.get("grip") else {}))
    except (ValueError, RuntimeError) as exc:
        return StepResult(ok=False, reason=str(exc))
    if not result["ok"]:
        return StepResult(ok=False, reason="; ".join(result["problems"]))
    written = [Path(result[key]) for key in ("glb", "sidecar", "provenance", "tint") if result.get(key)]
    record_file = Path(result["glb"]).with_suffix("").with_suffix(".materials.json")
    written += [record_file] if record_file.is_file() else []
    return StepResult(files=written, record={
        "glb": Path(result["glb"]).name, "sidecar": Path(result["sidecar"]).name})
