"""`checks`: the automatic checks of feature 7 on the row's asset, so it reaches the
review queue with its report and can be approved there.

The same checks `scripts/check_asset.py` and the studio's review sheet run
(`checks.registry.run_checks`), with the profile of the row's kind (`equipment` reads the
`equipment_part` profile), the row's picture as the silhouette reference and its
`size_cm` as the wanted height when that is what it measures. The report is written as
`<id>.checks.json` in the step's folder; the batch hands it to the review queue with the
asset. A failed check does not stop the row: the reviewer decides, unless the row says
`"checks": {"stop_on_fail": true}`.

Options, described in `step_catalog`: profile, set (thresholds, the shape of `--set`:
`{"budget.faces.max": 20000}`), reference, height, width, depth, stop_on_fail.
"""

from __future__ import annotations

import json
from typing import Any

from image_to_3dlab.batch_steps import StepContext, StepResult
from image_to_3dlab.checks import Asset, run_checks
from image_to_3dlab.checks.registry import load_profile, parse_overrides

PROFILES = {"prop": "prop", "creature": "creature", "equipment": "equipment_part"}


def profile_of(row) -> str:
    return row.options.get("checks", {}).get("profile") or PROFILES.get(row.kind, "prop")


def overrides_of(options: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`{"budget.faces.max": 20000}` as the checks read it: `{"budget.faces": {"max": 20000}}`."""
    return parse_overrides([f"{name}={json.dumps(value)}" for name, value in
                            (options.get("set") or {}).items()])


def preflight(rows) -> list[str]:
    problems = []
    for row in rows:
        try:
            load_profile(profile_of(row), overrides_of(row.options.get("checks", {})))
        except (ValueError, TypeError) as exc:
            problems.append(f"row {row.id}: checks: {exc}")
    return problems


def _reference(ctx: StepContext):
    """The row's picture, from its staged copy when there is one, or the one a step drew."""
    if ctx.row.images:
        copy = ctx.row_dir / "input" / ctx.row.images[0].name
        return copy if copy.is_file() else ctx.row.images[0]
    return ctx.picture or ctx.row.concept


def run(ctx: StepContext) -> StepResult:
    if ctx.asset is None:
        return StepResult(ok=False, reason="no asset to check")
    options = ctx.options
    numbers = {axis: options[axis] for axis in ("height", "width", "depth")
               if options.get(axis) is not None}
    if "height" not in numbers and ctx.row.size_cm is not None and (ctx.row.measure or "height") == "height":
        numbers["height"] = ctx.row.size_cm / 100
    asset = Asset(glb=ctx.asset, key=ctx.row.id, provenance=ctx.sidecar,
                  reference=_reference(ctx) if options.get("reference", True) else None,
                  reference_numbers=numbers or None)
    try:
        report = run_checks(asset, profile=profile_of(ctx.row), overrides=overrides_of(options))
    except (ValueError, TypeError) as exc:
        return StepResult(ok=False, reason=str(exc))
    report["asset_key"] = ctx.row.id
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.step_dir / f"{ctx.row.id}.checks.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [found["check"] for found in report.get("results", []) if found.get("status") == "fail"]
    ctx.log(f"checks: {report['status']}" + (f" ({', '.join(failed)})" if failed else ""))
    if report["status"] == "fail" and options.get("stop_on_fail"):
        return StepResult(ok=False, reason=f"checks fail: {', '.join(failed) or 'see the report'}")
    return StepResult(files=[path], record={"report": path.name, "status": report["status"]})
