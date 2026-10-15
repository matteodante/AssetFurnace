"""`finish`: the studio's Finish (`scripts/retopo_repaint.py`) at the row's face budget.

Retopology to `faces`, Pixel Match when the asset has a saved camera (a Pixal3D run's
`.svviews/`), the detail bake and texture compression. **No repaint unless a row asks for
it** with `"finish": {"paint": true}`: the repaint uses Hunyuan 2.1 Paint weights (Tencent
licence, not valid in the EU, UK or South Korea) and runs on Apple Silicon only.

The finished GLB gets its own `.provenance.json`: the licence it inherits from the model
it was made from, that model's record, and the settings used. Same shape as the props
record in `viewer/props_api.py`, plus the output's hash.

Options (`"finish": {...}` in the spec): paint, photo, atlas, texture_size, metallic,
roughness, ior, voxel, skip_bake, skip_compress.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_to_3dlab.batch_steps import StepContext, StepResult
from image_to_3dlab.blender import find_blender, missing_help
from image_to_3dlab.host import APPLE, host_platform
from image_to_3dlab.provenance import LICENSES, sha256_file

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "scripts" / "retopo_repaint.py"
# The worker's own default, used when a row sets no face budget.
DEFAULT_FACES = 40000
# Option -> flag; booleans are handled one by one below.
VALUE_FLAGS = {"atlas": "--atlas", "texture_size": "--texture-size", "metallic": "--metallic",
               "roughness": "--roughness", "ior": "--ior", "voxel": "--voxel"}
OPTIONS = {"paint", "photo", "skip_bake", "skip_compress", *VALUE_FLAGS}
REPAINT_OFF_MAC = ("the repaint runs on Apple Silicon only (it uses MLX); remove "
                   '"paint": true from the finish options')
# What the repaint runs (scripts/hunyuan_mlx_xiong_generate.py), and its licence.
PAINT_ROOT = REPO / "hunyuan_mlx" / "paint"
PAINT_PYTHON = PAINT_ROOT / ".venv" / "bin" / "python"
PAINT_WEIGHTS = PAINT_ROOT / "weights" / "hunyuan3d-paintpbr-v2-1"
PAINT_PROFILE = LICENSES["hunyuan-comfyui"]
# Worst last: a finished asset is as restricted as the most restricted model in it.
CLASS_ORDER = ("commercial-clear", "commercial-conditional", "territory-restricted",
               "research-only")


def preflight(rows) -> list[str]:
    problems = []
    for row in rows:
        unknown = sorted(set(row.options.get("finish", {})) - OPTIONS)
        if unknown:
            problems.append(f"row {row.id}: unknown finish option(s) {', '.join(unknown)}; "
                            f"options are {', '.join(sorted(OPTIONS))}")
    blender = find_blender()
    if blender is None or not Path(blender).is_file():
        problems.append(missing_help())
    if any(row.options.get("finish", {}).get("paint") for row in rows):
        if host_platform() != APPLE:
            problems.append(REPAINT_OFF_MAC)
        else:
            if not PAINT_PYTHON.is_file():
                problems.append(f"the repaint needs its environment: uv sync in {PAINT_ROOT}")
            if not PAINT_WEIGHTS.is_dir():
                problems.append(f"the repaint needs its weights in {PAINT_WEIGHTS}; the Setup "
                                "page installs them after asking")
    return problems


def source_image(ctx: StepContext) -> Path | None:
    """The concept the asset came from: the row's first image, else its record's input."""
    if ctx.row.images:
        copy = ctx.row_dir / "input" / ctx.row.images[0].name
        return copy if copy.is_file() else ctx.row.images[0]
    record = _read(ctx.sidecar)
    path = (record.get("input") or {}).get("path")
    return Path(path) if path and Path(path).is_file() else None


def views_for(asset: Path) -> Path | None:
    views = asset.with_suffix(".svviews")
    return views if (views / "transforms.json").is_file() else None


def command(ctx: StepContext, blender: Path) -> list[str]:
    options = ctx.options
    image = source_image(ctx)
    command = [sys.executable, "-u", str(WORKER), str(ctx.asset),
               str(image) if image else "", str(output_path(ctx)),
               "--faces", str(ctx.row.faces or DEFAULT_FACES),
               "--steps-dir", str(ctx.step_dir / "steps"),
               "--blender", str(blender)]
    for key, flag in VALUE_FLAGS.items():
        if options.get(key) is not None:
            command += [flag, str(options[key])]
    if not options.get("paint", False):
        command.append("--skip-paint")
    views = views_for(ctx.asset)
    if views is not None and options.get("photo", True):
        command += ["--views", str(views)]
    if options.get("skip_bake"):
        command.append("--skip-bake")
    if options.get("skip_compress"):
        command.append("--skip-compress")
    return command


def output_path(ctx: StepContext) -> Path:
    return ctx.step_dir / f"{ctx.row.id}.glb"


def run(ctx: StepContext) -> StepResult:
    if ctx.asset is None:
        return StepResult(ok=False, reason="no asset to finish")
    if source_image(ctx) is None:
        return StepResult(ok=False, reason="finish needs the source image: give the row "
                                           "an image, or an asset with its provenance record")
    blender = find_blender()
    if blender is None:
        return StepResult(ok=False, reason=missing_help())
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    log_path = ctx.step_dir / "finish.log"
    ctx.log(f"finish: {ctx.asset.name} at {ctx.row.faces or DEFAULT_FACES:,} faces "
            f"(log: {log_path})")
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command(ctx, Path(blender)), stdout=handle,
                                 stderr=subprocess.STDOUT, text=True, check=False)
    output = output_path(ctx)
    if process.returncode != 0 or not output.is_file():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-12:])
        return StepResult(ok=False, reason=f"retopo_repaint exited with code "
                                           f"{process.returncode}:\n{tail}")
    worker_record = output.with_suffix(".retopo-repaint.json")
    sidecar = output.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(
        provenance_record(ctx, output, _read(worker_record) or None), indent=2) + "\n",
        encoding="utf-8")
    steps = ctx.step_dir / "steps"
    diagnostics = sorted(path for path in steps.rglob("*") if path.is_file()) if steps.is_dir() else []
    files = [output, sidecar, *([worker_record] if worker_record.is_file() else []),
             log_path, *diagnostics]
    return StepResult(asset=output, sidecar=sidecar, files=files)


def provenance_record(ctx: StepContext, output: Path,
                      worker_record: dict[str, Any] | None) -> dict[str, Any]:
    """The finished asset's record: what went in, what came out, the licence it inherits.

    An asset without a record says so (`source: unknown`, no licence) rather than
    implying it is clear to ship.
    """
    source = _read(ctx.sidecar) or None
    classification = ((source or {}).get("output") or {}).get("classification")
    painted = bool(ctx.options.get("paint", False))
    components = []
    if painted:
        components.append({
            "component": "Hunyuan3D-2.1 Paint (hunyuan_mlx/paint)", "purpose": "repaint",
            "license": PAINT_PROFILE.license_name, "license_url": PAINT_PROFILE.license_url,
            "commercial_status": PAINT_PROFILE.classification,
            "conditions": list(PAINT_PROFILE.conditions),
        })
        classification = _worst(classification, PAINT_PROFILE.classification)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "batch finish: scripts/retopo_repaint.py",
        "input": {"path": str(ctx.asset), "sha256": sha256_file(ctx.asset),
                  "source": "generated" if source else "unknown"},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "license": (source or {}).get("license"),
        "classification": classification,
        "components": components,
        "parameters": {"faces": ctx.row.faces or DEFAULT_FACES,
                       "paint": bool(ctx.options.get("paint", False)),
                       "photo": views_for(ctx.asset) is not None
                       and bool(ctx.options.get("photo", True)),
                       **{key: value for key, value in ctx.options.items()
                          if key not in ("paint", "photo")}},
        "worker_record": worker_record,
        "source_record": source,
    }


def _worst(first: str | None, second: str) -> str:
    """The more restricted of two classifications; an unknown one counts as the worst."""
    def rank(name: str | None) -> int:
        return CLASS_ORDER.index(name) if name in CLASS_ORDER else len(CLASS_ORDER)

    return first if first is not None and rank(first) >= rank(second) else second


def _read(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
