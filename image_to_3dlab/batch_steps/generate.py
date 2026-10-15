"""`generate`: the row becomes a run manifest, run as `pipeline.py --run-manifest` runs it.

The row's images are copied into `rows/<id>/input/` first. Matting writes its cut-out
beside the image it reads, and a batch's reference folder (perhaps in another repository)
should not fill up with them; the copies also keep a row's folder complete on its own.

The manifest it writes is a normal schema-v1 manifest with an extra `batch` block (the
loader ignores fields it does not use), so one row can be re-run alone with

    python pipeline.py --run-manifest <batch>/rows/<id>/manifest.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from image_to_3dlab import manifest_run
from image_to_3dlab.batch_steps import StepContext, StepResult

MANIFEST = "manifest.json"


def preflight(rows) -> list[str]:
    problems: list[str] = []
    for backend in dict.fromkeys(row.backend for row in rows):
        generator = manifest_run.GENERATORS.get(backend)
        if generator is not None:
            problems += generator.readiness()
    return problems


def row_manifest(ctx: StepContext, images: list[Path]) -> dict:
    row = ctx.row
    return {
        "schema_version": 1,
        "_about": "Written by the batch runner for one row; see docs/batches.md.",
        **ctx.intent,
        "license_policy": ctx.license_policy,
        "input": {"path": str(images[0]), "source": "user-provided"},
        "model": {"backend": row.backend, "parameters": row.parameters},
        "output": {"directory": str(ctx.step_dir)},
        "batch": {"id": row.id, "kind": row.kind, "faces": row.faces,
                  "profile": row.profile, "steps": list(row.steps),
                  "size_cm": row.size_cm, "measure": row.measure,
                  "references": [str(image) for image in images[1:]]},
    }


def run(ctx: StepContext) -> StepResult:
    inputs = ctx.row_dir / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    images = []
    for image in ctx.row.images:
        copy = inputs / image.name
        shutil.copyfile(image, copy)
        images.append(copy)
    manifest = ctx.row_dir / MANIFEST
    manifest.write_text(json.dumps(row_manifest(ctx, images), indent=2) + "\n",
                        encoding="utf-8")
    try:
        glb, sidecar = manifest_run.run_manifest(manifest, log=ctx.log)
    except manifest_run.ManifestRunError as exc:
        return StepResult(ok=False, reason=str(exc))
    shutil.rmtree(ctx.step_dir / ".working", ignore_errors=True)
    companions = sorted(path for path in glb.parent.glob(f"{glb.stem}.*")
                        if path not in (glb, sidecar))
    return StepResult(asset=glb, sidecar=sidecar, files=[glb, sidecar, *companions, manifest],
                      record={"manifest": manifest.name})
