"""Reproducible, non-destructive S0/RIG0 checks on a concrete GLB artifact."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from image_to_3dlab.edit_glb import MAX_GLB_BYTES, EditGLB, EditGLBError
from image_to_3dlab.edit_math import (
    EditMathError,
    certify_bind,
    rebind_inverse,
    skin_matrices,
)

PROBE_VERSION = 1


def inspect_asset(data: bytes):
    """Return evidence and an optional rebuilt container, never an editable capability."""
    report = {
        "schema_version": 1, "probe_version": PROBE_VERSION,
        "source_sha256": hashlib.sha256(data).hexdigest(), "source_bytes": len(data),
        "status": "unsupported", "production_editing_ready": False,
        "pending_gates": ["canonical_document_blender_roundtrip", "seam_identity",
                          "real_gap_and_detach", "browser_preview_and_performance",
                          "authoritative_bind_recovery", "retarget_and_ue_export"],
    }
    try:
        asset = EditGLB(data)
        primitives, globals_, _ = asset.inspect_profile()
        rebuilt_bytes = asset.rebuild_container()
        rebuilt = EditGLB(rebuilt_bytes)
        rebuilt_parts, rebuilt_globals, _ = rebuilt.inspect_profile()
        exact = (asset.document == rebuilt.document and asset.binary == rebuilt.binary
                 and np.array_equal(globals_, rebuilt_globals))
        for before, after in zip(primitives, rebuilt_parts):
            exact &= np.array_equal(before["triangles"], after["triangles"])
            exact &= all(np.array_equal(value, after["attributes"][name])
                         for name, value in before["attributes"].items())
        if not exact or len(primitives) != len(rebuilt_parts):
            report.update(status="failed", error={"code": "roundtrip_mismatch", "message": "container/data changed"})
            return report, None
        report.update(status="inspection_passed", container_roundtrip={
            "json_equal": True, "binary_equal": True, "decoded_arrays_equal": True,
            "sha256": hashlib.sha256(rebuilt_bytes).hexdigest(),
            "scope": "JSON/BIN and decoded accessors; not a Blender or EditDocument roundtrip",
        })
        report["generator"] = asset.document["asset"].get("generator")
        report["extensions"] = asset.document.get("extensionsUsed", [])
        report["primitives"] = [{
            "id": p["id"], "vertices": len(p["attributes"]["POSITION"]),
            "triangles": len(p["triangles"]), "material": p["material"], "skin": p["skin"],
            "attributes": {name: {"dtype": str(a.dtype), "shape": list(a.shape),
                                  "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                           for name, a in p["attributes"].items()},
            "node_transform": p["transform"].tolist(),
        } for p in primitives]
        report["materials"] = len(asset.document.get("materials", []))
        report["images"] = len(asset.document.get("images", []))
        report["animations"] = len(asset.document.get("animations", []))
        report["skins"] = []
        for skin_index, skin in enumerate(asset.document.get("skins", [])):
            bind = globals_[skin["joints"]]
            inverse = (asset.accessor(skin["inverseBindMatrices"]) if "inverseBindMatrices" in skin
                       else np.repeat(np.eye(4)[None], len(bind), axis=0))
            result = {"skin": skin_index, "joint_count": len(bind), "pose_source": "default_node_transforms"}
            try:
                common = certify_bind(bind, inverse)
                result.update(bind_status="default_pose_consistent", common_bind=common.tolist(),
                              max_bind_error=float(np.max(np.abs(bind @ inverse - common))))
                # A reversible numerical experiment in memory, not a rig edit/export.
                new_bind = bind.copy()
                extent = max(float(np.ptp(bind[:, :3, 3], axis=0).max()), 1e-3)
                selected = min(1, len(bind) - 1)
                new_bind[selected, 0, 3] += extent * 0.01
                updated = rebind_inverse(bind, new_bind, inverse)
                errors = []
                for primitive in primitives:
                    if primitive["skin"] != skin_index:
                        continue
                    attrs = primitive["attributes"]
                    vertices = attrs["POSITION"][:32].astype(float)
                    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
                    original = skin_matrices(attrs["JOINTS_0"][:32], attrs["WEIGHTS_0"][:32], bind @ inverse)
                    # Redistributing to one joint tests common-bind invariance, rather
                    # than only the weaker fixed-weights compensation identity.
                    joints = np.full((len(vertices), 4), selected, dtype=np.int64)
                    weights = np.zeros((len(vertices), 4))
                    weights[:, 0] = 1
                    changed = skin_matrices(joints, weights, new_bind @ updated)
                    before = np.einsum("nij,nj->ni", original, homogeneous)
                    after = np.einsum("nij,nj->ni", changed, homogeneous)
                    errors.append(float(np.max(np.abs(before - after))))
                max_error = max(errors, default=0.0)
                tolerance = 1e-5 * max(extent, 1.0)
                result["reposition_then_reweight_probe"] = {
                    "joint_slot": selected, "max_rest_error": max_error,
                    "tolerance": tolerance, "passed": max_error <= tolerance,
                    "scope": "up to 32 vertices per primitive, in-memory matrices only",
                }
                if max_error > tolerance:
                    report["status"] = "failed"
            except EditMathError as exc:
                result.update(bind_status="needs_bind_adapter", error={"code": exc.code, "message": str(exc)})
            report["skins"].append(result)
        return report, rebuilt_bytes
    except EditGLBError as exc:
        report["error"] = {"code": exc.code, "message": str(exc)}
        return report, None


def run_probe(source: Path, output: Path):
    """Write evidence to a NEW directory. Existing files and source are never replaced."""
    source, output = source.resolve(), output.resolve()
    if source.stat().st_size > MAX_GLB_BYTES:
        raise ValueError("GLB exceeds inspection byte limit")
    with source.open("rb") as stream:
        data = stream.read(MAX_GLB_BYTES + 1)
    report, rebuilt = inspect_asset(data)
    report.update(source_path=str(source), created_at=datetime.now(timezone.utc).isoformat(),
                  numpy_version=np.__version__)
    # mkdir is exclusive and precedes every output write, including reports of errors.
    output.mkdir(parents=True, exist_ok=False)
    if rebuilt is not None:
        (output / "roundtrip.glb").write_bytes(rebuilt)
        sidecar = source.with_suffix(".provenance.json")
        report["provenance"] = {"source_sidecar": str(sidecar) if sidecar.is_file() else None,
                                "inherited": sidecar.is_file()}
        if sidecar.is_file():
            sidecar_data = sidecar.read_bytes()
            (output / "roundtrip.provenance.json").write_bytes(sidecar_data)
            report["provenance"]["sidecar_sha256"] = hashlib.sha256(sidecar_data).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
