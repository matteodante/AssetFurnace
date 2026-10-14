"""Round-trip canonical editing buffers through Blender without reimporting the GLB.

Run with Blender --background --factory-startup --python-exit-code 1 --python
scripts/blender_edit_bridge.py -- --input snapshot --output new-directory.
This validates the no-op mesh boundary, not armature/animation reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from image_to_3dlab.edit_document import EditDocument


def mesh_roundtrip(part):
    """Build explicitly from canonical arrays and read Blender's actual arrays back."""
    import bpy

    attrs, triangles = part["attributes"], part["triangles"]
    mesh = bpy.data.meshes.new("EditBoundary")
    obj = bpy.data.objects.new("EditBoundary", mesh)
    bpy.context.scene.collection.objects.link(obj)
    try:
        mesh.from_pydata(attrs["POSITION"].tolist(), [], triangles.tolist())
        mesh.update()
        identity = mesh.attributes.new("edit_vertex_id", "INT", "POINT")
        identity.data.foreach_set("value", np.arange(len(mesh.vertices), dtype=np.int32))
        face_identity = mesh.attributes.new("edit_face_id", "INT", "FACE")
        face_identity.data.foreach_set("value", np.arange(len(mesh.polygons), dtype=np.int32))
        loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        uv_errors = {}
        for name, values in attrs.items():
            if not name.startswith("TEXCOORD_"):
                continue
            layer = mesh.uv_layers.new(name=name)
            expected = values[loop_vertices].astype(np.float32)
            layer.data.foreach_set("uv", expected.reshape(-1))
            actual = np.empty(expected.size, dtype=np.float32)
            layer.data.foreach_get("uv", actual)
            uv_errors[name] = float(np.max(np.abs(actual.reshape(-1, 2) - expected)))
        normal_error = None
        if "NORMAL" in attrs:
            mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
            mesh.normals_split_custom_set_from_vertices(attrs["NORMAL"].tolist())
            actual = np.empty(len(mesh.loops) * 3, dtype=np.float32)
            mesh.corner_normals.foreach_get("vector", actual)
            expected = attrs["NORMAL"][loop_vertices].astype(float)
            lengths = np.linalg.norm(expected, axis=1, keepdims=True)
            if (lengths <= 1e-12).any():
                raise ValueError("zero normals require an explicit repair, not a no-op")
            normal_error = float(np.max(np.abs(actual.reshape(-1, 3) - expected / lengths)))
        weight_error = None
        if "WEIGHTS_0" in attrs:
            joints, weights = attrs["JOINTS_0"], attrs["WEIGHTS_0"]
            groups = {int(j): obj.vertex_groups.new(name=f"joint_{int(j)}") for j in np.unique(joints)}
            for vertex, (indices, values) in enumerate(zip(joints, weights)):
                combined = {}
                for joint, weight in zip(indices, values):
                    combined[int(joint)] = combined.get(int(joint), 0.0) + float(weight)
                for joint, weight in combined.items():
                    if weight:
                        groups[joint].add([vertex], weight, "REPLACE")
            group_to_joint = {group.index: joint for joint, group in groups.items()}
            weight_error = 0.0
            for vertex, (indices, values) in enumerate(zip(joints, weights)):
                expected = {}
                for joint, weight in zip(indices, values):
                    expected[int(joint)] = expected.get(int(joint), 0.0) + float(weight)
                actual = {group_to_joint[g.group]: g.weight for g in mesh.vertices[vertex].groups}
                weight_error = max(weight_error, max(abs(actual.get(joint, 0) - weight)
                                                     for joint, weight in expected.items()))
        positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", positions)
        ids = np.empty(len(mesh.vertices), dtype=np.int32)
        identity.data.foreach_get("value", ids)
        face_ids = np.empty(len(mesh.polygons), dtype=np.int32)
        face_identity.data.foreach_get("value", face_ids)
        if not np.array_equal(ids, np.arange(len(ids))) or not np.array_equal(face_ids, np.arange(len(face_ids))):
            raise ValueError("Blender changed canonical identity during no-op")
        positions = positions.reshape(-1, 3)
        read_triangles = loop_vertices.reshape(-1, 3).astype(np.uint32)
        positions_equal = np.array_equal(positions, attrs["POSITION"])
        triangles_equal = np.array_equal(read_triangles, triangles)
        passed = positions_equal and triangles_equal and all(v == 0 for v in uv_errors.values())
        passed &= weight_error is None or weight_error < 1e-6
        # Blender encodes custom normals internally; retain original normal bytes
        # for a no-op, while measuring the actual reconstructed corner normals.
        passed &= normal_error is None or normal_error < 5e-4
        result = {"part_id": part["id"], "passed": bool(passed), "positions_equal": positions_equal,
                  "triangles_equal": triangles_equal, "uv_max_error": uv_errors,
                  "weights_max_error": weight_error, "normals_max_error": normal_error}
        if not passed:
            raise ValueError(f"canonical no-op failed: {result}")
        output_attributes = {name: value.copy() for name, value in attrs.items()}
        output_attributes["POSITION"] = positions
        return output_attributes, read_triangles, result
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def main(argv=None):
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = EditDocument.read_snapshot(args.input)
    before = document.document_sha256
    parts = []
    for key, part in document.parts.items():
        attributes, triangles, report = mesh_roundtrip(part)
        document.replace_geometry(key, attributes, triangles)
        parts.append(report)
    if document.document_sha256 != before:
        raise ValueError("canonical document changed during Blender no-op")
    document.write_snapshot(args.output)
    exported = document.export_glb()
    (args.output / "roundtrip.glb").write_bytes(exported)
    report = {"blender_version": bpy.app.version_string, "source_sha256": document.source.source_sha256,
              "document_sha256": before, "parts": parts, "passed": True,
              "byte_exact_glb": exported == document.source.source,
              "scope": "mesh/UV/custom-normal/vertex-group boundary; skin and clips preserved by assembler",
              "pending": ["armature reconstruction", "topology edits", "seam reconstruction", "UE"]}
    (args.output / "bridge-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(args.output)}))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
