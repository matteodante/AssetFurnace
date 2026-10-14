"""Canonical mesh buffers and a copy-on-write GLB assembler for guided editing.

Identities are scoped to the source and topology revision. Node indices remain
stable; editing one instance clones its mesh instead of changing shared geometry.
Untouched glTF data (including images, skin and animation tracks) stays byte exact.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from image_to_3dlab.edit_glb import EditGLB, EditGLBError
from image_to_3dlab.glb_turn import build_glb

SCHEMA_VERSION = 1


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _array_hash(array):
    return _hash(_json({"shape": array.shape, "dtype": array.dtype.str}) + array.tobytes())


def _validate_geometry(attributes, triangles):
    if "POSITION" not in attributes:
        raise ValueError("POSITION is required")
    positions = np.asarray(attributes["POSITION"])
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("positions must be a nonempty (N,3) array")
    for name, values in attributes.items():
        array = np.asarray(values)
        if array.ndim != 2 or len(array) != len(positions) or array.dtype.kind not in "iuf":
            raise ValueError(f"invalid attribute: {name}")
        if array.shape[1] not in (2, 3, 4) or not np.isfinite(array).all():
            raise ValueError(f"invalid attribute shape/values: {name}")
        if name == "JOINTS_0" and (array.shape[1] != 4 or array.dtype.kind not in "iu"
                                   or (array < 0).any() or (array > 65535).any()):
            raise ValueError("invalid joint indices")
        if name == "WEIGHTS_0" and (array.shape[1] != 4 or (array < 0).any()
                                    or not np.allclose(array.sum(1), 1, atol=1e-5, rtol=0)):
            raise ValueError("invalid skin weights")
    triangles = np.asarray(triangles)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or triangles.dtype.kind not in "iu" or (triangles < 0).any()
            or (triangles >= len(positions)).any()):
        raise ValueError("invalid triangle indices")


class EditDocument:
    def __init__(self, source: bytes):
        self.source = EditGLB(source)
        parts, self.node_globals, self.parents = self.source.inspect_profile()
        for part in parts:
            part["triangles"] = part["triangles"].astype(np.uint32)
        self.parts = {part["id"]: part for part in parts}
        self._original = copy.deepcopy(self.parts)
        # Seams are NOT guessed from positions. The explicit source-vertex map is
        # lossless; seam reconstruction is a separate capability still under test.
        self.vertex_identity = "source_render_vertex"

    @property
    def topology_revision(self):
        return _hash(_json({key: {"vertices": len(p["attributes"]["POSITION"]),
                                  "triangles": _array_hash(p["triangles"])}
                            for key, p in self.parts.items()}))

    @property
    def document_sha256(self):
        return _hash(_json({"source": self.source.source_sha256, "topology": self.topology_revision,
                            "attributes": {key: {name: _array_hash(value)
                                                 for name, value in p["attributes"].items()}
                                           for key, p in self.parts.items()}}))

    def replace_geometry(self, part_id, attributes, triangles):
        part = self.parts[part_id]
        if set(attributes) != set(part["attributes"]):
            raise ValueError("an edit must preserve every attribute semantic")
        _validate_geometry(attributes, triangles)
        part["attributes"] = {key: np.array(value, copy=True) for key, value in attributes.items()}
        part["triangles"] = np.array(triangles, dtype=np.uint32, copy=True)

    def set_positions(self, part_id, indices, positions, *, expected_topology):
        if expected_topology != self.topology_revision:
            raise ValueError("topology revision conflict")
        part = self.parts[part_id]
        indices = np.asarray(indices)
        values = np.asarray(positions, dtype=np.float64)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu"
                or (indices < 0).any() or (indices >= len(part["attributes"]["POSITION"])).any()
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("invalid or repeated vertex indices")
        if values.shape != (len(indices), 3) or not np.isfinite(values).all():
            raise ValueError("invalid positions")
        if np.abs(values).max(initial=0) > np.finfo(np.float32).max:
            raise ValueError("positions cannot be represented as float32")
        part["attributes"]["POSITION"][indices] = values

    def manifest(self):
        return {
            "schema_version": SCHEMA_VERSION, "source_sha256": self.source.source_sha256,
            "topology_revision": self.topology_revision, "document_sha256": self.document_sha256,
            "vertex_identity": self.vertex_identity, "seam_editing_certified": False,
            "parts": [{"id": key, "node": p["node"], "mesh": p["mesh"], "primitive": p["primitive"],
                       "material": p["material"], "skin": p["skin"], "transform": p["transform"].tolist(),
                       "vertices": len(p["attributes"]["POSITION"]), "faces": len(p["triangles"]),
                       "attributes": {name: {"shape": list(a.shape), "dtype": a.dtype.str,
                                              "sha256": _array_hash(a)}
                                      for name, a in p["attributes"].items()}}
                      for key, p in self.parts.items()],
        }

    def write_snapshot(self, directory: Path):
        """New directory only; hashes cover source plus every canonical buffer file."""
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "source.glb").write_bytes(self.source.source)
        manifest = self.manifest()
        for index, (key, part) in enumerate(self.parts.items()):
            filename = f"primitive-{index}.npz"
            np.savez(directory / filename, triangles=part["triangles"], **part["attributes"])
            manifest["parts"][index]["buffer_file"] = filename
            manifest["parts"][index]["buffer_sha256"] = _hash((directory / filename).read_bytes())
        (directory / "document.json").write_bytes(_json(manifest))
        return manifest

    @classmethod
    def read_snapshot(cls, directory: Path):
        manifest = json.loads((directory / "document.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported EditDocument schema")
        document = cls((directory / "source.glb").read_bytes())
        if document.source.source_sha256 != manifest.get("source_sha256"):
            raise ValueError("source hash mismatch")
        if len(manifest["parts"]) != len(document.parts):
            raise ValueError("snapshot part count mismatch")
        for index, (key, part) in enumerate(document.parts.items()):
            item = manifest["parts"][index]
            if item["id"] != key or item["buffer_file"] != f"primitive-{index}.npz":
                raise ValueError("invalid buffer identity/path")
            data = (directory / item["buffer_file"]).read_bytes()
            if _hash(data) != item["buffer_sha256"]:
                raise ValueError("canonical buffer hash mismatch")
            with np.load(directory / item["buffer_file"], allow_pickle=False) as arrays:
                attributes = {name: arrays[name] for name in arrays.files if name != "triangles"}
                document.replace_geometry(key, attributes, arrays["triangles"])
        if document.topology_revision != manifest["topology_revision"]:
            raise ValueError("topology hash mismatch")
        if document.document_sha256 != manifest["document_sha256"]:
            raise ValueError("document hash mismatch")
        return document

    def export_glb(self):
        """Append replacement accessors and isolate edited instances; no glTF reimport.

        Callers must provide normals/tangents matching the changed shape. This method
        preserves attributes; it deliberately does not invent a surface refinement.
        """
        document = copy.deepcopy(self.source.document)
        binary = bytearray(self.source.binary[:self.source.buffer_size])
        mesh_clones = {}
        changed_any = False

        def append(array, original=None, *, indices=False, semantic=None):
            if indices:
                output = np.asarray(array, dtype="<u4").reshape(-1)
                component, kind = 5125, "SCALAR"
            else:
                is_joint = semantic == "JOINTS_0"
                if is_joint and (np.asarray(array).max(initial=0) > 65535 or np.asarray(array).min(initial=0) < 0):
                    raise ValueError("joint index exceeds unsigned short")
                output = np.asarray(array, dtype="<u2" if is_joint else "<f4")
                component = 5123 if is_joint else 5126
                kind = f"VEC{output.shape[1]}"
            if not np.isfinite(output).all():
                raise ValueError("attribute cannot be represented in GLB")
            binary.extend(b"\0" * (-len(binary) % 4))
            view = len(document["bufferViews"])
            document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": output.nbytes})
            binary.extend(output.tobytes())
            accessor = copy.deepcopy(original or {})
            for field in ("byteOffset", "sparse", "normalized", "min", "max"):
                accessor.pop(field, None)
            accessor.update(bufferView=view, componentType=component, count=len(output), type=kind)
            if semantic == "POSITION":
                accessor.update(min=output.min(axis=0).tolist(), max=output.max(axis=0).tolist())
            accessor_index = len(document["accessors"])
            document["accessors"].append(accessor)
            return accessor_index

        for key, part in self.parts.items():
            original = self._original[key]
            _validate_geometry(part["attributes"], part["triangles"])
            changed_attributes = [name for name, array in part["attributes"].items()
                                  if not np.array_equal(array, original["attributes"][name])]
            changed_indices = not np.array_equal(part["triangles"], original["triangles"])
            if not changed_attributes and not changed_indices:
                continue
            changed_any = True
            node = part["node"]
            if node not in mesh_clones:
                clone = copy.deepcopy(document["meshes"][part["mesh"]])
                mesh_clones[node] = clone
                document["nodes"][node]["mesh"] = len(document["meshes"])
                document["meshes"].append(clone)
            primitive = mesh_clones[node]["primitives"][part["primitive"]]
            for name in changed_attributes:
                old_accessor = self.source.accessors[primitive["attributes"][name]]
                primitive["attributes"][name] = append(part["attributes"][name], old_accessor, semantic=name)
            if changed_indices:
                primitive["indices"] = append(part["triangles"], indices=True)
        if not changed_any:
            return self.source.source
        document["buffers"][0]["byteLength"] = len(binary)
        result = build_glb(document, bytes(binary))
        try:
            EditGLB(result).inspect_profile()
        except EditGLBError as exc:
            raise ValueError(f"assembled GLB failed validation: {exc}") from exc
        return result
