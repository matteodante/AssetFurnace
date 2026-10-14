"""Bounded, read-only GLB inspection for the editor feasibility gate.

This is deliberately not EditDocument or a general glTF validator. Unsupported
features fail explicitly; nothing fetches external resources or rewrites an asset.
Accessor copies have stable source indices, never welded by coincident position.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import deque

import numpy as np

from image_to_3dlab.glb_turn import BIN_CHUNK, GLB_MAGIC, JSON_CHUNK, build_glb

MAX_GLB_BYTES = 512 * 1024 * 1024
MAX_ACCESSOR_BYTES = 128 * 1024 * 1024
_DTYPES = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
# These extensions only carry material/embedded texture information. No decoding
# or material conversion is claimed; the source JSON and image bytes are retained.
_PRESERVED_EXTENSIONS = {
    "KHR_materials_unlit", "KHR_materials_ior", "KHR_materials_specular",
    "KHR_materials_emissive_strength", "KHR_texture_transform", "EXT_texture_webp",
}


class EditGLBError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _fail(message, code="invalid_glb"):
    raise EditGLBError(code, message)


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _object(value, label):
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _objects(document, name):
    values = document.get(name, [])
    if not isinstance(values, list):
        _fail(f"{name} must be an array")
    return [_object(value, name) for value in values]


def _ref(values, index, label):
    _integer(index, label)
    if index >= len(values):
        _fail(f"{label} reference is out of range")
    return values[index]


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _extension_names(value):
    """Do not overlook compressed data just because extensionsUsed omitted it."""
    names = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if "extensions" in item:
                names.update(_object(item["extensions"], "extensions"))
            # extras are opaque application metadata, not glTF extension objects.
            pending.extend(v for k, v in item.items() if k != "extras")
        elif isinstance(item, list):
            pending.extend(item)
    return names


def _numbers(value, size, label):
    if not isinstance(value, list) or len(value) != size:
        _fail(f"{label} must contain {size} numbers")
    if any(type(v) not in (int, float) for v in value):
        _fail(f"{label} must be numeric")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        _fail(f"{label} must be finite")
    return array


def node_matrix(node):
    """glTF column-major matrix or T*R*S, without viewer centering/scaling."""
    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            _fail("node mixes matrix and TRS")
        matrix = _numbers(node["matrix"], 16, "node matrix").reshape(4, 4).T
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            _fail("node matrix is not affine")
        return matrix
    translation = _numbers(node.get("translation", [0, 0, 0]), 3, "translation")
    scale = _numbers(node.get("scale", [1, 1, 1]), 3, "scale")
    q = _numbers(node.get("rotation", [0, 0, 0, 1]), 4, "rotation")
    if not np.isclose(q @ q, 1, atol=1e-5, rtol=0):
        _fail("node quaternion is not normalized")
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


class EditGLB:
    """Validated container plus bounded accessors; inspect_profile runs semantic gates."""

    def __init__(self, data: bytes):
        if len(data) > MAX_GLB_BYTES:
            _fail("GLB exceeds inspection byte limit", "resource_limit")
        if len(data) < 20:
            _fail("truncated GLB header")
        magic, version, length = struct.unpack_from("<III", data)
        if magic != GLB_MAGIC or version != 2 or length != len(data):
            _fail("GLB magic/version/declared length mismatch")
        chunks, offset = [], 12
        while offset < len(data):
            if len(data) - offset < 8:
                _fail("truncated chunk header")
            size, kind = struct.unpack_from("<II", data, offset)
            if size % 4 or size > len(data) - offset - 8:
                _fail("invalid chunk length/alignment")
            chunks.append((kind, data[offset+8:offset+8+size]))
            offset += 8 + size
        if [kind for kind, _ in chunks] not in ([JSON_CHUNK], [JSON_CHUNK, BIN_CHUNK]):
            _fail("only one JSON and optional BIN chunk are supported", "unsupported_chunks")
        try:
            document = json.loads(chunks[0][1], object_pairs_hook=_json_object,
                                  parse_constant=lambda value: _fail(f"nonfinite JSON: {value}"))
        except (ValueError, UnicodeError) as exc:
            _fail(f"invalid GLB JSON: {exc}")
        self.document = _object(document, "glTF")
        asset = _object(document.get("asset"), "asset")
        if asset.get("version") != "2.0" or asset.get("minVersion", "2.0") != "2.0":
            _fail("only glTF 2.0 is supported", "unsupported_version")
        self.source = data
        self.source_sha256 = hashlib.sha256(data).hexdigest()
        self.binary = chunks[1][1] if len(chunks) == 2 else b""
        self.buffers = _objects(document, "buffers")
        if len(self.buffers) != 1 or "uri" in self.buffers[0]:
            _fail("one embedded buffer is required; external resources are not read",
                  "unsupported_buffers")
        size = _integer(self.buffers[0].get("byteLength"), "buffer length", 1)
        if not size <= len(self.binary) <= size + 3:
            _fail("BIN length disagrees with buffer including padding")
        self.buffer_size = size
        self.views = _objects(document, "bufferViews")
        self.accessors = _objects(document, "accessors")
        for view in self.views:
            if _integer(view.get("buffer"), "buffer reference") != 0:
                _fail("bufferView points outside embedded buffer")
            start = _integer(view.get("byteOffset", 0), "bufferView offset")
            length = _integer(view.get("byteLength"), "bufferView length", 1)
            if start + length > size:
                _fail("bufferView exceeds declared buffer")
            if "byteStride" in view:
                stride = _integer(view["byteStride"], "byteStride", 4)
                if stride > 252 or stride % 4:
                    _fail("byteStride must be a multiple of four <= 252")

    def accessor(self, index, *, normalized=True):
        """Independent array in source order. MAT4 rows are ordinary 4x4 matrices."""
        accessor = _ref(self.accessors, index, "accessor")
        if "sparse" in accessor:
            _fail("sparse accessors need a verified adapter", "unsupported_sparse")
        kind = accessor.get("type")
        component = accessor.get("componentType")
        if not isinstance(kind, str) or type(component) is not int or kind not in _WIDTHS or component not in _DTYPES:
            _fail("unsupported accessor component/type", "unsupported_accessor")
        if kind == "MAT4" and component != 5126:
            _fail("only float MAT4 is supported", "unsupported_accessor")
        dtype, width = np.dtype(_DTYPES[component]), _WIDTHS[kind]
        count = _integer(accessor.get("count"), "accessor count", 1)
        # Budget for the decoded representation too, including normalized float64.
        if count * width * 8 > MAX_ACCESSOR_BYTES:
            _fail("accessor exceeds decoded byte limit", "resource_limit")
        is_normalized = accessor.get("normalized", False)
        if type(is_normalized) is not bool or (is_normalized and component not in (5120, 5121, 5122, 5123)):
            _fail("invalid normalized accessor")
        if "bufferView" not in accessor:
            _fail("accessor without a bufferView needs a sparse/zero adapter", "unsupported_accessor")
        view = _ref(self.views, accessor["bufferView"], "bufferView")
        relative = _integer(accessor.get("byteOffset", 0), "accessor offset")
        start = view.get("byteOffset", 0) + relative
        element_bytes = dtype.itemsize * width
        stride = view.get("byteStride", element_bytes)
        if relative % dtype.itemsize or start % dtype.itemsize or stride < element_bytes:
            _fail("accessor alignment/stride mismatch")
        if relative + (count-1)*stride + element_bytes > view["byteLength"]:
            _fail("accessor exceeds its bufferView")
        rows = np.ndarray((count, width), dtype=dtype, buffer=self.binary,
                          offset=start, strides=(stride, dtype.itemsize)).copy()
        if not np.isfinite(rows).all():
            _fail("accessor contains nonfinite values")
        if normalized and is_normalized:
            rows = rows.astype(np.float64) / np.iinfo(dtype).max
            if dtype.kind == "i":
                rows = np.maximum(rows, -1)
        if kind == "MAT4":
            return rows.reshape(count, 4, 4).transpose(0, 2, 1).copy()
        return rows[:, 0] if kind == "SCALAR" else rows

    def node_globals(self):
        """Iterative tree validation also includes non-joint ancestors."""
        nodes = _objects(self.document, "nodes")
        if len(nodes) > 100_000:
            _fail("too many nodes for inspection", "resource_limit")
        parents = np.full(len(nodes), -1, dtype=np.int64)
        children = []
        for index, node in enumerate(nodes):
            branch = node.get("children", [])
            if not isinstance(branch, list):
                _fail("children must be an array")
            for child in branch:
                _ref(nodes, child, "child node")
                if parents[child] != -1:
                    _fail("node has multiple parents or repeated child")
                parents[child] = index
            children.append(branch)
        local = [node_matrix(node) for node in nodes]
        global_ = np.empty((len(nodes), 4, 4))
        queue = deque(np.flatnonzero(parents < 0).tolist())
        visited = 0
        while queue:
            index = queue.popleft()
            parent = parents[index]
            global_[index] = local[index] if parent < 0 else global_[parent] @ local[index]
            queue.extend(children[index])
            visited += 1
        if visited != len(nodes):
            _fail("node hierarchy has a cycle")
        if not np.isfinite(global_).all():
            _fail("nonfinite global node transform")
        return global_, parents

    def inspect_profile(self):
        """Validate the narrow S0 input profile, returning per-instance primitive data.

        Passing this gate does not enable production editing or certify animation,
        material rendering, seam identity, authoritative bind recovery or UE export.
        """
        used = self.document.get("extensionsUsed", [])
        required = self.document.get("extensionsRequired", [])
        if not isinstance(used, list) or not isinstance(required, list) or any(
                not isinstance(x, str) for x in [*used, *required]):
            _fail("invalid extension declaration")
        unknown = (set(used + required) | _extension_names(self.document)) - _PRESERVED_EXTENSIONS
        if unknown:
            _fail(f"extensions need a verified adapter: {', '.join(sorted(unknown))}", "unsupported_extension")
        for image in _objects(self.document, "images"):
            if "uri" in image:
                _fail("only embedded images are inspected", "unsupported_images")
            _ref(self.views, image.get("bufferView"), "image bufferView")
        # Scan all accessors, including animation data, before calling a round-trip safe.
        for index in range(len(self.accessors)):
            self.accessor(index)
        globals_, parents = self.node_globals()
        nodes = _objects(self.document, "nodes")
        meshes = _objects(self.document, "meshes")
        skins = _objects(self.document, "skins")
        materials = _objects(self.document, "materials")
        if len(skins) > 1:
            _fail("multiple independent skins need an adapter", "unsupported_skins")
        for skin in skins:
            joints = skin.get("joints")
            if not isinstance(joints, list) or not joints:
                _fail("skin joints must be a nonempty array")
            for joint in joints:
                _ref(nodes, joint, "skin joint")
            if len(set(joints)) != len(joints):
                _fail("duplicate skin joint")
            if "inverseBindMatrices" in skin:
                ibm_index = skin["inverseBindMatrices"]
                ibm_meta = _ref(self.accessors, ibm_index, "inverse bind accessor")
                if ibm_meta.get("type") != "MAT4" or ibm_meta.get("componentType") != 5126:
                    _fail("inverse bind accessor must be float MAT4")
                if self.accessor(ibm_index).shape != (len(joints), 4, 4):
                    _fail("inverse bind count differs from joints")
        primitives = []
        decoded_bytes = 0
        for node_index, node in enumerate(nodes):
            if "mesh" not in node:
                continue
            mesh = _ref(meshes, node["mesh"], "mesh")
            parts = _objects(mesh, "primitives")
            if not parts:
                _fail("mesh has no primitives")
            skin = _ref(skins, node["skin"], "skin") if "skin" in node else None
            for primitive_index, part in enumerate(parts):
                if part.get("mode", 4) != 4 or part.get("targets"):
                    _fail("only triangles without morph targets are inspected", "unsupported_primitive")
                attributes = _object(part.get("attributes"), "primitive attributes")
                if "POSITION" not in attributes:
                    _fail("primitive has no POSITION")
                if any(name.startswith(("JOINTS_", "WEIGHTS_")) and name not in ("JOINTS_0", "WEIGHTS_0")
                       for name in attributes):
                    _fail("only four skin influences are supported", "unsupported_influences")
                arrays = {name: self.accessor(index) for name, index in attributes.items()}
                positions = arrays["POSITION"]
                if positions.ndim != 2 or positions.shape[1] != 3 or positions.dtype != np.dtype("<f4"):
                    _fail("POSITION must be float VEC3")
                for name, array in arrays.items():
                    if len(array) != len(positions):
                        _fail(f"{name} count differs from POSITION")
                    decoded_bytes += array.nbytes
                    if decoded_bytes > MAX_GLB_BYTES:
                        _fail("instances exceed decoded memory budget", "resource_limit")
                    meta = self.accessors[attributes[name]]
                    if name in ("NORMAL", "TANGENT"):
                        expected = 3 if name == "NORMAL" else 4
                        if array.shape != (len(positions), expected) or meta["componentType"] != 5126:
                            _fail(f"{name} has an unsupported type")
                    if name.startswith("TEXCOORD_") and (
                        array.shape != (len(positions), 2) or array.dtype.kind != "f"
                    ):
                        _fail(f"{name} must be float/normalized VEC2")
                if "material" in part:
                    _ref(materials, part["material"], "material")
                if "indices" in part:
                    indices = self.accessor(part["indices"], normalized=False)
                    meta = self.accessors[part["indices"]]
                    if meta.get("normalized", False) or indices.ndim != 1 or indices.dtype.kind != "u":
                        _fail("triangle indices must be unsigned scalars")
                else:
                    indices = np.arange(len(positions), dtype=np.uint32)
                if len(indices) % 3 or (indices >= len(positions)).any():
                    _fail("triangle indices are incomplete or out of bounds")
                if ("JOINTS_0" in arrays) != ("WEIGHTS_0" in arrays):
                    _fail("JOINTS_0 and WEIGHTS_0 must appear together")
                if skin is not None:
                    if "JOINTS_0" not in arrays:
                        _fail("skinned primitive has no skin attributes")
                    joint_array, weights = arrays["JOINTS_0"], arrays["WEIGHTS_0"]
                    weight_meta = self.accessors[attributes["WEIGHTS_0"]]
                    if weight_meta["componentType"] not in (5121, 5123, 5126):
                        _fail("WEIGHTS_0 uses an invalid component type")
                    if joint_array.shape != (len(positions), 4) or joint_array.dtype.kind != "u":
                        _fail("JOINTS_0 must be unsigned VEC4")
                    if joint_array.dtype.itemsize not in (1, 2) or (joint_array >= len(skin["joints"])).any():
                        _fail("JOINTS_0 points outside the skin")
                    if weights.shape != joint_array.shape or weights.dtype.kind != "f":
                        _fail("WEIGHTS_0 must be float or normalized VEC4")
                    if (weights < 0).any() or not np.allclose(weights.sum(1), 1, atol=1e-5, rtol=0):
                        _fail("weights must be nonnegative and normalized", "invalid_weights")
                primitives.append({
                    "id": f"node:{node_index}/primitive:{primitive_index}", "node": node_index,
                    "mesh": node["mesh"], "primitive": primitive_index, "skin": node.get("skin"),
                    "material": part.get("material"), "attributes": arrays,
                    "triangles": indices.reshape(-1, 3), "transform": globals_[node_index].copy(),
                })
        if not primitives:
            _fail("no mesh instances to inspect")
        return primitives, globals_, parents

    def rebuild_container(self):
        """JSON/BIN container round-trip only; no Blender or canonical mesh conversion."""
        return build_glb(self.document, self.binary)
