"""Exercise decoded data, malformed boundaries and lossless container round trips."""

import copy
import struct

import numpy as np
import pytest

from image_to_3dlab.edit_glb import EditGLB, EditGLBError
from image_to_3dlab.glb_turn import build_glb, parse_glb


def fixture(*, skinned=True, interleaved=False, normalized=False):
    document = {
        "asset": {"version": "2.0", "extras": {"seed": 42}},
        "buffers": [], "bufferViews": [], "accessors": [],
        "nodes": [{"mesh": 0}, {"translation": [2, 0, 0], "children": [2]},
                  {"translation": [0, 1, 0]}],
        "materials": [{"name": "untouched", "extras": {"provenance": "fixture"}}],
        "meshes": [{"primitives": []}],
    }
    binary = bytearray()

    def append(array, kind, component, **extra):
        binary.extend(b"\0" * (-len(binary) % 4))
        view = len(document["bufferViews"])
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": array.nbytes})
        binary.extend(array.tobytes())
        index = len(document["accessors"])
        document["accessors"].append({"bufferView": view, "componentType": component,
                                      "count": len(array), "type": kind, **extra})
        return index

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    uv = np.array([[0, 0], [1, 0], [0, 1]], dtype="<f4")
    if interleaved:
        index = append(np.column_stack([positions, uv]).astype("<f4"), "VEC3", 5126)
        document["bufferViews"][0]["byteStride"] = 20
        document["accessors"].append({"bufferView": 0, "byteOffset": 12, "componentType": 5126,
                                      "count": 3, "type": "VEC2"})
        attributes = {"POSITION": index, "TEXCOORD_0": 1}
    else:
        attributes = {"POSITION": append(positions, "VEC3", 5126),
                      "TEXCOORD_0": append(uv, "VEC2", 5126)}
    if skinned:
        attributes["JOINTS_0"] = append(np.array([[0, 1, 0, 0]] * 3, dtype="u1"), "VEC4", 5121)
        weights = np.array([[0.2, 0.8, 0, 0]] * 3, dtype="<f4")
        if normalized:
            attributes["WEIGHTS_0"] = append(np.round(weights * 255).astype("u1"), "VEC4", 5121,
                                               normalized=True)
        else:
            attributes["WEIGHTS_0"] = append(weights, "VEC4", 5126)
        bind = np.array([np.eye(4), np.eye(4)])
        bind[:, 0, 3] = 2
        bind[1, 1, 3] = 1
        inverse = np.linalg.inv(bind).transpose(0, 2, 1).astype("<f4")
        ibm = append(inverse, "MAT4", 5126)
        document["skins"] = [{"joints": [1, 2], "inverseBindMatrices": ibm}]
        document["nodes"][0]["skin"] = 0
    indices = append(np.array([0, 1, 2], dtype="<u2"), "SCALAR", 5123)
    document["meshes"][0]["primitives"] = [{"attributes": attributes, "indices": indices, "material": 0}]
    # Two materials/primitives sharing accessors must retain that relationship.
    document["materials"].append({"name": "second"})
    second = copy.deepcopy(document["meshes"][0]["primitives"][0])
    second["material"] = 1
    document["meshes"][0]["primitives"].append(second)
    image_view = len(document["bufferViews"])
    binary.extend(b"\0" * (-len(binary) % 4))
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": 4})
    binary.extend(b"FAKE")  # Byte preservation fixture, deliberately no image decoding.
    document["images"] = [{"bufferView": image_view, "mimeType": "image/png"}]
    document["buffers"] = [{"byteLength": len(binary)}]
    return build_glb(document, bytes(binary))


def changed(change, *, skinned=True):
    document, binary = parse_glb(fixture(skinned=skinned))
    change(document)
    return build_glb(document, binary)


@pytest.mark.parametrize("interleaved,normalized", [(False, False), (True, False), (True, True)])
def test_real_accessors_and_container_preserve_materials_uv_skin_and_identity(interleaved, normalized):
    asset = EditGLB(fixture(interleaved=interleaved, normalized=normalized))
    primitives, global_, parents = asset.inspect_profile()
    assert len(primitives) == 2
    assert primitives[0]["id"] != primitives[1]["id"]
    assert primitives[0]["material"] == 0 and primitives[1]["material"] == 1
    assert np.allclose(primitives[0]["attributes"]["WEIGHTS_0"], [[0.2, 0.8, 0, 0]] * 3)
    assert np.array_equal(primitives[0]["triangles"], [[0, 1, 2]])
    assert np.array_equal(global_[2, :3, 3], [2, 1, 0])
    assert parents.tolist() == [-1, -1, 1]
    matrices = asset.accessor(asset.document["skins"][0]["inverseBindMatrices"])
    assert np.allclose(global_[[1, 2]] @ matrices, np.eye(4))
    rebuilt = EditGLB(asset.rebuild_container())
    assert rebuilt.document == asset.document
    assert rebuilt.binary == asset.binary
    # Working arrays cannot mutate source or another primitive sharing an accessor.
    primitives[0]["attributes"]["POSITION"][0] = [7, 8, 9]
    assert np.array_equal(primitives[1]["attributes"]["POSITION"][0], [0, 0, 0])
    assert asset.accessor(0)[0].tolist() == [0, 0, 0]


@pytest.mark.parametrize("data", [b"", b"glTF", fixture()[:-1], b"NOPE" + fixture()[4:]])
def test_truncated_or_wrong_container_fails(data):
    with pytest.raises(EditGLBError):
        EditGLB(data)


def test_declared_chunk_length_cannot_reach_past_file():
    data = bytearray(fixture())
    struct.pack_into("<I", data, 12, len(data) * 2)
    with pytest.raises(EditGLBError, match="chunk length"):
        EditGLB(bytes(data))


@pytest.mark.parametrize("change,reason", [
    (lambda d: d["buffers"][0].update(uri="../../private.bin"), "embedded buffer"),
    (lambda d: d["bufferViews"][0].update(byteLength=10**8), "bufferView exceeds"),
    (lambda d: d["bufferViews"][0].update(byteStride=5), "byteStride"),
    (lambda d: d["accessors"][0].update(byteOffset=2), "alignment"),
    (lambda d: d["accessors"][0].update(count=10), "accessor exceeds"),
    (lambda d: d["accessors"][0].update(count=10**9), "decoded byte limit"),
    (lambda d: d["accessors"][0].update(bufferView=-1), "integer"),
    (lambda d: d["accessors"][0].update(sparse={}), "sparse"),
    (lambda d: d["accessors"][0].update(normalized=True), "normalized"),
    (lambda d: d.update(extensionsUsed=["KHR_draco_mesh_compression"]), "extensions"),
    (lambda d: d["nodes"][2].update(children=[1]), "cycle"),
    (lambda d: d["nodes"][0].update(children=[2]), "multiple parents"),
    (lambda d: d["nodes"][0].update(mesh=99), "out of range"),
    (lambda d: d["nodes"][0].update(rotation=[0, 0, 0, 0]), "quaternion"),
    (lambda d: d["meshes"][0]["primitives"][0].update(mode=5), "only triangles"),
    (lambda d: d["meshes"][0]["primitives"][0].update(targets=[{}]), "morph targets"),
    (lambda d: d["skins"][0].update(joints=[1, 1]), "duplicate skin joint"),
    (lambda d: d["images"][0].update(uri="https://example.invalid/image.png"), "embedded images"),
])
def test_unsupported_and_malformed_inputs_fail_before_editing(change, reason):
    with pytest.raises(EditGLBError, match=reason):
        EditGLB(changed(change)).inspect_profile()


def test_instances_get_distinct_ids_and_keep_their_node_transform():
    asset = EditGLB(changed(lambda d: d["nodes"].append({"mesh": 0, "translation": [5, 0, 0]}),
                            skinned=False))
    parts, _, _ = asset.inspect_profile()
    assert len(parts) == 4
    assert len({p["id"] for p in parts}) == 4
    assert np.array_equal(parts[2]["transform"][:3, 3], [5, 0, 0])


def test_missing_indices_are_an_explicit_sequential_triangle_list():
    asset = EditGLB(changed(lambda d: d["meshes"][0]["primitives"][0].pop("indices")))
    assert asset.inspect_profile()[0][0]["triangles"].tolist() == [[0, 1, 2]]
