"""Canonical buffers are checked after reload; shared glTF accessors stay isolated."""

import copy
import json

import numpy as np
import pytest
from test_edit_glb import fixture

from image_to_3dlab.edit_document import EditDocument
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb, parse_glb


def test_noop_is_byte_exact_and_snapshot_is_reopenable(tmp_path):
    source = fixture(interleaved=True, normalized=True)
    document = EditDocument(source)
    assert document.export_glb() == source
    document.write_snapshot(tmp_path / "snapshot")
    restored = EditDocument.read_snapshot(tmp_path / "snapshot")
    assert restored.export_glb() == source
    assert restored.document_sha256 == document.document_sha256
    assert restored.vertex_identity == "source_render_vertex"


def test_editing_one_primitive_does_not_change_shared_materials_uv_skin_or_neighbor():
    source = fixture(interleaved=True)
    document = EditDocument(source)
    part_id = next(iter(document.parts))
    before_topology = document.topology_revision
    document.set_positions(part_id, [0], [[0, 0, 0.2]], expected_topology=before_topology)
    exported = EditGLB(document.export_glb())
    parts, _, _ = exported.inspect_profile()
    assert np.allclose(parts[0]["attributes"]["POSITION"][0], [0, 0, 0.2])
    assert parts[1]["attributes"]["POSITION"][0].tolist() == [0, 0, 0]
    original = EditGLB(source)
    for field in ("materials", "images", "skins"):
        assert exported.document[field] == original.document[field]
    assert exported.binary[:original.buffer_size] == original.binary[:original.buffer_size]
    assert document.topology_revision == before_topology


def test_shared_mesh_instances_are_isolated_by_cloning_only_the_edited_node():
    gltf, binary = parse_glb(fixture(skinned=False))
    gltf["nodes"].append({"mesh": 0, "translation": [4, 0, 0]})
    document = EditDocument(build_glb(gltf, binary))
    document.set_positions("node:3/primitive:0", [1], [[2, 0, 0]], expected_topology=document.topology_revision)
    exported = EditGLB(document.export_glb())
    assert exported.document["nodes"][0]["mesh"] == 0
    assert exported.document["nodes"][3]["mesh"] == 1
    parts, _, _ = exported.inspect_profile()
    assert parts[0]["attributes"]["POSITION"][1, 0] == 1
    assert parts[2]["attributes"]["POSITION"][1, 0] == 2
    assert parts[2]["transform"][0, 3] == 4


def test_topology_changes_allocate_new_identity_and_preserve_every_semantic():
    document = EditDocument(fixture())
    key = next(iter(document.parts))
    part = document.parts[key]
    before = document.topology_revision
    attributes = {name: np.concatenate([value, value[:1]]) for name, value in part["attributes"].items()}
    document.replace_geometry(key, attributes, np.array([[1, 2, 3]], dtype=np.uint32))
    assert document.topology_revision != before
    parts = EditGLB(document.export_glb()).inspect_profile()[0]
    assert parts[0]["triangles"].tolist() == [[1, 2, 3]]
    assert len(parts[0]["attributes"]["WEIGHTS_0"]) == 4
    with pytest.raises(ValueError, match="topology revision"):
        document.set_positions(key, [0], [[0, 0, 0]], expected_topology=before)


@pytest.mark.parametrize("indices,values", [([-1], [[0, 0, 0]]), ([0, 0], [[0, 0, 0]] * 2),
                                            ([0.5], [[0, 0, 0]]), ([0], [[np.nan, 0, 0]])])
def test_invalid_patch_does_not_mutate_document(indices, values):
    document = EditDocument(fixture())
    before = document.document_sha256
    with pytest.raises(ValueError):
        document.set_positions(next(iter(document.parts)), indices, values,
                               expected_topology=document.topology_revision)
    assert document.document_sha256 == before


def test_snapshot_detects_tampering_and_disallows_path_substitution(tmp_path):
    document = EditDocument(fixture())
    directory = tmp_path / "snapshot"
    document.write_snapshot(directory)
    manifest_path = directory / "document.json"
    original = json.loads(manifest_path.read_text())
    modified = copy.deepcopy(original)
    modified["parts"][0]["buffer_file"] = "../private.npz"
    manifest_path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="identity/path"):
        EditDocument.read_snapshot(directory)
    manifest_path.write_text(json.dumps(original))
    (directory / "primitive-0.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="buffer hash"):
        EditDocument.read_snapshot(directory)
