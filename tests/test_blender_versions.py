"""Manual edits keep their source, provenance and lineage across server restarts."""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_edit_glb import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import assets_api
import blender_versions as bv

from image_to_3dlab.blender_exchange import inspect_glb
from image_to_3dlab.glb_turn import build_glb, parse_glb


@pytest.fixture
def library(tmp_path):
    source = tmp_path / "knight" / "knight.glb"
    source.parent.mkdir()
    source.write_bytes(fixture())
    record = {"license": {"name": "source conditions"}, "classification": "research-only", "motion": {"license": "motion conditions"}}
    source.with_suffix(".provenance.json").write_text(json.dumps(record))
    return tmp_path, source, assets_api.list_assets(tmp_path)[0], record


def test_snapshot_and_revision_keep_original_and_full_provenance(library):
    output, source, asset, record = library
    original = source.read_bytes()
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "1" * 32)
    source.write_bytes(b"changed after opening")
    doc, binary = parse_glb(original)
    doc["asset"]["generator"] = "Blender test export"
    edited = build_glb(doc, binary)
    revision = bv.import_version(output, session["id"], edited, "edited.glb")
    assert source.read_bytes() == b"changed after opening"
    assert revision["parent_sha256"] == hashlib.sha256(original).hexdigest()
    assert (output / revision["file"]).read_bytes() == edited
    saved = json.loads((output / revision["provenance"]).read_text())
    assert saved["source_record"] == record
    assert saved["license"] == record["license"]
    assert saved["classification"] == "research-only"
    assert "source_changed" in revision["warning_codes"]
    listed = assets_api.list_assets(output)
    assert len(listed) == 1 and listed[0]["id"] == asset["id"]
    assert listed[0]["blender_versions"][0]["id"] == revision["id"]


def test_concurrent_retries_publish_one_revision(library):
    output, source, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "2" * 32)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: bv.import_version(output, session["id"], source.read_bytes(), "edit.glb"), range(2)))
    assert results[0]["id"] == results[1]["id"]
    assert len(bv.list_versions(output)) == 1
    assert bv.create_session(output, asset["id"], "knight/knight.glb", "2" * 32)["id"] == session["id"]
    with pytest.raises(ValueError, match="different"):
        bv.import_version(output, session["id"], b"different", "edit.glb")


@pytest.mark.parametrize("relative", ["../outside.glb", "C:/outside.glb", "knight/../knight/knight.glb", "knight\\knight.glb", "/knight/knight.glb"])
def test_source_paths_are_strict_and_asset_bound(library, relative):
    output, _, asset, _ = library
    with pytest.raises(ValueError):
        bv.create_session(output, asset["id"], relative, "3" * 32)
    assert not list((output / "blender-edits/sessions").glob("*/session.json"))


def test_other_asset_cannot_be_used_as_parent(library):
    output, source, asset, _ = library
    other = output / "other" / "other.glb"
    other.parent.mkdir()
    other.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="belong"):
        bv.create_session(output, asset["id"], "other/other.glb", "4" * 32)


def test_bad_export_never_publishes_or_replaces_original(library):
    output, source, asset, _ = library
    before = source.read_bytes()
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "5" * 32)
    with pytest.raises(ValueError):
        bv.import_version(output, session["id"], b"not glb", "export.glb")
    assert bv.list_versions(output) == []
    assert source.read_bytes() == before


def test_revision_can_parent_another_revision_without_legacy_clips(library):
    output, source, asset, _ = library
    first = bv.create_session(output, asset["id"], "knight/knight.glb", "6" * 32)
    v1 = bv.import_version(output, first["id"], source.read_bytes(), "one.glb")
    second = bv.create_session(output, asset["id"], v1["file"], "7" * 32)
    doc, binary = parse_glb(source.read_bytes())
    doc.pop("skins")
    for node in doc["nodes"]:
        node.pop("skin", None)
    v2 = bv.import_version(output, second["id"], build_glb(doc, binary), "two.glb")
    assert v2["parent_version"] == v1["id"]
    assert v2["facts"]["skins"] == 0
    assert "rig_removed" in v2["warning_codes"]
    assert len(assets_api.list_assets(output)[0]["blender_versions"]) == 2


def test_external_resources_and_compression_are_refused_before_launch():
    doc, binary = parse_glb(fixture())
    doc["images"][0]["uri"] = "https://example.com/secret.png"
    with pytest.raises(ValueError, match="embedded"):
        inspect_glb(build_glb(doc, binary))
    doc["images"][0].pop("uri")
    doc["extensionsRequired"] = ["KHR_draco_mesh_compression"]
    with pytest.raises(ValueError, match="compression"):
        inspect_glb(build_glb(doc, binary))


def test_snapshot_tampering_is_not_silently_imported(library):
    output, source, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "8" * 32)
    (bv.session_path(output, session["id"]) / "input/source.glb").write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot"):
        bv.import_version(output, session["id"], source.read_bytes(), "edit.glb")


def test_sparse_accessor_logical_allocation_is_bounded():
    doc, binary = parse_glb(fixture())
    doc["accessors"][0]["count"] = 10 ** 12
    doc["accessors"][0]["sparse"] = {"count": 1}
    with pytest.raises(ValueError, match="memory limit"):
        inspect_glb(build_glb(doc, binary))


def test_versions_survive_missing_original(library):
    output, source, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "a" * 32)
    version = bv.import_version(output, session["id"], source.read_bytes(), "edited.glb")
    source.unlink()
    listed = assets_api.list_assets(output)
    assert len(listed) == 1 and listed[0]["id"] == asset["id"]
    assert listed[0]["blender_versions"][0]["id"] == version["id"]


def test_failed_publication_is_invisible_and_retryable(library, monkeypatch):
    output, source, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "b" * 32)
    original_writer = bv.write_json
    def fail(path, value):
        if path.name == "version.json":
            raise OSError("disk full")
        original_writer(path, value)
    with monkeypatch.context() as context:
        context.setattr(bv, "write_json", fail)
        with pytest.raises(OSError, match="disk full"):
            bv.import_version(output, session["id"], source.read_bytes(), "edited.glb")
    assert bv.list_versions(output) == []
    bv.import_version(output, session["id"], source.read_bytes(), "edited.glb")
    assert len(bv.list_versions(output)) == 1


def test_legacy_sidecar_output_path_is_preserved(library):
    output, source, asset, record = library
    record.pop("classification")
    record["output"] = "old-output.glb"
    source.with_suffix(".provenance.json").write_text(json.dumps(record))
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "c" * 32)
    version = bv.import_version(output, session["id"], source.read_bytes(), "edited.glb")
    saved = json.loads((output / version["provenance"]).read_text())
    assert saved["source_record"] == record
    assert saved["license"] == record["license"]
    assert saved["classification"] is None
