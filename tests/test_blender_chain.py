"""A version made in Blender goes on through Finish, Rig and Animate (gap 9 of
docs/plans/2026-10-17-step-parity-design.md): the studio shows the steps on a chosen
version, Rig and Animate take it like a model they made, and what is made from it stays
in the asset it came from. Blender, SkinTokens and Kimodo are not run here."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_edit_glb import fixture

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import assets_api  # noqa: E402
import autorig_api as aa  # noqa: E402
import blender_versions as bv  # noqa: E402

from image_to_3dlab.glb_turn import build_glb, parse_glb  # noqa: E402

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _edited(data: bytes, *, skin: bool = True) -> bytes:
    doc, binary = parse_glb(data)
    doc["asset"]["generator"] = "Blender test export"
    if not skin:
        doc.pop("skins")
        for node in doc["nodes"]:
            node.pop("skin", None)
    return build_glb(doc, binary)


@pytest.fixture
def library(tmp_path):
    """A generated knight, and a Blender version of it (skinned unless asked)."""
    source = tmp_path / "knight" / "knight.glb"
    source.parent.mkdir()
    doc, binary = parse_glb(fixture())
    doc["scenes"] = [{"nodes": [0, 1]}]
    doc["scene"] = 0
    source.write_bytes(build_glb(doc, binary))
    source.with_suffix(".provenance.json").write_text(json.dumps({"license": {"name": "source conditions"}}))
    asset = assets_api.list_assets(tmp_path)[0]

    def version(skin=True, parent="knight/knight.glb", session="1" * 32):
        opened = bv.create_session(tmp_path, asset["id"], parent, session)
        return bv.import_version(tmp_path, opened["id"], _edited(source.read_bytes(), skin=skin), "edited.glb")

    return tmp_path, asset, version


def test_what_is_made_from_a_version_stays_in_its_asset(library):
    output, asset, version = library
    made = version()
    data = (output / made["file"]).read_bytes()
    finish = output / "finish" / "knight__finish__20261017-120000"
    (finish / "input").mkdir(parents=True)
    (finish / "input" / "source.glb").write_bytes(data)
    (finish / "knight_40k.glb").write_bytes(b"finished")
    rig = output / "animate" / "knight__rig__20261017-130000"
    rig.mkdir(parents=True)
    (rig / "knight_rigged.glb").write_bytes(data)
    (rig / "knight_rigged.provenance.json").write_text(json.dumps(
        {"input": {"sha256": hashlib.sha256(data).hexdigest()}}))
    (listed,) = assets_api.list_assets(output)
    assert listed["id"] == asset["id"]
    assert listed["finished"] == "finish/knight__finish__20261017-120000/knight_40k.glb"
    assert listed["rigged"] == "animate/knight__rig__20261017-130000/knight_rigged.glb"
    assert [v["id"] for v in listed["blender_versions"]] == [made["id"]]


def test_a_version_of_a_version_is_followed_too(library):
    output, asset, version = library
    first = version()
    second = version(parent=first["file"], session="2" * 32)
    data = (output / second["file"]).read_bytes()
    finish = output / "finish" / "knight__finish__20261017-140000"
    (finish / "input").mkdir(parents=True)
    (finish / "input" / "source.glb").write_bytes(data)
    (finish / "knight_20k.glb").write_bytes(b"finished from the second")
    (listed,) = assets_api.list_assets(output)
    assert listed["finished"].endswith("knight_20k.glb") and len(listed["blender_versions"]) == 2


def test_rig_and_animate_offer_the_versions(library):
    output, _, version = library
    made = version()
    offered = {m["path"]: m["kind"] for m in aa.pickable_models(output)}
    assert offered[made["file"]] == "version"
    assert aa.resolve_model(made["file"], output) == output / made["file"]
    record = aa.source_record_for(output / made["file"])
    assert record["license"] == {"name": "source conditions"}   # the version's provenance travels


def test_a_rigged_version_becomes_a_rig_once_and_is_animated(library, tmp_path):
    output, _, version = library
    made = version()
    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "walk.npz").write_bytes(b"x")
    (motions / "presets.json").write_text(json.dumps({"presets": [{"id": "walk", "label": "Walk"}]}))
    manager = aa.AnimateJobManager(output / "animate", motions)
    job = manager.create_animation(output / made["file"], "walk", 0)
    run = job.result.parent
    assert aa.RUN_DIRECTORY.fullmatch(run.name) and job.result.name == "edited_walk.glb"
    rigged = run / "edited_rigged.glb"
    assert rigged.read_bytes() == (output / made["file"]).read_bytes()   # the version itself, not rigged again
    provenance = json.loads(rigged.with_suffix(".provenance.json").read_text())
    assert provenance["input"]["sha256"] == made["sha256"] and provenance["blender_version"] == made["id"]
    assert provenance["license"] == {"name": "source conditions"}
    manager.finish(job)
    again = manager.create_animation(output / made["file"], "walk", 0)
    assert again.result.parent == run   # the same rig, not a second copy
    (listed,) = assets_api.list_assets(output)
    assert listed["rigged"] == rigged.relative_to(output).as_posix()


def test_a_version_without_a_skeleton_is_rigged_first(library, tmp_path):
    output, _, version = library
    made = version(skin=False)
    manager = aa.AnimateJobManager(output / "animate", tmp_path)
    with pytest.raises(ValueError, match="no skeleton"):
        manager.create_animation(output / made["file"], "walk", 0)


@needs_node
def test_the_studio_shows_the_steps_on_a_version():
    url = (REPO / "viewer" / "studio" / "library.js").as_uri()
    program = (
        f"import * as m from {json.dumps(url)};\n"
        "const asset = { id: 'a', kind: 'model', stage: 'finished', picture: 'p.png', model: 'm.glb', finished: 'finish/r/f.glb',"
        " rigged: null, clips: [], props: [], made: { model: { seed: 1 } }, stale: {} };\n"
        "const v = (facts, parent) => ({ id: 'v', file: 'blender-edits/versions/v/model.glb', parent_file: parent, facts });\n"
        "const out = {\n"
        "  rigged: m.versionAsset(asset, v({ skins: 1 }, 'm.glb')),\n"
        "  fromModel: m.versionAsset(asset, v({ skins: 0 }, 'm.glb')),\n"
        "  fromFinish: m.versionAsset(asset, v({ skins: 0 }, 'finish/r/f.glb')),\n"
        "  props: m.versionAsset({ ...asset, kind: 'prop set' }, v({ skins: 0 }, 'm.glb')),\n"
        "};\n"
        "out.redo = ['model', 'finished', 'rigged'].map((s) => m.canRedo(out.fromFinish, s));\n"
        "console.log(JSON.stringify(out));")
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True,
                            text=True, encoding="utf-8")
    out = json.loads(result.stdout)
    file = "blender-edits/versions/v/model.glb"
    assert out["rigged"]["stage"] == "rigged" and out["rigged"]["rigged"] == file and out["rigged"]["kind"] == "character"
    assert out["fromModel"]["stage"] == "model" and out["fromModel"]["model"] == file and not out["fromModel"]["finished"]
    assert out["fromFinish"]["stage"] == "finished" and out["fromFinish"]["finished"] == file
    assert out["fromFinish"]["picture"] == "p.png" and out["fromFinish"]["made"] == {}
    assert out["props"] is None
    assert out["redo"] == [False, False, False]   # a redo would draw over the edit


def _rig_only(tmp_path):
    """An asset that is only a rig (a GLB uploaded to Animate), with one move."""
    run = tmp_path / "animate" / "knight__rig__20261017-100000"
    run.mkdir(parents=True)
    doc, binary = parse_glb(fixture())
    doc["scenes"] = [{"nodes": [0, 1]}]
    doc["scene"] = 0
    data = build_glb(doc, binary)
    (run / "knight_rigged.glb").write_bytes(data)
    (run / "knight_rigged.provenance.json").write_text(json.dumps({"license": {"name": "source conditions"}}))
    doc["asset"]["generator"] = "a walk"
    (run / "knight_walk.glb").write_bytes(build_glb(doc, binary))
    return run


def test_a_rig_made_from_a_version_keeps_the_assets_id(tmp_path):
    run = _rig_only(tmp_path)
    (asset,) = assets_api.list_assets(tmp_path)
    opened = bv.create_session(tmp_path, asset["id"], f"animate/{run.name}/knight_rigged.glb", "3" * 32)
    made = bv.import_version(tmp_path, opened["id"], _edited((run / "knight_rigged.glb").read_bytes()), "edited.glb")
    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "walk.npz").write_bytes(b"x")
    (motions / "presets.json").write_text(json.dumps({"presets": [{"id": "walk", "label": "Walk"}]}))
    aa.AnimateJobManager(tmp_path / "animate", motions).create_animation(tmp_path / made["file"], "walk", 0)
    (listed,) = assets_api.list_assets(tmp_path)
    assert listed["id"] == asset["id"] and listed["rigged"].endswith("edited_rigged.glb")
    assert [v["id"] for v in listed["blender_versions"]] == [made["id"]]


def test_what_is_made_from_a_version_of_a_move_stays_in_its_asset(tmp_path):
    run = _rig_only(tmp_path)
    (asset,) = assets_api.list_assets(tmp_path)
    opened = bv.create_session(tmp_path, asset["id"], f"animate/{run.name}/knight_walk.glb", "4" * 32)
    made = bv.import_version(tmp_path, opened["id"], _edited((run / "knight_walk.glb").read_bytes(), skin=False),
                             "walk_edit.glb")
    finish = tmp_path / "finish" / "knight__finish__20261017-150000"
    (finish / "input").mkdir(parents=True)
    (finish / "input" / "source.glb").write_bytes((tmp_path / made["file"]).read_bytes())
    (finish / "knight_20k.glb").write_bytes(b"finished from the move's version")
    (listed,) = assets_api.list_assets(tmp_path)
    assert listed["id"] == asset["id"] and listed["finished"].endswith("knight_20k.glb")
