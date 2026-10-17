"""The final review of the step parity work (Codex, read only, on main after #109), one test per
defect found and confirmed. Blender, Unreal, SkinTokens, sd-cli and the judge are stand-ins."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from batch_fakes import tiny_glb, tiny_png

from image_to_3dlab import batch, piece_materials, rigging
from image_to_3dlab.batch_steps import StepContext
from image_to_3dlab.batch_steps import checks as checks_step
from image_to_3dlab.batch_steps import finish as finish_step

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
NODE = shutil.which("node")


def _row(tmp_path, **fields):
    values = {"id": "SM_Cat", "images": (tiny_png(tmp_path / "refs" / "cat.png"),), "asset": None, "kind": "prop",
              "faces": 3000, "profile": None, "steps": ("generate", "finish"), "backend": "fake",
              "parameters": {}, "options": {}}
    values.update(fields)
    return batch.BatchRow(**values)


def _context(tmp_path, row, **fields):
    row_dir = tmp_path / "rows" / row.id
    values = dict(row=row, row_dir=row_dir, step_dir=row_dir / "step", asset=None, sidecar=None,
                  intent={"use_case": "game"}, license_policy={"allow_conditional": True}, options={},
                  log=lambda line: None)
    values.update(fields)
    return StepContext(**values)


# ------------------------------------------------------------------------- batches

def test_a_row_from_words_is_not_refused_by_finish_before_its_picture_is_drawn(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    row = _row(tmp_path, images=(), prompt="a red barrel", steps=("picture", "generate", "finish"))
    assert not any("source image" in problem for problem in finish_step.preflight([row]))


def test_checks_and_the_repaint_use_the_picture_the_model_was_made_from(tmp_path):
    row = _row(tmp_path, steps=("concept", "generate", "finish", "checks"))
    concept = tiny_png(tmp_path / "rows" / "SM_Cat" / "concept" / "s4.png")
    ctx = _context(tmp_path, row, picture=concept)
    assert checks_step._reference(ctx) == concept
    assert finish_step.source_image(ctx) == concept
    plain = _context(tmp_path, row, picture=row.images[0])   # no step drew one: the row's own
    assert checks_step._reference(plain) == row.images[0] and finish_step.source_image(plain) == row.images[0]


def test_a_replaced_animate_keeps_the_clips_made_before_it(tmp_path):
    family = tiny_glb(tmp_path / "rows" / "SM_Cat" / "clips" / "Idle.glb")
    mine = tiny_glb(tmp_path / "mine" / "Jump.glb")
    row = _row(tmp_path, replace={"animate": [str(mine)]})
    ctx = _context(tmp_path, row, step_dir=tmp_path / "rows" / "SM_Cat" / "animate", clips=[family])
    result = batch.REPLACED.run(ctx)
    assert [clip.name for clip in result.clips] == ["Idle.glb", "Jump.glb"]


def test_the_rig_runs_on_absolute_paths_whatever_folder_it_was_asked_from(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(rigging, "installed", lambda: True)
    monkeypatch.setattr(rigging, "run_commands", lambda commands, *a, **k: seen.extend(commands) or 1)
    source = tiny_glb(tmp_path / "model.glb")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError):
        rigging.rig_model(source, Path("rigged"), "model", origin="test", tool="test")
    command = seen[0][0]
    for flag in ("--input", "--output"):
        assert Path(command[command.index(flag) + 1]).is_absolute(), flag


def test_a_picture_asked_for_in_a_relative_folder_lands_there(tmp_path, monkeypatch):
    from image_to_3dlab import text_image

    seen = {}

    class Stop(Exception):
        pass

    def build(prompt, settings, output_path, weights):
        seen["out"] = output_path
        raise Stop

    monkeypatch.setattr(text_image, "prepare_prompt", lambda prompt, settings, **k: (prompt, {}))
    monkeypatch.setattr(text_image, "build_command", build)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Stop):
        text_image.generate("a barrel", {}, Path("pictures") / "a.png", weights={"x": Path("w")})
    assert seen["out"] == (tmp_path / "pictures" / "a.png").resolve()


# -------------------------------------------------------------- materials, pieces

def test_the_words_are_found_through_a_rig_and_a_pixal3d_record(tmp_path):
    picture = tiny_png(tmp_path / "wolf.png")
    (tmp_path / "wolf.provenance.json").write_text(json.dumps({"prompt": "a wolf with grey fur"}))
    model = tiny_glb(tmp_path / "wolf_3d.glb")
    (tmp_path / "wolf_3d.json").write_text(json.dumps({"input": {"path": str(picture)}}))   # Pixal3D's record
    assert piece_materials.prompt_for(model) == "a wolf with grey fur"
    rigged = tiny_glb(tmp_path / "wolf_rigged.glb")
    (tmp_path / "wolf_rigged.provenance.json").write_text(json.dumps(
        {"input": {"name": "wolf_3d.glb", "sha256": "x"}, "source_record": {"input": {"path": str(picture)}}}))
    assert piece_materials.prompt_for(rigged) == "a wolf with grey fur"
    assert piece_materials.propose(kind="creature", words=piece_materials.prompt_for(rigged))["material"] == "fur"


def test_the_outfit_pick_passes_the_command_line():
    from image_to_3dlab.slot_parts import pieces

    assert pieces.parse_picks("concept=4,helmet=7") == {"concept": 4, "helmet": 7}


def _two_bodies(tmp_path):
    from image_to_3dlab.slot_parts import body
    from tests.test_slot_parts_cli import _body

    body_json, *_ = _body(tmp_path)
    tall = body.load(body_json)
    tall["name"] = "tall"
    return body_json, body.write(tall, tmp_path / "bodies")


def test_a_carried_set_keeps_its_material_corrections(tmp_path):
    import slot_parts as cli

    body_json, tall_json = _two_bodies(tmp_path)
    first = tmp_path / "sets" / "red" / "stick"
    (first / "fit").mkdir(parents=True)
    (first / "red.parts.json").write_text(json.dumps({"set": "red", "body": {"name": "stick"}}))
    (first / piece_materials.CORRECTIONS).write_text(json.dumps({"helmet": "leather"}))
    out = tmp_path / "sets" / "red" / "tall" / "fit"
    cli.carried_record(first / "fit", tall_json, out, {"files": []})
    assert json.loads((out.parent / piece_materials.CORRECTIONS).read_text()) == {"helmet": "leather"}


def test_fit_again_on_a_carried_set_carries_it_again(tmp_path):
    from image_to_3dlab.slot_parts import runs

    _two_bodies(tmp_path)
    for name, extra in (("stick", {}), ("tall", {"carried_from": "stick"})):
        folder = tmp_path / "sets" / "red" / name
        (folder / "fit").mkdir(parents=True)
        (folder / "fit" / "fit.json").write_text("{}")
        (folder / "red.parts.json").write_text(json.dumps({"set": "red", "body": {"name": name}, **extra}))
    again = runs.fit_command(tmp_path, {"body": "tall", "set": "red"})
    assert again == runs.refit_command(tmp_path, {"body": "stick", "set": "red", "to": "tall"})


# ------------------------------------------------------------------------- studio

def test_a_clip_of_a_blender_version_does_not_claim_the_versions_hash():
    rig = {"input": {"source": "v.glb", "sha256": "a"}, "output": {"sha256": "the version's"}, "license": {"name": "MIT"}}
    record = rigging.animation_record(rig, {"id": "walk", "label": "Walk"}, 0.0)
    assert "output" not in record and record["license"] == {"name": "MIT"}


@pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")
def test_a_version_of_a_finished_version_is_finished_too():
    url = (REPO / "viewer" / "studio" / "library.js").as_uri()
    program = (
        f"import * as m from {json.dumps(url)};\n"
        "const first = { id: 'v1', file: 'blender-edits/versions/v1/model.glb', parent_file: 'finish/r/f.glb', facts: { skins: 0 } };\n"
        "const second = { id: 'v2', file: 'blender-edits/versions/v2/model.glb', parent_file: first.file, parent_version: 'v1', facts: { skins: 0 } };\n"
        "const asset = { id: 'a', kind: 'model', stage: 'finished', picture: 'p.png', model: 'm.glb', finished: 'finish/r/f.glb',"
        " rigged: null, clips: [], props: [], made: {}, stale: {}, blender_versions: [first, second] };\n"
        "console.log(JSON.stringify(m.versionAsset(asset, second)));")
    out = json.loads(subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True, encoding="utf-8").stdout)
    assert out["stage"] == "finished" and out["finished"] == "blender-edits/versions/v2/model.glb"


# ------------------------------------------------------------------------- Unreal

# ------------------------------------------------------------------------ engines

def test_a_pixal3d_install_does_not_download_moge_without_its_own_consent(monkeypatch, capsys):
    import bootstrap_pixal3d as boot

    asked = []
    monkeypatch.setattr(boot, "install_moge", lambda *a, **k: asked.append("moge"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: None)
    monkeypatch.setattr(boot, "build_present", lambda: True)
    monkeypatch.setattr(boot, "current_kind", lambda key, prefer_compile=False: "prebuilt")
    assert boot.main(["--yes", "--weights-only"]) == 0
    assert asked == [] and "--moge" in capsys.readouterr().out


# ---------------------------------------------------------------------------- MCP

def test_a_review_report_outside_the_shared_folders_is_withheld(tmp_path, monkeypatch):
    from image_to_3dlab import mcp_server
    from image_to_3dlab.checks import review

    secret = tmp_path / "secret.checks.json"
    secret.write_text(json.dumps({"private": True}))
    monkeypatch.setattr(review, "item_with_report", lambda log, key: {
        "key": key, "state": "to_review", "report": {"private": True}, "report_file": str(secret)})
    held = mcp_server._ROOTS.set([])
    try:
        item = mcp_server.review_item("k")
    finally:
        mcp_server._ROOTS.reset(held)
    assert item["report"] is None and "outside" in item["report_withheld"]
    assert mcp_server.review_item("k")["report"] == {"private": True}   # not over MCP: as before


def test_the_picture_an_asset_record_names_is_held_too(tmp_path):
    from image_to_3dlab import mcp_server

    outside = tmp_path / "private.png"
    row = SimpleNamespace(images=(), asset=None, replace={}, options={}, concept=outside)
    loaded = SimpleNamespace(path=mcp_server.OUTPUT / "lot.json", directory=mcp_server.OUTPUT / "lot", rows=[row])
    held = mcp_server._ROOTS.set([])
    try:
        assert any(str(outside) in problem for problem in mcp_server._outside_spec(loaded))
    finally:
        mcp_server._ROOTS.reset(held)


def test_run_steps_never_rewrites_the_spec_of_a_running_batch(tmp_path, monkeypatch):
    from image_to_3dlab import mcp_server

    monkeypatch.setattr(batch, "DEFAULT_OUTPUT", tmp_path / "batches")
    monkeypatch.setattr(mcp_server, "AGENT_SPECS", tmp_path / "batches" / ".agent")
    spec = mcp_server.AGENT_SPECS / "a.json"
    spec.parent.mkdir(parents=True)
    spec.write_text('{"first": true}')
    monkeypatch.setattr(batch, "lock_holder", lambda directory: 4242)
    result = mcp_server.run_steps("a", ["checks"], asset=str(tiny_glb(tmp_path / "a.glb")))
    assert result["started"] is False and "running" in result["problems"][0]
    assert spec.read_text() == '{"first": true}'


def test_picks_made_at_once_are_all_kept(tmp_path, monkeypatch):
    from image_to_3dlab import mcp_server

    directory = tmp_path / "lot"
    directory.mkdir()
    waiting = {"status": "waiting", "steps": {"picture": {"status": "waiting", "key": "k",
                                                          "candidates": [{"choice": 1, "file": "a.png"},
                                                                         {"choice": 2, "file": "b.png"}]}}}
    batch.write_json(directory / batch.STATE_FILE, {"rows": {"r1": waiting, "r2": waiting}})
    monkeypatch.setattr(batch, "lock_holder", lambda d: 4242)   # running: the picks wait for the next run
    write = batch.write_json

    def slow(path, data):
        time.sleep(0.2)
        write(path, data)

    monkeypatch.setattr(batch, "write_json", slow)
    threads = [threading.Thread(target=mcp_server.pick_candidate, args=(str(directory), row, "picture", 2))
               for row in ("r1", "r2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert set(json.loads((directory / batch.PICKS_FILE).read_text())) == {"r1", "r2"}


def test_a_link_to_a_file_with_a_hash_in_its_name_reads_that_file(tmp_path, monkeypatch):
    from mcp.client.client import Client

    from image_to_3dlab import mcp_server

    monkeypatch.setattr(mcp_server, "REPO", tmp_path / "repo")
    monkeypatch.setattr(mcp_server, "OUTPUT", tmp_path / "repo" / "output")
    made = tmp_path / "repo" / "output" / "asset#1.glb"
    made.parent.mkdir(parents=True)
    made.write_bytes(b"the first")
    (made.parent / "asset").write_bytes(b"not this one")
    monkeypatch.setattr(mcp_server, "list_outputs", lambda batch_ref, row_id=None: {"rows": [{"asset": str(made)}]})

    async def work():
        async with Client(mcp_server.build_server(), mode="legacy") as session:
            result = await session.call_tool("list_outputs", {"batch_ref": str(made.parent)})
            link = next(item for item in result.content if item.type == "resource_link")
            return await session.read_resource(str(link.uri))

    import base64

    assert base64.b64decode(asyncio.run(work()).contents[0].blob) == b"the first"


def test_a_shared_network_folder_keeps_its_server():
    from image_to_3dlab import mcp_server

    assert mcp_server._uri_path("file://server/share/project/x.glb") == Path("//server/share/project/x.glb")
