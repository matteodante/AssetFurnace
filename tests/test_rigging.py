"""Rig and animate outside the studio: `image_to_3dlab/rigging.py` (rig_model,
animate_model), `scripts/autorig.py`, and the batch steps `rig` and `animate`
(docs/plans/2026-10-17-step-parity-design.md, section 6). SkinTokens and the retarget are
stand-ins: small Python programs that write the GLB they were asked for."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest
from batch_fakes import FakeFinish, fake_step, install_fake_generator, tiny_glb, write_spec

from image_to_3dlab import batch, gpu_lock, rigging, step_catalog
from image_to_3dlab.batch_steps import animate as animate_step
from image_to_3dlab.batch_steps import export_unreal as export_step
from image_to_3dlab.batch_steps import rig as rig_step

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import autorig  # noqa: E402

WRITE_GLB = (
    "import json, pathlib, struct, sys\n"
    "out = pathlib.Path(sys.argv[1]); lock = pathlib.Path(sys.argv[2])\n"
    "doc = json.dumps({'asset': {'version': '2.0'}, 'nodes': [{}], 'skins': [{'joints': [0]}],\n"
    "                  'extras': {'locked': lock.exists()}}).encode()\n"
    "doc += b' ' * (-len(doc) % 4)\n"
    "body = struct.pack('<II', len(doc), 0x4E4F534A) + doc\n"
    "out.write_bytes(struct.pack('<4sII', b'glTF', 2, 12 + len(body)) + body)\n"
    "print('wrote', out.name)\n")


def skinned(path: Path) -> Path:
    document = json.dumps({"asset": {"version": "2.0"}, "nodes": [{}], "skins": [{"joints": [0]}]}).encode()
    document += b" " * (-len(document) % 4)
    body = struct.pack("<II", len(document), 0x4E4F534A) + document
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


def _extras(glb: Path) -> dict:
    from image_to_3dlab.glb_turn import parse_glb

    return parse_glb(glb.read_bytes())[0].get("extras", {})


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """SkinTokens, the weight clean-up and the retarget as stand-ins; a test GPU lock."""
    script = tmp_path / "write_glb.py"
    script.write_text(WRITE_GLB)
    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(gpu_lock, "LOCK", lock)
    monkeypatch.setattr(rigging, "installed", lambda root=None: True)
    calls = []

    def rig_commands(source, result):
        calls.append(("rig", source))
        return [([sys.executable, str(script), str(result), str(lock)], REPO),
                ([sys.executable, "-c", "print('weights cleaned')"], REPO)]

    def animate_command(rigged, motion, result, arm_spread, speed=1.0):
        calls.append(("animate", motion.stem, arm_spread, speed))
        return [sys.executable, str(script), str(result), str(lock)]

    monkeypatch.setattr(rigging, "rig_commands", rig_commands)
    monkeypatch.setattr(rigging, "animate_command", animate_command)
    return calls


def test_rig_model_holds_the_gpu_for_skintokens_and_keeps_the_licence(tmp_path, tools):
    source = tiny_glb(tmp_path / "knight.glb", "source")
    source.with_suffix(".provenance.json").write_text(json.dumps(
        {"license": {"name": "MIT"}, "classification": "commercial-clear"}))
    rigged, sidecar = rigging.rig_model(source, tmp_path / "out", "knight", origin="test", tool="test rig")
    assert rigged.name == "knight_rigged.glb" and _extras(rigged)["locked"] is True
    assert not (tmp_path / "gpu.lock").exists()
    record = json.loads(sidecar.read_text())
    assert record["license"] == {"name": "MIT"} and record["classification"] == "commercial-clear"
    assert record["rig"]["tool"] == "SkinTokens" and record["tool"] == "test rig"
    assert (tmp_path / "out" / "input" / "source.glb").read_bytes() == source.read_bytes()
    assert "weights cleaned" in (tmp_path / "out" / "rig.log").read_text()


def test_a_failed_rig_says_why(tmp_path, tools, monkeypatch):
    monkeypatch.setattr(rigging, "rig_commands", lambda source, result: [
        ([sys.executable, "-c", "print('CUDA out of memory'); raise SystemExit(4)"], REPO)])
    with pytest.raises(RuntimeError, match="code 4(.|\n)*CUDA out of memory"):
        rigging.rig_model(tiny_glb(tmp_path / "a.glb"), tmp_path / "out", "a", origin="t", tool="t")


def test_nothing_runs_without_skintokens(tmp_path, monkeypatch):
    monkeypatch.setattr(rigging, "installed", lambda root=None: False)
    with pytest.raises(RuntimeError, match="Setup page"):
        rigging.rig_model(tiny_glb(tmp_path / "a.glb"), tmp_path / "out", "a", origin="t", tool="t")


def test_animate_model_plays_presets_without_the_gpu(tmp_path, tools):
    rigged = skinned(tmp_path / "knight_rigged.glb")
    rigged.with_suffix(".provenance.json").write_text(json.dumps({"license": {"name": "MIT"}}))
    clips = rigging.animate_model(rigged, ["idle", "walk"], tmp_path / "clips", arm_spread=10, tool="t")
    assert [clip.name for clip in clips] == ["knight_idle.glb", "knight_walk.glb"]
    assert all(_extras(clip)["locked"] is False for clip in clips)   # CPU only: no GPU lock
    record = json.loads(clips[1].with_suffix(".provenance.json").read_text())
    assert record["license"] == {"name": "MIT"} and record["motion"]["preset"] == "walk"
    assert record["motion"]["arm_spread_degrees"] == 10 and record["motion"]["speed"] == 1.0
    with pytest.raises(ValueError, match="no such preset: moonwalk"):
        rigging.animate_model(rigged, ["moonwalk"], tmp_path / "x", tool="t")
    with pytest.raises(ValueError, match="no skeleton"):
        rigging.animate_model(tiny_glb(tmp_path / "flat.glb"), ["idle"], tmp_path / "x", tool="t")


def test_the_command_line_rigs_and_animates(tmp_path, tools, capsys):
    source = tiny_glb(tmp_path / "wolf.glb")
    assert autorig.main(["rig", str(source), "--out", str(tmp_path / "rig")]) == 0
    rigged = tmp_path / "rig" / "wolf_rigged.glb"
    assert rigged.is_file() and str(rigged) in capsys.readouterr().out
    assert autorig.main(["animate", str(rigged), "--presets", "idle", "walk", "--speed", "1.5",
                         "--arm-spread", "5", "--out", str(tmp_path / "clips")]) == 0
    assert ("animate", "walk", 5.0, 1.5) in tools
    assert autorig.main(["animate", str(rigged), "--presets", "moonwalk"]) == 1
    assert autorig.main(["presets"]) == 0 and "idle" in capsys.readouterr().out


# ----------------------------------------------------------------- batch steps

def test_rig_and_animate_inside_a_batch_and_the_clips_reach_the_export(tmp_path, monkeypatch, tools):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    exported = []

    def export(source, **kwargs):
        exported.append({"source": source, **kwargs})
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        glb = tiny_glb(out / "SK_Wolf.glb")
        (out / "SK_Wolf.unreal.json").write_text("{}")
        (out / "SK_Wolf.provenance.json").write_text("{}")
        return {"ok": True, "glb": str(glb), "sidecar": str(out / "SK_Wolf.unreal.json"),
                "provenance": str(out / "SK_Wolf.provenance.json"), "tint": None, "problems": []}

    monkeypatch.setattr(export_step, "export_unreal", export)
    monkeypatch.setattr(export_step, "find_blender", lambda: Path("blender.exe"))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "rig", "animate", "export_unreal"])
    spec = json.loads(spec_path.read_text())
    spec["defaults"].update(kind="creature", size_cm=120, animate={"presets": ["idle", "walk"]})
    spec_path.write_text(json.dumps(spec))
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0, report.failed
    directory = spec_path.parent / "out" / "test-batch"
    entry = json.loads((directory / batch.STATE_FILE).read_text())["rows"]["SM_row01"]
    assert entry["asset"] == "rows/SM_row01/rig/SM_row01_rigged.glb"   # the animate step keeps the rig
    assert entry["steps"]["animate"]["record"] == {"clips": ["SM_row01_idle.glb", "SM_row01_walk.glb"]}
    (call,) = exported
    assert call["source"].name == "SM_row01_rigged.glb"
    assert [clip.name for clip in call["animations"]] == ["SM_row01_idle.glb", "SM_row01_walk.glb"]
    # a new clip list reruns the animation and the export, not the rig
    tools.clear()
    spec["defaults"]["animate"] = {"presets": ["idle"]}
    spec_path.write_text(json.dumps(spec))
    batch.run_batch(batch.load_batch(spec_path))
    assert [call[0] for call in tools] == ["animate"]
    assert [clip.name for clip in exported[-1]["animations"]] == ["SM_row01_idle.glb"]


def test_animate_settings_are_checked_before_the_batch(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "rig", "animate"])
    spec = json.loads(spec_path.read_text())
    spec["defaults"]["animate"] = {"presets": ["moonwalk"], "speed": 9}
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(batch.BatchError) as raised:
        batch.load_batch(spec_path)
    text = str(raised.value)
    assert "moonwalk" in text and "speed" in text


def test_rig_and_animate_preflights_name_a_missing_install(tmp_path, monkeypatch):
    monkeypatch.setattr(rigging, "installed", lambda root=None: False)
    row = batch.BatchRow(id="A", images=(), asset=None, kind="creature", faces=None, profile=None,
                         steps=("generate", "rig", "animate"), backend="fake",
                         options={"animate": {"presets": ["idle"]}})
    assert rig_step.preflight([row]) == [rigging.MISSING]
    assert rigging.MISSING in animate_step.preflight([row])
    no_presets = batch.BatchRow(**{**row.__dict__, "options": {}})
    assert any("presets" in p for p in animate_step.preflight([no_presets]))


def test_rig_and_animate_are_described_with_their_command_line():
    rig, animate = step_catalog.get("rig"), step_catalog.get("animate")
    assert rig.cli == "scripts/autorig.py rig" and rig.gpu and rig.replace == ("glb",)
    assert animate.cli == "scripts/autorig.py animate" and not animate.gpu
    presets = animate.param("presets")
    assert presets.type == "choices" and "idle" in step_catalog.choices(presets)
    flags = {flag for action in autorig.build_parser()._subparsers._group_actions[0].choices["animate"]._actions
             for flag in action.option_strings}
    for param in animate.params:
        assert f"--{param.name.replace('_', '-')}" in flags, param.name


def test_an_interrupted_command_is_stopped_before_the_gpu_is_let_go(tmp_path, monkeypatch):
    """Ctrl+C or a cancelled batch during SkinTokens: the child (in its own process group)
    must not go on using the GPU once the lock is free."""
    stopped, lock = [], tmp_path / "gpu.lock"

    class Running:
        pid = 4242

        def wait(self, timeout=None):
            if timeout is None:
                assert lock.exists()   # still held while the child is stopped
                raise KeyboardInterrupt
            return -1

    monkeypatch.setattr(rigging.processes, "terminate_group", lambda pid: stopped.append((pid, lock.exists())))
    with pytest.raises(KeyboardInterrupt):
        rigging.run_commands([(["skintokens"], REPO)], tmp_path / "rig.log", gpu={0}, lock=lock,
                             popen=lambda *a, **k: Running())
    assert stopped == [(4242, True)] and not lock.exists()


def test_clips_with_an_export_need_a_creature(tmp_path, monkeypatch):
    monkeypatch.setattr(rigging, "installed", lambda root=None: True)
    row = batch.BatchRow(id="A", images=(), asset=None, kind="prop", faces=None, profile=None,
                         steps=("generate", "rig", "animate", "export_unreal"), backend="fake",
                         options={"animate": {"presets": ["idle"]}})
    assert any("creature" in p for p in animate_step.preflight([row]))


def test_the_rig_record_names_the_bpy_skintokens_ran(tmp_path, monkeypatch):
    from image_to_3dlab import rigging

    site = tmp_path / ".venv" / "Lib" / "site-packages"
    (site / "bpy-5.0.1.dist-info").mkdir(parents=True)
    assert rigging.bpy_version(tmp_path) == "5.0.1"
    assert rigging.bpy_version(tmp_path / "none") is None
    monkeypatch.setattr(rigging, "bpy_version", lambda *a: "5.0.1")
    record = rigging.rig_record("a.glb", b"x", None, "upload")
    assert record["rig"]["bpy"] == "5.0.1"
