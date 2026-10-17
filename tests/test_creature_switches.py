"""Creature families' clips can be switched (gap 5 of
docs/plans/2026-10-17-step-parity-design.md): the test clips after a rig, the CC0 library,
the longer GenMM takes and their factor and seed, on the command line
(`creature_clips.py`, `chain_retarget.py`), in the studio's Rig step and in the batch step
`clips`. Blender and SkinTokens are not run here."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from batch_fakes import FakeFinish, fake_step, install_fake_generator, tiny_glb, write_spec

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "viewer"))
import chain_retarget  # noqa: E402
import creature_clips  # noqa: E402

from image_to_3dlab import batch, rigging, step_catalog  # noqa: E402
from image_to_3dlab import clip_library as cl  # noqa: E402
from image_to_3dlab.batch_steps import clips as clips_step  # noqa: E402

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def test_the_longer_takes_can_be_left_out():
    every = creature_clips.plan("digitigrade", chosen=False, legs=2, installed=True)
    plain = creature_clips.plan("digitigrade", chosen=False, legs=2, installed=True, variations=False)
    assert any(motion.endswith(cl.LONGER) for motion, _ in every)
    assert not any(motion.endswith(cl.LONGER) for motion, _ in plain)
    assert {m for m, _ in plain} == {m for m, _ in every if not m.endswith(cl.LONGER)}


def test_the_library_can_be_left_out():
    assert creature_clips.plan("digitigrade", chosen=False, legs=2, installed=True, library=False) == [
        ("walk", None), ("attack", None)]


def test_a_varied_take_records_the_factor_and_seed_it_was_made_with():
    record = cl.motion_record("kaiju", "walk" + cl.LONGER, factor=3.0, seed=7)
    assert "3x as long, seed 7" in record["varied_with"]


def test_the_clips_command_line_takes_the_switches():
    args = creature_clips.build_parser().parse_args(
        ["--rig", "w.glb", "--no-library", "--no-variations", "--variation-factor", "3", "--variation-seed", "7"])
    assert args.library is False and args.variations is False
    assert args.variation_factor == 3.0 and args.variation_seed == 7
    defaults = creature_clips.build_parser().parse_args(["--rig", "w.glb"])
    assert defaults.library and defaults.variations
    assert (defaults.variation_factor, defaults.variation_seed) == (cl.LONGER_FACTOR, cl.LONGER_SEED)


def test_rigging_builds_the_clip_command_with_the_switches():
    command, _ = rigging.clip_command(Path("py"), Path("w_rigged.glb"), "auto", library=False, variations=True,
                                      variation_factor=3.0, variation_seed=7)
    assert "--no-library" in command and "--no-variations" not in command
    assert command[command.index("--variation-factor") + 1] == "3" and command[command.index("--variation-seed") + 1] == "7"
    plain, _ = rigging.clip_command(Path("py"), Path("w_rigged.glb"))
    assert not {"--no-library", "--no-variations", "--variation-factor"} & set(plain)


def test_chain_retarget_has_a_command_line_that_lists_the_library(capsys):
    assert chain_retarget.main(["--list"]) == 0
    listed = capsys.readouterr().out
    assert "kaiju" in listed and "walk" in listed
    args = chain_retarget.build_parser().parse_args(["--rig", "w_rigged.glb", "--source", "kaiju", "--clip", "walk",
                                                     "--longer", "--factor", "3", "--seed", "7"])
    assert args.source == "kaiju" and args.longer and args.factor == 3.0 and args.seed == 7


def test_chain_retarget_refuses_an_unknown_clip_before_blender(capsys):
    assert chain_retarget.main(["--rig", "w_rigged.glb", "--source", "kaiju", "--clip", "moonwalk"]) == 1
    assert "moonwalk" in capsys.readouterr().err


# ------------------------------------------------------------------------ studio

def test_the_studio_rig_can_skip_the_clips_or_their_longer_takes():
    import autorig_api

    with_clips = autorig_api.rig_commands(Path("s.glb"), Path("r.glb"), "auto")
    assert any("creature_clips.py" in " ".join(c) for c, _ in with_clips)
    without = autorig_api.rig_commands(Path("s.glb"), Path("r.glb"), "auto", clips=False)
    (family,) = [c for c, _ in without if "creature_clips.py" in " ".join(c)]   # the family file, no clips
    assert "--no-clips" in family
    short = autorig_api.rig_commands(Path("s.glb"), Path("r.glb"), "auto", variations=False)
    assert any("--no-variations" in c for c, _ in short)


def test_without_clips_only_the_family_is_planned():
    assert creature_clips.plan("digitigrade", True, 4, installed=True, clips=False) == []
    assert creature_clips.build_parser().parse_args(["--rig", "r.glb", "--no-clips"]).clips is False
    command, _ = rigging.clip_command(Path("py"), Path("r.glb"), clips=False)
    assert "--no-clips" in command


@needs_node
def test_the_rig_step_reads_its_switches_before_the_card_changes():
    """The progress card replaces the controls as soon as a step starts: the switches are
    read with the other settings, before it."""
    url = (REPO / "viewer" / "studio" / "steps.js").as_uri()
    program = (f"import * as m from {json.dumps(url)};\n"
               "const boxes = { clips: { checked: false }, variations: { checked: false } };\n"
               "const card = { querySelector: (q) => { const k = /data-k=\"(\\w+)\"/.exec(q); return k ? boxes[k[1]] || null : null; } };\n"
               "const s = m.readSettings(card, { engines: [] });\n"
               "console.log(JSON.stringify({ clips: s.clips, variations: s.variations }));")
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True,
                            text=True, encoding="utf-8")
    assert json.loads(result.stdout) == {"clips": False, "variations": False}


@needs_node
def test_the_rig_step_offers_the_switches():
    url = (REPO / "viewer" / "studio" / "creature-family.js").as_uri()
    result = subprocess.run([NODE, "--input-type=module", "--eval",
                             f"import * as m from {json.dumps(url)};\nconsole.log(JSON.stringify(m.familyPicker()));"],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    html = json.loads(result.stdout)
    assert 'data-k="clips" checked' in html and 'data-k="variations" checked' in html


# ------------------------------------------------------------------------- batch

@pytest.fixture
def fake_clips(monkeypatch):
    """creature_clips.py as a stand-in: two clips and a family file beside the rig."""
    seen = []

    def run_commands(commands, log_path, **kwargs):
        (command, _), = commands
        seen.append(command)
        rigged = Path(command[command.index("--rig") + 1])
        stem = rigged.stem.removesuffix("_rigged")
        for name in ("digitigrade-walk", "digitigrade-idle"):
            tiny_glb(rigged.with_name(f"{stem}_{name}.glb"), name)
        rigged.with_name(rigged.stem + ".family.json").write_text(json.dumps(
            {"schema_version": 1, "family": "digitigrade", "clips": ["digitigrade-walk", "digitigrade-idle"],
             "clips_error": None}))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n")
        return 0

    monkeypatch.setattr(rigging, "run_commands", run_commands)
    monkeypatch.setattr(rigging, "installed", lambda root=None: True)
    return seen


def test_the_clips_step_writes_its_own_folder_and_hands_the_clips_on(tmp_path, monkeypatch, fake_clips):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    exported = []

    def export_step(ctx):
        from image_to_3dlab.batch_steps import StepResult
        exported.append((ctx.asset.name, [c.name for c in ctx.clips]))
        return StepResult()

    monkeypatch.setitem(batch.STEPS, "export_unreal", fake_step(export_step))

    def rig_step(ctx):
        from test_rigging import skinned

        from image_to_3dlab.batch_steps import StepResult
        rigged = skinned(ctx.step_dir / f"{ctx.row.id}_rigged.glb")
        sidecar = rigged.with_suffix(".provenance.json")
        sidecar.write_text(json.dumps({"license": {"name": "MIT"}}))
        return StepResult(asset=rigged, sidecar=sidecar, files=[rigged, sidecar])

    monkeypatch.setitem(batch.STEPS, "rig", fake_step(rig_step))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "rig", "clips", "export_unreal"])
    spec = json.loads(spec_path.read_text())
    spec["defaults"].update(kind="creature", size_cm=200, clips={"family": "digitigrade", "variations": False})
    spec_path.write_text(json.dumps(spec))
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0, report.failed
    (command,) = fake_clips
    assert command[command.index("--family") + 1] == "digitigrade" and "--no-variations" in command
    assert exported == [("SM_row01_rigged.glb", ["SM_row01_digitigrade-walk.glb", "SM_row01_digitigrade-idle.glb"])]
    folder = spec_path.parent / "out" / "test-batch" / "rows" / "SM_row01" / "clips"
    assert (folder / "SM_row01_rigged.family.json").is_file()   # beside the asset the export reads
    record = json.loads((folder / "SM_row01_digitigrade-walk.provenance.json").read_text())
    assert record["license"] == {"name": "MIT"}
    assert json.loads((folder / "SM_row01_digitigrade-walk.provenance.json").read_text())["motion"]["clip"]


def test_animate_after_clips_keeps_the_family_clips(tmp_path, monkeypatch, fake_clips):
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    exported = []

    def export_step(ctx):
        from image_to_3dlab.batch_steps import StepResult
        exported.append(sorted(c.name for c in ctx.clips))
        return StepResult()

    def rig_step(ctx):
        from test_rigging import skinned

        from image_to_3dlab.batch_steps import StepResult
        rigged = skinned(ctx.step_dir / f"{ctx.row.id}_rigged.glb")
        sidecar = rigged.with_suffix(".provenance.json")
        sidecar.write_text(json.dumps({"license": {"name": "MIT"}}))
        return StepResult(asset=rigged, sidecar=sidecar, files=[rigged, sidecar])

    def animate_model(rigged, presets, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        made = []
        for preset in presets:
            clip = tiny_glb(folder / f"{rigged.stem}_{preset}.glb", preset)
            clip.with_suffix(".provenance.json").write_text("{}")
            made.append(clip)
        (folder / "animate.log").write_text("ok\n")
        return made

    monkeypatch.setattr(rigging, "animate_model", animate_model)
    monkeypatch.setitem(batch.STEPS, "export_unreal", fake_step(export_step))
    monkeypatch.setitem(batch.STEPS, "rig", fake_step(rig_step))
    spec_path = write_spec(tmp_path, rows=1, steps=["generate", "finish", "rig", "clips", "animate", "export_unreal"])
    spec = json.loads(spec_path.read_text())
    spec["defaults"].update(kind="creature", size_cm=200, animate={"presets": ["wave"]})
    spec_path.write_text(json.dumps(spec))
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0, report.failed
    assert exported == [["SM_row01_digitigrade-idle.glb", "SM_row01_digitigrade-walk.glb", "SM_row01_rigged_wave.glb"]]


def test_a_clips_row_that_is_not_a_creature_is_told(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(rigging, "installed", lambda root=None: True)
    row = SimpleNamespace(id="r", kind="object", steps=["rig", "clips", "export_unreal"], options={})
    (problem,) = clips_step.preflight([row])
    assert '"kind": "creature"' in problem
    assert clips_step.preflight([SimpleNamespace(**{**vars(row), "kind": "creature"})]) == []


def test_the_clips_step_is_described_and_checked():
    spec = step_catalog.get("clips")
    assert spec.feature == 6 and spec.cli == "scripts/creature_clips.py"
    assert step_catalog.check("clips", {"family": "wizard"})[0].startswith("family must be one of")
    assert step_catalog.check("clips", {"variation_factor": 50})[0].startswith("variation_factor must be")


def test_every_clips_setting_is_a_flag_of_the_clips_script():
    flags = {flag for action in creature_clips.build_parser()._actions for flag in action.option_strings}
    described = {param.cli_flag for param in step_catalog.get("clips").params if param.cli_flag}
    assert described <= flags
    assert flags - described - {"-h", "--help", "--rig", "--blend"} == set()


def test_the_catalogues_families_are_the_creature_families():
    from image_to_3dlab import creature_family

    assert step_catalog.get("clips").param("family").choices == ("auto", *creature_family.FAMILIES)
