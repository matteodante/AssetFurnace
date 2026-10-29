"""`scripts/creature_rig.py`: its arguments, the UniRig run it stages, the read step's second
try, and the five steps end to end with Blender and the skeleton step stood in for."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import creature_rig as cli
import pytest


def test_arguments_and_their_defaults(tmp_path):
    args = cli.parse_args(["dragon.glb", str(tmp_path / "out" / "dragon_rigged.glb")])
    assert (args.draws, args.seed, args.turn_deg) == (4, 12345, None)
    assert args.work == tmp_path / "out" / "dragon_rigged.rig-work"
    args = cli.parse_args(["dragon.fbx", "r.glb", "--draws", "6", "--turn-deg", "-20", "--work", "W"])
    assert (args.draws, args.turn_deg, args.work) == (6, -20.0, Path("W"))
    assert cli.joints_path(Path("out/dragon.glb")) == Path("out/dragon.joints.json")


@pytest.mark.parametrize("argv", [["dragon.glb", "r.glb", "--draws", "0"], ["dragon.usd", "r.glb"],
                                  ["dragon.glb", "r.fbx"]])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


def fake_unirig(folder: Path) -> Path:
    (folder / ".venv" / "Scripts").mkdir(parents=True)
    (folder / ".venv" / "Scripts" / "python.exe").write_text("")
    (folder / "run.py").write_text("")
    return folder


def test_unirig_is_found_only_when_installed(tmp_path):
    assert cli.find_unirig({}, tmp_path / "none") is None
    home = fake_unirig(tmp_path / "UniRig")
    assert cli.find_unirig({}, home) == home
    assert cli.unirig_python(home) == home / ".venv" / "Scripts" / "python.exe"
    assert cli.find_unirig({cli.UNIRIG_ENV: str(home)}, tmp_path / "none") == home
    with pytest.raises(cli.RigError, match="not a UniRig install"):
        cli.find_unirig({cli.UNIRIG_ENV: str(tmp_path / "elsewhere")}, home)


def test_unirig_never_downloads_during_a_rig():
    env = cli.unirig_env({"PATH": "x"})
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1" and env["PATH"] == "x"


def test_the_unirig_commands(tmp_path):
    py = Path("py.exe")
    extract = cli.extract_command(py, tmp_path, "stamp")
    assert extract[:3] == ["py.exe", "-m", "src.data.extract"]
    assert f"--input_dir={tmp_path / 'src'}" in extract and f"--output_dir={tmp_path / 'npz'}" in extract
    assert "--faces_target_count=50000" in extract and "--time=stamp" in extract
    draw = cli.skeleton_command(py, tmp_path, 7)
    assert draw[:2] == ["py.exe", "run.py"] and "--seed=7" in draw
    assert f"--task={cli.UNIRIG_TASK}" in draw
    assert {f"--input_dir={tmp_path / 'in'}", f"--output_dir={tmp_path / 'out'}",
            f"--npz_dir={tmp_path / 'npz'}"} <= set(draw)


def test_one_run_draws_as_many_skeletons_as_copies(tmp_path):
    """The model is read once, then copied under N names: UniRig draws each copy afresh."""
    unirig = fake_unirig(tmp_path / "UniRig")
    model = tmp_path / "unirig_input.glb"
    model.write_bytes(b"glb")
    calls = []

    def run(cmd, cwd, env, stdout, stderr, check):
        calls.append((cmd, cwd, env))
        args = dict(a.split("=", 1) for a in cmd if a.startswith("--") and "=" in a)
        if "src.data.extract" in cmd:
            (Path(args["--output_dir"]) / "model").mkdir(parents=True)
            (Path(args["--output_dir"]) / "model" / "raw_data.npz").write_bytes(b"npz")
        else:
            for glb in sorted(Path(args["--input_dir"]).glob("*.glb"))[:-1]:   # one draw fails
                assert (Path(args["--npz_dir"]) / glb.stem / "raw_data.npz").is_file()
                (Path(args["--output_dir"]) / glb.stem).mkdir(parents=True)
                (Path(args["--output_dir"]) / glb.stem / "skeleton.fbx").write_bytes(b"fbx")
        return SimpleNamespace(returncode=0)

    said = []
    files = cli.unirig_draws(model, tmp_path / "W", 3, 12345, unirig, run=run, log=said.append)
    assert [f.parent.name for f in files] == ["draw0", "draw1"]
    assert len(calls) == 2 and all(cwd == str(unirig) and env["HF_HUB_OFFLINE"] == "1" for _, cwd, env in calls)
    assert "--seed=12345" in calls[1][0]
    assert any("2 of 3" in s for s in said)


def test_a_model_unirig_cannot_read_gives_no_draws(tmp_path):
    unirig = fake_unirig(tmp_path / "UniRig")
    model = tmp_path / "m.glb"
    model.write_bytes(b"glb")
    said = []
    files = cli.unirig_draws(model, tmp_path / "W", 2, 1, unirig, run=lambda *a, **k: None, log=said.append)
    assert files == [] and "could not read" in said[0]


def test_a_skeleton_blender_cannot_read_is_read_by_unirigs_python(tmp_path, monkeypatch):
    work = tmp_path / "W"
    work.mkdir()
    fbx = [tmp_path / f"draw{k}" / "skeleton.fbx" for k in range(3)]
    commands = []

    def stream(cmd, log=print, cwd=None, env=None):
        commands.append(cmd)
        if cmd[0] == "blender.exe":
            (work / "draws" / "draw0.json").write_text("{}")      # Blender stopped after the first
        elif cmd[-1] == "1":
            (work / "draws" / "draw1.json").write_text("{}")      # the second try reads draw 1 only
        return 0, []

    monkeypatch.setattr(cli, "stream", stream)
    said = []
    got = cli.read_draws(Path("blender.exe"), work, fbx, Path("unirig-python.exe"), said.append)
    assert [p.name for p in got] == ["draw0.json", "draw1.json"]
    assert commands[1] == ["unirig-python.exe", str(cli.READ_SCRIPT), str(work), str(fbx[1]), "--start", "1"]
    assert commands[2][-1] == "2" and any("could not be read" in s for s in said)


def test_the_blender_command_and_its_report():
    cmd = cli.blender_command(Path("b.exe"), Path("s.py"), [Path("W"), "--turn-deg", 20.0])
    assert cmd == ["b.exe", "-b", "--factory-startup", "--python", "s.py", "--", "W", "--turn-deg", "20.0"]
    assert cli.report_line(["noise", 'TAG::{"a": 1}', "Blender quit"], "TAG") == {"a": 1}
    assert cli.report_line(["noise"], "TAG") is None


def test_a_failing_blender_step_says_why(monkeypatch):
    monkeypatch.setattr(cli, "stream", lambda cmd, log=print: (1, ["Traceback", "KeyError: 'Head'"]))
    with pytest.raises(cli.RigError, match="KeyError"):
        cli.run_script(Path("b.exe"), cli.BUILD_SCRIPT, ["W"], "I2L_CREATURE_BUILD")


def test_the_streamed_lines_a_job_log_shows(tmp_path):
    import sys

    said = []
    code, lines = cli.stream([sys.executable, "-c", "print('creature-rig x: 1'); print('other')"], said.append)
    assert code == 0 and lines == ["creature-rig x: 1", "other"] and said == ["creature-rig x: 1"]


class StandIn:
    """The skeleton step, as image_to_3dlab.creature_rig.pipeline promises it."""

    def __init__(self):
        self.calls = []

    def _write(self, work, body, chosen):
        spec = {"body": body, "height": 1.0, "facing": "-Y", "chosen": chosen, "notes": ["a note"],
                "bones": [{"name": "Root", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 0.1], "kind": "root",
                           "side": "", "deform": False}], "draws": []}
        (Path(work) / "joints.json").write_text(json.dumps(spec))
        return spec

    def joints_from_draws(self, work, draws):
        self.calls.append(("draws", list(draws)))
        return self._write(work, "quadruped", 0)

    def joints_without_unirig(self, work):
        self.calls.append(("centreline",))
        return self._write(work, "centreline", None)


def fake_blender(monkeypatch, tmp_path):
    blender = tmp_path / "blender.exe"
    blender.write_text("")
    ran = []

    def run_script(blender_, script, args, tag, log=print):
        ran.append((script.name, [str(a) for a in args]))
        work = Path(args[0]) if script != cli.PREPARE_SCRIPT else Path(args[1])
        if script == cli.PREPARE_SCRIPT:
            (work / "unirig_input.glb").write_bytes(b"glb")
            return {"turn_deg": 21.0, "front_cue": "toes"}
        (work / "rigged.glb").write_bytes(b"rigged")
        return {"empty_bones": [], "voxel_copy": False}

    monkeypatch.setattr(cli, "run_script", run_script)
    return blender, ran


def test_without_unirig_the_centre_line_skeleton_is_rigged(tmp_path, monkeypatch):
    blender, ran = fake_blender(monkeypatch, tmp_path)
    pipeline = StandIn()
    monkeypatch.setattr(cli, "pipeline_module", lambda: pipeline)
    monkeypatch.setattr(cli, "find_unirig", lambda: None)
    model = tmp_path / "dragon.glb"
    model.write_bytes(b"model")
    said = []
    summary = cli.main([str(model), str(tmp_path / "out" / "dragon_rig.glb"), "--blender", str(blender),
                        "--turn-deg", "20"], log=said.append)
    assert pipeline.calls == [("centreline",)]
    assert [name for name, _ in ran] == [cli.PREPARE_SCRIPT.name, cli.BUILD_SCRIPT.name]
    assert ran[0][1][2:] == ["--turn-deg", "20.0"]
    assert (tmp_path / "out" / "dragon_rig.glb").read_bytes() == b"rigged"
    assert json.loads((tmp_path / "out" / "dragon_rig.joints.json").read_text())["body"] == "centreline"
    assert summary["body"] == "centreline" and summary["draws"] == 0 and summary["turn_deg"] == 21.0
    assert any("UniRig is not installed" in s for s in said) and any("a note" in s for s in said)
    assert said[-1].startswith("I2L_CREATURE_RIG::")


def test_with_unirig_its_draws_are_read_and_chosen_from(tmp_path, monkeypatch):
    blender, _ = fake_blender(monkeypatch, tmp_path)
    pipeline = StandIn()
    monkeypatch.setattr(cli, "pipeline_module", lambda: pipeline)
    unirig = fake_unirig(tmp_path / "UniRig")
    work = tmp_path / "W"
    seen = {}

    def draws(model_glb, work_, count, seed, unirig_, log=print):
        seen.update(model=model_glb, count=count, seed=seed, unirig=unirig_)
        return [work_ / "unirig" / "out" / f"draw{k}" / "skeleton.fbx" for k in range(count)]

    def read(blender_, work_, skeletons, python, log=print):
        seen["python"] = python
        return [work_ / "draws" / f"draw{k}.json" for k in range(len(skeletons))]

    monkeypatch.setattr(cli, "unirig_draws", draws)
    monkeypatch.setattr(cli, "read_draws", read)
    model = tmp_path / "dragon.glb"
    model.write_bytes(b"model")
    summary = cli.main([str(model), str(tmp_path / "r.glb"), "--blender", str(blender), "--work", str(work),
                        "--unirig", str(unirig), "--draws", "6", "--seed", "3"], log=lambda s: None)
    assert seen["model"] == work / "unirig_input.glb" and (seen["count"], seen["seed"]) == (6, 3)
    assert seen["python"] == unirig / ".venv" / "Scripts" / "python.exe"
    assert pipeline.calls == [("draws", [work / "draws" / f"draw{k}.json" for k in range(6)])]
    assert summary["draws"] == 6 and summary["chosen"] == 0 and set(summary["seconds"]) == {
        "prepare", "draw", "read", "joints", "build"}


def test_a_missing_model_fails_before_blender_starts(tmp_path):
    with pytest.raises(cli.RigError, match="no model"):
        cli.main([str(tmp_path / "none.glb"), str(tmp_path / "r.glb")])


def test_the_command_line_help_runs():
    import sys

    done = subprocess.run([sys.executable, str(cli.SCRIPTS / "creature_rig.py"), "--help"], capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0 and "--draws" in done.stdout
