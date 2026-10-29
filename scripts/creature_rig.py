#!/usr/bin/env python3
"""Rig a generated creature automatically: UniRig draws skeletons, the best is checked on the mesh and named, Blender skins it.

python scripts/creature_rig.py MODEL.glb RESULT.glb [--draws 4] [--work DIR] [--turn-deg D] [--seed 12345]
    [--unirig DIR] [--blender EXE]

Five steps, all in one work folder (default: RESULT's name + `.rig-work`, beside it):

1. prepare (Blender): the model turned to face -Y, on the floor, and its watertight shell
   (`blender_creature_rig_prepare.py`);
2. draw (UniRig): several skeletons in one run, the same model under several names;
3. read (Blender): the skeleton files into draw files in the model's frame
   (`blender_creature_rig_read.py`; UniRig's own Python reads any file Blender cannot);
4. joints: each draw cleaned and scored, the best repaired and named by body plan
   (`image_to_3dlab.creature_rig.pipeline`); without UniRig, a skeleton traced from the
   body's centre line instead;
5. build (Blender): armature, bone heat skin, rigged GLB and a poses sheet
   (`blender_creature_rig_build.py`).

RESULT gets the rigged GLB and RESULT's `.joints.json` the skeleton the two-camera fit reads.
Each step prints `creature-rig:` lines for a job log; the last line is
`I2L_CREATURE_RIG::{json}`. UniRig runs offline: its weights come with its install
(scripts/bootstrap_unirig.py), never from a rig run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPTS = REPO / "scripts"
PREPARE_SCRIPT = SCRIPTS / "blender_creature_rig_prepare.py"
READ_SCRIPT = SCRIPTS / "blender_creature_rig_read.py"
BUILD_SCRIPT = SCRIPTS / "blender_creature_rig_build.py"
UNIRIG_ENV = "I2L_UNIRIG"
UNIRIG_HOME = REPO / "vendor" / "UniRig"
UNIRIG_TASK = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
UNIRIG_DATA = "configs/data/quick_inference.yaml"
UNIRIG_FACES = 50000        # UniRig simplifies the mesh to this many faces before drawing
MODEL_SUFFIXES = (".glb", ".gltf", ".fbx", ".obj", ".blend")
STEPS = 5


class RigError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="creature_rig.py", description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="the creature: .glb, .gltf, .fbx, .obj or .blend")
    ap.add_argument("result", type=Path, help="the rigged GLB to write")
    ap.add_argument("--draws", type=int, default=4, help="UniRig skeletons to choose from (6 when a leg "
                    "comes out wrong)")
    ap.add_argument("--work", type=Path, default=None, help="the work folder (default: beside RESULT)")
    ap.add_argument("--turn-deg", type=float, default=None, help="turn about Z that makes the creature "
                    "face -Y (default: found from its symmetry and its feet)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--unirig", type=Path, default=None, help=f"UniRig's folder (default: ${UNIRIG_ENV}, "
                    "then vendor/UniRig)")
    ap.add_argument("--blender", type=Path, default=None)
    ap.add_argument("--reuse", type=Path, default=None, help="an earlier run's work folder: its prepared model, "
                    "shell and UniRig draws are reused, so only the skeleton and the skin are made again")
    ap.add_argument("--draw", type=int, default=None, help="keep this draw (0..) rather than the best scored "
                    "one; the studio's \"Use draw N\"")
    args = ap.parse_args(argv)
    if args.draws < 1:
        ap.error("--draws must be 1 or more")
    if args.model.suffix.lower() not in MODEL_SUFFIXES:
        ap.error(f"cannot read {args.model.suffix or 'a file without a suffix'}: use one of "
                 + ", ".join(MODEL_SUFFIXES))
    if args.result.suffix.lower() != ".glb":
        ap.error("RESULT must be a .glb")
    if args.work is None:
        args.work = default_work(args.result)
    return args


def default_work(result: Path) -> Path:
    """`out/dragon.glb` -> `out/dragon.rig-work`."""
    return result.with_name(result.stem + ".rig-work")


def joints_path(result: Path) -> Path:
    """`out/dragon.glb` -> `out/dragon.joints.json`, the skeleton the fit reads."""
    return result.with_suffix(".joints.json")


# --- UniRig -----------------------------------------------------------------------------------


def unirig_python(folder: Path) -> Path | None:
    """UniRig's own Python (its venv), or None."""
    for p in (folder / ".venv" / "Scripts" / "python.exe", folder / ".venv" / "bin" / "python"):
        if p.is_file():
            return p
    return None


def find_unirig(env: dict | None = None, home: Path = UNIRIG_HOME) -> Path | None:
    """UniRig's folder when it is installed (run.py and its venv), else None. `$I2L_UNIRIG` wins;
    a wrong one is an error, not a quiet fall back to the centre line."""
    env = os.environ if env is None else env
    given = env.get(UNIRIG_ENV)
    if given:
        folder = Path(given).expanduser()
        if not (folder / "run.py").is_file() or unirig_python(folder) is None:
            raise RigError(f"{UNIRIG_ENV}={given} is not a UniRig install (run.py and .venv expected)")
        return folder
    if (home / "run.py").is_file() and unirig_python(home) is not None:
        return home
    return None


def unirig_env(base: dict | None = None) -> dict:
    """UniRig's environment: offline (weights come with the install, never with a run), UTF-8."""
    env = dict(os.environ if base is None else base)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return env


def extract_command(python: Path, run: Path, stamp: str) -> list[str]:
    """UniRig reads the model once (its mesh, simplified, into an npz)."""
    return [str(python), "-m", "src.data.extract", f"--config={UNIRIG_DATA}",
            "--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm", "--force_override=true", "--num_runs=1",
            "--id=0", f"--time={stamp}", f"--faces_target_count={UNIRIG_FACES}",
            f"--input_dir={run / 'src'}", f"--output_dir={run / 'npz'}"]


def skeleton_command(python: Path, run: Path, seed: int) -> list[str]:
    """UniRig draws a skeleton for every model in `in/`; one seed for the run, so each copy
    of the model gets a different draw."""
    return [str(python), "run.py", f"--task={UNIRIG_TASK}", f"--seed={seed}", f"--input_dir={run / 'in'}",
            f"--output_dir={run / 'out'}", f"--npz_dir={run / 'npz'}"]


def stage_draws(run: Path, count: int) -> None:
    """The model read once, then `count` copies under the names draw0.. (the GLB in `in/`, its
    npz folder beside it), so one UniRig run draws `count` skeletons."""
    read = run / "npz" / "model"
    if not (read / "raw_data.npz").is_file():
        raise RigError(f"UniRig could not read the model (see {run / 'extract.log'})")
    for k in range(count):
        shutil.copytree(read, run / "npz" / f"draw{k}", dirs_exist_ok=True)
        shutil.copy2(run / "src" / "model.glb", run / "in" / f"draw{k}.glb")


def skeleton_files(run: Path, count: int) -> list[Path]:
    """The skeleton files UniRig wrote, in draw order (a draw it failed on is missing)."""
    return [p for k in range(count) for p in [run / "out" / f"draw{k}" / "skeleton.fbx"] if p.is_file()]


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def unirig_draws(model_glb: Path, work: Path, count: int, seed: int, unirig: Path | None = None,
                 run: Callable = subprocess.run, log: Callable = print) -> list[Path]:
    """`count` skeletons drawn by UniRig for one model, from one run: the skeleton files made
    (none when UniRig failed; its logs are in work/unirig)."""
    unirig = unirig or find_unirig()
    if unirig is None:
        raise RigError("UniRig is not installed")
    python = unirig_python(unirig)
    if python is None:
        raise RigError(f"UniRig at {unirig} has no .venv")
    folder = Path(work) / "unirig"
    if folder.exists():
        shutil.rmtree(folder)
    for d in ("src", "in", "npz", "out"):
        (folder / d).mkdir(parents=True)
    shutil.copy2(model_glb, folder / "src" / "model.glb")
    env = unirig_env()
    with open(folder / "extract.log", "wb") as f:
        run(extract_command(python, folder, time.strftime("%Y_%m_%d_%H_%M_%S")), cwd=str(unirig), env=env,
            stdout=f, stderr=subprocess.STDOUT, check=False)
    try:
        stage_draws(folder, count)
    except RigError as exc:
        log(f"creature-rig: {exc}\n{_tail(folder / 'extract.log')}")
        return []
    with open(folder / "unirig.log", "wb") as f:
        done = run(skeleton_command(python, folder, seed), cwd=str(unirig), env=env, stdout=f,
                   stderr=subprocess.STDOUT, check=False)
    files = skeleton_files(folder, count)
    if len(files) < count:
        log(f"creature-rig: UniRig drew {len(files)} of {count} skeletons (exit {getattr(done, 'returncode', '?')}, "
            f"see {folder / 'unirig.log'})\n{_tail(folder / 'unirig.log')}")
    return files


# --- Blender ----------------------------------------------------------------------------------


def blender_command(blender: Path, script: Path, args: list) -> list[str]:
    return [str(blender), "-b", "--factory-startup", "--python", str(script), "--", *[str(a) for a in args]]


def report_line(lines: list[str], tag: str) -> dict | None:
    """The JSON a script printed after `tag::`, or None."""
    for line in reversed(lines):
        if line.startswith(tag + "::"):
            return json.loads(line.split("::", 1)[1])
    return None


def stream(cmd: list[str], log: Callable = print, cwd: Path | None = None, env: dict | None = None) -> tuple[int, list[str]]:
    """Run a command, passing its `creature-rig` lines on as they come; its exit code and output."""
    proc = subprocess.Popen(cmd, cwd=None if cwd is None else str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        lines.append(line)
        if line.startswith("creature-rig"):
            log(line)
    return proc.wait(), lines


def run_script(blender: Path, script: Path, args: list, tag: str, log: Callable = print) -> dict:
    code, lines = stream(blender_command(blender, script, args), log)
    report = report_line(lines, tag)
    if code != 0 or report is None:
        raise RigError(f"{script.name} failed (exit {code}):\n" + "\n".join(lines[-25:]))
    return report


def read_draws(blender: Path, work: Path, skeletons: list[Path], fallback_python: Path | None,
               log: Callable = print) -> list[Path]:
    """Draw files for the skeleton files, numbered in order. Blender reads them all; a file it
    could not read gets a second try with UniRig's Python (its `bpy` wrote the file); a file
    neither reads is left out."""
    folder = work / "draws"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    if not skeletons:
        return []
    stream(blender_command(blender, READ_SCRIPT, [work, *skeletons]), log)
    wanted = [folder / f"draw{k}.json" for k in range(len(skeletons))]
    for k, (fbx, out) in enumerate(zip(skeletons, wanted)):
        if out.is_file() or fallback_python is None:
            continue
        log(f"creature-rig: Blender could not read {fbx.parent.name}; trying UniRig's Python")
        stream([str(fallback_python), str(READ_SCRIPT), str(work), str(fbx), "--start", str(k)], log)
    got = [p for p in wanted if p.is_file()]
    if len(got) < len(wanted):
        log(f"creature-rig: {len(wanted) - len(got)} skeleton file(s) could not be read")
    return got


def pipeline_module():
    """The skeleton step's module (imported first thing, so a broken install fails in a second,
    not after the slow steps)."""
    from image_to_3dlab.creature_rig import pipeline

    return pipeline


def joints_step(pipeline, work: Path, draws: list[Path], prefer: int | None = None) -> dict:
    """The skeleton: the best UniRig draw (or the one asked for), cleaned, repaired and
    named; or the centre line."""
    if not draws:
        return pipeline.joints_without_unirig(work)
    return pipeline.joints_from_draws(work, draws, prefer) if prefer is not None else pipeline.joints_from_draws(work, draws)


REUSED = ("model.blend", "unirig_input.glb", "shell.npz")


def reuse_work(earlier: Path, work: Path) -> list[Path]:
    """Copy an earlier run's prepared model, shell and draws into `work`; returns the draws."""
    missing = [name for name in REUSED if not (earlier / name).is_file()]
    if missing:
        raise RigError(f"{earlier} has no {', '.join(missing)}: rig the model again instead")
    work.mkdir(parents=True, exist_ok=True)
    for name in REUSED:
        shutil.copy2(earlier / name, work / name)
    if (work / "draws").exists():
        shutil.rmtree(work / "draws")
    if (earlier / "draws").is_dir():
        shutil.copytree(earlier / "draws", work / "draws")
    return sorted((work / "draws").glob("draw*.json"), key=lambda p: int(p.stem[4:] or 0))


def find_blender_or_fail(explicit: Path | None) -> Path:
    from image_to_3dlab.blender import find_blender, missing_help

    path = explicit or find_blender()
    if path is None or not Path(path).is_file():
        raise RigError(missing_help())
    return Path(path)


def main(argv: list[str] | None = None, log: Callable = print) -> dict:
    args = parse_args(argv)
    model, result, work = args.model.resolve(), args.result.resolve(), args.work.resolve()
    if not model.is_file():
        raise RigError(f"no model at {model}")
    blender = find_blender_or_fail(args.blender)
    pipeline = pipeline_module()

    unirig = None if args.reuse is not None else (args.unirig.resolve() if args.unirig else find_unirig())
    work.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}

    def step(n: int, name: str, what: str):
        log(f"creature-rig: {n}/{STEPS} {name}: {what}")
        return time.time()

    draws: list[Path] = []
    skeletons: list[Path] = []
    prepared: dict = {}
    if args.reuse is not None:
        log(f"creature-rig: 1-3/{STEPS} reuse: the prepared model and the draws of {args.reuse}")
        draws = reuse_work(args.reuse.resolve(), work)
    else:
        t = step(1, "prepare", f"{model.name} turned, placed and sealed")
        prep_args = [model, work] + (["--turn-deg", args.turn_deg] if args.turn_deg is not None else [])
        prepared = run_script(blender, PREPARE_SCRIPT, prep_args, "I2L_CREATURE_PREPARE", log)
        times["prepare"] = time.time() - t
        if unirig is None:
            log(f"creature-rig: 2/{STEPS} draw: UniRig is not installed, the skeleton comes from the centre line")
        else:
            t = step(2, "draw", f"{args.draws} UniRig skeleton(s), seed {args.seed}")
            skeletons = unirig_draws(work / "unirig_input.glb", work, args.draws, args.seed, unirig, log=log)
            times["draw"] = time.time() - t
            log(f"creature-rig: 2/{STEPS} draw: {len(skeletons)} skeleton(s) in {times['draw']:.0f} s")

    if skeletons:
        t = step(3, "read", f"{len(skeletons)} skeleton file(s) into the model's frame")
        draws = read_draws(blender, work, skeletons, unirig_python(unirig), log)
        times["read"] = time.time() - t

    t = step(4, "joints", f"choosing from {len(draws)} draw(s)" if draws else "the centre-line skeleton")
    spec = joints_step(pipeline, work, draws, args.draw)
    times["joints"] = time.time() - t
    try:   # every draw from the side and the front, for a person to pick another
        from image_to_3dlab.creature_rig import sheet
        sheet.draws_sheet(work)
    except Exception as exc:
        log(f"creature-rig: the draws sheet could not be drawn ({exc})")
    log(f"creature-rig: 4/{STEPS} joints: {spec.get('body')} with {len(spec.get('bones', []))} bones"
        + (f", draw {spec['chosen']} chosen" if spec.get("chosen") is not None else ""))
    for note in spec.get("notes", []):
        log(f"creature-rig:   {note}")

    t = step(5, "build", "armature, skin weights, rigged GLB and poses sheet")
    built = run_script(blender, BUILD_SCRIPT, [work], "I2L_CREATURE_BUILD", log)
    times["build"] = time.time() - t

    result.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "rigged.glb", result)
    shutil.copy2(work / "joints.json", joints_path(result))
    summary = {"result": str(result), "joints": str(joints_path(result)), "work": str(work),
               "body": spec.get("body"), "bones": len(spec.get("bones", [])), "chosen": spec.get("chosen"),
               "draws": len(draws), "turn_deg": prepared.get("turn_deg"), "front_cue": prepared.get("front_cue"),
               "empty_bones": built.get("empty_bones", []), "voxel_copy": built.get("voxel_copy"),
               "sheet": str(work / "poses-sheet.png") if (work / "poses-sheet.png").is_file() else None,
               "draws_sheet": str(work / "draws-sheet.png") if (work / "draws-sheet.png").is_file() else None,
               "seconds": {k: round(v, 1) for k, v in times.items()}}
    log("I2L_CREATURE_RIG::" + json.dumps(summary))
    return summary


if __name__ == "__main__":
    try:
        main()
    except RigError as exc:
        print(f"creature-rig: failed: {exc}", file=sys.stderr)
        sys.exit(1)
