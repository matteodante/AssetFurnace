"""Giving a model a skeleton and playing preset clips on it: the commands and records
shared by the studio's Animate tab (`viewer/autorig_api.py`), `scripts/autorig.py` and the
batch steps `rig` and `animate`.

* **rig**: SkinTokens (`vendor/SkinTokens/demo.py`) fits a skeleton and skin weights to a
  T-posed model (about a minute, ~8 GB of GPU memory), then `scripts/clean_skin_weights.py`
  takes fingertips off the toes. The creature family's clips (`scripts/creature_clips.py`)
  are a step of their own, `clip_command`.
* **animate**: `scripts/kimodo_retarget.py` plays a preset clip (pre-made with Kimodo,
  shipped in `image_to_3dlab/motions/`) on the skeleton. About ten seconds, CPU only, in
  SkinTokens' environment, which carries Blender's `bpy`.

Nothing here downloads: SkinTokens runs with `HF_HUB_OFFLINE=1`, and a missing install is
named for the Setup page.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from image_to_3dlab import creature_family, gpu_lock, processes

REPO = Path(__file__).resolve().parents[1]
SKINTOKENS = REPO / "vendor" / "SkinTokens"
RETARGET = REPO / "scripts" / "kimodo_retarget.py"
CLEAN_WEIGHTS = REPO / "scripts" / "clean_skin_weights.py"
CLIPS_SCRIPT = REPO / "scripts" / "creature_clips.py"
MOTIONS = REPO / "image_to_3dlab" / "motions"
PRESET_ID = re.compile(r"^[a-z0-9_]{1,40}$")
MAX_ARM_SPREAD = 30.0
SPEED_RANGE = (0.5, 2.0)
AUTO = "auto"
FAMILY_CHOICES = (AUTO, *creature_family.FAMILIES)
# What a run of SkinTokens is told, so it never fetches weights on its own.
ENV = {"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"}

SKINTOKENS_LICENSE = {
    "name": "MIT",
    "url": "https://github.com/VAST-AI-Research/SkinTokens/blob/main/LICENSE",
    "model": "VAST-AI/SkinTokens",
}
KIMODO_LICENSE = {
    "name": "NVIDIA Open Model License",
    "url": "https://www.nvidia.com/en-us/agreements/enterprise-software/"
           "nvidia-open-model-license/",
    "model": "nvidia/Kimodo-SOMA-RP-v1.1",
    "note": "Clip pre-generated with Kimodo. NVIDIA claims no ownership of outputs.",
}


def venv_python(root: Path = SKINTOKENS) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def installed(root: Path = SKINTOKENS) -> bool:
    return venv_python(root).exists() and (root / "demo.py").is_file()


MISSING = ("the auto-rig needs SkinTokens in vendor/SkinTokens with its environment; the "
           "studio's Setup page installs it after asking")


def presets(root: Path = MOTIONS) -> list[dict[str, Any]]:
    """The shipped clips, in the order `presets.json` lists them, that exist on disk."""
    index = _read_json(root / "presets.json") or {}
    return [dict(p) for p in index.get("presets", [])
            if PRESET_ID.fullmatch(str(p.get("id", ""))) and (root / f"{p['id']}.npz").is_file()]


def preset_ids(root: Path = MOTIONS) -> tuple[str, ...]:
    return tuple(p["id"] for p in presets(root))


# ------------------------------------------------------------------------- commands

def rig_commands(source: Path, result: Path) -> list[tuple[list[str], Path]]:
    """SkinTokens, then a skin-weight clean-up: it sometimes weights fingertips to a toe,
    which stretches them into spikes once the hand moves (scripts/clean_skin_weights.py)."""
    return [
        ([str(venv_python()), "demo.py", "--input", str(source), "--output", str(result),
          "--use_transfer"], SKINTOKENS),
        ([str(venv_python()), str(CLEAN_WEIGHTS), "--in", str(result), "--out", str(result)], REPO),
    ]


def parse_family(value: Any) -> str:
    """The family a person chose for a rig, or "auto". Anything else is refused."""
    family = str(value or AUTO).strip().lower()
    if family not in FAMILY_CHOICES:
        raise ValueError(f"unknown family {family!r}; choose one of {', '.join(FAMILY_CHOICES)}")
    return family


def clip_command(python: Path, rigged: Path, family: str = AUTO, *, library: bool = True,
                 variations: bool = True, variation_factor: float | None = None,
                 variation_seed: int | None = None) -> tuple[list[str], Path]:
    """The step after rigging: recognise the family, write the test clips (the CC0
    library's, then procedural ones; `library=False` only the procedural; `variations=False`
    no longer GenMM takes)."""
    command = [str(python), str(CLIPS_SCRIPT), "--rig", str(rigged), "--family", parse_family(family)]
    if not library:
        command.append("--no-library")
    if not variations:
        command.append("--no-variations")
    if variation_factor is not None:
        command += ["--variation-factor", f"{variation_factor:g}"]
    if variation_seed is not None:
        command += ["--variation-seed", str(int(variation_seed))]
    return command, REPO


def animate_command(rigged: Path, motion: Path, result: Path, arm_spread: float,
                    speed: float = 1.0) -> list[str]:
    return [str(venv_python()), str(RETARGET), "--rig", str(rigged), "--motion", str(motion),
            "--out", str(result), "--arm-spread", f"{arm_spread:g}", "--speed", f"{speed:g}"]


def clamp_speed(value: Any, preset: dict[str, Any]) -> float:
    """The playback speed: the caller's, else the preset's own default, else 1.

    Kimodo's strikes are slow next to real ones, so kick and punch presets ship faster.
    """
    if value is None or value == "":
        value = preset.get("speed", 1.0)
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("speed must be a number") from exc
    low, high = SPEED_RANGE
    if not low <= speed <= high:
        raise ValueError(f"speed must be between {low:g} and {high:g}")
    return speed


def clamp_spread(value: Any) -> float:
    try:
        spread = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("arm_spread must be a number of degrees") from exc
    if not 0 <= spread <= MAX_ARM_SPREAD:
        raise ValueError(f"arm_spread must be between 0 and {MAX_ARM_SPREAD:g} degrees")
    return spread


# ------------------------------------------------------------------------- records

def rig_record(name: str, data: bytes, source: dict[str, Any] | None, origin: str,
               tool: str = "viewer Animate tab: SkinTokens auto-rig") -> dict[str, Any]:
    """Provenance for a rigged model: the source's licence carries over, and the rigger's
    is listed beside it rather than mixed in."""
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": tool,
        "input": {"name": name, "sha256": hashlib.sha256(data).hexdigest(),
                  "source": origin},
        "license": (source or {}).get("license"),
        "classification": ((source or {}).get("output") or {}).get("classification")
        or (source or {}).get("classification"),
        "rig": {"tool": "SkinTokens", "license": SKINTOKENS_LICENSE},
        "source_record": source,
    }


def animation_record(rig: dict[str, Any] | None, preset: dict[str, Any], arm_spread: float,
                     tool: str = "viewer Animate tab: scripts/kimodo_retarget.py") -> dict[str, Any]:
    record = dict(rig or {})
    record.update({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": tool,
        "motion": {"preset": preset["id"], "label": preset.get("label"),
                   "made_with": "Kimodo", "license": KIMODO_LICENSE,
                   "arm_spread_degrees": arm_spread},
    })
    return record


def clip_files(rigged: Path) -> list[Path]:
    """The test clips the clip step wrote for this rig, as its family file lists them."""
    found = creature_family.read_family(rigged) or {}
    stem = rigged.stem.removesuffix("_rigged")
    return [p for p in (rigged.with_name(f"{stem}_{name}.glb") for name in found.get("clips") or []) if p.is_file()]


def record_clips(rigged: Path, rig_record: dict[str, Any],
                 tool: str = "viewer Animate tab: scripts/creature_clips.py") -> list[Path]:
    """A provenance sidecar per test clip: the rig's, plus how the motion was made."""
    found = creature_family.read_family(rigged) or {}
    written = []
    for clip in clip_files(rigged):
        name = clip.stem.removeprefix(rigged.stem.removesuffix("_rigged") + "_")
        record = creature_family.clip_record(rig_record, name, found.get("family"), tool=tool, output=clip,
                                             made=(found.get("clip_sources") or {}).get(name))
        path = clip.with_suffix(".provenance.json")
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        written.append(path)
    return written


def source_record_for(glb: Path) -> dict[str, Any] | None:
    """A licence record kept beside a model: a provenance sidecar, a run manifest, or a
    finish record. The first that names a licence wins."""
    for sidecar in (glb.with_suffix(".provenance.json"), glb.with_suffix(".json"),
                    glb.with_suffix(".retopo-repaint.json")):
        record = _read_json(sidecar)
        if isinstance(record, dict) and record.get("license"):
            return record
    return None


# ------------------------------------------------------------------------- running

def run_commands(commands: list[tuple[list[str], Path]], log_path: Path, *, gpu: set[int] | None = None,
                 session: str = "rig", popen: Callable[..., Any] = subprocess.Popen,
                 lock: Path | None = None) -> int:
    """Run the commands in order to the end, their output into `log_path`; the exit code
    of the first that fails, else 0. The commands whose index is in `gpu` hold the shared
    GPU lock while they run, and only them. Interrupted, the running command and what it
    started are stopped before the lock is let go."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        for index, (command, cwd) in enumerate(commands):
            log.write(f"$ {' '.join(command)}\n")
            log.flush()

            def run(command: list[str] = command, cwd: Path = cwd) -> int:
                process = popen(command, cwd=str(cwd), env={**os.environ, **ENV}, stdout=log,
                                stderr=subprocess.STDOUT, **processes.group_popen_kwargs())
                try:
                    return process.wait()
                except BaseException:
                    # Stopped (Ctrl+C, a cancelled batch): the child is in its own group and
                    # would go on using the GPU after the lock is let go. Stop it first.
                    processes.terminate_group(process.pid)
                    try:
                        process.wait(timeout=30)
                    except Exception:  # noqa: BLE001 - already stopping; the interrupt goes on
                        pass
                    raise

            if index in (gpu or set()):
                with gpu_lock.gpu(session, path=lock or gpu_lock.LOCK):
                    code = run()
            else:
                code = run()
            if code != 0:
                return code
    return 0


def rig_model(source: Path, out_dir: Path, name: str, *, origin: str, tool: str,
              log: Callable[[str], None] = lambda line: None, session: str = "rig") -> tuple[Path, Path]:
    """Rig one model into `out_dir`: `<name>_rigged.glb` and its `.provenance.json` (the
    source's licence, SkinTokens' beside it), with `rig.log`. The source is copied in
    first, so the run never depends on it. RuntimeError with the log's tail on failure."""
    if not installed():
        raise RuntimeError(MISSING)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy = out_dir / "input" / "source.glb"
    copy.parent.mkdir(exist_ok=True)
    data = Path(source).read_bytes()
    copy.write_bytes(data)
    result = out_dir / f"{name}_rigged.glb"
    log_path = out_dir / "rig.log"
    log(f"rig: SkinTokens on {Path(source).name}")
    code = run_commands(rig_commands(copy, result), log_path, gpu={0}, session=session)
    if code != 0 or not result.is_file():
        raise RuntimeError(f"the auto-rig {'exited with code ' + str(code) if code else 'wrote no GLB'}:\n"
                           f"{log_tail(log_path)}")
    record = rig_record(Path(source).name, data, source_record_for(Path(source)), origin, tool=tool)
    sidecar = result.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return result, sidecar


def animate_model(rigged: Path, preset_ids_: list[str], out_dir: Path, *, speed: float | None = None,
                  arm_spread: float = 0.0, tool: str, log: Callable[[str], None] = lambda line: None,
                  motions: Path = MOTIONS) -> list[Path]:
    """Play preset clips on a rigged model: `<stem>_<preset>.glb` each, with a
    `.provenance.json` (the rig's record plus the motion's licence), into `out_dir`."""
    if not installed():
        raise RuntimeError(MISSING)
    known = {preset["id"]: preset for preset in presets(motions)}
    unknown = [preset for preset in preset_ids_ if preset not in known]
    if unknown:
        raise ValueError(f"no such preset: {', '.join(unknown)}; presets are {', '.join(known)}")
    if not has_skin(rigged):
        raise ValueError(f"{Path(rigged).name} has no skeleton: rig it first")
    spread = clamp_spread(arm_spread)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(rigged).stem.removesuffix("_rigged")
    rig = _read_json(Path(rigged).with_suffix(".provenance.json"))
    made = []
    for preset_id in preset_ids_:
        preset = known[preset_id]
        pace = clamp_speed(speed, preset)
        result = out_dir / f"{stem}_{preset_id}.glb"
        log_path = out_dir / "animate.log"
        log(f"animate: {preset_id}")
        code = run_commands([(animate_command(Path(rigged), motions / f"{preset_id}.npz", result, spread, pace),
                              REPO)], log_path)
        if code != 0 or not result.is_file():
            raise RuntimeError(f"{preset_id}: the retarget {'exited with code ' + str(code) if code else 'wrote no GLB'}"
                               f":\n{log_tail(log_path)}")
        record = animation_record(rig, preset, spread, tool=tool)
        record["motion"]["speed"] = pace
        result.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        made.append(result)
    return made


def has_skin(glb: Path) -> bool:
    """Whether a GLB carries a skeleton (a glTF skin)."""
    from image_to_3dlab.glb_turn import parse_glb

    try:
        document, _ = parse_glb(Path(glb).read_bytes())
    except (OSError, ValueError, KeyError):
        return False
    return bool(document.get("skins"))


def log_tail(log_path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
