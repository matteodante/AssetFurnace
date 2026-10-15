"""Run a manifest through a registered generator: Pixal3D today.

`pipeline.py --run-manifest` handles SF3D, TRELLIS.2 and Hunyuan in `cli.py`; a backend
listed in `GENERATORS` comes here instead. Batches use the same function for every row, so
one row of a batch and a single run of that row's manifest are the same code.

A generator is registered with a readiness check that only looks at the disk. It runs
before the generator, and a missing install stops the run with the command to fix it:
nothing here downloads weights (AGENTS.md). That is also why SF3D and TRELLIS.2 are not
registered yet: both fetch weights from Hugging Face on first use.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .provenance import finalize_output, load_run_manifest, validate_run_policy

REPO = Path(__file__).resolve().parents[1]


class ManifestRunError(Exception):
    """A run that did not happen. `code` is the exit code: 2 for a bad manifest or a
    missing install, 1 for a generator that failed."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


@dataclass
class Generated:
    glb: Path
    parameters: dict[str, Any]   # what really ran, for the provenance record
    # Files the generator writes beside its GLB, by suffix. They move with it, so the
    # final GLB keeps them under its own name (`<stem>.json`, `<stem>.svviews/`).
    companions: dict[str, Path] = field(default_factory=dict)
    # Models the generator ran besides its own (a background remover), for provenance.
    components: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Generator:
    name: str
    run: Callable[[list[Path], Path, dict[str, Any], Callable[[str], None]], Generated]
    readiness: Callable[[], list[str]]   # what is missing; empty when ready
    parameters: frozenset[str]           # the manifest parameters it accepts
    max_images: int = 1
    repo: Path | None = None             # backend checkout, for its git revision
    # What these inputs would need beyond the install (a remover's weights); empty when fine.
    input_check: Callable[[list[Path], dict[str, Any]], list[str]] = lambda images, parameters: []
    # Wrong parameter values, found before the run rather than by the generator's argparse.
    check_parameters: Callable[[dict[str, Any]], list[str]] = lambda parameters: []


# ------------------------------------------------------------------------------ Pixal3D

PIXAL3D_SCRIPT = REPO / "scripts" / "pixal3d_generate.py"
PIXAL3D_ROOT = REPO / "vendor" / "pixal3d-cpp"
PIXAL3D_PARAMETERS = frozenset({"seed", "steps", "gss", "gsh", "fov", "matte"})
PIXAL3D_BOOTSTRAP = "python scripts/bootstrap_pixal3d.py (8.4 GB of weights; it asks first)"


def _pixal3d_module():
    spec = importlib.util.spec_from_file_location("pixal3d_generate", PIXAL3D_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pixal3d_state() -> dict[str, Any]:
    """`readiness()` from the wrapper itself, so "ready" has one definition."""
    return _pixal3d_module().readiness()


def _pixal3d_steps_problem(steps: int | None) -> str | None:
    """The wrapper's own `steps_problem`: an unpatched or stale build refuses `--steps`."""
    return _pixal3d_module().steps_problem(steps)


def pixal3d_readiness() -> list[str]:
    state = _pixal3d_state()
    if state.get("ready"):
        return []
    problem = (f"Pixal3D is not ready (built: {state.get('cli_built')}, weight files: "
               f"{state.get('weights_present')} of 9). Install it with {PIXAL3D_BOOTSTRAP}")
    return [problem]


def pixal3d_check_parameters(parameters: dict[str, Any]) -> list[str]:
    def number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    problems = []
    seed, steps = parameters.get("seed"), parameters.get("steps")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        problems.append(f"seed must be a whole number from 0, not {seed!r}")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int)
                              or not 1 <= steps <= 50):
        problems.append(f"steps must be a whole number from 1 to 50, not {steps!r}")
    for key in ("gss", "gsh"):
        value = parameters.get(key)
        if value is not None and not (number(value) and value > 0):
            problems.append(f"{key} must be a positive number, not {value!r}")
    fov = parameters.get("fov")
    if fov is not None and not (number(fov) and 0 < fov < 3.14159):
        problems.append(f"fov must be an angle in radians between 0 and pi, not {fov!r}")
    matte = parameters.get("matte")
    if matte is not None and not isinstance(matte, bool):
        problems.append(f"matte must be true, false or left out, not {matte!r}")
    return problems


def pixal3d_input_check(images: list[Path], parameters: dict[str, Any]) -> list[str]:
    """An image that is not cut out gets matted, and rembg downloads a missing remover on
    its own: that must be an explicit install, never a side effect of a run."""
    from PIL import Image, UnidentifiedImageError

    from .matte import is_matted, matte_model, model_file

    steps = parameters.get("steps")
    if isinstance(steps, int) and not isinstance(steps, bool):
        problem = _pixal3d_steps_problem(steps)
        if problem:
            return [problem]
    try:   # a broken picture fails here, not minutes into a batch
        with Image.open(images[0]) as opened:
            opened.load()
            matted = is_matted(opened)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return [f"{images[0].name} is not a readable picture: {exc}"]
    if parameters.get("matte") is False or model_file(matte_model()).is_file():
        return []
    if parameters.get("matte") is not True and matted:
        return []
    return [f"{images[0].name} needs its background removed and no remover is installed: "
            "install BiRefNet-lite with python scripts/bootstrap_matte.py (224 MB; it asks "
            'first), give a cut-out image, or set "matte": false']


def pixal3d_command(image: Path, output: Path, parameters: dict[str, Any]) -> list[str]:
    command = [sys.executable, "-u", str(PIXAL3D_SCRIPT), str(image), str(output)]
    for key in ("seed", "steps", "gss", "gsh", "fov"):
        if parameters.get(key) is not None:
            command += [f"--{key}", str(parameters[key])]
    if parameters.get("matte") is True:
        command.append("--matte")
    elif parameters.get("matte") is False:
        command.append("--no-matte")
    return command


def run_worker(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """The generator's subprocess, by a name a test can replace on its own."""
    return subprocess.run(command, **kwargs)


def run_pixal3d(images: list[Path], working: Path, parameters: dict[str, Any],
                log: Callable[[str], None]) -> Generated:
    """`scripts/pixal3d_generate.py` in a subprocess, as the studio runs it."""
    working.mkdir(parents=True, exist_ok=True)
    output = (working / f"{images[0].stem}.glb").resolve()
    log_path = working / "pixal3d.log"
    log(f"pixal3d: {images[0].name} -> {output.name} (log: {log_path})")
    with log_path.open("w", encoding="utf-8") as handle:
        process = run_worker(pixal3d_command(images[0], output, parameters),
                             stdout=handle, stderr=subprocess.STDOUT, text=True,
                                 check=False)
    if process.returncode != 0 or not output.is_file():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-12:])
        raise RuntimeError(f"pixal3d exited with code {process.returncode}:\n{tail}")
    record = output.with_name(f"{output.stem}.json")
    companions = {".json": record} if record.is_file() else {}
    views = output.with_suffix(".svviews")
    if views.is_dir():
        companions[".svviews"] = views
    data = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
    return Generated(glb=output, parameters=dict(data.get("parameters", parameters)),
                     companions=companions, components=list(data.get("components", [])))


GENERATORS: dict[str, Generator] = {
    "pixal3d": Generator(name="pixal3d", run=run_pixal3d, readiness=pixal3d_readiness,
                         parameters=PIXAL3D_PARAMETERS, repo=PIXAL3D_ROOT,
                         input_check=pixal3d_input_check,
                         check_parameters=pixal3d_check_parameters),
}


# ------------------------------------------------------------------------- the run

def run_manifest(path: Path, log: Callable[[str], None] = print) -> tuple[Path, Path]:
    """Generate, then file the result with its `.provenance.json`. Returns both paths."""
    source = path.expanduser().resolve()
    try:
        data = load_run_manifest(source)
        backend = data["model"]["backend"]
        generator = GENERATORS[backend]
        manifest_input = data["input"]
        names = manifest_input["views"] if "views" in manifest_input else [manifest_input["path"]]
        parameters = dict(data["model"].get("parameters", {}))
        intent = {
            "use_case": data.get("use_case", "showcase"),
            "distribution": data.get("distribution", "private"),
            "commercial_intent": data.get("commercial_intent", False),
        }
        policy = data.get("license_policy", {})
        output = data.get("output", {})
        if not isinstance(output, dict) or not isinstance(output.get("directory", "output"), str):
            raise ValueError('output must be an object like {"directory": "output"}')
        output_root = Path(output.get("directory", "output"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestRunError(f"invalid run manifest: {exc}") from exc

    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ManifestRunError("invalid run manifest: input.path and input.views must be "
                               "file paths")
    images = [(source.parent / name).resolve() for name in names]
    if not images:
        raise ManifestRunError("the manifest names no input image")
    if len(images) > generator.max_images:
        raise ManifestRunError(f"{backend} takes one image" if generator.max_images == 1
                               else f"{backend} takes at most {generator.max_images} images")
    missing = [str(image) for image in images if not image.is_file()]
    if missing:
        raise ManifestRunError(f"input image does not exist: {', '.join(missing)}")
    unknown = sorted(set(parameters) - generator.parameters)
    if unknown:
        raise ManifestRunError(f"{backend} does not take parameter(s) {', '.join(unknown)}; "
                               f"it takes {', '.join(sorted(generator.parameters))}")
    wrong = generator.check_parameters(parameters)
    if wrong:
        raise ManifestRunError(f"{backend}: {'; '.join(wrong)}")
    if not isinstance(intent["commercial_intent"], bool):
        raise ManifestRunError("commercial_intent must be true or false")
    if not isinstance(policy, dict) or not all(
            isinstance(policy.get(key, True), bool)
            for key in ("allow_conditional", "allow_research_only")):
        # "false" as a string is truthy: the licence gate must not fail open on a typo
        raise ManifestRunError("license_policy.allow_conditional and allow_research_only "
                               "must be true or false")
    unknown_policy = sorted(set(policy) - {"allow_conditional", "allow_research_only"})
    if unknown_policy:
        # a misspelt key would leave the permissive default in force
        raise ManifestRunError(f"license_policy: unknown field(s) {', '.join(unknown_policy)}")
    try:
        profile = validate_run_policy(backend, intent["use_case"], intent["distribution"],
                                      policy.get("allow_conditional", True))
    except ValueError as exc:
        raise ManifestRunError(f"license policy: {exc}") from exc
    problems = generator.readiness() + generator.input_check(images, parameters)
    if problems:
        raise ManifestRunError("; ".join(problems))

    working = output_root / ".working"
    try:
        generated = generator.run(images, working, parameters, log)
    except (RuntimeError, OSError, ValueError) as exc:
        raise ManifestRunError(f"{backend}: {exc}", code=1) from exc
    result, sidecar = finalize_output(
        generated=generated.glb, image=images[0], output_root=output_root, backend=backend,
        profile=profile, intent=intent, parameters=generated.parameters,
        source_manifest=source, backend_repo=generator.repo,
    )
    for suffix, companion in generated.companions.items():
        if companion.exists():
            shutil.move(str(companion), str(result.with_suffix(suffix)))
    if generated.components:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        record["components"] = [*record.get("components", []), *generated.components]
        sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result, sidecar
