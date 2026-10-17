"""One description per pipeline step, read by every route: the studio, the command line,
the batches and the MCP server.

    from image_to_3dlab import step_catalog
    step_catalog.check("finish", {"atlas": 1024})     # -> [] or the problems
    step_catalog.describe()                           # -> plain JSON, for the studio and agents

A step's description says what it takes and leaves, its parameters (name, type, default,
range or choices), whether it can be skipped, what file of your own can stand in for its
output, and which row fields it depends on. Nothing here runs a step: each one has a single
implementation, named in `run`, that every route calls. Why it is built this way:
`docs/plans/2026-10-17-step-parity-design.md`.

Parameters with `row=True` are fields of a batch row (`faces`, `size_cm`) rather than
options under the step's name; they are described here so the studio and agents see the
whole step, and checked where the row is read.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TYPES = ("int", "float", "bool", "text", "choice", "file", "files", "vector3", "object")
FILE_TYPES = ("file", "files")


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    default: Any = None
    range: tuple[float, float] | None = None
    choices: tuple[str, ...] | None = None
    row: bool = False          # a batch row field, not an option under the step's name
    help: str = ""


@dataclass(frozen=True)
class StepSpec:
    name: str
    label: str
    takes: tuple[str, ...]           # what it reads from the chain: asset, picture, rig...
    makes: str | None                # what it leaves: asset, report, export; None: nothing new
    params: tuple[Param, ...]
    row_fields: tuple[str, ...]      # the batch row fields its result depends on
    skip: str | None                 # what happens when it is left out; None: it cannot be
    replace: tuple[str, ...]         # file types that can stand in for its output
    cli: str                         # its command line
    run: str                         # its one implementation, module:function
    feature: int | None = None       # the feature number in the Auroconts plan, if any
    candidates: str | None = None    # the parameter that makes several to pick from
    gpu: bool = False
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def param(self, name: str) -> Param | None:
        return next((param for param in self.params if param.name == name), None)

    @property
    def options(self) -> tuple[Param, ...]:
        """The settings a batch row gives under the step's own name."""
        return tuple(param for param in self.params if not param.row)


SIZE_LIMIT_CM = 10_000_000   # as batch.SIZE_LIMIT_CM: 100 km is not a game asset


def _profiles() -> tuple[str, ...]:
    from image_to_3dlab.checks.registry import profile_names

    return tuple(profile_names())


CATALOG: dict[str, StepSpec] = {spec.name: spec for spec in (
    StepSpec(
        name="generate", label="3D model", takes=("picture",), makes="asset",
        params=(
            Param("backend", "choice", "pixal3d", choices=("pixal3d",), row=True,
                  help="the image-to-3D model; batches run the ones that check their weights "
                       "without downloading them"),
            Param("seed", "int", None, range=(0, 2**31 - 1), help="same seed, same model"),
            Param("steps", "int", None, range=(1, 50), help="diffusion steps"),
            Param("gss", "float", None, range=(0, math.inf), help="shape guidance"),
            Param("gsh", "float", None, range=(0, math.inf), help="shape guidance, second stage"),
            Param("fov", "float", None, range=(0, math.pi), help="camera field of view (radians)"),
            Param("matte", "bool", None, help="cut the picture out first; unset: only when it "
                                              "has no transparent background"),
        ),
        row_fields=("backend", "parameters"),
        skip=None, replace=("glb",),
        cli="pipeline.py --run-manifest", run="image_to_3dlab.batch_steps.generate:run",
        gpu=True,
        note="Generator settings go in the row's `parameters` (and `seed`). A model of your "
             "own stands in as the row's `asset`, with no generate step.",
    ),
    StepSpec(
        name="finish", label="Finish", takes=("asset", "picture"), makes="asset",
        params=(
            Param("faces", "int", 40000, range=(1000, 200000), row=True, help="triangle budget"),
            Param("paint", "bool", False, help="repaint with Hunyuan 2.1 Paint (Apple Silicon; "
                                               "licence not valid in the EU, UK, South Korea)"),
            Param("photo", "bool", True, help="Pixel Match, when the model kept its camera"),
            Param("atlas", "int", None, range=(256, 8192), help="texture atlas size"),
            Param("texture_size", "int", None, range=(256, 8192), help="final texture size"),
            Param("metallic", "float", None, range=(0.0, 1.0)),
            Param("roughness", "float", None, range=(0.0, 1.0)),
            Param("ior", "float", None, range=(1.0, 3.0)),
            Param("voxel", "float", None, range=(0.0, 0.1), help="remesh voxel size"),
            Param("skip_bake", "bool", False, help="no detail bake"),
            Param("skip_compress", "bool", False, help="no texture compression"),
        ),
        row_fields=("faces",),
        skip="the model goes on as it came out of the generator", replace=("glb",),
        cli="scripts/retopo_repaint.py", run="image_to_3dlab.batch_steps.finish:run",
        gpu=True,
    ),
    StepSpec(
        name="export_unreal", label="Unreal export", feature=5, takes=("asset",), makes="export",
        params=(
            Param("kind", "choice", "prop", choices=("prop", "creature", "equipment"), row=True),
            Param("size_cm", "float", None, range=(0, SIZE_LIMIT_CM), row=True,
                  help="real size in centimetres: a generated model has none of its own"),
            Param("measure", "choice", "height", choices=("height", "longest"), row=True,
                  help="what size_cm measures"),
            Param("name", "text", None, help="asset name; default the row's id"),
            Param("collision", "choice", None, choices=("box", "convex", "unreal"),
                  help="objects only; default box"),
            Param("lods", "files", None, help="detail levels; default the _LODn files beside "
                                              "the model, [] for none"),
            Param("tint", "file", None, help="tint mask; default the material step's, beside "
                                             "the model"),
            Param("forward", "vector3", None, help="which way the model faces, in Blender axes; "
                                                   "default from the creature family, else 0 -1 0"),
            Param("body", "file", None, help="equipment: the rigged body it belongs to"),
            Param("body_name", "text", None, help="equipment: the body's export name"),
            Param("animations", "files", None, help="creatures: more animated GLBs"),
        ),
        row_fields=("kind", "size_cm", "measure", "profile"),
        skip="no Unreal files are written", replace=(),
        cli="scripts/export_unreal.py", run="image_to_3dlab.batch_steps.export_unreal:run",
        note="The export is an extra output: the row's asset stays the finished model.",
    ),
    StepSpec(
        name="checks", label="Checks", feature=7, takes=("asset", "picture"), makes="report",
        params=(
            Param("profile", "choice", None, choices=None,
                  help="thresholds profile; default from the row's kind"),
            Param("set", "object", None, help='thresholds, as --set: {"budget.faces.max": 20000}'),
            Param("reference", "bool", True, help="compare the silhouette with the row's picture"),
            Param("height", "float", None, range=(0, math.inf),
                  help="wanted height in metres; default size_cm when it is a height"),
            Param("width", "float", None, range=(0, math.inf), help="wanted width (X), metres"),
            Param("depth", "float", None, range=(0, math.inf), help="wanted depth (Z), metres"),
            Param("stop_on_fail", "bool", False, help="a failed check fails the row; otherwise "
                                                      "the reviewer decides"),
        ),
        row_fields=("kind", "size_cm", "measure"),
        skip="the row reaches the review queue without a report, and cannot be approved "
             "until the checks run", replace=(),
        cli="scripts/check_asset.py", run="image_to_3dlab.batch_steps.checks:run",
    ),
)}


def get(name: str) -> StepSpec:
    """The step's description; KeyError for a name the catalogue does not have."""
    return CATALOG[name]


def choices(param: Param) -> tuple[str, ...] | None:
    if param.name == "profile" and param.type == "choice" and param.choices is None:
        return _profiles()   # read from the checks' profile files, so a new one needs no edit
    return param.choices


# ------------------------------------------------------------------------- checking

def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _in(value: float, span: tuple[float, float] | None) -> bool:
    return span is None or span[0] <= value <= span[1]


def _span(span: tuple[float, float]) -> str:
    low, high = span
    return f"above {low}" if high == math.inf else f"from {low} to {high}"


def value_problem(param: Param, value: Any) -> str | None:
    """Why `value` will not do for `param`, or None. A None value means "the default"."""
    if value is None:
        return None
    name, kind = param.name, param.type
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int) or not _in(value, param.range):
            return f"{name} must be a whole number {_span(param.range or (-math.inf, math.inf))}, not {value!r}"
    elif kind == "float":
        if not _number(value) or not _in(value, param.range):
            return f"{name} must be a number {_span(param.range or (-math.inf, math.inf))}, not {value!r}"
    elif kind == "bool":
        if not isinstance(value, bool):
            return f"{name} must be true or false, not {value!r}"
    elif kind == "text":
        if not isinstance(value, str) or not value.strip():
            return f"{name} must be text, not {value!r}"
    elif kind == "choice":
        allowed = choices(param) or ()
        if value not in allowed:
            return f"{name} must be one of {', '.join(allowed)}, not {value!r}"
    elif kind == "vector3":
        if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(map(_number, value)):
            return f"{name} must be three numbers, like [0, -1, 0], not {value!r}"
    elif kind == "object":
        if not isinstance(value, dict):
            return f"{name} must be an object, not {value!r}"
    elif kind == "file":
        if not isinstance(value, str) or not Path(value).is_file():
            return f"{name}: file does not exist: {value}"
    elif kind == "files":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return f"{name} must be a list of file paths, not {value!r}"
        missing = [item for item in value if not Path(item).is_file()]
        if missing:
            return f"{name}: file does not exist: {', '.join(missing)}"
    return None


def check(name: str, options: dict[str, Any]) -> list[str]:
    """What is wrong with a step's options: unknown names, wrong types, out of range."""
    spec = get(name)
    known = {param.name: param for param in spec.options}
    problems = []
    unknown = sorted(set(options) - set(known))
    if unknown:
        problems.append(f"unknown {name} option(s) {', '.join(unknown)}; options are "
                        f"{', '.join(sorted(known))}")
    problems += [problem for key, value in options.items() if key in known
                 for problem in [value_problem(known[key], value)] if problem]
    return problems


def resolve_files(name: str, options: dict[str, Any], base: Path) -> dict[str, Any]:
    """The step's options with relative file paths made absolute against `base` (a batch
    spec's folder). Anything that is not a file option, or not a string, is left as it is
    for `check` to report."""
    if name not in CATALOG:
        return dict(options)
    resolved = dict(options)
    for param in get(name).options:
        value = options.get(param.name)
        if param.type == "file" and isinstance(value, str):
            resolved[param.name] = str((base / value).resolve())
        elif param.type == "files" and isinstance(value, list):
            resolved[param.name] = [str((base / item).resolve()) if isinstance(item, str) else item
                                    for item in value]
    return resolved


def file_options(name: str, options: dict[str, Any]) -> list[str]:
    """The names of a step's options that hold files, among those given."""
    if name not in CATALOG:
        return []
    return [param.name for param in get(name).options
            if param.type in FILE_TYPES and param.name in options]


# ------------------------------------------------------------------------- describing

def _plain(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return None
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def describe_step(spec: StepSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["params"] = [{key: _plain(value) for key, value in {
        **asdict(param), "choices": choices(param)}.items()} for param in spec.params]
    return {key: _plain(value) if key != "params" else value for key, value in data.items()}


def describe() -> list[dict[str, Any]]:
    """Every step as plain JSON, in chain order: what the studio builds its forms from and
    what an agent reads to know what it can ask for."""
    return [describe_step(spec) for spec in CATALOG.values()]
