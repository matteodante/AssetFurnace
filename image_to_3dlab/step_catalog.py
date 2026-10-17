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

TYPES = ("int", "float", "bool", "text", "choice", "choices", "file", "files", "vector3", "object")
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
    off: bool = False          # a file option that `false` turns off (no file, not the default)
    # its flag on the step's command line when it is not --<name with dashes>; "" for none
    # (a positional argument, or a choice the command line makes another way)
    flag: str | None = None

    @property
    def cli_flag(self) -> str | None:
        return (self.flag if self.flag is not None else f"--{self.name.replace('_', '-')}") or None


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


SAMPLERS = ("euler", "euler_a", "heun", "dpm2", "dpmpp2m", "lcm")   # text_image.SAMPLERS
# text_image.T2I_DEFAULTS and MAX_SIZE: the Turbo's card draws text alone at 2048, up to its widest preset
PICTURE_SIZE, PICTURE_MAX = 2048, 2752
# image_to_3dlab.concept.prompts, written out so the catalogue imports nothing heavy
MODES = ("redraw", "dress", "add_piece", "other_view", "complete_piece")
STRUCTURES = ("depth", "lineart")
GUIDES = ("body", "pose")
VIEWS = ("side", "back")
# image_to_3dlab.creature_family.FAMILIES, with "auto" first
FAMILIES = ("auto", "humanoid", "digitigrade", "quadruped", "multileg", "winged", "serpent", "plant")

CATALOG: dict[str, StepSpec] = {spec.name: spec for spec in (
    StepSpec(
        name="picture", label="Picture", takes=(), makes="picture",
        params=(
            Param("prompt", "text", None, row=True, help="what to draw", flag=""),
            Param("seed", "int", 42, range=(0, 2**31 - 1), help="same seed, same picture"),
            Param("count", "int", 1, range=(1, 8), help="pictures to pick from, seeds seed, seed+1..."),
            Param("pick", "int", None, range=(0, 2**31 - 1),
                  help="the seed of the picture to keep, when there are several", flag=""),
            Param("width", "int", PICTURE_SIZE, range=(256, PICTURE_MAX),
                  help="rounded to a multiple of 32; the card's presets: 2048x2048, 2400x1792, 1792x2400, "
                       "2528x1696, 1696x2528, 2752x1536, 1536x2752 (1024 is about 5x faster)"),
            Param("height", "int", PICTURE_SIZE, range=(256, PICTURE_MAX)),
            Param("transparent", "bool", True, flag="--opaque",
                  help="the card's template for a transparent background (--opaque turns it off)"),
            Param("rewrite", "bool", True, flag="--no-rewrite",
                  help="the card's prompt rewriting, on this machine (--no-rewrite turns it off)"),
            Param("cfg_scale", "float", 1.0, range=(0.0, 20.0)),
            Param("sampler", "choice", "euler", choices=SAMPLERS),
            Param("negative_prompt", "text", None, help="what to keep out; only with cfg_scale above 1 (at 1 the "
                                                        "engine leaves it out)"),
            Param("bible", "text", None, help="a style bible's words after the prompt: - for the active "
                                              "one, or its name; none by default"),
            Param("kind", "text", None, row=True, help="the asset's kind, for the bible's words for that kind"),
        ),
        row_fields=(),
        skip="give the row its picture as images instead", replace=("png", "jpg"),
        cli="scripts/text_image.py", run="image_to_3dlab.batch_steps.picture:run",
        candidates="count", gpu=True,
        note="Qwen-Image 2.1 Turbo, the studio's Create engine. With count above 1 the row "
             "waits for a pick; the candidates are kept, so picking draws nothing again.",
    ),
    StepSpec(
        name="concept", label="Concept from a reference", feature=2, takes=("picture",), makes="picture",
        params=(
            Param("mode", "choice", None, choices=MODES, flag="",
                  help="redraw, dress, add_piece, other_view or complete_piece"),
            Param("shape", "files", None, help="shape pictures, in order; default the row's picture"),
            Param("style_refs", "files", None, flag="--style-ref", help="style pictures, after the shape ones"),
            Param("subject", "text", None, help="redraw: what to draw"),
            Param("colours", "text", None, help="redraw: its colours"),
            Param("parts", "text", None, help="redraw: parts to keep"),
            Param("structure", "choice", None, choices=STRUCTURES, help="redraw: what the shape picture is"),
            Param("outfit", "text", None, help="dress: the outfit"),
            Param("guide", "choice", None, choices=GUIDES, help="dress: follow the body or only its pose"),
            Param("piece", "text", None, help="add_piece, complete_piece: the piece"),
            Param("view", "choice", None, choices=VIEWS, help="other_view: which side"),
            Param("depth", "bool", None, help="other_view: the second shape picture is that side's depth"),
            Param("style", "text", None, help="style words"),
            Param("bible", "text", None, help="a style bible's words and up to three of its pictures: - for "
                                              "the active one, or its name; none by default"),
            Param("kind", "text", None, row=True, help="the asset's kind, for the bible's words for that kind"),
            Param("seed", "int", 42, range=(0, 2**31 - 1)),
            Param("count", "int", 1, range=(1, 4), flag="--attempts", help="concepts to pick from, seed, seed+1..."),
            Param("pick", "int", None, range=(0, 2**31 - 1), flag="",
                  help="the seed of the concept to keep, when there are several"),
        ),
        row_fields=(),
        skip="the row's own picture goes on as it is", replace=("png", "jpg"),
        cli="scripts/reference_concept.py make", run="image_to_3dlab.batch_steps.concept:run",
        candidates="count", gpu=True,
        note="Qwen-Image 2.1 in edit mode, as the studio's Draw from a reference. The shape "
             "pictures are the row's picture unless the settings name others.",
    ),
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
        name="rig", label="Rig", takes=("asset",), makes="asset",
        params=(),
        row_fields=(),
        skip="the model stays without a skeleton", replace=("glb",),
        cli="scripts/autorig.py rig", run="image_to_3dlab.batch_steps.rig:run", gpu=True,
        note="SkinTokens and the skin-weight clean-up, as the studio's Animate tab. The "
             "rigged GLB becomes the row's asset.",
    ),
    StepSpec(
        name="clips", label="Family clips", feature=6, takes=("asset",), makes="clips",
        params=(
            Param("family", "choice", "auto", choices=FAMILIES, help="the creature's family; auto: recognised"),
            Param("clips", "bool", True, flag="--no-clips",
                  help="the test clips; off: the family alone (the Unreal export's facing)"),
            Param("library", "bool", True, flag="--no-library", help="the CC0 library's clips, carried by chains"),
            Param("variations", "bool", True, flag="--no-variations",
                  help="the library's loops also as longer varied takes (GenMM)"),
            Param("variation_factor", "float", 2.0, range=(1.0, 8.0), help="how much longer a varied take is"),
            Param("variation_seed", "int", 0, range=(0, 2**31 - 1), help="the variation's seed"),
        ),
        row_fields=(),
        skip="no test clips; the rig goes on as it is", replace=(),
        cli="scripts/creature_clips.py", run="image_to_3dlab.batch_steps.clips:run",
        note="After rig. The clips reach a creature's Unreal export; the family file beside "
             "the rig gives it its facing.",
    ),
    StepSpec(
        name="animate", label="Animate", takes=("asset",), makes="clips",
        params=(
            Param("presets", "choices", None, help="the preset clips to play, e.g. [\"idle\", \"walk\"]"),
            Param("speed", "float", None, range=(0.5, 2.0), help="playback speed; default each preset's own"),
            Param("arm_spread", "float", 0.0, range=(0.0, 30.0), help="arms out from the body, degrees"),
        ),
        row_fields=(),
        skip="no clips; a creature is exported with what its rig has", replace=("glb",),
        cli="scripts/autorig.py animate", run="image_to_3dlab.batch_steps.animate:run",
        note="Kimodo clips retargeted on the rig, CPU only. The row's asset stays the rigged "
             "model; the export takes the clips.",
    ),
    StepSpec(
        name="materials", label="Material", feature=8, takes=("asset",), makes="material",
        params=(
            Param("material", "choice", None, choices=None,
                  help="the asset's one material (metal, leather, cloth...); default the rule's, from "
                       "the words it was made from or its kind"),
        ),
        row_fields=("kind",),
        skip="the export keeps the model's own roughness and metal", replace=(),
        cli="scripts/export_unreal.py", run="image_to_3dlab.batch_steps.materials:run",
        note="One material per asset or piece (function 8); no zones. The export packs its roughness and "
             "metal into the ORM. A set's pieces: slot_parts.py materials and pieces_materials.",
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
            Param("tint", "file", None, off=True, help="tint mask; default the material step's, beside "
                                                       "the model; false for none"),
            Param("forward", "vector3", None, help="which way the model faces, in Blender axes; "
                                                   "default from the creature family, else 0 -1 0"),
            Param("body", "file", None, help="equipment, or an object on a socket: the rigged body it belongs to"),
            Param("socket", "text", None, help="an object held by the body (a weapon, a shield): NAME=BONE, "
                                               "the socket of its skeleton it hangs from in Unreal"),
            Param("body_name", "text", None, help="equipment: the body's export name"),
            Param("animations", "files", None, help="creatures: more animated GLBs"),
            Param("material", "choice", None, choices=None,
                  help="the one material of a piece (metal, leather, cloth...): its roughness "
                       "and metal in the ORM"),
            Param("two_sided", "bool", None, help="draw both faces of every triangle; unset: cloth is two "
                                                  "sided, the rest one sided"),
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
                                                      "the reviewer decides", flag=""),
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
    if param.name == "presets" and param.choices is None:
        from image_to_3dlab.rigging import preset_ids

        return preset_ids()   # the shipped clips, as image_to_3dlab/motions lists them
    if param.name == "material" and param.type == "choice" and param.choices is None:
        from image_to_3dlab.piece_materials import MATERIALS

        return tuple(MATERIALS)   # the game materials' own list
    return param.choices


# ------------------------------------------------------------------------- checking

def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _in(value: float, span: tuple[float, float] | None) -> bool:
    return span is None or span[0] <= value <= span[1]


def _span(span: tuple[float, float]) -> str:
    low, high = span
    return f"above {low}" if high == math.inf else f"from {low} to {high}"


def value_problem(param: Param, value: Any, files: bool = True) -> str | None:
    """Why `value` will not do for `param`, or None. A None value means "the default".
    `files=False` checks a file option's shape but not that the file is there."""
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
    elif kind == "choices":
        allowed = choices(param) or ()
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return f"{name} must be a list of names, not {value!r}"
        unknown = [item for item in value if item not in allowed]
        if unknown:
            return f"{name}: {', '.join(unknown)} not one of {', '.join(allowed)}"
    elif kind == "vector3":
        if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(map(_number, value)):
            return f"{name} must be three numbers, like [0, -1, 0], not {value!r}"
    elif kind == "object":
        if not isinstance(value, dict):
            return f"{name} must be an object, not {value!r}"
    elif kind == "file":
        if value is False and param.off:
            return None
        if not isinstance(value, str):
            return f"{name} must be a file path, not {value!r}"
        if files and not Path(value).is_file():
            return f"{name}: file does not exist: {value}"
    elif kind == "files":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return f"{name} must be a list of file paths, not {value!r}"
        missing = [item for item in value if not Path(item).is_file()]
        if files and missing:
            return f"{name}: file does not exist: {', '.join(missing)}"
    return None


def check(name: str, options: dict[str, Any], files: bool = True) -> list[str]:
    """What is wrong with a step's options: unknown names, wrong types, out of range, and
    with `files`, files that are not there. A batch checks the files only for the steps it
    will run: a finished row does not need the body it was exported with any more."""
    spec = get(name)
    known = {param.name: param for param in spec.options}
    problems = []
    unknown = sorted(set(options) - set(known))
    if unknown:
        problems.append(f"unknown {name} option(s) {', '.join(unknown)}; options are "
                        f"{', '.join(sorted(known))}")
    problems += [problem for key, value in options.items() if key in known
                 for problem in [value_problem(known[key], value, files)] if problem]
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
