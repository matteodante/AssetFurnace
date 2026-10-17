"""The step catalogue: one description per pipeline step, read by every route
(docs/plans/2026-10-17-step-parity-design.md, section 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from image_to_3dlab import batch, manifest_run, step_catalog

REPO = Path(__file__).resolve().parents[1]


def test_every_batch_step_with_a_module_is_in_the_catalogue():
    for name, available in batch.available_steps().items():
        if available:
            assert name in step_catalog.CATALOG, name


def test_the_catalogue_is_plain_json_for_the_studio_and_agents():
    described = step_catalog.describe()
    assert json.loads(json.dumps(described)) == described
    names = [step["name"] for step in described]
    assert names[:4] == ["picture", "concept", "generate", "finish"]
    finish = next(step for step in described if step["name"] == "finish")
    faces = next(param for param in finish["params"] if param["name"] == "faces")
    assert faces == {"name": "faces", "type": "int", "default": 40000, "range": [1000, 200000],
                     "choices": None, "row": True, "help": faces["help"], "flag": None, "off": False}
    assert finish["skip"] and finish["replace"] == ["glb"]


def test_check_reports_unknown_options_and_bad_values():
    problems = step_catalog.check("finish", {"atlass": 1, "atlas": "large", "photo": "yes",
                                             "roughness": 2.0, "voxel": 0.004})
    assert any("atlass" in p and "options are" in p for p in problems)
    assert any(p.startswith("atlas must be a whole number from 256 to 8192") for p in problems)
    assert any(p.startswith("photo must be true or false") for p in problems)
    assert any(p.startswith("roughness must be a number from 0.0 to 1.0") for p in problems)
    assert not any(p.startswith("voxel") for p in problems)
    assert step_catalog.check("finish", {"atlas": 1024, "paint": False}) == []


def test_row_fields_are_not_options():
    """`faces` is the row's own field: under "finish": {...} it would be silently ignored."""
    assert any("faces" in p for p in step_catalog.check("finish", {"faces": 3000}))


def test_check_validates_files_vectors_and_choices(tmp_path):
    lod = tmp_path / "crate_LOD1.glb"
    lod.write_bytes(b"glTF")
    assert step_catalog.check("export_unreal", {"lods": [str(lod)], "collision": "convex",
                                                "forward": [0, -1, 0], "name": "Crate"}) == []
    problems = step_catalog.check("export_unreal", {
        "lods": [str(tmp_path / "gone.glb")], "collision": "sphere", "forward": [0, 1],
        "name": 7, "tint": str(tmp_path / "nope.png")})
    assert any("gone.glb" in p for p in problems)
    assert any(p.startswith("collision must be one of box, convex, unreal") for p in problems)
    assert any(p.startswith("forward must be three numbers") for p in problems)
    assert any(p.startswith("name must be text") for p in problems)
    assert any("nope.png" in p for p in problems)


def test_a_step_not_in_the_catalogue_cannot_be_checked():
    with pytest.raises(KeyError):
        step_catalog.check("no_such_step", {})


def test_relative_file_options_resolve_against_a_folder(tmp_path):
    options = step_catalog.resolve_files("export_unreal", {"lods": ["a_LOD1.glb"], "tint": "t.png",
                                                           "name": "keep.png"}, tmp_path)
    assert options["lods"] == [str((tmp_path / "a_LOD1.glb").resolve())]
    assert options["tint"] == str((tmp_path / "t.png").resolve())
    assert options["name"] == "keep.png"


# ---------------------------------------------------------------------- parity

def test_generate_params_are_the_generators_own():
    described = {param.name for param in step_catalog.CATALOG["generate"].params if not param.row}
    assert described == set(manifest_run.GENERATORS["pixal3d"].parameters)


def _flags(script: str) -> set[str]:
    sys.path.insert(0, str(REPO / "scripts"))
    module = __import__(script)
    return {flag for action in module.build_parser()._actions for flag in action.option_strings}


def test_every_finish_param_reaches_the_finish_worker():
    flags = _flags("retopo_repaint")
    from image_to_3dlab.batch_steps import finish as finish_step

    for param in step_catalog.CATALOG["finish"].params:
        if param.name in ("paint", "photo"):   # the worker's --skip-paint and --views
            continue
        flag = finish_step.VALUE_FLAGS.get(param.name) or f"--{param.name.replace('_', '-')}"
        assert flag in flags, param.name


def test_every_export_param_is_a_flag_of_the_export_script():
    flags = _flags("export_unreal")
    for param in step_catalog.CATALOG["export_unreal"].params:
        assert f"--{param.name.replace('_', '-')}" in flags, param.name


# Flags that are about where and how a script runs, not settings of the step.
NOT_SETTINGS = {"-h", "--help", "--out", "--blender", "--check-in-unreal", "--json", "--no-queue", "--log",
                "--key", "--provenance"}


@pytest.mark.parametrize("step, script", [("export_unreal", "export_unreal"), ("checks", "check_asset"),
                                          ("picture", "text_image")])
def test_every_setting_of_the_script_is_in_the_catalogue(step, script):
    """The other way round: a flag added to the script and not to the catalogue would be
    out of reach of the batches, the agents and the studio's forms."""
    described = {param.cli_flag for param in step_catalog.CATALOG[step].params if param.cli_flag}
    missing = sorted(flag for flag in _flags(script) - NOT_SETTINGS - described if flag.startswith("--"))
    assert missing == [], f"{script} has settings the catalogue does not describe: {missing}"


def test_every_checks_param_is_a_flag_of_the_checks_script():
    flags = _flags("check_asset")
    for param in step_catalog.CATALOG["checks"].params:
        if param.cli_flag:   # stop_on_fail is a batch decision, not a check setting
            assert param.cli_flag in flags, param.name


def test_every_picture_param_is_a_flag_of_the_picture_script():
    flags = _flags("text_image")
    for param in step_catalog.CATALOG["picture"].params:
        if param.cli_flag:   # the words are its argument, and the pictures it writes are the pick
            assert param.cli_flag in flags, param.name


def test_every_concept_param_is_a_flag_of_the_concept_script():
    sys.path.insert(0, str(REPO / "scripts"))
    import reference_concept

    make = reference_concept.parser()._subparsers._group_actions[0].choices["make"]
    flags = {flag for action in make._actions for flag in action.option_strings}
    described = {param.cli_flag for param in step_catalog.CATALOG["concept"].params if param.cli_flag}
    assert described <= flags, described - flags
    assert flags - described - {"-h", "--help"} == set()


def test_the_concept_choices_are_the_prompts_own():
    from image_to_3dlab.concept import prompts

    spec = step_catalog.CATALOG["concept"]
    assert spec.param("mode").choices == prompts.MODES
    assert spec.param("structure").choices == tuple(prompts.STRUCTURES)
    assert spec.param("guide").choices == prompts.GUIDES
    assert spec.param("view").choices == tuple(prompts.VIEWS)


def test_the_picture_settings_match_the_engine():
    from image_to_3dlab import text_image

    spec = step_catalog.CATALOG["picture"]
    assert spec.param("sampler").choices == text_image.SAMPLERS
    assert spec.param("seed").default == text_image.DEFAULTS["seed"]
    assert spec.param("width").range == (text_image.MIN_SIZE, text_image.MAX_SIZE)


def test_a_list_of_choices_is_checked_name_by_name():
    assert step_catalog.check("animate", {"presets": ["idle", "walk"], "arm_spread": 10}) == []
    problems = step_catalog.check("animate", {"presets": ["idle", "moonwalk"]})
    assert problems == [f"presets: moonwalk not one of {', '.join(step_catalog.choices(step_catalog.get('animate').param('presets')))}"]
    assert step_catalog.check("animate", {"presets": "idle"})[0].startswith("presets must be a list")
