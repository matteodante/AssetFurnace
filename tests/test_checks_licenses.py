"""Licences from provenance: every record shape in use, and the lineage of a finished model."""

from __future__ import annotations

import json
import shutil

import _check_fixtures as fx
import pytest

from image_to_3dlab import checks
from image_to_3dlab.checks import licenses


def check(glb, **kw):
    return checks.run_checks(checks.Asset(glb=glb, **kw), only=["license.components"])["results"][0]


def test_finalize_output_record(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {"license": {"name": "MIT (TRELLIS.2) + DINOv3 License", "conditions": [
        "BRIA RMBG-2.0 is disabled and must not be loaded in commercial runs."]},
        "output": {"classification": "commercial-conditional"},
        "components": [{"component": "BRIA RMBG-2.0", "loaded": False, "commercial_status": "blocked"}]})
    found = check(glb)
    assert found["status"] == "pass"
    assert any("DINOv3" in c for c in found["value"]["conditions"])


def test_pixal3d_record_is_read_from_its_plain_json(tmp_path):
    glb = fx.clean(tmp_path / "run.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD, suffix=".json")
    found = check(glb)
    assert found["status"] == "pass"
    assert "MIT (BiRefNet)" in found["value"]["licenses"]


def test_rig_record_nests_its_sources(tmp_path):
    glb = fx.clean(tmp_path / "x_rigged.glb")
    fx.sidecar(glb, {"license": "MIT + DINOv3 License + Qwen Research License Agreement",
                     "classification": "commercial-conditional",
                     "rig": {"license": {"name": "MIT"}},
                     "source_record": {"models": [{"id": "qwen-image-2.1", "license": "Qwen Research License Agreement",
                                                   "attribution": "Built with Qwen"}]}})
    found = check(glb)
    assert found["status"] == "warn"           # Qwen's licence: non-commercial only
    assert "Attribution: say 'Built with Qwen'." in found["value"]["conditions"]


def test_conditional_licences_can_be_made_to_warn(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    found = checks.run_checks(checks.Asset(glb=glb), only=["license.components"],
                              overrides={"license.components": {"conditional": "warn"}})["results"][0]
    assert found["status"] == "warn"


def test_a_blocked_component_that_was_loaded_fails(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {"license": "MIT", "components": [{"component": "BRIA RMBG-2.0", "commercial_status": "blocked"}]})
    found = check(glb)
    assert found["status"] == "fail"
    assert "BRIA" in found["message"]


def test_an_unknown_licence_warns(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {"license": "Some Bespoke Research Licence"})
    assert check(glb)["status"] == "warn"


def test_a_finished_model_without_a_sidecar_is_traced_to_its_generated_run(tmp_path):
    output = tmp_path / "output"
    run = output / "knight__pixal3d__20261013-184959"
    run.mkdir(parents=True)
    generated = fx.clean(run / f"{run.name}.glb")
    fx.sidecar(generated, fx.PIXAL3D_RECORD, suffix=".json")
    finish = output / "finish" / f"{run.name}__finish__20261013-185337"
    (finish / "input").mkdir(parents=True)
    shutil.copy(generated, finish / "input" / "source.glb")
    finished = fx.clean(finish / "knight_9k.glb")
    found = check(finished)
    # Codex review 7: nothing in a finish run binds the finished bytes, so at most a warn.
    assert found["status"] == "warn" and "not bound" in found["message"]
    assert "inherited from the generated run" in found["details"]["found"]
    # A different source model breaks the trace: no record, so it fails.
    fx.cube(finish / "input" / "source.glb")
    assert check(finished)["status"] == "fail"


def test_a_sidecar_beside_a_finished_model_wins_over_the_trace(tmp_path):
    finish = tmp_path / "output" / "finish" / "knight__pixal3d__20261013-184959__finish__20261013-185337"
    finish.mkdir(parents=True)
    glb = fx.clean(finish / "knight_9k.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    assert check(glb)["details"]["found"] == "beside the model"


def test_an_explicit_record_is_used_as_given(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    record = tmp_path / "elsewhere.json"
    record.write_text(json.dumps(fx.HUNYUAN_RECORD))
    assert check(glb, provenance=record)["status"] == "fail"


def test_classify_reads_combined_names():
    classes, conditions = licenses.classify("MIT (code + flow weights); DINOv3 License (bundled encoder)")
    assert classes == ["conditional", "clear"]
    assert conditions


def test_a_sidecar_for_another_file_is_not_trusted(tmp_path):
    """Codex review: a declared output hash must match the GLB it sits beside."""
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {**fx.PIXAL3D_RECORD, "output": {"classification": "commercial-conditional", "sha256": "0" * 64}})
    found = check(glb)
    assert found["status"] == "fail" and "does not match" in found["message"]
    fx.sidecar(glb, {**fx.PIXAL3D_RECORD, "output": {"sha256": checks.Asset(glb=glb).sha256}})
    assert check(glb)["status"] == "pass"


def test_a_record_without_the_models_own_licence_fails(tmp_path):
    """Codex review: a rig record with `license: null` must not pass on the rig tool's MIT."""
    glb = fx.clean(tmp_path / "x_rigged.glb")
    fx.sidecar(glb, {"license": None, "classification": None, "rig": {"tool": "SkinTokens", "license": {"name": "MIT"}},
                     "source_record": None})
    found = check(glb)
    assert found["status"] == "fail" and "own licence" in found["message"]


def test_a_blender_version_takes_its_licence_from_its_source_record(tmp_path):
    glb = fx.clean(tmp_path / "model.glb")
    fx.sidecar(glb, {"license": None, "tool": "Blender", "source_record": fx.PIXAL3D_RECORD})
    assert check(glb)["status"] == "pass"


def test_an_explicit_record_for_another_file_is_not_trusted(tmp_path):
    """Codex review 2: the --provenance path checks the declared hash too."""
    glb = fx.clean(tmp_path / "a.glb")
    record = tmp_path / "elsewhere.json"
    record.write_text(json.dumps({**fx.PIXAL3D_RECORD, "output": {"sha256": "0" * 64}}))
    found = check(glb, provenance=record)
    assert found["status"] == "fail" and "does not match" in found["message"]


def test_a_research_only_model_warns_and_says_it_is_non_commercial(tmp_path):
    """Research-only is non-commercial. AssetFurnace is never commercial, so it warns and
    says so (it reaches the credits and the review) instead of failing (user, 2026-10-17)."""
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {"license": "MIT", "classification": "research-only"})
    found = check(glb)
    assert found["status"] == "warn" and "non-commercial" in found["message"]


def test_a_research_only_model_with_a_blocked_one_still_fails(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {"license": "Tencent Hunyuan 3D 2.1 Community License", "classification": "research-only"})
    assert check(glb)["status"] == "fail"



@pytest.mark.parametrize("name, status", [
    ("CC-BY-NC 4.0", "fail"), ("Creative Commons Attribution-NonCommercial 4.0", "fail"),
    ("CC BY-NC-SA 4.0", "fail"), ("CC-BY 4.0", "pass"),
])
def test_non_commercial_creative_commons_is_blocked(tmp_path, name, status):
    """Codex review 2: the NC variants must not match the plain CC-BY pattern."""
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {"license": name})
    assert check(glb)["status"] == status


def _prop_run(tmp_path, source_matches=True):
    run = tmp_path / "output" / "props" / "tavern__pixal3d__20261013-234430__props__20261013-234841"
    prop = run / "finished" / "prop_01"
    prop.mkdir(parents=True)
    source = fx.clean(run / "source.glb")
    sha = checks.Asset(glb=source).sha256 if source_matches else "0" * 64
    (run / "props.provenance.json").write_text(json.dumps({
        "schema_version": 1, "input": {"sha256": sha, "source": "generated"},
        "license": fx.PIXAL3D_RECORD["license"], "classification": "commercial-conditional",
        "source_record": fx.PIXAL3D_RECORD}))
    return fx.clean(prop / "prop_01_LOD0.glb")


def test_a_prop_is_traced_to_its_props_run_but_only_warns(tmp_path):
    """Codex reviews 4 and 6: props keep their licence in <run>/props.provenance.json, which
    names no hash per prop, so the inherited licence is not bound to the file: a warn."""
    found = check(_prop_run(tmp_path))
    assert "props.provenance.json" in found["details"]["records"][0]
    assert found["status"] == "warn" and "not bound" in found["message"]


def test_a_prop_with_its_own_hash_bound_sidecar_passes(tmp_path):
    prop = _prop_run(tmp_path)
    fx.sidecar(prop, {**fx.PIXAL3D_RECORD, "output": {"sha256": checks.Asset(glb=prop).sha256}})
    assert check(prop)["status"] == "pass"


def test_a_props_run_whose_source_changed_is_not_trusted(tmp_path):
    found = check(_prop_run(tmp_path, source_matches=False))
    assert found["status"] == "fail" and "source.glb" in found["message"]


@pytest.mark.parametrize("name, status", [
    ("MIT + Proprietary Research License", "warn"),
    ("MIT (code + flow weights); DINOv3 License (bundled encoder)", "pass"),
    ("MIT + DINOv3 License + Qwen Research License Agreement", "warn"),
    ("MIT code / Apache-2.0 U-2-Net", "pass"),
])
def test_every_part_of_a_composite_licence_is_classified(tmp_path, name, status):
    """Codex review 6: one recognised part must not clear an unknown one."""
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, {"license": name})
    assert check(glb)["status"] == status


def test_a_finished_model_is_traced_to_a_source_in_a_licence_class_folder(tmp_path):
    """Codex review 7: pipeline.py writes output/<licence folder>/<stem>.glb, not output/<stem>/."""
    output = tmp_path / "output"
    stem = "knight__pixal3d__commercial-conditional__0a1b2c3d4e5f"
    (output / "conditional").mkdir(parents=True)
    generated = fx.clean(output / "conditional" / f"{stem}.glb")
    fx.sidecar(generated, {**fx.PIXAL3D_RECORD, "output": {"sha256": checks.Asset(glb=generated).sha256}})
    finish = output / "finish" / f"{stem}__finish__20261013-185337"
    (finish / "input").mkdir(parents=True)
    shutil.copy(generated, finish / "input" / "source.glb")
    found = check(fx.clean(finish / "knight_9k.glb"))
    assert found["status"] == "warn" and "conditional" in found["details"]["found"]
