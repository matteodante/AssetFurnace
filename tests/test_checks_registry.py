"""The checks registry: one contract for every check, and a report that never lies by omission."""

from __future__ import annotations

import json

import _check_fixtures as fx
import pytest

from image_to_3dlab import checks
from image_to_3dlab.checks import registry


def test_result_builds_the_contract():
    r = registry.result("x.y", "warn", 3, {"warn": 1, "fail": 5}, "three")
    assert r == {"check": "x.y", "status": "warn", "value": 3,
                 "threshold": {"warn": 1, "fail": 5}, "message": "three"}
    assert registry.result("x.y", "pass", 1, None, "ok", {"a": 1})["details"] == {"a": 1}


@pytest.mark.parametrize("bad", [
    {"check": "x", "status": "skip", "value": 1, "threshold": None, "message": "m"},
    {"check": "x", "status": "pass", "value": 1, "threshold": None},
    {"check": "", "status": "pass", "value": 1, "threshold": None, "message": "m"},
    {"check": "x", "status": "pass", "value": 1, "threshold": None, "message": 5},
])
def test_validate_result_rejects_anything_off_contract(bad):
    with pytest.raises((ValueError, TypeError)):
        registry.validate_result(bad)


def test_grading_a_limit_and_a_band():
    assert registry.grade_max(0.5, warn=1, fail=3) == "pass"
    assert registry.grade_max(2, warn=1, fail=3) == "warn"
    assert registry.grade_max(4, warn=1, fail=3) == "fail"
    assert registry.grade_max(4, warn=None, fail=3) == "fail"
    assert registry.grade_band(0.8, 0.65, 0.95) == "pass"
    assert registry.grade_band(0.5, 0.65, 0.95) == "fail"
    assert registry.grade_band(0.99, 0.65, 0.95) == "fail"


def test_worst_status_wins():
    assert registry.worst([]) == "pass"
    assert registry.worst(["pass", "warn"]) == "warn"
    assert registry.worst(["warn", "fail", "pass"]) == "fail"


def test_registering_a_name_twice_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        registry.register("mesh.holes")(lambda asset, config: None)


def test_every_generic_check_is_registered():
    names = set(checks.REGISTRY)
    assert {"mesh.flipped_faces", "mesh.holes", "mesh.loose_parts", "mesh.nonmanifold", "uv.overlap",
            "budget.faces", "budget.materials", "budget.textures", "budget.texel_density",
            "measure.height", "measure.footprint", "silhouette", "gltf.valid",
            "license.components"} <= names


def test_profiles_merge_over_the_defaults_and_accept_overrides():
    prop = registry.load_profile("prop")
    assert prop["name"] == "prop"
    assert "mesh.holes" in prop["checks"]
    tuned = registry.load_profile("prop", {"budget.faces": {"max": 123}})
    assert tuned["checks"]["budget.faces"]["max"] == 123
    assert tuned["checks"]["mesh.holes"] == prop["checks"]["mesh.holes"]
    with pytest.raises(ValueError, match="unknown profile"):
        registry.load_profile("../etc")


def test_a_check_without_its_inputs_is_skipped_not_passed(tmp_path):
    asset = checks.Asset(glb=fx.clean(tmp_path / "a.glb"))
    report = checks.run_checks(asset, profile="prop", only=["silhouette", "measure.height"])
    assert report["results"] == []
    assert {s["check"] for s in report["skipped"]} == {"silhouette", "measure.height"}
    assert all(s["reason"] for s in report["skipped"])


def test_a_crashing_check_fails_instead_of_vanishing(tmp_path, monkeypatch):
    def boom(asset, config):
        raise RuntimeError("kaput")
    monkeypatch.setitem(registry.REGISTRY, "test.boom", registry.CheckSpec("test.boom", boom, ("glb",)))
    report = checks.run_checks(checks.Asset(glb=fx.clean(tmp_path / "a.glb")), only=["test.boom"])
    assert report["status"] == "fail"
    assert "kaput" in report["results"][0]["message"]


def test_a_check_returning_off_contract_output_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, "test.sloppy",
                        registry.CheckSpec("test.sloppy", lambda a, c: {"status": "ok"}, ("glb",)))
    report = checks.run_checks(checks.Asset(glb=fx.clean(tmp_path / "a.glb")), only=["test.sloppy"])
    assert report["results"][0]["status"] == "fail"


def test_warn_only_turns_a_fail_into_a_warn_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, "test.strict", registry.CheckSpec(
        "test.strict", lambda a, c: registry.result("test.strict", "fail", 1, None, "bad"), ("glb",)))
    report = checks.run_checks(checks.Asset(glb=fx.clean(tmp_path / "a.glb")), only=["test.strict"],
                               overrides={"test.strict": {"warn_only": True}})
    assert report["results"][0]["status"] == "warn"
    assert "warn only" in report["results"][0]["message"]


def test_a_disabled_check_is_listed_as_skipped(tmp_path):
    report = checks.run_checks(checks.Asset(glb=fx.clean(tmp_path / "a.glb")), only=["mesh.holes"],
                               overrides={"mesh.holes": {"enabled": False}})
    assert report["skipped"] == [{"check": "mesh.holes", "reason": "disabled in the profile"}]


def test_the_report_is_written_beside_the_glb(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    report = checks.run_checks(checks.Asset(glb=glb, key="mob_0412"), only=["mesh.holes"])
    path = checks.write_report(report)
    assert path == tmp_path / "a.checks.json"
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 1
    assert saved["asset_key"] == "mob_0412"
    assert saved["asset_sha256"] == report["asset_sha256"]
    assert saved["status"] in ("pass", "warn", "fail")


def test_a_reference_is_recorded_with_its_hash(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    ref = fx.clean(tmp_path / "ref.glb")
    report = checks.run_checks(checks.Asset(glb=glb, reference=ref), only=["mesh.holes"])
    assert report["reference"]["kind"] == "glb"
    assert len(report["reference"]["sha256"]) == 64


@pytest.mark.parametrize("numbers", [{"height": -1.8}, {"width": 0}, {"depth": "tall"}])
def test_reference_measurements_must_be_positive_numbers(tmp_path, numbers):
    """Codex review: a negative height made every size check pass."""
    with pytest.raises(ValueError, match="positive"):
        checks.Asset(glb=tmp_path / "a.glb", reference_numbers=numbers)
