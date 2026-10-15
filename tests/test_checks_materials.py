"""The materials checks as the checks registry runs them, on the small synthetic sphere."""

from __future__ import annotations

from test_materials import banded_albedo, lit, make_glb

from image_to_3dlab import checks
from image_to_3dlab.checks import material_checks

NAMES = ("material.tint_zones", "material.baked_light_direction", "material.baked_light_highlights")


def by_name(report):
    return {r["check"]: r for r in report["results"]}


def test_the_materials_checks_are_registered():
    assert set(NAMES) <= set(checks.REGISTRY)


def test_a_clean_banded_asset_passes_them(tmp_path):
    asset = checks.Asset(glb=make_glb(tmp_path / "clean.glb", banded_albedo()))
    found = by_name(checks.run_checks(asset, profile="creature", only=NAMES))
    assert {name: found[name]["status"] for name in NAMES} == dict.fromkeys(NAMES, "pass")
    assert found["material.tint_zones"]["value"] == 3


def test_painted_light_is_failed_by_the_registry_too(tmp_path):
    asset = checks.Asset(glb=make_glb(tmp_path / "lit.glb", lit(banded_albedo(), ambient=0.3)))
    found = by_name(checks.run_checks(asset, only=NAMES))
    assert found["material.baked_light_direction"]["status"] == "fail"
    assert found["material.baked_light_direction"]["details"]["direction"]


def test_the_analysis_runs_once_per_asset(tmp_path, monkeypatch):
    calls = []
    real = material_checks.materials.run_checks

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    material_checks._records.cache_clear()
    monkeypatch.setattr(material_checks.materials, "run_checks", counting)
    asset = checks.Asset(glb=make_glb(tmp_path / "once.glb", banded_albedo()))
    checks.run_checks(asset, only=NAMES)
    assert len(calls) == 1


def test_an_unmeasurable_light_check_is_a_warning_never_a_pass():
    record = {"check": "baked_light_direction", "status": "skipped", "value": None,
              "threshold": {"warn": 1.3, "fail": 1.45}, "message": "no colour group is large enough"}
    out = material_checks.as_registry_result("material.baked_light_direction", record)
    assert out["status"] == "warn"
    assert "could not measure" in out["message"]


def test_tint_zones_are_asked_of_creatures_and_equipment_not_of_props(tmp_path):
    """A rock or a barrel need not be recolourable; a monster or an armour piece must be."""
    plain = checks.Asset(glb=make_glb(tmp_path / "plain.glb", banded_albedo(colours=((150, 120, 90),))))
    prop = checks.run_checks(plain, profile="prop", only=NAMES)
    assert "material.tint_zones" in {s["check"] for s in prop["skipped"]}
    assert prop["status"] == "pass"
    for profile in ("creature", "equipment_part"):
        found = by_name(checks.run_checks(plain, profile=profile, only=NAMES))
        assert found["material.tint_zones"]["status"] == "fail"
