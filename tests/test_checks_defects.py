"""Acceptance for feature 7: an asset broken on purpose is stopped by the right check.

Each fixture in `_check_fixtures.py` carries one defect. The clean asset passes every
check; each broken copy fails the check meant to catch it, and (where the defect is
isolated) no other.
"""

from __future__ import annotations

import _check_fixtures as fx
import pytest

from image_to_3dlab import checks

SIZE = {"height": fx.HEIGHT, "width": 2 * fx.RADIUS, "depth": 2 * fx.RADIUS}
# The silhouette band is warn-only until the pilot calibrates it; these tests hold it to fail.
STRICT = {"silhouette": {"warn_only": False}}


@pytest.fixture
def reference(tmp_path):
    """A variant of the clean asset: recognisably the same, 15% wider."""
    return fx.stretched(tmp_path / "reference.glb", x=1.15)


def run(glb, reference, overrides=None, numbers=SIZE):
    fx.sidecar(glb, fx.PIXAL3D_RECORD)
    asset = checks.Asset(glb=glb, reference=reference, reference_numbers=numbers)
    return checks.run_checks(asset, profile="prop", overrides={**STRICT, **(overrides or {})})


def failing(report):
    return {r["check"] for r in report["results"] if r["status"] == "fail"}


def test_the_clean_asset_passes_every_check(tmp_path, reference):
    report = run(fx.clean(tmp_path / "clean.glb"), reference)
    assert failing(report) == set()
    assert {r["status"] for r in report["results"]} == {"pass"}, report["results"]
    ran = {r["check"] for r in report["results"]} | {s["check"] for s in report["skipped"]}
    assert ran >= set(checks.REGISTRY) - {"gltf.valid"}


@pytest.mark.parametrize("build, stopped_by", [
    (fx.flipped, {"mesh.flipped_faces"}),
    (fx.inside_out, {"mesh.flipped_faces"}),
    (fx.holed, {"mesh.holes"}),
    (lambda p: fx.with_specks(p, count=5), {"mesh.loose_parts"}),
    (fx.overlapping_uv, {"uv.overlap"}),
    (fx.many_materials, {"budget.materials"}),
    (fx.big_texture, {"budget.textures"}),
    (lambda p: fx.scaled(p, 1.3), {"measure.height", "measure.footprint"}),
])
def test_each_defect_is_stopped_by_its_own_check(tmp_path, reference, build, stopped_by):
    report = run(build(tmp_path / "broken.glb"), reference)
    assert report["status"] == "fail"
    assert failing(report) == stopped_by, [r for r in report["results"] if r["status"] != "pass"]


def test_one_speck_warns_and_many_fail(tmp_path, reference):
    one = run(fx.with_specks(tmp_path / "one.glb", count=1), reference)
    status = {r["check"]: r["status"] for r in one["results"]}
    assert status["mesh.loose_parts"] == "warn"


def test_too_many_faces_for_the_rows_budget(tmp_path, reference):
    report = run(fx.clean(tmp_path / "a.glb"), reference, {"budget.faces": {"max": 1000}})
    assert failing(report) == {"budget.faces"}


def test_a_copy_of_the_reference_is_stopped_by_the_silhouette(tmp_path, reference):
    report = run(fx.stretched(tmp_path / "copy.glb", x=1.15), reference,
                 numbers={"height": fx.HEIGHT, "width": 2 * fx.RADIUS * 1.15, "depth": 2 * fx.RADIUS})
    assert failing(report) == {"silhouette"}
    assert "copy" in next(r for r in report["results"] if r["check"] == "silhouette")["message"]


def test_a_different_shape_is_stopped_by_the_silhouette(tmp_path, reference):
    report = run(fx.cube(tmp_path / "cube.glb"), reference,
                 numbers={"height": fx.HEIGHT, "width": fx.HEIGHT, "depth": fx.HEIGHT})
    assert "silhouette" in failing(report)
    assert "not recognisable" in next(r for r in report["results"] if r["check"] == "silhouette")["message"]


def test_a_bigger_copy_is_judged_by_size_not_by_silhouette(tmp_path, reference):
    """Size is the measurements' job; the silhouette compares shape at a common scale."""
    report = run(fx.scaled(tmp_path / "big.glb", 1.3), reference)
    by_name = {r["check"]: r for r in report["results"]}
    assert by_name["silhouette"]["status"] == "pass"
    assert by_name["measure.height"]["status"] == "fail"


def test_no_provenance_is_stopped_by_the_licence_check(tmp_path, reference):
    glb = fx.clean(tmp_path / "a.glb")
    report = checks.run_checks(checks.Asset(glb=glb, reference=reference, reference_numbers=SIZE),
                               profile="prop", overrides=STRICT)
    assert failing(report) == {"license.components"}


def test_a_hunyuan_model_is_stopped_by_the_licence_check(tmp_path, reference):
    glb = fx.clean(tmp_path / "a.glb")
    fx.sidecar(glb, fx.HUNYUAN_RECORD)
    report = checks.run_checks(checks.Asset(glb=glb, reference=reference, reference_numbers=SIZE),
                               profile="prop", overrides=STRICT)
    assert failing(report) == {"license.components"}


def test_one_material_with_many_draw_calls_is_over_budget(tmp_path, reference):
    """Codex review: draw calls count against the budget, not only materials."""
    glb = fx.one_material_many_draws(tmp_path / "many.glb")
    found = checks.run_checks(checks.Asset(glb=glb), only=["budget.materials"])["results"][0]
    assert found["value"] == 1
    assert found["status"] == "fail" and "draw call" in found["message"]


def test_a_smaller_inside_out_piece_is_not_hidden_by_a_bigger_good_one(tmp_path, reference):
    """Codex review: winding is judged per closed piece, not on the summed volume."""
    glb = fx.with_inward_piece(tmp_path / "two.glb")
    found = checks.run_checks(checks.Asset(glb=glb), only=["mesh.flipped_faces"])["results"][0]
    assert found["status"] == "fail" and "inward" in found["message"]


def test_the_inner_wall_of_a_cavity_is_not_a_flipped_piece(tmp_path):
    """An inward shell enclosed by the body is a cavity's wall (generated models have them)."""
    glb = fx.with_cavity(tmp_path / "hollow.glb")
    found = checks.run_checks(checks.Asset(glb=glb), only=["mesh.flipped_faces"])["results"][0]
    assert found["status"] == "pass", found
