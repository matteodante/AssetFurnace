"""Silhouette: shape at a common scale, four views, a band, and overlays for the studio."""

from __future__ import annotations

import _check_fixtures as fx
import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import checks
from image_to_3dlab.checks import silhouette


def scores(tmp_path, build_asset, build_reference):
    asset = checks.Asset(glb=build_asset(tmp_path / "a.glb"), reference=build_reference(tmp_path / "r.glb"))
    return silhouette.compare(asset)[0]


def test_a_uniformly_scaled_copy_scores_one_in_every_view(tmp_path):
    found = scores(tmp_path, fx.clean, lambda p: fx.scaled(p, 2.5))
    assert set(found) == set(silhouette.VIEWS)
    assert all(v == pytest.approx(1.0, abs=0.01) for v in found.values())


def test_a_stretch_shows_in_the_views_that_see_it(tmp_path):
    found = scores(tmp_path, fx.clean, lambda p: fx.stretched(p, x=1.5))
    assert found["side"] == pytest.approx(1.0, abs=0.01)  # the side view does not see width
    assert found["front"] < 0.75 and found["top"] < 0.75


def test_a_different_shape_scores_low(tmp_path):
    found = scores(tmp_path, fx.clean, fx.cube)
    assert np.mean(list(found.values())) < 0.5


def test_a_picture_reference_compares_the_front_view(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    mask = silhouette.mask_of_mesh(*checks.Asset(glb=glb).positions, "front", 256)
    picture = np.zeros((400, 300, 4), dtype=np.uint8)  # a cut-out: the subject is the opaque part
    picture[..., :3] = 200
    picture[60:316, 20:276, 3] = mask * 255
    Image.fromarray(picture).save(tmp_path / "concept.png")
    found = silhouette.compare(checks.Asset(glb=glb, reference=tmp_path / "concept.png"))[0]
    assert list(found) == ["front"]
    assert found["front"] > 0.9


def test_a_picture_without_alpha_uses_the_background(tmp_path):
    picture = np.full((100, 100, 3), 240, dtype=np.uint8)
    picture[20:80, 40:60] = (90, 40, 20)
    Image.fromarray(picture).save(tmp_path / "flat.jpg")
    mask = silhouette.subject_of_image(tmp_path / "flat.jpg")
    assert mask[50, 50] and not mask[5, 5]
    assert 1000 < mask.sum() < 1400


def test_the_check_writes_one_overlay_per_view(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    result = checks.run_checks(checks.Asset(glb=glb, reference=fx.stretched(tmp_path / "r.glb", x=1.2)),
                               only=["silhouette"])["results"][0]
    overlays = result["details"]["overlays"]
    assert set(overlays) == set(silhouette.VIEWS)
    for relative in overlays.values():
        assert (tmp_path / relative).is_file()


def test_one_weak_view_warns_even_when_the_mean_passes(tmp_path):
    glb = fx.clean(tmp_path / "a.glb")
    asset = checks.Asset(glb=glb, reference=fx.stretched(tmp_path / "r.glb", x=1.5))
    result = checks.run_checks(asset, only=["silhouette"], overrides={"silhouette": {
        "min": 0.5, "max": 0.99, "min_view": 0.7, "warn_only": False}})["results"][0]
    assert result["status"] == "warn"
    assert "view is off" in result["message"]


def test_fit_band_separates_approved_from_sent_back():
    labelled = [(0.95, False), (0.99, False), (0.72, True), (0.8, True), (0.88, True), (0.4, False)]
    low, high = silhouette.fit_band(labelled)
    assert 0.4 < low <= 0.72 and 0.88 <= high < 0.95
    with pytest.raises(ValueError):
        silhouette.fit_band([(0.3, False)])
