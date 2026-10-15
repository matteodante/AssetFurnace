"""The real round trip, when Blender is installed: render a small sphere unlit, carry the
colours back to its atlas, and find the texture again."""

from __future__ import annotations

import numpy as np
import pytest
from test_materials import banded_albedo, make_glb, two_material_glb

from image_to_3dlab import materials as m
from image_to_3dlab import multiview as mv
from image_to_3dlab.blender import find_blender
from image_to_3dlab.edit_glb import EditGLB

BLENDER = find_blender()
needs_blender = pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")


@needs_blender
def test_colours_carried_back_from_the_views_match_the_texture(tmp_path):
    albedo = banded_albedo()
    asset = make_glb(tmp_path / "ball.glb", albedo)
    views = mv.render_views(asset, tmp_path / "views", size=96, around=8, blender=BLENDER)
    assert len(views) == 10
    surface = m.surfaces(EditGLB(asset.read_bytes()))[0]
    mean, seen = mv.reproject(views, surface, lambda view: view.color).mean()
    assert seen.mean() > 0.5
    error = np.abs(mean[seen] - albedo[seen].astype(float))
    assert np.median(error) < 6 and np.percentile(error, 95) < 40


@needs_blender
def test_each_pixel_says_which_material_it_shows(tmp_path):
    asset = two_material_glb(tmp_path / "two.glb", ["arm", "arm"])
    views = mv.render_views(asset, tmp_path / "views", size=48, around=4, blender=BLENDER)
    shown = set(np.unique(np.concatenate([v.material.ravel() for v in views])))
    assert {-1, 0} <= shown or {-1, 1} <= shown       # background and a tagged material
    assert shown <= {-1, 0, 1}
