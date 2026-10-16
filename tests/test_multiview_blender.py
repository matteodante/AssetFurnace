"""The real round trip, when Blender is installed: render a small sphere unlit, carry the
colours back to its atlas, and find the texture again."""

from __future__ import annotations

import numpy as np
import pytest
from test_materials import SIZE, banded_albedo, make_glb, sphere, two_material_glb

from image_to_3dlab import materials as m
from image_to_3dlab import multiview as mv
from image_to_3dlab.blender import find_blender
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb, parse_glb

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


def quadrants(size: int = 128) -> np.ndarray:
    """Four flat colours, so a wrong UV set or a missed transform lands on the wrong one."""
    image = np.zeros((size, size, 3), np.uint8)
    half = size // 2
    image[:half, :half], image[:half, half:] = (220, 40, 30), (40, 200, 60)
    image[half:, :half], image[half:, half:] = (40, 60, 210), (230, 210, 40)
    return image


def add_uv_set(path, uv):
    document, binary = parse_glb(path.read_bytes())
    data = np.asarray(uv, np.float32).tobytes()
    binary = binary + b"\0" * (-len(binary) % 4)
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
    binary += data
    document["buffers"][0]["byteLength"] = len(binary)
    document["accessors"].append({"bufferView": len(document["bufferViews"]) - 1, "componentType": 5126,
                                  "count": len(uv), "type": "VEC2"})
    document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_1"] = len(document["accessors"]) - 1
    return document, binary


@needs_blender
def test_the_base_colours_own_uv_set_is_read_through_its_transform(tmp_path):
    albedo = quadrants()
    asset = make_glb(tmp_path / "two_uv.glb", albedo)
    _, uv0, _ = sphere()
    document, binary = add_uv_set(asset, np.c_[1.0 - uv0[:, 0], uv0[:, 1]])     # set 1: mirrored
    document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {
        "index": 0, "texCoord": 1, "extensions": {"KHR_texture_transform": {"offset": [0.1, 0.0]}}}
    document["extensionsUsed"] = ["KHR_texture_transform"]
    asset.write_bytes(build_glb(document, binary))
    views = mv.render_views(asset, tmp_path / "views", size=96, around=8, blender=BLENDER)
    surface = m.surfaces(EditGLB(asset.read_bytes()))[0]
    mean, seen = mv.reproject(views, surface, lambda view: view.color).mean()
    error = np.abs(mean[seen] - albedo[seen].astype(float))
    assert seen.mean() > 0.3 and np.median(error) < 6


@needs_blender
def test_cut_out_texels_are_not_rendered_as_solid(tmp_path):
    rgba = np.dstack([banded_albedo(), np.full((SIZE, SIZE), 255, np.uint8)])
    rgba[SIZE // 2:, :, 3] = 0                    # the lower hemisphere is cut out
    asset = make_glb(tmp_path / "cut.glb", rgba)
    document, binary = parse_glb(asset.read_bytes())
    document["materials"][0]["alphaMode"] = "MASK"
    asset.write_bytes(build_glb(document, binary))
    views = mv.render_views(asset, tmp_path / "views", size=96, around=8, blender=BLENDER)
    surface = m.surfaces(EditGLB(asset.read_bytes()))[0]
    _, seen = mv.reproject(views, surface, lambda view: view.color).mean()
    assert seen[: SIZE // 2 - 4].mean() > 0.3
    assert seen[SIZE // 2 + 4:].mean() < 0.01     # nothing behind a cut-out is a texel of it


@needs_blender
def test_mask_cut_off_follows_the_cutoff_not_the_raw_alpha(tmp_path):
    """Alpha 0.63 is kept and 0.39 is discarded at the default cutoff 0.5, never half-shown."""
    rgba = np.dstack([banded_albedo(), np.full((SIZE, SIZE), 160, np.uint8)])
    rgba[SIZE // 2:, :, 3] = 100
    asset = make_glb(tmp_path / "soft.glb", rgba)
    document, binary = parse_glb(asset.read_bytes())
    document["materials"][0]["alphaMode"] = "MASK"
    asset.write_bytes(build_glb(document, binary))
    views = mv.render_views(asset, tmp_path / "views", size=96, around=8, blender=BLENDER)
    surface = m.surfaces(EditGLB(asset.read_bytes()))[0]
    _, seen = mv.reproject(views, surface, lambda view: view.color).mean()
    assert seen[: SIZE // 2 - 4].mean() > 0.5
    assert seen[SIZE // 2 + 4:].mean() < 0.01


@needs_blender
def test_ring_views_keep_the_worlds_up_at_the_top_of_the_picture(tmp_path):
    # A 2D model shown a figure lying on its side sees something it was not trained on.
    asset = make_glb(tmp_path / "ball.glb", banded_albedo())
    views = mv.render_views(asset, tmp_path / "views", size=64, around=8, blender=BLENDER)
    for view in views[:-2]:
        rows = np.flatnonzero((view.material >= 0).any(axis=1))
        top, bottom = view.normal[rows[1]], view.normal[rows[-2]]
        shown = view.material[rows[1]] >= 0
        assert top[shown, 2].mean() > 0.5 and bottom[view.material[rows[-2]] >= 0, 2].mean() < -0.5


@needs_blender
def test_a_lit_picture_shows_shading_from_the_upper_left_and_the_data_passes_stay_flat(tmp_path):
    flat = np.full((SIZE, SIZE, 3), 180, np.uint8)
    asset = make_glb(tmp_path / "ball.glb", flat)
    views = mv.render_views(asset, tmp_path / "views", size=64, around=4, lit=True, blender=BLENDER)
    view = views[0]
    assert view.lit is not None and view.lit.shape == view.color.shape
    shown = view.material >= 0
    rows, cols = np.nonzero(shown)
    upper_left = shown & (np.arange(64)[:, None] < rows.mean()) & (np.arange(64)[None, :] < cols.mean())
    lower_right = shown & (np.arange(64)[:, None] > rows.mean()) & (np.arange(64)[None, :] > cols.mean())
    assert view.lit[upper_left].mean() > view.lit[lower_right].mean() + 20
    assert np.ptp(view.color[shown], axis=0).max() <= 2            # the unlit pass is still flat
    assert mv.render_views(asset, tmp_path / "plain", size=32, around=4, blender=BLENDER)[0].lit is None
