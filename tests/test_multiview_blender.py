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
