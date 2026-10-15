"""Tests for `image_to_3dlab/glb_skin.py`: skeleton, rest mesh and posed skinning of a GLB."""

import numpy as np
import pytest

import _skinned_glb as fx
from image_to_3dlab.glb_skin import SkinnedGLB


def test_skeleton_and_rest_mesh(tmp_path):
    glb = SkinnedGLB(fx.write(tmp_path / "strip.glb"))
    parents, heads = glb.skeleton()
    assert parents == {"bone_0": None, "bone_1": "bone_0"}
    assert np.allclose(heads["bone_1"], [0, 1, 0])
    assert np.allclose(glb.mesh.vertices, fx.VERTICES)
    assert glb.mesh.faces.shape == (4, 3) and glb.animations() == []


def test_a_clip_bends_the_top_row(tmp_path):
    glb = SkinnedGLB(fx.write(tmp_path / "bend.glb", {"bend": [(0.0, fx.turn_z(0)), (1.0, fx.turn_z(90))]}))
    assert glb.animations() == [("bend", 1.0)]
    posed = glb.deform(glb.pose(0, 1.0))
    assert np.allclose(posed[5], [-1.0, 1.5, 0.0], atol=1e-5)  # (0.5, 2) turned 90 degrees about (0, 1)
    assert np.allclose(posed[:2], fx.VERTICES[:2], atol=1e-6)  # the bottom row stays put
    half = glb.deform(glb.pose(0, 0.5))[4]
    assert np.allclose(half, [-np.sin(np.pi / 4), 1 + np.cos(np.pi / 4), 0], atol=1e-5)
    assert np.allclose(glb.deform(glb.pose(0, 5.0)), posed)  # past the end holds the last key


def test_no_skin_is_refused(tmp_path):
    path = fx.write(tmp_path / "strip.glb")
    from image_to_3dlab.glb_turn import build_glb, parse_glb
    doc, binary = parse_glb(path.read_bytes())
    doc.pop("skins")
    doc["nodes"][0].pop("skin")
    path.write_bytes(build_glb(doc, binary))
    with pytest.raises(LookupError):
        SkinnedGLB(path)
