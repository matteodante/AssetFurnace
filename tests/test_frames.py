"""image_to_3dlab/frames.py: glTF's frame and Blender's, written once (code read 18/10/2026:
four copies, in the fit, B0's segmentation, the Unreal profile and the export's grip)."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_3dlab import frames, unreal_profile
from image_to_3dlab.slot_parts import fit, segment
from image_to_3dlab.slot_parts.camera import Camera


def test_blenders_importer_puts_a_gltf_point_at_x_minus_z_y_and_back():
    points = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 4.0]])
    assert frames.gltf_to_blender(points).tolist() == [[1.0, -3.0, 2.0], [-0.5, -4.0, 0.0]]
    assert frames.blender_to_gltf(frames.gltf_to_blender(points)) == pytest.approx(points)
    assert frames.gltf_to_blender([0, 0, 1]).tolist() == [0, -1, 0]   # glTF's front is Blender's -Y


def test_every_caller_turns_the_same_way():
    point = np.array([[0.1, 0.2, 0.3]])
    assert fit.to_blender(point) == pytest.approx(frames.gltf_to_blender(point))
    assert unreal_profile.gltf_to_blender([0.1, 0.2, 0.3]) == pytest.approx(frames.gltf_to_blender([0.1, 0.2, 0.3]))
    cam = Camera(fov=0.8, dist=2.8, size=400)
    assert segment.project(cam, frames.gltf_to_blender(point)) == pytest.approx(cam.project(point))


def test_one_orm_packing_for_the_material_step_and_the_unreal_export():
    from image_to_3dlab import materials

    rng = np.random.default_rng(0)
    mr = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    ao = rng.integers(0, 256, (4, 4), dtype=np.uint8)
    expected = materials.pack_orm(ao / 255.0, mr[..., 1] / 255.0, mr[..., 2] / 255.0)
    assert np.array_equal(unreal_profile.pack_orm(mr, ao), expected)
    assert np.array_equal(unreal_profile.pack_orm(mr, ao)[..., 0], ao)   # 8-bit values come back exactly
    assert fit.COVER_GAP == 0.06
