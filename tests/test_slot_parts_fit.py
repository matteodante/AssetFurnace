"""The pure half of fitting: placement through the body's camera, paired pieces per side,
turning a gauntlet onto the forearm, and putting cage spikes back."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_3dlab.slot_parts import fit
from image_to_3dlab.slot_parts.camera import Camera

CAM = Camera(fov=0.349, dist=2.8, size=100, scale=2.0, offset=(10.0, 20.0))


def test_gltf_points_become_blender_points():
    assert fit.to_blender(np.array([[1.0, 2.0, 3.0]]))[0] == pytest.approx([1.0, -3.0, 2.0])


def test_unproject_undoes_the_camera():
    gltf = np.array([[0.12, -0.3, 0.05]])
    (u, v), = CAM.project(gltf)
    x, z = fit.unproject(CAM, u, v, depth=-0.05)  # Blender y = -glTF z
    assert (x, z) == pytest.approx((0.12, -0.3), abs=1e-9)


def test_the_midline_maps_to_the_image_centre_line():
    assert fit.midline_px(CAM, 0.0) == pytest.approx(50 * 2.0 + 10.0)


def test_mirror_box_about_a_line():
    assert fit.mirror_box([10, 5, 30, 25], 50) == [70, 5, 90, 25]


def test_concept_boxes_scale_into_the_body_picture():
    assert fit.concept_to_body([100, 50, 200, 150], (1024, 1024), (768, 768)) == [75, 37.5, 150, 112.5]


def _cube(size=(0.2, 0.1, 0.4), centre=(0.0, 0.0, 0.0)):
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    return corners * np.array(size) + np.array(centre)


def test_placement_fills_the_box_at_the_given_depth():
    box_world = (-0.1, 0.2, 0.1, 0.6)  # x0, z0, x1, z1 the piece should cover (its own aspect)
    px = [CAM.project(np.array([[x, z, 0.0]]))[0] for x, z in ((box_world[0], box_world[3]), (box_world[2], box_world[1]))]
    box_px = [px[0][0], px[0][1], px[1][0], px[1][1]]
    m, info = fit.placement(_cube(), box_px, CAM, depth=0.0)
    moved = _cube() @ m[:3, :3].T + m[:3, 3]
    assert moved[:, 0].min() == pytest.approx(-0.1, abs=1e-6) and moved[:, 0].max() == pytest.approx(0.1, abs=1e-6)
    assert moved[:, 2].min() == pytest.approx(0.2, abs=1e-6) and moved[:, 2].max() == pytest.approx(0.6, abs=1e-6)
    assert info["aspect_mismatch"] == pytest.approx(1.0)


def test_a_thin_piece_is_stretched_in_depth_to_cover_the_body():
    box_px = [100, 100, 110, 120]
    _, info = fit.placement(_cube(size=(0.2, 0.01, 0.4)), box_px, CAM, depth=0.0, body_depth=0.3)
    assert info["depth_scale"] > info["scale"]
    _, whole = fit.placement(_cube(), box_px, CAM, depth=0.0, body_depth=0.3, frame=(-1, -1, 1, 1, 0.0))
    assert whole["depth_scale"] == whole["scale"]  # a whole-character frame keeps its depth


def test_rotation_between_turns_a_onto_b_even_when_opposite():
    for a, b in (([1, 0, 0], [0, 0, 1]), ([0, 1, 0], [0, -1, 0]), ([1, 1, 0], [1, 1, 0])):
        r = fit.rotation_between(np.array(a, float), np.array(b, float))
        assert r @ (np.array(a) / np.linalg.norm(a)) == pytest.approx(np.array(b) / np.linalg.norm(b))
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_a_vertical_gauntlet_turns_onto_a_horizontal_forearm():
    glove = _cube(size=(0.05, 0.05, 0.3), centre=(0.5, 0.0, 1.0))
    m, degrees = fit.turn_onto(glove, np.array([-1.0, 0.0, 0.0]))
    turned = glove @ m[:3, :3].T + m[:3, 3]
    extent = turned.max(0) - turned.min(0)
    assert extent[0] == pytest.approx(0.3, abs=1e-6) and degrees == pytest.approx(90.0)
    assert turned.mean(0) == pytest.approx(glove.mean(0))


def test_centre_across_leaves_the_move_along_the_limb():
    shift = fit.centre_across(np.zeros(3), np.array([0.3, 0.1, 0.2]), axis=np.array([1.0, 0, 0]))
    assert shift == pytest.approx([0.0, 0.1, 0.2])


def _strip(n=10):
    rest = np.array([[i * 0.01, 0.0, 0.0] for i in range(n)])
    edges = np.array([[i, i + 1] for i in range(n - 1)])
    return rest, edges


def test_a_flung_vertex_is_a_spike_and_goes_back():
    rest, edges = _strip()
    fitted = rest + [0.0, 0.002, 0.0]
    fitted[5] = [0.05, 0.3, 0.0]
    bad = fit.spikes(rest, fitted, edges)
    assert bad.tolist() == [i == 5 for i in range(10)]
    fixed = fit.smooth_spikes(fitted, edges, bad)
    assert fixed[5] == pytest.approx((fitted[4] + fitted[6]) / 2)
    assert np.allclose(fixed[[0, 9]], fitted[[0, 9]])


def test_a_spike_drawn_out_as_a_strip_is_found_by_its_stretched_edges():
    rest, edges = _strip(20)
    fitted = rest + [0.0, 0.002, 0.0]
    for k, i in enumerate(range(8, 12)):  # four neighbours flung out together, further and further
        fitted[i] = rest[i] + [0.0, 0.1 * (k + 1), 0.0]
    assert not fit.spikes(rest, fitted, edges)[9:11].all()  # their neighbours moved as far
    bad = fit.stretched(rest, fitted, edges)
    assert bad[8:12].all() and not bad[:6].any() and not bad[14:].any()
    fixed = fit.smooth_spikes(fitted, edges, bad)
    assert np.abs(fixed[8:12, 1]).max() < 0.01


def test_an_even_move_is_not_a_spike():
    rest, edges = _strip()
    assert not fit.spikes(rest, rest + [0.0, 0.2, 0.0], edges).any()


def test_coverage_report_takes_worst_and_mean():
    frames = [{"clipping": {"helmet": 0.01, "torso": 0.02}, "holes": 0.01},
              {"clipping": {"helmet": 0.03, "torso": 0.00}, "holes": 0.02}]
    report = fit.coverage_report(frames)
    assert report["clipping"]["helmet"] == {"max": 0.03, "mean": pytest.approx(0.02)}
    assert report["holes"]["max"] == 0.02 and report["worst_frame"] == 1
