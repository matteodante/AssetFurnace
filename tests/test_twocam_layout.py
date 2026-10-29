"""Tests for `image_to_3dlab/twocam/layout.py`: the two-camera frame and its pixel maths."""

import numpy as np
import pytest
from twocam_toy import LAYOUT, toy_rig

from image_to_3dlab.twocam.layout import DIVIDER, Layout, LayoutError


def test_the_layout_comes_from_the_rig():
    layout = Layout.from_rig_info(toy_rig())
    assert (layout.w_front, layout.w_side, layout.m_per_px, layout.ground_v) == (640, 640, 0.004, 630)


def test_a_rig_without_a_layout_is_refused_rather_than_given_the_wolfs():
    info = toy_rig()
    del info["layout"]
    with pytest.raises(LayoutError, match="another creature's scale"):
        Layout.from_rig_info(info)


def test_an_incomplete_layout_is_refused():
    with pytest.raises(LayoutError):
        Layout.from_dict({"W_FRONT": 640})


def test_the_ground_sits_on_the_ground_row_in_both_views():
    layout = Layout.from_dict(LAYOUT)
    front, side = layout.project(np.array([[0.3, -0.2, 0.0]]))
    assert front[0, 1] == pytest.approx(630)
    assert side[0, 1] == pytest.approx(630)


def test_projecting_and_unprojecting_round_trips():
    layout = Layout.from_dict(LAYOUT)
    p = np.random.default_rng(0).uniform(-1, 1, (20, 3)) + [0, 0, 1]
    front, side = layout.project(p)
    xz = layout.unproject_front(front)
    yz = layout.unproject_side(side)
    np.testing.assert_allclose(xz, p[:, [0, 2]], atol=1e-12)
    np.testing.assert_allclose(yz, p[:, [1, 2]], atol=1e-12)


def test_the_head_is_on_the_right_of_the_side_view():
    layout = Layout.from_dict(LAYOUT)
    _, side = layout.project(np.array([[0, -0.5, 1.0], [0, 0.5, 1.0]]))  # head is -Y
    assert side[0, 0] > side[1, 0]


def test_split_cuts_at_the_front_views_width_and_checks_the_size():
    layout = Layout.from_dict({**LAYOUT, "W_FRONT": 420, "W_SIDE": 860})
    front, side = layout.split(np.zeros((720, 1280, 3)))
    assert front.shape[1] == 420 and side.shape[1] == 860
    with pytest.raises(LayoutError):
        layout.split(np.zeros((720, 1000, 3)))


def test_compose_lays_both_renders_on_the_backdrop_with_a_divider():
    layout = Layout.from_dict(LAYOUT)
    opaque = np.full((720, 640, 4), 255, np.uint8)
    frame = layout.compose(opaque, opaque)
    assert frame.shape == (720, 1280, 3)
    assert tuple(frame[10, 640]) == DIVIDER
    assert tuple(frame[10, 100]) == (255, 255, 255)
