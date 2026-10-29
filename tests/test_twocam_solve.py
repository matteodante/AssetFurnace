"""Tests for `image_to_3dlab/twocam/solve.py`: the fit, and the checks before it.

Each fix over AnyMove's solver has a test here that fails on AnyMove's behaviour: the
layout read from the rig, limbs found without the wolf's names, a locked joint axis,
swaps decided over the whole clip, and a root that turns past half a turn.
"""

import copy
import math

import numpy as np
import pytest
import torch
from twocam_toy import toy_rig

from image_to_3dlab.twocam import exam, solve
from image_to_3dlab.twocam.layout import Layout
from image_to_3dlab.twocam.motion import matrices_to_quaternions


def _clip(info, seed=0, frames=24, noise=0.0):
    truth = exam.make_motion(info, seed=seed, frames=frames)
    joints = exam.motion_joints(info, truth)
    return truth, joints, exam.synthetic_detections(joints, Layout.from_rig_info(info), noise, seed=seed + 1)


def _names(info):
    return [k["name"] for k in info["keypoints"]]


# --- the rig ---------------------------------------------------------------------------


def test_limbs_are_found_from_the_rig_not_from_the_wolfs_names():
    info = toy_rig()
    names = _names(info)
    pairs = solve.limb_pairs(info)
    assert set(pairs) == {"leg_front", "leg_hind"}
    assert [(names[a], names[b]) for a, b in pairs["leg_front"]] == [
        ("shoulder_L", "shoulder_R"), ("elbow_L", "elbow_R"), ("paw_L", "paw_R"), ("toe_L", "toe_R")]


def test_the_feet_are_each_limbs_lowest_left_right_pair():
    assert solve.ground_joints(toy_rig()) == ["toe_L", "toe_R", "heel_L", "heel_R"]


def test_a_rig_a_little_lopsided_at_rest_still_gets_both_feet_of_a_pair():
    """A height band alone took 5 of the spider's 8 feet: one side stood a little higher."""
    info = toy_rig()
    names = _names(info)
    info["keypoints"][names.index("heel_R")]["rest"][2] = 0.06  # 4 cm above its twin
    assert {"heel_L", "heel_R"} <= set(solve.ground_joints(info))


def test_a_rig_without_limbs_falls_back_to_the_joints_on_the_floor():
    info = toy_rig()
    for b in info["bones"]:
        b["name"] = b["name"].replace("_L", "_l").replace("_R", "_r")
    for k in info["keypoints"]:
        k["name"] = k["name"].replace("_L", "_l").replace("_R", "_r")
    assert set(solve.ground_joints(info)) >= {"toe_l", "toe_r", "heel_l", "heel_r"}


@pytest.mark.parametrize("break_it, message", [
    (lambda i: i["bones"][3].update(parent=7), "parents must come before"),
    (lambda i: i["bones"][2].update(parent=-1), "only the first bone"),
    (lambda i: i["bones"][1].update(limits_deg=[[10, -10], [0, 0], [0, 0]]), "limits_deg"),
    (lambda i: i["keypoints"][0].update(end="middle"), "'head' or 'tail'"),
])
def test_a_rig_the_fit_cannot_use_is_refused_with_a_reason(break_it, message):
    info = toy_rig()
    break_it(info)
    with pytest.raises(solve.RigInfoError, match=message):
        solve.validate_rig_info(info)


def test_rest_pose_kinematics_put_every_joint_at_its_rest_position():
    info = toy_rig()
    rest = {"fps": 24, "bones": [b["name"] for b in info["bones"]], "trans": [[0, 0, 0]],
            "root_quat": [[1, 0, 0, 0]], "euler": np.zeros((1, len(info["bones"]), 3)).tolist()}
    joints = exam.motion_joints(info, rest)[0]
    np.testing.assert_allclose(joints, [k["rest"] for k in info["keypoints"]], atol=1e-9)


# --- the fit -------------------------------------------------------------------------------


def test_triangulation_inverts_the_projection():
    info = toy_rig()
    _, joints, d = _clip(info)
    np.testing.assert_allclose(solve.triangulate(d, Layout.from_rig_info(info)), joints, atol=1e-9)


def test_a_made_up_motion_is_recovered_within_two_centimetres():
    info = toy_rig()
    truth, joints, d = _clip(info, noise=2.0)
    motion = solve.capture(d, info, truth["fps"], iters=400)
    assert exam.score(info, joints, np.array(motion["joints"]))["mean_cm"] < 2.0
    assert motion["layout"]["M_PER_PX"] == 0.004
    assert motion["fps"] == truth["fps"]


def test_the_fit_uses_the_rigs_scale_not_the_wolfs():
    """AnyMove read the scale from an environment variable and fell back to the wolf's:
    any other creature was fitted at the wrong size without a word."""
    info = toy_rig()
    _, _, d = _clip(info)
    tri = solve.triangulate(d, Layout.from_rig_info(info))
    wolf = solve.triangulate(d, Layout())
    hips = _names(info).index("hips")
    rest_height = info["keypoints"][hips]["rest"][2]
    assert abs(tri[0, hips, 2] - rest_height) < 0.15
    assert abs(wolf[0, hips, 2] - rest_height) > 0.15


def test_a_locked_axis_does_not_turn_the_fit_into_nan():
    """limits [0, 0] made AnyMove divide by a zero span: every angle of every bone NaN."""
    info = toy_rig()
    shin = [b["name"] for b in info["bones"]].index("shin_front_L")
    info["bones"][shin]["limits_deg"][1] = [0, 0]
    truth, _, d = _clip(info)
    motion = solve.capture(d, info, truth["fps"], iters=150)
    euler = np.array(motion["euler"])
    assert np.isfinite(euler).all()
    assert np.abs(euler[:, shin, 1]).max() == 0


def test_the_fit_refuses_detections_for_another_rig():
    info = toy_rig()
    truth, _, d = _clip(info)
    d["front_uv"] = d["front_uv"][:, :-1]
    with pytest.raises(solve.DetectionError, match="this rig's reader"):
        solve.capture(d, info, truth["fps"], iters=10)


def test_a_clip_without_a_frame_rate_is_refused():
    info = toy_rig()
    _, _, d = _clip(info)
    with pytest.raises(solve.DetectionError, match="frame rate"):
        solve.capture(d, info, 0.0, iters=10)


def test_the_root_track_stays_continuous_past_half_a_turn():
    info = toy_rig()
    rig = solve.Rig(info)
    rest = np.array([k["rest"] for k in info["keypoints"]])
    hips = rest[_names(info).index("hips")]
    angles = np.radians(np.linspace(0, 270, 40))
    tri = []
    for a in angles:
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        tri.append((rest - hips) @ rz.T + hips)
    aa = solve.root_from_triangulation(np.array(tri), np.ones((40, len(rest))), rig, info)
    steps = np.linalg.norm(np.diff(aa, axis=0), axis=1)
    assert steps.max() < np.radians(15)
    turned = solve.axis_angle(torch.tensor(aa[-3])).numpy()
    brot = rig.B[0, :3, :3].numpy()
    world = brot @ turned @ brot.T
    q = matrices_to_quaternions(world)
    assert abs(abs(q[3]) - abs(math.sin(angles[-3] / 2))) < 0.05


# --- checks before the fit -------------------------------------------------------------------


def test_a_side_view_swap_is_found_and_put_back():
    info = toy_rig()
    _, _, d = _clip(info, frames=40)
    clean = copy.deepcopy(d)
    pairs = solve.limb_pairs(info)
    L, R = (np.array(x) for x in zip(*pairs["leg_front"]))
    for k in ("side_uv", "side_conf"):
        d[k][10:20, L], d[k][10:20, R] = d[k][10:20, R].copy(), d[k][10:20, L].copy()
    swapped = solve.fix_swaps(d, pairs)
    assert swapped == {"leg_front": 10, "leg_hind": 0}
    np.testing.assert_allclose(d["side_uv"], clean["side_uv"])


def test_one_unclear_frame_does_not_swap_a_leg():
    """AnyMove decided frame by frame; here every change of mind costs, so a single
    frame whose evidence is weaker than the cost of switching in and out stays put."""
    cost_keep = np.zeros(30)
    cost_swap = np.full(30, 5.0)
    cost_keep[15], cost_swap[15] = 9.0, 0.0  # one frame prefers swapping by 9 px
    assert not solve.viterbi_two_states(cost_keep, cost_swap, switch=8.0).any()
    cost_keep[12:19], cost_swap[12:19] = 9.0, 0.0  # seven frames do
    assert solve.viterbi_two_states(cost_keep, cost_swap, switch=8.0)[12:19].all()


def test_swaps_on_a_rig_without_limbs_do_nothing():
    info = toy_rig()
    _, _, d = _clip(info)
    assert solve.fix_swaps(d, {}) == {}


def test_views_played_out_of_step_are_re_timed():
    info = toy_rig()
    _, _, d = _clip(info, frames=60)
    lag = 4
    d["front_uv"] = np.concatenate([d["front_uv"][lag:], np.repeat(d["front_uv"][-1:], lag, 0)])
    before = solve.sync_report(d)
    assert before["best_lag_frames"] == -lag
    shift = solve.align_views(d, _names(info))
    after = solve.sync_report(d)
    assert after["mean_height_gap_px"] < 0.5 * before["mean_height_gap_px"]
    assert np.median(shift[10:-10]) == pytest.approx(-lag, abs=1)


@pytest.mark.parametrize("gap, feet, level", [(2.0, 0.95, "good"), (8.0, 0.95, "doubtful"),
                                              (2.0, 0.6, "doubtful"), (20.0, 0.95, "bad"), (2.0, 0.3, "bad")])
def test_the_verdict(gap, feet, level):
    v = solve.verdict({"mean_height_gap_px": gap}, {"score": feet}, height_px=250)
    assert v["level"] == level
    assert bool(v["reasons"]) == (level == "bad")


def test_a_clip_played_out_of_step_is_judged_after_re_timing():
    """Judged as generated, two halves merely out of step looked hopeless, and `fit`
    refused the very clips re-timing fixes."""
    info = toy_rig()
    _, _, d = _clip(info, frames=60, seed=3)
    lag = 4
    d["front_uv"] = np.concatenate([d["front_uv"][lag:], np.repeat(d["front_uv"][-1:], lag, 0)])
    _, report = solve.assess(d, info)
    assert report["front_shift_frames"] is not None
    assert report["mean_height_gap_px"] < report["before"]["mean_height_gap_px"]
    order = ["good", "doubtful", "bad"]
    assert order.index(report["verdict"]["level"]) <= order.index(report["verdict_as_generated"]["level"])


def test_the_root_quaternion_keeps_one_sign_through_half_a_turn():
    info = toy_rig()
    truth = exam.make_motion(info, seed=0, frames=30)
    yaw = np.radians(np.linspace(150, 210, 30))  # through half a turn about the pelvis' own Z
    truth["root_quat"] = [[math.cos(a / 2), 0, 0, math.sin(a / 2)] for a in yaw]
    joints = exam.motion_joints(info, truth)
    d = exam.synthetic_detections(joints, Layout.from_rig_info(info))
    motion = solve.capture(d, info, 24.0, iters=150)
    q = np.array(motion["root_quat"])
    assert (np.sum(q[1:] * q[:-1], axis=1) > 0).all()
