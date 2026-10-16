"""Chain retarget: a clip made for one creature, carried onto another skeleton by roles."""

from __future__ import annotations

import chain_retarget as cr
import creature_motion as cm
import numpy as np
import pytest

UP, FORWARD, LEFT = np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
FRAME = {"up": UP.tolist(), "forward": FORWARD.tolist(), "left": LEFT.tolist()}


def _rig(rows):
    parents = {n: p for n, p, _ in rows}
    heads = {n: np.array(h, float) for n, _, h in rows}
    order = [n for n, p, _ in rows if p is None]
    i = 0
    while i < len(order):
        order += [n for n, p, _ in rows if p == order[i]]
        i += 1
    return cm.Rig(parents, order, heads)


def _quadruped(prefix="", scale=1.0, shin=0.5):
    """Hips at the back, spine forward to a neck; four legs of three segments and a tail."""
    s = scale
    rows = [(f"{prefix}hips", None, (0, 1.0 * s, -0.4 * s)), (f"{prefix}spine", f"{prefix}hips", (0, 1.0 * s, 0.0)),
            (f"{prefix}chest", f"{prefix}spine", (0, 1.0 * s, 0.4 * s)),
            (f"{prefix}neck", f"{prefix}chest", (0, 1.2 * s, 0.6 * s)), (f"{prefix}head", f"{prefix}neck", (0, 1.4 * s, 0.7 * s)),
            (f"{prefix}tail0", f"{prefix}hips", (0, 1.0 * s, -0.6 * s)), (f"{prefix}tail1", f"{prefix}tail0", (0, 0.9 * s, -1.0 * s))]
    legs = []
    for end, z, parent in (("front", 0.4, "chest"), ("back", -0.4, "hips")):
        for side, x in (("L", 0.2), ("R", -0.2)):
            names = [f"{prefix}{end}{side}{k}" for k in range(4)]
            knee = 1.0 - (1.0 - shin)
            pts = [(x * s, 0.95 * s, z * s), (x * s, knee * s, (z + 0.05) * s), (x * s, 0.1 * s, z * s),
                   (x * s, 0.0, (z + 0.1) * s)]
            prev = f"{prefix}{parent}"
            for n, p in zip(names, pts):
                rows.append((n, prev, p))
                prev = n
            legs.append({"bones": names, "side": side, "end": end, "ankle": names[2]})
    roles = {"family": "quadruped", "legs": legs, "arms": [], "wings": [],
             "spine": [f"{prefix}hips", f"{prefix}spine", f"{prefix}chest"],
             "neck": [f"{prefix}neck", f"{prefix}head"], "tail": [f"{prefix}tail0", f"{prefix}tail1"], "body": [],
             "frame": FRAME, "ground": 0.0}
    return _rig(rows), roles


def _world(rig, turn, root):
    """Bone world matrices for one frame of `turn` (rest rotations are identity)."""
    total = cm.accumulate(rig, turn)
    heads = cm.posed_heads(rig, total, root)
    out = {}
    for b in rig.order:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = total[b], heads[b]
        out[b] = m
    return out


def _source(rig, roles, frames):
    rest = {b: np.block([[np.eye(3), rig.heads[b][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for b in rig.order}
    return cr.Source(parents=rig.parents, rest=rest, poses=[_world(rig, t, r) for t, r in frames], roles=roles)


def _walkish(rig, n=6):
    """Hips bob and roll a little, one front leg swings, the neck nods: a few frames."""
    frames = []
    for k in range(n):
        a = 20.0 * np.sin(2 * np.pi * k / n)
        turn = {"hips": cm.axis_angle(FORWARD, a / 4), "frontL0": cm.axis_angle(LEFT, a),
                "neck": cm.axis_angle(LEFT, -a / 2), "tail0": cm.axis_angle(UP, a)}
        root = rig.heads["hips"] + UP * 0.03 * np.cos(2 * np.pi * k / n)
        frames.append((turn, root))
    return frames


def _posed(rig, frame):
    turn, root = frame
    return cm.posed_heads(rig, cm.accumulate(rig, turn), root)


def test_the_same_skeleton_gets_the_source_motion_back():
    rig, roles = _quadruped()
    frames = _walkish(rig)
    out = cr.retarget(rig, roles, _source(rig, roles, frames))
    assert len(out) == len(frames)
    for want, got in zip(frames, out):
        a, b = _posed(rig, want), _posed(rig, got)
        for bone in rig.order:
            assert np.allclose(a[bone], b[bone], atol=1e-6), bone


def test_a_bigger_creature_facing_elsewhere_moves_the_same_way():
    src_rig, src_roles = _quadruped()
    frames = _walkish(src_rig)
    turn90 = cm.axis_angle(UP, 90.0)  # the target faces +X
    big, roles = _quadruped(prefix="t_", scale=2.0)
    big.heads = {b: turn90 @ h for b, h in big.heads.items()}
    roles["frame"] = {"up": UP.tolist(), "forward": (turn90 @ FORWARD).tolist(), "left": (turn90 @ LEFT).tolist()}
    out = cr.retarget(big, roles, _source(src_rig, src_roles, frames))
    for want, got in zip(frames, out):
        a, b = _posed(src_rig, want), _posed(big, got)
        for bone in src_rig.order:
            assert np.allclose(turn90 @ (2.0 * a[bone]), b["t_" + bone], atol=1e-5), bone


def test_a_planted_foot_stays_planted_on_a_leg_of_other_proportions():
    src_rig, src_roles = _quadruped()
    frames = []
    for k in range(5):  # the body sinks and rises over planted feet: only the knees bend
        drop = 0.05 * k
        frames.append(({}, src_rig.heads["hips"] - UP * drop))
    source = _source(src_rig, src_roles, frames)
    # the source's feet would sink with the body; plant them the way a real clip does
    for pose, (_, root) in zip(source.poses, frames):
        for leg in src_roles["legs"]:
            pts = [src_rig.heads[b] + (root - src_rig.heads["hips"]) for b in leg["bones"]]
            solved = cm.fabrik(pts, src_rig.heads[leg["bones"][-1]], bend=FORWARD)
            for b, p in zip(leg["bones"], solved):
                pose[b][:3, 3] = p
            for i, b in enumerate(leg["bones"][:-1]):
                pose[b][:3, :3] = cm.rotation_between(src_rig.heads[leg["bones"][i + 1]] - src_rig.heads[b],
                                                      solved[i + 1] - solved[i])
    rig, roles = _quadruped(prefix="t_", shin=0.3)  # a longer thigh and a shorter shin
    out = cr.retarget(rig, roles, source)
    for got in out:
        posed = _posed(rig, got)
        for leg in roles["legs"]:
            tip = leg["bones"][-1]
            assert np.allclose(posed[tip], rig.heads[tip], atol=2e-3), tip
            for a, b in zip(leg["bones"], leg["bones"][1:]):
                assert np.isclose(np.linalg.norm(posed[b] - posed[a]), np.linalg.norm(rig.heads[b] - rig.heads[a]))


@pytest.mark.parametrize(("n_target", "n_source", "rows"), [
    (3, 4, [0, 2, 3]), (4, 4, [0, 1, 2, 3]), (1, 2, [1]), (5, 4, [0, 1, 2, 2, 3]), (2, 1, [0, 0])])
def test_rows_of_legs_are_matched_front_to_back(n_target, n_source, rows):
    assert cr.pick_rows(n_target, n_source) == rows


def test_a_bone_on_no_chain_follows_its_parent():
    src_rig, src_roles = _quadruped()
    rig, roles = _quadruped()
    rig.parents["horn"], rig.heads["horn"] = "head", np.array([0, 1.6, 0.7])
    rig.order.append("horn")
    out = cr.retarget(rig, roles, _source(src_rig, src_roles, _walkish(src_rig)))
    for turn, _ in out:
        assert np.allclose(turn.get("horn", np.eye(3)), np.eye(3))


def test_a_long_body_follows_a_short_source_body_without_jumps():
    """A serpent of twelve bones bends like a source of four: same tip direction, and no
    bone turns much more than its neighbours."""
    src = _rig([(f"s{k}", f"s{k - 1}" if k else None, (0, 0.1, -0.5 * k)) for k in range(5)])
    tgt = _rig([(f"t{k}", f"t{k - 1}" if k else None, (0, 0.1, -2.0 * k / 12)) for k in range(13)])
    src_roles = {"family": "serpent", "legs": [], "arms": [], "wings": [], "spine": [], "neck": [], "tail": [],
                 "body": [f"s{k}" for k in range(5)], "frame": FRAME, "ground": 0.0}
    roles = {**src_roles, "body": [f"t{k}" for k in range(13)]}
    bend = [({f"s{k}": cm.axis_angle(UP, 20.0) for k in range(4)}, src.heads["s0"])]
    out = cr.retarget(tgt, roles, _source(src, src_roles, bend))
    a, b = _posed(src, bend[0]), _posed(tgt, out[0])
    want = (a["s4"] - a["s3"]) / np.linalg.norm(a["s4"] - a["s3"])
    got = (b["t12"] - b["t11"]) / np.linalg.norm(b["t12"] - b["t11"])
    assert want @ got > 0.99
    angles = [np.degrees(np.arccos(np.clip((np.trace(out[0][0].get(f"t{k}", np.eye(3))) - 1) / 2, -1, 1)))
              for k in range(1, 12)]
    assert max(angles) < 12.0


def test_a_leg_pointing_elsewhere_still_lifts_when_the_source_leg_lifts():
    """A spider's front leg reaches forward; the rig's points straight out to the side.
    The source's lift must stay a lift on the rig, not turn into a sideways sweep."""
    def creature(prefix, out):
        rows = [(f"{prefix}body", None, (0, 0.3, 0))]
        for side, x in (("L", 1.0), ("R", -1.0)):
            d = np.array([x * out[0], 0, out[1]])
            pts = [(0.1 * x, 0.3, 0), tuple(np.array([0.1 * x, 0.45, 0]) + 0.3 * d),
                   tuple(np.array([0.1 * x, 0.0, 0]) + 0.6 * d)]
            prev = f"{prefix}body"
            for k, p in enumerate(pts):
                rows.append((f"{prefix}{side}{k}", prev, p))
                prev = f"{prefix}{side}{k}"
        roles = {"family": "multileg", "arms": [], "wings": [], "spine": [], "neck": [], "tail": [], "body": [],
                 "legs": [{"bones": [f"{prefix}{s}{k}" for k in range(3)], "side": s, "end": "front"} for s in "LR"],
                 "frame": FRAME, "ground": 0.0}
        return _rig(rows), roles

    src, src_roles = creature("", (0.5, 0.85))  # forward and out
    tgt, roles = creature("t_", (1.0, 0.0))  # straight out
    lift = []
    for k in range(3):  # raise the left leg about the axis across its own direction
        leg_dir = src.heads["L2"] - src.heads["L0"]
        axis = np.cross(leg_dir, UP)
        lift.append(({"L0": cm.axis_angle(axis, 15.0 * k)}, src.heads["body"]))
    source = _source(src, src_roles, lift)
    source.roles["ground"] = -1.0  # no ground in reach: only the turn matters here
    roles["ground"] = -1.0
    out = cr.retarget(tgt, roles, source)
    rest_tip = tgt.heads["t_L2"]
    tip = _posed(tgt, out[-1])["t_L2"]
    assert tip[1] - rest_tip[1] > 0.1  # it rose
    assert abs(tip[2] - rest_tip[2]) < 0.05  # and did not sweep forward or back


def test_legs_that_all_start_at_one_hub_are_ordered_by_their_feet():
    heads = {"hub": np.zeros(3), "a0": np.zeros(3), "a1": np.array([0.5, 0, -0.5]),
             "b0": np.zeros(3), "b1": np.array([0.5, 0, 0.5])}
    legs = [{"bones": ["a0", "a1"], "side": "L"}, {"bones": ["b0", "b1"], "side": "L"}]
    assert [leg["bones"][0] for leg in cr._ordered(legs, heads, FORWARD, "L")] == ["b0", "a0"]


def test_two_chains_in_one_mesh_leg_follow_the_same_source_leg():
    """SkinTokens sometimes puts two chains in one leg; driven by two source legs they
    would tear the skin between them, as the procedural clips learned (`twins`)."""
    src_rig, src_roles = _quadruped()
    rig, roles = _quadruped(prefix="t_")
    extra = [f"t_twin{k}" for k in range(4)]
    prev = "t_hips"
    for name, bone in zip(extra, roles["legs"][3]["bones"]):  # beside the back right leg
        rig.parents[name], rig.heads[name] = prev, rig.heads[bone] + np.array([0.01, 0, 0])
        rig.order.append(name)
        prev = name
    roles["legs"].append({"bones": extra, "side": "R", "end": "back", "ankle": extra[2]})
    source = _source(src_rig, src_roles, _walkish(src_rig))
    pairs = {mine[0]: (kind, theirs[0]) for kind, mine, theirs in cr.match_chains(rig, roles, source)
             if kind != "body"}
    assert pairs["t_twin0"] == ("twin", "t_backR0") and pairs["t_backR0"] == ("leg", "backR0")
    assert pairs["t_frontR0"] == ("leg", "frontR0")
    for turn, _ in cr.retarget(rig, roles, source):  # the two chains stay parallel
        total = cm.accumulate(rig, turn)
        for a, b in zip(extra, roles["legs"][3]["bones"]):
            assert np.allclose(total[a], total[b], atol=1e-9)


def test_steps_are_scaled_by_leg_length_not_hip_height():
    """A flat spider with long legs: its hips are low, its steps long. Scaled by hip height
    a squat rig would take steps its legs cannot reach."""
    src_rig, src_roles = _quadruped()
    rig, roles = _quadruped(prefix="t_")
    roles["ground"] = -0.5  # the same legs on a higher-hipped body: hip height up, leg length not
    source = _source(src_rig, src_roles, [({}, src_rig.heads["hips"] + FORWARD * 0.1)])
    _, root = cr.retarget(rig, roles, source)[0]
    assert np.allclose(root - rig.heads["t_hips"], FORWARD * 0.1, atol=1e-6)


def test_a_body_roll_does_not_bend_a_limb_that_points_elsewhere():
    """Only the body rolls; the source's arm hangs on unbent. The rig's arm points another
    way at rest, and the heading turn must not make the roll into an arm-to-body bend."""
    def creature(prefix, heading):
        rows = [(f"{prefix}body", None, (0, 1.0, 0)), (f"{prefix}chest", f"{prefix}body", (0, 1.0, 0.4))]
        arms = []
        for side, x in (("L", 1.0), ("R", -1.0)):
            d = np.array([x * heading[0], 0, heading[1]])
            names = [f"{prefix}arm{side}{k}" for k in range(3)]
            for k, n in enumerate(names):
                rows.append((n, names[k - 1] if k else f"{prefix}chest", tuple(np.array([0, 1.0, 0.4]) + 0.3 * k * d)))
            arms.append({"bones": names, "side": side})
        roles = {"family": "x", "legs": [], "arms": arms, "wings": [], "spine": [f"{prefix}body", f"{prefix}chest"],
                 "neck": [], "tail": [], "body": [], "frame": FRAME, "ground": 0.0}
        return _rig(rows), roles

    src, src_roles = creature("", (1.0, 0.0))  # arms straight out
    tgt, roles = creature("t_", (0.3, 0.95))  # arms reaching forward
    roll = [({"body": cm.axis_angle(FORWARD, 30.0)}, src.heads["body"])]
    turn, _ = cr.retarget(tgt, roles, _source(src, src_roles, roll))[0]
    for k in range(3):
        assert np.allclose(turn.get(f"t_armL{k}", np.eye(3)), np.eye(3), atol=1e-9)


def test_the_library_spider_has_all_eight_legs():
    from image_to_3dlab import clip_library as cl

    legs = cl.SOURCES["spider"]["roles"]["legs"]
    assert len(legs) == 8 and len({leg["bones"][0] for leg in legs}) == 8
