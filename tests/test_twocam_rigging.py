"""Tests for `image_to_3dlab/twocam/rigging.py`: getting any rigged model ready for capture.

The skeletons here are built the way SkinTokens names them (`bone_N`, in no useful order),
so nothing can lean on names: a quadruped turned to three-quarters, a humanoid, an
eight-legged spider, and legs that start at the middle of the hips.
"""

import itertools
import math

import numpy as np
import pytest

from image_to_3dlab.twocam import rigging


class Builder:
    def __init__(self):
        self.names, self.parents, self.heads, self.tails = [], [], [], []

    def bone(self, parent, head, tail):
        self.names.append(f"bone_{len(self.names)}")
        self.parents.append(parent)
        self.heads.append(head)
        self.tails.append(tail)
        return len(self.names) - 1

    def chain(self, parent, points):
        for a, b in itertools.pairwise(points):
            parent = self.bone(parent, a, b)
        return parent

    def skeleton(self):
        return rigging.Skeleton(self.names, self.parents, np.array(self.heads, float), np.array(self.tails, float))


def quadruped(legs_from_middle=False):
    """Facing -Y, left is +X: hips at the back, a neck and head ahead, four legs, a tail."""
    b = Builder()
    root = b.bone(-1, (0, 0.3, 0.8), (0, 0.0, 0.85))
    chest = b.bone(root, (0, 0.0, 0.85), (0, -0.3, 0.85))
    b.chain(chest, [(0, -0.3, 0.85), (0, -0.5, 1.05), (0, -0.7, 1.1)])           # neck, head
    b.chain(root, [(0, 0.35, 0.8), (0, 0.6, 0.75), (0, 0.85, 0.7)])            # tail
    for sx in (1, -1):
        start = (0, -0.3, 0.8) if legs_from_middle else (0.15 * sx, -0.3, 0.75)
        b.chain(chest, [start, (0.16 * sx, -0.32, 0.42), (0.16 * sx, -0.34, 0.05), (0.16 * sx, -0.46, 0.0)])
        start = (0, 0.3, 0.75) if legs_from_middle else (0.14 * sx, 0.3, 0.75)
        b.chain(root, [start, (0.15 * sx, 0.22, 0.42), (0.15 * sx, 0.32, 0.05), (0.15 * sx, 0.2, 0.0)])
    return b.skeleton()


def humanoid():
    b = Builder()
    root = b.bone(-1, (0, 0, 1.0), (0, 0, 1.2))
    chest = b.bone(root, (0, 0, 1.2), (0, 0, 1.45))
    b.chain(chest, [(0, 0, 1.45), (0, 0, 1.6), (0, 0, 1.75)])
    for sx in (1, -1):
        b.chain(chest, [(0.18 * sx, 0, 1.42), (0.4 * sx, 0, 1.2), (0.55 * sx, 0, 0.98)])
        b.chain(root, [(0.1 * sx, 0, 0.95), (0.11 * sx, 0, 0.5), (0.11 * sx, 0, 0.08), (0.11 * sx, -0.15, 0.0)])
    return b.skeleton()


def spider():
    b = Builder()
    root = b.bone(-1, (0, 0, 0.4), (0, -0.2, 0.4))
    for k, y in enumerate((-0.15, -0.05, 0.05, 0.15)):
        for sx in (1, -1):
            b.chain(root, [(0.08 * sx, y, 0.4), (0.4 * sx, y * 2, 0.55), (0.7 * sx, y * 3, 0.0)])
    return b.skeleton()


def body_points(skeleton, per_bone=40, seed=0):
    """A stand-in mesh: points scattered along every bone, as a symmetric creature's surface."""
    rng = np.random.default_rng(seed)
    pts = []
    for h, t in zip(skeleton.heads, skeleton.tails):
        s = rng.uniform(0, 1, (per_bone, 1))
        pts.append(h + (t - h) * s)
    pts = np.concatenate(pts)
    return np.concatenate([pts, pts * [-1, 1, 1]])  # exactly mirror-symmetric


def turned(skeleton, verts, degrees):
    m = np.eye(4)
    a = math.radians(degrees)
    m[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    return skeleton.transformed(m), verts @ m[:3, :3].T


@pytest.mark.parametrize("degrees", [0, 35, 64, 120, 200, 300])
def test_a_creature_turned_any_way_is_turned_back_to_face_minus_y(degrees):
    sk, verts = turned(quadruped(), body_points(quadruped()), degrees)
    m, _ = rigging.normalising_matrix(verts, sk, length_m=1.6)
    back = sk.transformed(m)
    head = back.tails[3]  # the head's tip (bones: hips, chest, neck, head, tail...)
    hips = back.heads[0]
    assert head[1] < hips[1] - 0.3, "the head ends up at -Y"
    assert abs(head[0]) < 0.05, "and on the middle line"


def test_the_creature_is_sized_and_stood_on_the_ground():
    sk = quadruped()
    verts = body_points(sk)
    m, _ = rigging.normalising_matrix(verts, sk, length_m=2.0)
    v = verts @ m[:3, :3].T + m[:3, 3]
    assert np.ptp(v[:, 1]) == pytest.approx(2.0)
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert abs(v[:, 0].min() + v[:, 0].max()) < 1e-9


def test_feet_say_which_way_is_front_even_when_the_head_is_low():
    sk = quadruped()
    assert rigging.forward_sign(sk) == 1.0
    flipped = sk.transformed(np.diag([-1.0, -1.0, 1.0, 1.0]))  # half a turn: now facing +Y
    assert rigging.forward_sign(flipped) == -1.0


def test_quadruped_limbs_are_paired_named_front_first_and_painted():
    plan = rigging.plan_bones(quadruped(), tolerance=0.03)
    named = {n for n in plan.names if n.startswith("limb")}
    assert {"limb1_1_L", "limb1_1_R", "limb2_3_L", "limb2_3_R"} <= named
    assert plan.paint["1L"]["label"] == "front-left leg" and plan.paint["1L"]["colour_name"] == "pink"
    assert plan.paint["2R"]["label"] == "hind-right leg" and plan.paint["2R"]["colour_name"] == "cyan"
    assert rigging.colour_code(plan.paint).startswith("front-left leg pink, front-right leg blue")


def test_left_is_plus_x_when_facing_minus_y():
    sk = quadruped()
    plan = rigging.plan_bones(sk, tolerance=0.03)
    for i, name in enumerate(plan.names):
        if name.endswith("_L"):
            assert sk.heads[i, 0] > 0
        if name.endswith("_R"):
            assert sk.heads[i, 0] < 0


def test_legs_that_start_at_the_middle_of_the_hips_are_still_paired():
    """SkinTokens may start both legs at one point: judged by the first joint, they looked
    like part of the body."""
    plan = rigging.plan_bones(quadruped(legs_from_middle=True), tolerance=0.03)
    assert sum(n.endswith("_L") for n in plan.names) == 6
    assert sum(n.endswith("_R") for n in plan.names) == 6


def test_a_humanoid_gets_arms_over_legs():
    plan = rigging.plan_bones(humanoid(), tolerance=0.03)
    assert plan.paint["1L"]["label"] == "left arm" and plan.paint["2L"]["label"] == "left leg"


def test_a_spider_gets_four_pairs_of_legs_front_to_back():
    sk = spider()
    plan = rigging.plan_bones(sk, tolerance=0.03)
    roots = sorted((n, sk.heads[i, 1]) for i, n in enumerate(plan.names) if n.endswith("_1_L"))
    assert [n for n, _ in roots] == ["limb1_1_L", "limb2_1_L", "limb3_1_L", "limb4_1_L"]
    assert [y for _, y in roots] == sorted(y for _, y in roots), "numbered from the front"
    assert plan.paint["4L"]["label"] == "fourth left limb"


def test_a_lone_bone_off_the_middle_stays_in_the_middle():
    b = Builder()
    root = b.bone(-1, (0, 0, 1), (0, -0.3, 1))
    b.bone(root, (0.2, 0, 1), (0.4, 0, 1))  # a horn on one side only
    plan = rigging.plan_bones(b.skeleton(), tolerance=0.03)
    assert plan.names == ["bone_0", "bone_1"] and plan.sides == ["", ""]


def test_tails_join_up_so_each_joint_is_read_once():
    sk = quadruped()
    plan = rigging.plan_bones(sk, tolerance=0.03)
    kps, ends = rigging.keypoints(plan.names, sk.parents, sk.heads, plan.tails, sk.children)
    names = [k["name"] for k in kps]
    assert len(names) == len(set(names))
    leaves = [i for i, c in enumerate(sk.children) if not c]
    assert len(kps) == len(sk.names) + len(leaves)
    head_of = dict(ends)
    root_tail = head_of[plan.names[0]]
    assert root_tail == plan.names[1], "the hips' tail is the spine's head, not a tip of its own"


def test_limits_follow_the_role():
    plan = rigging.plan_bones(quadruped(), tolerance=0.03)
    by_name = dict(zip(plan.names, plan.limits))
    assert by_name["bone_0"] == rigging.ROOT_LIMITS
    assert by_name["limb1_1_L"] == rigging.LIMB_ROOT_LIMITS
    assert by_name["limb1_3_L"] == rigging.LIMB_END_LIMITS


def test_the_roll_keeps_bending_about_x_for_a_bone_in_the_body_plane():
    z = rigging.roll_vector((0, 0, 1), (0, -0.5, 1))  # a bone along the spine
    assert abs(z[0]) < 1e-9 and abs(np.linalg.norm(z) - 1) < 1e-9


def test_the_layout_leaves_room_to_rear_up_and_fits_the_frame():
    sk = quadruped()
    verts = body_points(sk)
    m, _ = rigging.normalising_matrix(verts, sk, length_m=1.6)
    v = verts @ m[:3, :3].T + m[:3, 3]
    lay = rigging.layout_for(v)
    assert lay["W_FRONT"] + lay["W_SIDE"] == 1280 and lay["W_FRONT"] % 2 == 0
    assert 630 * lay["M_PER_PX"] >= 1.75 * np.ptp(v[:, 2]) - 1e-9
    assert lay["W_SIDE"] * lay["M_PER_PX"] >= 1.45 * np.ptp(v[:, 1]) - 1e-9


def test_a_skeleton_with_two_roots_is_refused():
    with pytest.raises(rigging.RiggingError, match="one root"):
        rigging.Skeleton(["a", "b"], [-1, -1], np.zeros((2, 3)), np.ones((2, 3)))


def test_a_neck_can_lift_the_head_further_than_the_back_bends():
    """A howling wolf pinned its head at the one limit every middle bone shared."""
    sk = quadruped()
    plan = rigging.plan_bones(sk, tolerance=0.03)
    roles = rigging.middle_roles(sk, plan.sides)
    assert roles == {1: "spine", 2: "neck", 3: "neck", 4: "tail", 5: "tail"}
    assert plan.limits[2] == rigging.NECK_LIMITS and plan.limits[1] == rigging.SPINE_LIMITS
    assert plan.limits[4] == rigging.TAIL_LIMITS


def test_a_humanoid_neck_starts_above_the_shoulders():
    sk = humanoid()
    plan = rigging.plan_bones(sk, tolerance=0.03)
    roles = rigging.middle_roles(sk, plan.sides)
    assert roles[1] == "spine" and roles[2] == "neck" and roles[3] == "neck"


def test_fingers_keep_their_own_names_all_the_way_down():
    """SkinTokens gives humanoids five-fingered hands: only each finger's first bone had a
    letter, so the second bones of all five fingers shared one name."""
    b = Builder()
    root = b.bone(-1, (0, 0, 1.0), (0, 0, 1.4))
    for sx in (1, -1):
        hand = b.chain(root, [(0.2 * sx, 0, 1.4), (0.5 * sx, 0, 1.2), (0.6 * sx, 0, 1.1)])
        for f, y in enumerate((-0.03, 0.0, 0.03)):
            b.chain(hand, [(0.6 * sx, y, 1.1), (0.64 * sx, y, 1.08), (0.67 * sx, y, 1.06)])
    plan = rigging.plan_bones(b.skeleton(), tolerance=0.03)
    assert len(set(plan.names)) == len(plan.names)
    assert {"limb1_3a_L", "limb1_4a_L", "limb1_4c_R"} <= set(plan.names)


def quadruped_with(ears=False, tail_to_floor=False, long_left_leg=False):
    """The toy quadruped plus the parts real rigs have: ears, a tail resting on the floor,
    or one front leg a bone longer than the other (SkinTokens does that)."""
    b = Builder()
    root = b.bone(-1, (0, 0.3, 0.8), (0, 0.0, 0.85))
    chest = b.bone(root, (0, 0.0, 0.85), (0, -0.3, 0.85))
    head = b.chain(chest, [(0, -0.3, 0.85), (0, -0.5, 1.05), (0, -0.7, 1.1)])
    if ears:
        for sx in (1, -1):
            b.chain(head, [(0.05 * sx, -0.68, 1.15), (0.08 * sx, -0.66, 1.25)])
    tail_end = (0, 1.1, 0.0) if tail_to_floor else (0, 0.85, 0.7)
    b.chain(root, [(0, 0.35, 0.8), (0, 0.6, 0.6), tail_end])
    for sx in (1, -1):
        leg = [(0.15 * sx, -0.3, 0.75), (0.16 * sx, -0.32, 0.42), (0.16 * sx, -0.34, 0.05), (0.16 * sx, -0.46, 0.0)]
        if long_left_leg and sx == 1:
            leg.insert(2, (0.16 * sx, -0.33, 0.25))
        b.chain(chest, leg)
        b.chain(root, [(0.14 * sx, 0.3, 0.75), (0.15 * sx, 0.22, 0.42), (0.15 * sx, 0.32, 0.05), (0.15 * sx, 0.2, 0.0)])
    return b.skeleton()


def test_ears_are_named_left_and_right_but_are_not_limbs():
    """Counted as limbs, a wolf's ears made the prompt say "six limbs", took the legs'
    colours, and moved the neck's start back to the shoulders."""
    sk = quadruped_with(ears=True)
    plan = rigging.plan_bones(sk, tolerance=0.03)
    assert [g["label"] for g in plan.paint.values()] == [
        "front-left leg", "front-right leg", "hind-left leg", "hind-right leg"]
    assert {"part1_1_L", "part1_1_R"} <= set(plan.names)
    roles = rigging.middle_roles(sk, plan.sides, plan.limbs)
    assert roles[2] == "neck" and roles[3] == "neck"


def test_a_tail_on_the_floor_does_not_turn_the_creature_round():
    sk = quadruped_with(tail_to_floor=True)
    assert rigging.forward_sign(sk) == 1.0


def test_legs_a_bone_apart_are_still_a_pair():
    plan = rigging.plan_bones(quadruped_with(long_left_leg=True), tolerance=0.03)
    labels = [g["label"] for g in plan.paint.values()]
    assert labels[:2] == ["front-left leg", "front-right leg"]
    assert any(n.endswith("x_L") for n in plan.names), "the extra bone keeps its leg's side"
    assert not plan.unpaired


def test_a_branch_off_the_middle_with_no_mirror_is_reported():
    b = Builder()
    root = b.bone(-1, (0, 0, 1), (0, -0.3, 1))
    b.chain(root, [(0.2, 0, 1), (0.6, 0, 0.6), (0.8, 0, 0.1)])  # one arm, nothing on the other side
    plan = rigging.plan_bones(b.skeleton(), tolerance=0.03)
    assert plan.unpaired == ["bone_1"]
