"""Tests for `scripts/creature_motion.py`: the test clips keep their feet planted, their
lengths and their loop, on skeletons from `_creature_skeletons.py`."""

import numpy as np
import pytest
from _creature_skeletons import humanoid, quadruped, werewolf

import creature_motion as cm
from image_to_3dlab import creature_family as cf
from kimodo_retarget import pose_matrices


def rig_of(skeleton, family=None):
    parents, heads, names = skeleton
    order = [b for b, p in parents.items() if p is None]
    for b in order:
        order += [c for c, p in parents.items() if p == b]
    rig = cm.Rig(parents, order, heads)
    return rig, cf.classify(parents, heads, family=family), names


def positions(rig, frame):
    turn, root = frame
    return cm.posed_heads(rig, cm.accumulate(rig, turn), root)


def test_fabrik_reaches_and_keeps_lengths():
    points = [np.array(p, float) for p in ((0, 1, 0), (0, 0.5, 0.05), (0, 0, 0))]
    out = cm.fabrik(points, np.array([0.2, 0.1, 0.1]))
    assert np.allclose(out[0], points[0]) and np.allclose(out[-1], [0.2, 0.1, 0.1], atol=1e-5)
    assert np.allclose([np.linalg.norm(b - a) for a, b in zip(out, out[1:])], [np.linalg.norm(points[1] - points[0]),
                                                                              np.linalg.norm(points[2] - points[1])])
    far = cm.fabrik(points, np.array([0, -5.0, 0]))
    reach = sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:]))
    assert np.isclose(np.linalg.norm(far[-1] - far[0]), reach)  # straightened, not stretched


def test_fabrik_bends_a_straight_chain():
    points = [np.array(p, float) for p in ((0, 1, 0), (0, 0.5, 0), (0, 0, 0))]
    out = cm.fabrik(points, np.array([0, 0.1, 0]), bend=np.array([0, 0, 1.0]))
    assert np.allclose(out[-1], [0, 0.1, 0], atol=1e-4)
    assert out[1][2] > 0  # the knee went the way it was told
    assert np.allclose([np.linalg.norm(b - a) for a, b in zip(out, out[1:])], [0.5, 0.5])


@pytest.mark.parametrize("skeleton,family", [(werewolf, "digitigrade"), (humanoid, None)])
def test_biped_walk_plants_its_feet_and_loops(skeleton, family):
    rig, fam, n = rig_of(skeleton(), family)
    frames = cm.biped_walk(rig, fam)
    up, forward, left = (np.array(fam["frame"][k]) for k in ("up", "forward", "left"))
    legs = {leg["side"]: leg for leg in fam["legs"]}
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-6) for b in rig.order)  # loops without a pop
    toes = {s: np.array([positions(rig, f)[leg["bones"][-1]] for f in frames]) for s, leg in legs.items()}
    for side, track in toes.items():
        rest = rig.heads[legs[side]["bones"][-1]]
        assert track[:, 1].min() >= rest[1] - 1e-4  # never through the floor
        planted = np.abs(track[:, 1] - rest[1]) < 1e-6
        assert 0.45 < planted.mean() < 0.75
        travel = np.diff(track[planted] @ forward)
        steady = travel[np.abs(travel) < 0.1]  # within one stance, not across the swing
        assert np.allclose(np.diff(steady), 0, atol=1e-4)  # planted feet slide at a constant speed
    assert np.all((toes["L"] - toes["R"]) @ left > 0)  # feet never cross


def test_digitigrade_feet_stay_on_their_toes():
    rig, fam, n = rig_of(werewolf(), "digitigrade")
    leg = next(leg for leg in fam["legs"] if leg["side"] == "L")
    rest = rig.heads[leg["bones"][-1]] - rig.heads[leg["ankle"]]
    for frame in cm.biped_walk(rig, fam):
        p = positions(rig, frame)
        foot = p[leg["bones"][-1]] - p[leg["ankle"]]
        if abs(p[leg["bones"][-1]][1] - rig.heads[leg["bones"][-1]][1]) < 1e-6:  # planted
            assert np.allclose(foot, rest, atol=1e-4)
        assert foot @ np.array([0, 1, 0]) < 0  # the hock stays above the toes


def test_rotations_match_the_blender_pose_maths():
    rig, fam, _ = rig_of(werewolf(), "digitigrade")
    rng = np.random.default_rng(1)
    rest_rot = {}
    for b in rig.order:  # any rest orientation: the bones' own axes must not matter
        q = rng.normal(size=4)
        rest_rot[b] = cm.axis_angle(q[:3], float(q[3]) * 90)
    rest = {b: np.block([[rest_rot[b], rig.heads[b][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
            for b in rig.order}
    frame = cm.biped_walk(rig, fam)[7]
    world_rot, root = cm.frame_rotations(rig, rest_rot, *frame)
    pose = pose_matrices(rest, rig.parents, rig.order, world_rot, root)
    expected = positions(rig, frame)
    assert all(np.allclose(pose[b][:3, 3], expected[b], atol=1e-9) for b in rig.order)


def test_trot_moves_diagonal_pairs_together():
    rig, fam, _ = rig_of(quadruped())
    frames = cm.quadruped_trot(rig, fam)
    slot = {cm._trot_slot(leg): leg for leg in fam["legs"]}
    lifted = {k: np.array([positions(rig, f)[leg["bones"][-1]][1] - rig.heads[leg["bones"][-1]][1] > 1e-4
                           for f in frames]) for k, leg in slot.items()}
    assert (lifted["FL"] == lifted["BR"]).all() and (lifted["FR"] == lifted["BL"]).all()
    assert not (lifted["FL"] & lifted["FR"]).any()  # never both front feet off the ground
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-6) for b in rig.order)


@pytest.mark.parametrize("skeleton,family", [(werewolf, "digitigrade"), (quadruped, None)])
def test_attacks_start_and_end_at_rest(skeleton, family):
    rig, fam, _ = rig_of(skeleton(), family)
    frames = cm.make("attack", rig, fam)
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-6) for b in rig.order)  # back where it started
    for leg in fam["legs"]:  # standing where it stood (arms may start lowered from a T-pose)
        assert np.allclose(first[leg["bones"][-1]], rig.heads[leg["bones"][-1]], atol=1e-4)
    middle = positions(rig, frames[len(frames) // 2])
    assert max(np.linalg.norm(middle[b] - rig.heads[b]) for b in rig.order) > 0.05 * fam["height"]


def test_quadruped_attack_lifts_only_the_forefeet():
    rig, fam, _ = rig_of(quadruped())
    rear = positions(rig, cm.quadruped_attack(rig, fam)[13])
    for leg in fam["legs"]:
        rise = rear[leg["bones"][-1]][1] - rig.heads[leg["bones"][-1]][1]
        assert (rise > 0.1 * fam["height"]) if leg["end"] == "front" else abs(rise) < 1e-6


def test_clip_names():
    assert cm.clips_for("digitigrade") == ("walk", "attack")
    assert cm.clips_for("kraken") == ()  # not a family
    assert cm.clip_name("quadruped", "trot") == "quadruped-trot"
    with pytest.raises(ValueError):
        cm.make("moonwalk", *rig_of(werewolf())[:2])
    with pytest.raises(ValueError):
        cm.quadruped_trot(*rig_of(werewolf())[:2])
