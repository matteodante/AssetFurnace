"""Tests for the many-legged and serpent clips of `scripts/creature_motion.py`."""

import numpy as np
import pytest
from _creature_skeletons import build, chain, reroot, snake, spider
from test_creature_motion import positions, rig_of

import creature_motion as cm


def lifted(rig, fam, frames, eps=1e-4):
    """{leg index: bool per frame}: which feet are off the ground."""
    return {i: np.array([positions(rig, f)[leg["bones"][-1]][1] - rig.heads[leg["bones"][-1]][1] > eps
                         for f in frames]) for i, leg in enumerate(fam["legs"])}


@pytest.mark.parametrize("legs", [6, 8])
def test_many_legs_walk_in_two_alternating_groups(legs):
    rig, fam, _ = rig_of(spider(legs))
    frames = cm.multileg_walk(rig, fam)
    up = lifted(rig, fam, frames)
    order = cm.leg_rows(fam)
    for side in ("L", "R"):
        row = order[side]
        for a, b in zip(row, row[1:]):  # neighbours on one side never swing together
            assert not (up[a] & up[b]).any()
    for a, b in zip(order["L"], order["R"]):  # nor do the two legs of a pair
        assert not (up[a] & up[b]).any()
    standing = np.sum([~v for v in up.values()], axis=0)
    assert standing.min() >= legs // 2  # always half the legs on the ground
    assert any(v.any() for v in up.values())  # and the legs do step
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-6) for b in rig.order)


def test_many_legs_attack_raises_the_front_pair():
    rig, fam, _ = rig_of(spider(8))
    frames = cm.make("attack", rig, fam)
    rows = cm.leg_rows(fam)
    front = {rows["L"][0], rows["R"][0]}
    rear = positions(rig, frames[13])
    for i, leg in enumerate(fam["legs"]):
        rise = rear[leg["bones"][-1]][1] - rig.heads[leg["bones"][-1]][1]
        assert (rise > 0.1 * fam["height"]) if i in front else abs(rise) < 1e-4
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], rig.heads[b], atol=1e-4) for b in rig.order)
    assert all(np.allclose(last[b], rig.heads[b], atol=1e-4) for b in rig.order)


@pytest.mark.parametrize("root", ["seg0", "seg11", "seg5"])
def test_serpent_slither_waves_more_at_the_tail(root):
    rig, fam, n = rig_of(reroot(snake(), root))
    assert fam["family"] == "serpent"
    frames = cm.serpent_slither(rig, fam)
    body = fam["body"]
    sway = {b: max(abs(positions(rig, f)[b][0] - rig.heads[b][0]) for f in frames) for b in body}
    assert sway[body[-1]] > 0.1 * fam["height"] or sway[body[-1]] > 0.05
    assert sway[body[0]] < 0.4 * sway[body[-1]]  # the head stays steady
    for f in frames:
        p = positions(rig, f)
        assert min(p[b][1] for b in body) >= min(rig.heads[b][1] for b in body) - 1e-6  # never underground
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-6) for b in rig.order)


@pytest.mark.parametrize("root", ["seg0", "seg11"])
def test_serpent_strike_lunges_the_head_and_keeps_the_tail(root):
    rig, fam, n = rig_of(reroot(snake(), root))
    frames = cm.make("attack", rig, fam)
    body = fam["body"]
    forward = np.array(fam["frame"]["forward"])
    # with the tail held and the body's length kept, the head lunges out of a coil, not past its rest
    along = [float(positions(rig, f)[body[0]] @ forward) for f in frames]
    rise = max(positions(rig, f)[body[0]][1] - rig.heads[body[0]][1] for f in frames)
    coiled = int(np.argmin(along))
    assert max(along[coiled:]) - along[coiled] > 0.1 * sum(np.linalg.norm(rig.heads[b] - rig.heads[a])
                                                         for a, b in zip(body, body[1:]))
    assert rise > 0
    tail = [positions(rig, f)[body[-1]] for f in frames]
    assert all(np.allclose(t, rig.heads[body[-1]], atol=1e-6) for t in tail)  # the tail is the anchor
    for frame in (frames[0], frames[-1]):
        assert all(np.allclose(positions(rig, frame)[b], rig.heads[b], atol=1e-6) for b in rig.order)


def test_new_families_have_clips():
    assert cm.clips_for("multileg") == ("walk", "attack")
    assert cm.clips_for("serpent") == ("slither", "attack")


def test_two_chains_inside_one_leg_move_as_one():
    rig, fam, n = rig_of(spider(8, twin=1))
    frames = cm.multileg_walk(rig, fam)
    legs = {leg["bones"][-1]: leg for leg in fam["legs"]}
    twin, own = n["twin3"], n["legL13"]
    assert twin in legs and own in legs
    for f in frames:
        p = positions(rig, f)
        assert np.allclose(p[twin] - rig.heads[twin], p[own] - rig.heads[own], atol=1e-4)


@pytest.mark.parametrize("root", ["seg0", "seg11", "seg5", "seg6"])
def test_a_serpent_rooted_anywhere_keeps_its_anchor(root):
    rig, fam, n = rig_of(reroot(snake(), root))
    body = fam["body"]
    for f in cm.serpent_slither(rig, fam):  # the head is held
        assert np.linalg.norm(positions(rig, f)[body[0]] - rig.heads[body[0]]) < 1e-3
    for f in cm.serpent_strike(rig, fam):  # the tail is held
        assert np.linalg.norm(positions(rig, f)[body[-1]] - rig.heads[body[-1]]) < 1e-3


def test_a_strike_never_drives_the_head_into_the_ground():
    parents, heads, names = snake()
    heads = dict(heads)
    heads[names["seg0"]] = heads[names["seg1"]] + np.array([0, 0.001, 0.15])  # head barely raised
    rig, fam, _ = rig_of((parents, heads, names))
    floor = min(rig.heads[b][1] for b in fam["body"])
    for f in cm.serpent_strike(rig, fam):
        p = positions(rig, f)
        assert min(p[b][1] for b in fam["body"]) >= floor - 1e-6


def _turned(rig, frame, bone):
    return cm.accumulate(rig, frame[0])[bone]


@pytest.mark.parametrize("root", ["seg5", "seg6"])
def test_a_mid_body_root_does_not_roll_over(root):
    rig, fam, n = rig_of(reroot(snake(), root))
    up = np.array(fam["frame"]["up"])
    for clip in (cm.serpent_slither, cm.serpent_strike):
        frames = clip(rig, fam)
        turns = [_turned(rig, f, n[root]) for f in frames]
        # stays right side up and never tips from one frame to the next (a strike may swing
        # the body round fast about the vertical, which is not rolling over)
        tilts = [np.degrees(np.arccos(np.clip(float((t @ up) @ up), -1, 1))) for t in turns]
        assert max(tilts) < 10 and max(abs(a - b) for a, b in zip(tilts, tilts[1:])) < 5


def test_a_long_serpent_keeps_its_neck_off_the_ground():
    rows = chain("seg", None, [(0, 0.06 if i == 0 else 0.05, 1.0 - 2.0 * i / 51) for i in range(52)])
    rig, fam, _ = rig_of(build(rows))
    floor = min(rig.heads[b][1] for b in fam["body"])
    for f in cm.serpent_strike(rig, fam):
        p = positions(rig, f)
        assert min(p[b][1] for b in fam["body"]) >= floor - 1e-6
