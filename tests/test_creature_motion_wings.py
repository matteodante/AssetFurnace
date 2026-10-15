"""Tests for the winged and plant clips of `scripts/creature_motion.py`."""

import numpy as np
import pytest
from _creature_skeletons import build, chain, plant, quadruped
from test_creature_motion import positions, rig_of

import creature_motion as cm


def tip(fam, side):
    return next(w for w in fam["wings"] if w["side"] == side)["bones"][-1]


def test_wings_beat_down_fast_and_up_slow_in_mirror():
    rig, fam, _ = rig_of(quadruped(wings=True))
    assert fam["family"] == "winged"
    frames = cm.winged_hover(rig, fam)
    up = np.array(fam["frame"]["up"])
    heights = {s: np.array([positions(rig, f)[tip(fam, s)] @ up for f in frames]) for s in ("L", "R")}
    assert np.allclose(heights["L"], heights["R"], atol=1e-6)  # the two wings beat together
    span = heights["L"].max() - heights["L"].min()
    assert span > 0.2 * fam["height"]
    top, bottom = int(np.argmax(heights["L"][:-1])), int(np.argmin(heights["L"][:-1]))
    down = (bottom - top) % (len(frames) - 1)
    assert down < (len(frames) - 1) / 2  # the downstroke is the shorter half
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-6) for b in rig.order)


def test_hovering_lifts_the_feet_and_the_tail_follows():
    rig, fam, n = rig_of(quadruped(wings=True))
    frames = cm.winged_hover(rig, fam)
    for leg in fam["legs"]:
        foot = leg["bones"][-1]
        assert min(positions(rig, f)[foot][1] for f in frames) > rig.heads[foot][1] + 0.02
    tail = fam["tail"][-1]
    assert max(abs(positions(rig, f)[tail][0] - rig.heads[tail][0]) for f in frames) > 0.01


def test_a_winged_attack_spreads_the_wings_and_rears():
    rig, fam, _ = rig_of(quadruped(wings=True))
    frames = cm.make("attack", rig, fam)
    rear = positions(rig, frames[13])
    assert rear[tip(fam, "L")][1] > rig.heads[tip(fam, "L")][1] + 0.05
    front = [leg for leg in fam["legs"] if leg["end"] == "front"]
    assert all(rear[leg["bones"][-1]][1] > rig.heads[leg["bones"][-1]][1] + 0.05 for leg in front)
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-4) for b in rig.order)


@pytest.mark.parametrize("clip", ["sway", "attack"])
def test_a_plant_keeps_its_roots_and_moves_its_head(clip):
    rig, fam, n = rig_of(plant())
    assert fam["family"] == "plant"
    frames = cm.make(clip, rig, fam)
    base, top = fam["body"][0], fam["body"][-1]
    floor = min(rig.heads[b][1] for b in rig.order)
    for f in frames:
        p = positions(rig, f)
        assert np.allclose(p[base], rig.heads[base], atol=1e-6)
        assert min(p[b][1] for b in fam["body"]) >= floor - 1e-6
    reach = max(np.linalg.norm(positions(rig, f)[top] - rig.heads[top]) for f in frames)
    assert reach > 0.05 * fam["height"]
    first, last = positions(rig, frames[0]), positions(rig, frames[-1])
    assert all(np.allclose(first[b], last[b], atol=1e-6) for b in rig.order)


def test_a_plant_snaps_forward_after_drawing_back():
    rig, fam, _ = rig_of(plant())
    frames = cm.make("attack", rig, fam)
    forward = np.array(fam["frame"]["forward"])
    top = fam["body"][-1]
    along = [float((positions(rig, f)[top] - rig.heads[top]) @ forward) for f in frames]
    back, ahead = int(np.argmin(along)), int(np.argmax(along))
    assert along[back] < 0 < along[ahead] and back < ahead


def test_the_new_families_have_clips():
    assert cm.clips_for("winged") == ("hover", "attack")
    assert cm.clips_for("plant") == ("sway", "attack")


def test_a_plant_rooted_at_its_flower_keeps_its_roots_down():
    rows = [("head", None, (0, 1.1, 0.2))]
    rows += chain("stem", "head", [(0, 0.9, 0.1), (0, 0.6, 0.05), (0, 0.3, 0.02), (0, 0.06, 0)])
    for k, (x, z) in enumerate(((0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25), (0.18, 0.18))):
        rows += chain(f"root{k}", "stem3", [(x * 0.5, 0.02, z * 0.5), (x, 0.0, z)])
    rig, fam, n = rig_of(build(rows))
    for clip in ("sway", "attack"):
        frames = cm.make(clip, rig, fam)
        for f in frames:
            p = positions(rig, f)
            assert np.allclose(p[n["stem3"]], rig.heads[n["stem3"]], atol=1e-4)
            for k in range(5):  # the roots stay where they grew
                assert np.linalg.norm(p[n[f"root{k}1"]] - rig.heads[n[f"root{k}1"]]) < 0.02
        assert max(np.linalg.norm(positions(rig, f)[n["head"]] - rig.heads[n["head"]]) for f in frames) > 0.05
