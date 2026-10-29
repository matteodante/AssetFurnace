"""Joint limits from the creature rig's bone names: hinges fold one way, each part its range."""

from __future__ import annotations

import numpy as np

from image_to_3dlab.creature_rig import limits as lm

DEFAULT = [[-80.0, 80.0], [-20.0, 20.0], [-45.0, 45.0]]


def _rig(bones):
    """bones: (name, parent name or None, head, tail) in the prepared frame (Z up, facing -Y)."""
    names = [b[0] for b in bones]
    parents = [names.index(b[1]) if b[1] else -1 for b in bones]
    heads = np.array([b[2] for b in bones], float)
    tails = np.array([b[3] for b in bones], float)
    return names, parents, heads, tails


def _quadruped(hind_knee_y=0.3, front=(0.0, 0.0)):
    """Hips at the back, legs straight down unless told otherwise, toes forward."""
    bones = [("Root", None, (0, 0.3, 0.0), (0, 0.3, 0.1)), ("Hips", "Root", (0, 0.3, 0.8), (0, 0.0, 0.85)),
             ("Spine1", "Hips", (0, 0.0, 0.85), (0, -0.3, 0.85)), ("Neck1", "Spine1", (0, -0.3, 0.85), (0, -0.5, 1.05)),
             ("Head", "Neck1", (0, -0.5, 1.05), (0, -0.7, 1.1)), ("Tail1", "Hips", (0, 0.35, 0.8), (0, 0.6, 0.75))]
    for s, x in (("L", 0.15), ("R", -0.15)):
        bones += [(f"Thigh.{s}", "Hips", (x, 0.3, 0.75), (x, hind_knee_y, 0.42)),
                  (f"Shin.{s}", f"Thigh.{s}", (x, hind_knee_y, 0.42), (x, 0.3, 0.12)),
                  (f"HindPaw.{s}", f"Shin.{s}", (x, 0.3, 0.12), (x, 0.3, 0.03)),
                  (f"HindToes.{s}", f"HindPaw.{s}", (x, 0.3, 0.03), (x, 0.2, 0.0)),
                  (f"UpperArm.{s}", "Spine1", (x, -0.3, 0.75), (x, -0.3 + front[0], 0.42)),
                  (f"Forearm.{s}", f"UpperArm.{s}", (x, -0.3 + front[0], 0.42), (x, -0.3 + front[1], 0.12)),
                  (f"FrontPaw.{s}", f"Forearm.{s}", (x, -0.3 + front[1], 0.12), (x, -0.3, 0.03)),
                  (f"FrontToes.{s}", f"FrontPaw.{s}", (x, -0.3, 0.03), (x, -0.42, 0.0))]
    return _rig(bones)


def _limits(rig):
    names, parents, heads, tails = rig
    return dict(zip(names, lm.named_limits(names, parents, heads, tails, [DEFAULT] * len(names))))


def _wide_way(rig, name, lim):
    """Which way (world) the bone's tip goes turning into the wide side of its bend range."""
    names, parents, heads, tails = rig
    i = names.index(name)
    own = tails[i] - heads[i]
    frame = lm._frame(heads[i], tails[i])
    parent = tails[parents[i]] - heads[parents[i]]
    axis = lm.bend_axis(frame, own / np.linalg.norm(own), parent / np.linalg.norm(parent))
    low, high = lim[axis]
    sign = 1.0 if high > -low else -1.0
    return lm._tip_shift(frame, own, axis, 0.3 * sign)


def test_a_knee_folds_back_and_an_elbow_forward_one_way():
    rig = _quadruped()
    lim = _limits(rig)
    for s in "LR":
        assert _wide_way(rig, f"Shin.{s}", lim[f"Shin.{s}"])[1] > 0          # the shin swings back
        assert _wide_way(rig, f"Forearm.{s}", lim[f"Forearm.{s}"])[1] < 0    # the forearm comes forward
    shin = lim["Shin.L"]
    assert max(abs(v) for v in shin[0]) == lm.HINGE_FOLD and min(abs(v) for v in shin[0]) <= lm.HINGE_PAST + 1e-6
    assert shin[1] == [-26.0, 26.0] and lim["Forearm.L"][1] == [-80.0, 80.0]   # a forearm turns the palm over


def test_a_knee_standing_bent_can_straighten_and_folds_further_its_way():
    rig = _quadruped(hind_knee_y=0.15)          # the knee forward of the hip: the shin runs back and down
    lim = _limits(rig)
    low, _high = sorted(lim["Shin.L"][0], key=abs)
    assert abs(low) > 15                        # room to straighten past the rest pose
    assert _wide_way(rig, "Shin.L", lim["Shin.L"])[1] > 0


def test_a_birds_knee_folds_forward_and_its_foot_gets_a_paws_range():
    bones = [("Root", None, (0, 0, 0), (0, 0, 0.1)), ("Hips", "Root", (0, 0, 0.9), (0, -0.3, 1.0))]
    for s, x in (("L", 0.15), ("R", -0.15)):
        bones += [(f"Thigh.{s}", "Hips", (x, 0, 0.85), (x, 0.15, 0.5)),       # back and down to the "knee"
                  (f"Shin.{s}", f"Thigh.{s}", (x, 0.15, 0.5), (x, -0.05, 0.1)),  # forward and down
                  (f"Foot.{s}", f"Shin.{s}", (x, -0.05, 0.1), (x, -0.05, 0.02)),
                  (f"Toes.{s}", f"Foot.{s}", (x, -0.05, 0.02), (x, -0.2, 0.0))]
    rig = _rig(bones)
    lim = _limits(rig)
    assert _wide_way(rig, "Shin.L", lim["Shin.L"])[1] < 0                     # the rest pose says forward
    assert sorted(abs(v) for v in lim["Foot.L"][0]) == [100.0, 100.0]         # paw, not foot (45)


def test_a_paw_folds_its_own_way_and_only_twenty_degrees_the_other():
    lim = _limits(_quadruped())
    for name in ("HindPaw.L", "FrontPaw.L"):
        assert sorted(abs(v) for v in lim[name][0]) == [lm.PAW_AGAINST, 100.0], name
    rig = _quadruped()
    assert _wide_way(rig, "FrontPaw.L", lim["FrontPaw.L"])[1] > 0            # a wrist folds back
    assert _wide_way(rig, "HindPaw.L", lim["HindPaw.L"])[1] < 0              # a hock forward


def test_each_named_part_gets_its_range_and_an_unknown_bone_keeps_its_role():
    names, parents, heads, tails = _quadruped()
    names = [n.replace("Tail1", "bone_9") for n in names]
    lim = dict(zip(names, lm.named_limits(names, parents, heads, tails, [DEFAULT] * len(names))))
    assert lim["Neck1"][1] == [-50.0, 50.0] and lim["Head"][1] == [-50.0, 50.0]
    assert lim["Spine1"][1] == [-30.0, 30.0] and lim["HindToes.L"][1] == [-10.0, 10.0]
    assert lim["bone_9"] == DEFAULT


def test_an_arm_swings_back_only_so_far_but_a_front_leg_freely():
    arms = [("Root", None, (0, 0, 0), (0, 0, 0.1)), ("Hips", "Root", (0, 0, 1.0), (0, 0, 1.3)),
            ("Spine1", "Hips", (0, 0, 1.3), (0, 0, 1.5))]
    for s, x in (("L", 0.2), ("R", -0.2)):
        arms += [(f"UpperArm.{s}", "Spine1", (x, 0, 1.45), (1.5 * x, 0, 1.15)),
                 (f"LowerArm.{s}", f"UpperArm.{s}", (1.5 * x, 0, 1.15), (1.8 * x, 0, 0.9)),
                 (f"Thigh.{s}", "Hips", (x / 2, 0, 1.0), (x / 2, 0, 0.5)),
                 (f"Shin.{s}", f"Thigh.{s}", (x / 2, 0, 0.5), (x / 2, 0, 0.05))]
    rig = _rig(arms)
    lim = _limits(rig)
    names, _parents, heads, tails = rig
    i = names.index("UpperArm.L")
    channel, back = lm.swing_back(lm._frame(heads[i], tails[i]), tails[i] - heads[i])
    assert lim["UpperArm.L"][channel][1 if back > 0 else 0] == lm.ARM_BACK * back      # back: 75 at most
    assert abs(lim["UpperArm.L"][channel][0 if back > 0 else 1]) > lm.ARM_BACK         # forward: free
    front_leg = _limits(_quadruped())["UpperArm.L"]
    assert front_leg == DEFAULT                          # a leg it stands on: its role's limits


def test_a_jaw_only_opens():
    bones = [("Root", None, (0, 0, 0), (0, 0, 0.1)), ("Head", "Root", (0, 0, 1.0), (0, -0.3, 1.05)),
             ("Jaw", "Head", (0, -0.05, 0.98), (0, -0.28, 0.95))]
    rig = _rig(bones)
    lim = _limits(rig)
    _names, _parents, heads, tails = rig
    own = tails[2] - heads[2]
    frame = lm._frame(heads[2], tails[2])
    axis = lm.bend_axis(frame, own / np.linalg.norm(own), (tails[1] - heads[1]) / np.linalg.norm(tails[1] - heads[1]))
    low, high = lim["Jaw"][axis]
    assert sorted([abs(low), abs(high)]) == [3.0, 40.0] and low * high > 0       # one way, not through rest
    sign = 1.0 if high > 0 else -1.0
    assert lm._tip_shift(frame, own, axis, 0.3 * sign)[2] < 0                   # the chin goes down


def test_a_neck_and_head_bend_up_and_down_even_when_they_meet_a_little_to_one_side():
    """A drawn head that leans to one side of the neck made its sideways axis the bend axis:
    the head could nod only 35 degrees and swing 60 to the side."""
    names, parents, heads, tails = _quadruped()
    i = names.index("Head")
    tails[i] = heads[i] + np.array([0.1, -0.15, 0.1])     # leaning 35 degrees to one side
    lim = dict(zip(names, lm.named_limits(names, parents, heads, tails, [DEFAULT] * len(names))))
    assert lim["Head"][0] == [-60.0, 60.0] and lim["Head"][2] == [-35.0, 35.0]
