"""Body analysis (feet, head, tail, spine, limb pairs, props), the wing test, names by plan."""

from __future__ import annotations

import numpy as np
import pytest
from creature_rig_shapes import (
    biped,
    biped_draw,
    box,
    mirrored,
    quadruped,
    quadruped_draw,
)

from image_to_3dlab.creature_rig import body as B
from image_to_3dlab.creature_rig.draws import clean
from image_to_3dlab.creature_rig.shell import Shell
from image_to_3dlab.creature_rig.skeleton import Tree


@pytest.fixture(scope="module")
def quad():
    return Shell(quadruped(), 1.05)


@pytest.fixture(scope="module")
def man():
    return Shell(biped(), 1.2)


@pytest.fixture(scope="module")
def bat():
    return Shell(biped(wings=True), 1.2)


def _tree(d: dict, shell: Shell) -> Tree:
    return clean(Tree.from_bones(d["parents"], d["heads"], d["tails"]), shell)[0]


def _raw(d: dict) -> Tree:
    return Tree.from_bones(d["parents"], d["heads"], d["tails"])


def _bones(d: dict, shell: Shell, cleaned: bool = True):
    tree = _tree(d, shell) if cleaned else _raw(d)
    body = B.analyse(tree, shell)
    bones, _ = B.name_bones(tree, body, shell)
    return {b.name: b for b in bones}, body


# --- analysis ----------------------------------------------------------------------------


def test_a_quadruped_has_four_feet_a_head_a_tail_and_two_leg_pairs(quad):
    tree = _tree(quadruped_draw(), quad)
    body = B.analyse(tree, quad)
    p = tree.points
    assert len(body.feet) == 4                            # narrow hind legs 0.1 h off the middle stay feet
    assert p[body.head] == pytest.approx([0, -0.92, 0.95], abs=0.03)
    assert p[body.tail][1] > 0.9
    assert body.kind == "quadruped"
    hind, front = body.leg_pairs
    assert p[hind[0].end][1] > 0 > p[front[0].end][1]    # back to front
    assert {hind[0].side, hind[1].side} == {1, -1}
    assert body.upper_pairs == [] and body.singles == [] and body.props == []


def test_a_tail_lying_on_the_floor_curled_to_one_side_is_no_foot(quad):
    tree = _raw(quadruped_draw(move={"tail2": (0.05, 0.8, 0.3)}, ends={"tail2": (0.12, 1.15, 0.01)}))
    body = B.analyse(tree, quad)
    assert len(body.feet) == 4
    assert tree.points[body.tail] == pytest.approx([0.12, 1.15, 0.01], abs=0.03)
    assert body.kind == "quadruped"


def test_a_biped_and_the_wing_test(man, bat):
    tree = _tree(biped_draw(), man)
    body = B.analyse(tree, man)
    assert body.kind == "biped" and body.upper_kinds == ["arm"]
    assert body.flatness[0] < 2.5                         # round arms
    wtree = _tree(biped_draw(wings=True), bat)
    wing = B.analyse(wtree, bat)
    assert wing.kind == "winged" and wing.upper_kinds == ["wing"]
    assert wing.flatness[0] > 5


def test_an_unpaired_end_off_the_middle_is_held_and_gets_no_bones(quad):
    plain, _ = _bones(quadruped_draw(), quad)
    held, body = _bones(quadruped_draw(extra={"grip": ((0.3, -0.38, 0.6), "chest")},
                                       ends={"grip": (0.45, -0.38, 0.6)}), quad, cleaned=False)
    assert len(body.props) == 1
    assert sorted(held) == sorted(plain)
    assert any("held" in n for n in body.notes)


def test_short_pairs_at_the_head_are_horns_not_arms(quad):
    parts = {"snout": ((0, -0.8, 0.95), "head"),
             "horn.L": ((0.05, -0.62, 1.0), "head"), "horn.R": ((-0.05, -0.62, 1.0), "head")}
    d = quadruped_draw(extra=parts, ends={"snout": (0, -0.92, 0.95), "horn.L": (0.12, -0.55, 1.12),
                                          "horn.R": (-0.12, -0.55, 1.12)})
    names, body = _bones(d, quad, cleaned=False)
    assert body.kind == "quadruped" and body.upper_pairs == []
    horns = [b for n, b in names.items() if "X" in n]
    assert len(horns) == 2 and all(b.kind in ("part", "head") for b in horns)
    assert not any(n.startswith(("Shoulder", "Arm")) for n in names)


# --- names -------------------------------------------------------------------------------


def test_quadruped_bones_are_named_by_plan(quad):
    names, _ = _bones(quadruped_draw(), quad)
    assert sorted(names) == sorted([
        "Root", "Hips", "Tail1", "Tail2", "Spine1", "Spine2", "Neck1", "Head",
        *[f"{n}.{s}" for s in "LR" for n in ("Thigh", "Shin", "HindPaw", "HindToes",
                                             "UpperArm", "Forearm", "FrontPaw", "FrontToes")]])
    assert names["Root"].deform is False and names["Root"].head[2] == pytest.approx(0.0)
    assert names["Root"].head[:2] == pytest.approx(names["Hips"].head[:2])
    assert names["Hips"].head == pytest.approx([0, 0.35, 0.62])
    assert names["Hips"].tail == pytest.approx([0, 0.5, 0.6])          # to the first tail joint
    assert names["Tail1"].parent == "Hips" and names["Tail2"].parent == "Tail1"
    assert names["Spine1"].parent == "Hips"
    assert names["Thigh.L"].head == pytest.approx([0.1, 0.38, 0.55]) and names["Thigh.L"].side == "L"
    assert names["Shin.L"].head == pytest.approx([0.1, 0.38, 0.36])
    assert names["UpperArm.R"].head == pytest.approx([-0.1, -0.38, 0.55])
    assert names["UpperArm.R"].parent == "Spine2"
    assert names["Neck1"].kind == "neck" and names["Head"].kind == "head" and names["Tail1"].kind == "tail"
    assert names["Thigh.L"].kind == "hind" and names["UpperArm.L"].kind == "front"


def test_biped_bones_and_a_hand_whose_fingers_folded(man):
    names, _ = _bones(biped_draw(fingers=True), man)
    for n in ("Thigh", "Shin", "Foot", "Toes", "Arm", "LowerArm", "Hand"):
        assert f"{n}.L" in names and f"{n}.R" in names
    assert "Shoulder.L" not in names                       # three arm joints: no room for the top bone
    assert names["Hand.L"].tail == pytest.approx([0.69, 0.0, 0.87], abs=0.03)
    assert names["Arm.L"].parent == "Spine2"


def test_a_forearm_drawn_as_two_bones_side_by_side_keeps_one(man):
    d = biped_draw()
    extra = {"ulna.L": ((0.44, 0.02, 0.87), "elbow.L"), "ulna.R": ((-0.44, 0.02, 0.87), "elbow.R")}
    names, _ = _bones(_with(d, extra, {"ulna.L": (0.52, 0.02, 0.87), "ulna.R": (-0.52, 0.02, 0.87)}), man,
                      cleaned=False)
    assert not any("X" in n for n in names)
    assert names["Hand.L"].head == pytest.approx([0.6, 0.0, 0.87])


def _with(d: dict, extra: dict, ends: dict) -> dict:
    out = {k: list(v) for k, v in d.items()}
    for name, (pos, parent) in extra.items():
        out["names"].append(name)
        out["parents"].append(out["names"].index(parent))
        out["heads"].append(list(pos))
        out["tails"].append(list(ends[name]))
    return out


def test_wings_keep_their_rays_as_bones(bat):
    d = biped_draw(wings=True)
    rays = {f"ray{k}.{s}": ((sx * 0.8, 0.05, 0.875), f"wrist.{s}") for k in range(2) for s, sx in (("L", 1), ("R", -1))}
    ends = {f"ray{k}.{s}": (sx * 0.98, 0.05 + 0.2 * (k + 1), 0.875) for k in range(2) for s, sx in (("L", 1), ("R", -1))}
    d = _with(d, rays, ends)
    for i, n in enumerate(d["names"]):                     # the wrist forks: its own end moves onto the first ray
        if n.startswith("wrist"):
            d["tails"][i] = d["heads"][i]
    names, body = _bones(d, bat)
    assert body.kind == "winged"
    rays_l = [b for n, b in names.items() if n.startswith("WingHandX") and n.endswith(".L")]
    assert len(rays_l) >= 2 and all(b.kind == "wing" for b in rays_l)


def test_choose_matches_joints_to_the_plan_in_order():
    assert B.choose([0.3, 0.66, 0.88], [0.32, 0.66, 0.88]) == {0: 0, 1: 1, 2: 2}
    assert B.choose([0.1, 0.33, 0.5, 0.65, 0.9], [0.32, 0.66, 0.88]) == {0: 1, 1: 3, 2: 4}
    assert B.choose([0.6], [0.32, 0.66, 0.88]) == {1: 0}   # one joint: the knee it is nearest


def test_extra_joints_on_a_limb_are_segments(quad):
    d = quadruped_draw(extra={"mid.L": ((0.1, 0.38, 0.45), "hip.L"), "mid.R": ((-0.1, 0.38, 0.45), "hip.R")})
    for i, n in enumerate(d["names"]):                     # knees now hang from the extra joint
        if n.startswith("knee."):
            d["parents"][i] = d["names"].index("mid." + n[-1])
    names, _ = _bones(d, quad)
    assert any(n.startswith("HindSeg") and n.endswith(".L") for n in names)
    assert {"Thigh.L", "Shin.L", "HindPaw.L", "HindToes.L"} <= set(names)


def test_several_pairs_of_one_kind_are_numbered_from_the_front():
    def limb(at, side):
        return B.Limb(0, [0, 1, 2], at, side, True, 1.0)
    pairs = [(limb(0, 1), limb(0, -1)), (limb(5, 1), limb(5, -1)), (limb(9, 1), limb(9, -1))]
    body = B.Body(list(range(12)), 11, None, [], pairs, [(limb(10, 1), limb(10, -1))] * 2, ["arm", "arm"],
                  [1.0, 1.0], [], [], "centaur")
    plan = B.limb_plan(body)
    assert [(k, n) for k, _, n in plan] == [("hind", None), ("front", 1), ("front", 2), ("arm", 1), ("arm", 2)]
    assert plan[1][1][0].at == 9                           # front 1 is the pair nearest the head


# --- the centre-line skeleton --------------------------------------------------------------


def test_centre_line_bones_of_a_quadruped(quad):
    bones, body = B.centre_line_bones(quad)
    names = {b.name for b in bones}
    assert body.kind == "quadruped"
    assert {"Root", "Hips", "Head", "Thigh.L", "Shin.R", "HindToes.L", "UpperArm.L", "FrontToes.R"} <= names
    assert {"Tail1", "Tail2", "Tail3"} <= names
    feet = [b for b in bones if b.name.endswith("Toes.L") or b.name.endswith("Toes.R")]
    assert all(b.tail[2] < 0.1 for b in feet)


def test_centre_line_bones_of_a_biped_and_a_winged_body(man, bat):
    bones, body = B.centre_line_bones(man)
    assert body.kind == "biped"
    assert {"Thigh.L", "Toes.R", "Shoulder.L", "Arm.R", "Hand.L", "Head"} <= {b.name for b in bones}
    wbones, wbody = B.centre_line_bones(bat)
    assert wbody.kind == "winged"
    assert {"WingShoulder.L", "WingUpper.R", "WingHand.L"} <= {b.name for b in wbones}


def test_a_centaur_has_two_leg_pairs_and_arms():
    horse = [p for k, p in enumerate(quadruped()) if k not in (1, 3)]          # no horse neck or head
    rider = [box([-0.1, -0.55, 0.7], [0.1, -0.35, 1.3]), box([-0.07, -0.52, 1.32], [0.07, -0.38, 1.5]),
             box([-0.03, -0.47, 1.28], [0.03, -0.43, 1.34])]
    arms = mirrored([box([0.1, -0.49, 1.17], [0.5, -0.41, 1.25])])
    shell = Shell(horse + rider + arms, 1.5)
    extra = {"chest2": ((0, -0.45, 0.95), "chest"), "neck2": ((0, -0.45, 1.2), "chest2"),
             "head2": ((0, -0.45, 1.36), "neck2"),
             "sh.L": ((0.12, -0.45, 1.21), "neck2"), "el.L": ((0.3, -0.45, 1.21), "sh.L"),
             "wr.L": ((0.45, -0.45, 1.21), "el.L")}
    extra.update({k[:-2] + ".R": ((-v[0][0], v[0][1], v[0][2]), v[1][:-2] + ".R" if v[1].endswith(".L") else v[1])
                  for k, v in extra.items() if k.endswith(".L")})
    d = quadruped_draw(drop=("neck",), extra=extra,
                       ends={"head2": (0, -0.47, 1.48), "wr.L": (0.5, -0.45, 1.21), "wr.R": (-0.5, -0.45, 1.21)})
    names, body = _bones(d, shell)
    assert body.kind == "centaur"
    assert {"Thigh.L", "UpperArm.L", "Arm.L", "LowerArm.R", "Hand.L", "Head"} <= set(names)
    assert names["Head"].head[2] > 1.3


def test_is_trunk():
    assert B.is_trunk(B.Bone("Head", "Neck1", np.zeros(3), np.ones(3), "head"))
    assert not B.is_trunk(B.Bone("Arm.L", "Spine2", np.zeros(3), np.ones(3), "arm", "L"))


def test_no_bone_is_shorter_than_a_cell_and_hips_point_along_the_trunk(quad):
    d = quadruped_draw(move={"tail1": (0, 0.351, 0.62)})       # the first tail joint drawn on the hips
    names, _ = _bones(d, quad, cleaned=False)
    assert all(b.length >= B.MIN_BONE * quad.height for b in names.values())
    hips = names["Hips"]
    assert hips.tail == pytest.approx([0, 0.0, 0.62])          # on to the next trunk joint, not onto Tail1
    assert names["Tail1"].tail == pytest.approx([0, 0.75, 0.6])
