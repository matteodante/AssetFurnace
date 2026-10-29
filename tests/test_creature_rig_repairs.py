"""Repairs to the chosen skeleton: missed legs, twins, the middle plane, soles, the head, a jaw."""

from __future__ import annotations

import numpy as np
import pytest
from creature_rig_shapes import biped, biped_draw, box, quadruped, quadruped_draw

from image_to_3dlab.creature_rig import body as B
from image_to_3dlab.creature_rig import repairs as R
from image_to_3dlab.creature_rig.draws import clean, score
from image_to_3dlab.creature_rig.shell import Shell
from image_to_3dlab.creature_rig.skeleton import Bone, Tree


@pytest.fixture(scope="module")
def quad():
    return Shell(quadruped(), 1.05)


@pytest.fixture(scope="module")
def centre(quad):
    return B.centre_line_bones(quad)[0]


@pytest.fixture(scope="module")
def man():
    return Shell(biped(), 1.2)


def _named(d: dict, shell: Shell, cleaned: bool = True):
    tree = Tree.from_bones(d["parents"], d["heads"], d["tails"])
    if cleaned:
        tree = clean(tree, shell)[0]
    body = B.analyse(tree, shell)
    bones, _ = B.name_bones(tree, body, shell)
    return bones


def _by(bones):
    return {b.name: b for b in bones}


def _with_missing_legs(d: dict, shell: Shell, centre):
    tree = clean(Tree.from_bones(d["parents"], d["heads"], d["tails"]), shell)[0]
    body = B.analyse(tree, shell)
    missed = score(tree, shell)["missed"]
    tree, body, notes = R.add_missing_legs(tree, body, shell, missed, centre, B.MIDDLE_DRAWN)
    bones, _ = B.name_bones(tree, body, shell)
    return _by(bones), body, notes, missed


# --- missing legs ------------------------------------------------------------------------


def test_a_leg_the_draw_missed_comes_from_the_centre_line(quad, centre):
    names, body, notes, missed = _with_missing_legs(quadruped_draw(drop=("shoulder.L",)), quad, centre)
    assert len(missed) == 1
    assert body.kind == "quadruped"
    assert {"UpperArm.L", "Forearm.L", "FrontPaw.L", "FrontToes.L", "UpperArm.R"} <= set(names)
    assert names["FrontToes.L"].tail[:2] == pytest.approx([0.1, -0.38], abs=0.06)
    assert names["UpperArm.L"].kind == "front"
    assert any("centre line" in n for n in notes)
    assert not any(n.startswith("Chain") for n in names)


def test_front_legs_drawn_only_to_the_knee_are_replaced_not_kept_as_arms(quad, centre):
    d = quadruped_draw(drop=("wrist.L", "wrist.R"), ends={"elbow.L": (0.1, -0.38, 0.3), "elbow.R": (-0.1, -0.38, 0.3)})
    names, body, notes, missed = _with_missing_legs(d, quad, centre)
    assert len(missed) == 2
    assert body.kind == "quadruped" and body.upper_pairs == []
    assert {"FrontToes.L", "FrontToes.R"} <= set(names)
    assert not any(n.startswith(("Arm", "Shoulder", "Chain")) for n in names)
    assert any("drawn short" in n for n in notes)


def test_no_missed_contact_changes_nothing(quad, centre):
    tree = Tree.from_bones(*[quadruped_draw()[k] for k in ("parents", "heads", "tails")])
    body = B.analyse(tree, quad)
    same, body2, notes = R.add_missing_legs(tree, body, quad, [], centre, B.MIDDLE_DRAWN)
    assert same is tree and body2 is body and notes == []


# --- twins and the middle plane ----------------------------------------------------------


def test_twins_on_a_mirrored_body_become_exact_mirrors(quad):
    bones = _named(quadruped_draw(move={"knee.L": (0.12, 0.36, 0.38), "hock.L": (0.09, 0.4, 0.17)}), quad)
    notes = R.exact_twins(bones, quad)
    by = _by(bones)
    for n in ("Thigh", "Shin", "HindPaw", "HindToes"):
        left, right = by[n + ".L"], by[n + ".R"]
        assert left.head == pytest.approx(quad.mirror(right.head), abs=1e-9)
        assert left.tail == pytest.approx(quad.mirror(right.tail), abs=1e-9)
    assert by["Shin.L"].head == pytest.approx([0.11, 0.37, 0.37], abs=0.015)   # the mean of the two
    assert any("exact mirrors" in n for n in notes)


def test_twins_on_a_lopsided_body_are_only_made_equally_long():
    pack = box([0.15, 0.2, 0.0], [0.35, 0.6, 0.5])           # something beside the left hind leg only
    shell = Shell(quadruped() + [pack], 1.05)
    bones = _named(quadruped_draw(move={"knee.L": (0.1, 0.3, 0.38)}), shell, cleaned=False)
    by = _by(bones)
    before = {n: (by[n].tail - by[n].head) / by[n].length for n in ("Thigh.L", "Shin.L")}
    R.exact_twins(bones, shell)
    for n in ("Thigh", "Shin", "HindPaw", "HindToes"):
        assert by[n + ".L"].length == pytest.approx(by[n + ".R"].length, rel=1e-6)
    for n, u in before.items():                                # each kept its own direction
        assert (by[n].tail - by[n].head) / by[n].length == pytest.approx(u, abs=1e-9)
    assert by["Shin.L"].head == pytest.approx(by["Thigh.L"].tail, abs=1e-9)    # still joined


def test_twins_unlike_each_other_stay_as_drawn():
    pack = box([0.15, 0.2, 0.0], [0.35, 0.6, 0.5])
    shell = Shell(quadruped() + [pack], 1.05)
    bones = _named(quadruped_draw(), shell, cleaned=False)
    by = _by(bones)
    by["Thigh.L"].tail = by["Thigh.L"].head + (by["Thigh.L"].tail - by["Thigh.L"].head) * 3.0   # one giant thigh
    before = by["Thigh.R"].tail.copy()
    notes = R.exact_twins(bones, shell)
    assert by["Thigh.R"].tail == pytest.approx(before)
    assert any("as drawn" in n for n in notes)


def test_middle_bones_go_onto_the_middle_plane(quad):
    bones = _named(quadruped_draw(move={"back": (0.03, 0.0, 0.62)}), quad, cleaned=False)
    assert R.middle_plane(bones, quad) >= 1
    by = _by(bones)
    assert by["Spine1"].head[0] == pytest.approx(quad.mid)


# --- feet and head -----------------------------------------------------------------------


def test_a_standing_legs_last_bone_runs_along_its_sole_to_the_toes(man):
    bones = _named(biped_draw(), man)
    by = _by(bones)
    by["Toes.L"].tail = by["Toes.L"].head + [0, 0.0, -0.04]   # drawn as a stub straight down
    notes = R.soles(bones, man)
    assert by["Toes.L"].tail[1] == pytest.approx(-0.2, abs=0.02)      # the front of the foot
    assert by["Toes.L"].tail[2] < 0.03
    assert any("soles" in n for n in notes)


def test_a_head_drawn_ending_in_an_ear_runs_to_the_face(quad):
    d = quadruped_draw(drop=("head",), extra={"earL": ((0.04, -0.66, 0.98), "neck"),
                                              "earR": ((-0.04, -0.66, 0.98), "neck")},
                       ends={"earL": (0.07, -0.68, 1.04), "earR": (-0.07, -0.68, 1.04)})
    bones = _named(d, quad, cleaned=False)
    head = _by(bones)["Head"]
    assert abs(head.tail[0]) > 0.04
    notes = R.head_to_face(bones, quad)
    by = _by(bones)
    head = by["Head"]
    assert head.head[0] == pytest.approx(quad.mid) and head.tail[1] < -0.9
    ears = [b for b in bones if b.name.startswith("HeadX")]
    assert len(ears) == 1 and ears[0].parent == "Head"
    assert [b.name for b in bones].index("Head") < [b.name for b in bones].index(ears[0].name)
    assert notes


def test_an_open_mouth_gets_a_jaw_and_a_closed_one_does_not(quad):
    shell = Shell(quadruped(mouth=True), 1.05)
    skull = {"snout": ((0, -0.7, 0.95), "head"), "stub": ((0, -0.68, 0.93), "head")}   # a jaw drawn as a stub
    bones = _named(quadruped_draw(extra=skull, ends={"snout": (0, -0.92, 0.95), "stub": (0, -0.68, 0.84)}),
                   shell, cleaned=False)
    stub = next(b.name for b in bones if b.head[2] == pytest.approx(0.93))
    notes = R.open_mouth_jaw(bones, shell)
    by = _by(bones)
    jaw = by["Jaw"]
    assert jaw.parent == "Head" and jaw.kind == "jaw"
    assert -0.76 < jaw.head[1] < -0.68                         # the mouth's back corner
    assert jaw.tail[1] < -0.88 and jaw.tail[2] < jaw.head[2]   # the chin, in front and below
    assert stub not in by                                      # the drawn stub went
    assert notes and "stub" in notes[0]
    closed = _named(quadruped_draw(), quad)
    assert R.open_mouth_jaw(closed, quad) == []


def test_a_short_bone_on_one_side_of_the_head_goes_but_a_pair_stays(quad):
    bones = _named(quadruped_draw(), quad)
    head = _by(bones)["Head"]
    brow = Bone("HeadX1", "Head", head.head + [0.05, -0.05, 0.05], head.head + [0.07, -0.1, 0.07], "head")
    bones.append(brow)
    assert R.lone_head_bones(bones, quad)
    assert "HeadX1" not in _by(bones)
    ears = [Bone(f"HeadX{k}", "Head", head.head + [s * 0.03, 0, 0.05], head.head + [s * 0.06, 0, 0.09], "head")
            for k, s in ((1, 1), (2, -1))]
    bones += ears
    assert R.lone_head_bones(bones, quad) == []
    assert {"HeadX1", "HeadX2"} <= set(_by(bones))


def test_repair_bones_runs_them_all(quad):
    bones = _named(quadruped_draw(), quad)
    notes = R.repair_bones(bones, quad)
    assert any("soles" in n for n in notes) and any("mirror" in n for n in notes)
    assert np.all([b.head[0] == pytest.approx(quad.mid) for b in bones if b.kind in ("spine", "neck", "tail")])


def test_a_head_stopping_short_of_the_face_reaches_its_front(quad):
    bones = _named(quadruped_draw(ends={"head": (0, -0.72, 0.97)}), quad, cleaned=False)
    head = _by(bones)["Head"]
    start = head.head.copy()
    notes = R.head_to_face(bones, quad)
    assert head.head == pytest.approx(start)
    assert head.tail[1] < -0.93 and abs(head.tail[0]) < 0.04
    assert "short of the face" in notes[0]
    assert R.head_to_face(bones, quad) == []                   # once there, it stays
