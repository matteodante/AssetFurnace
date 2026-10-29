"""Cleaning a drawn skeleton, scoring it on the body, choosing the best of several."""

from __future__ import annotations

import json

import numpy as np
import pytest
from creature_rig_shapes import biped, biped_draw, quadruped, quadruped_draw

from image_to_3dlab.creature_rig import draws
from image_to_3dlab.creature_rig.shell import Shell
from image_to_3dlab.creature_rig.skeleton import Tree


@pytest.fixture(scope="module")
def quad():
    return Shell(quadruped(), 1.05)


@pytest.fixture(scope="module")
def man():
    return Shell(biped(), 1.2)


def _tree(d: dict) -> Tree:
    return Tree.from_bones(d["parents"], d["heads"], d["tails"])


def test_read_draw(tmp_path):
    path = tmp_path / "draw0.json"
    path.write_text(json.dumps(quadruped_draw()), encoding="utf-8")
    tree = draws.read_draw(path)
    assert tree.joints() == 23
    assert tree.tip.sum() == 6


def test_a_root_stub_above_the_pelvis_goes():
    d = quadruped_draw()
    d["names"].append("stub")
    d["heads"].append([0, 0.35, 0.75])
    d["tails"].append([0, 0.35, 0.62])
    d["parents"].append(-1)
    d["parents"][d["names"].index("hips")] = len(d["names"]) - 1
    tree = draws.drop_root_stubs(_tree(d))
    assert tree.points[tree.root] == pytest.approx([0, 0.35, 0.62])
    assert tree.joints() == 23


def test_fingers_fold_into_one_end_at_the_wrist(man):
    tree = _tree(biped_draw(fingers=True))
    folded, count = draws.fold_digits(tree, man.height)
    assert count == 8                                         # four fingers on each hand
    assert folded.joints() == tree.joints() - 16
    ends = folded.points[folded.tip]
    hand = ends[ends[:, 0] > 0.5]
    assert len(hand) == 1
    assert hand[0] == pytest.approx([0.69, 0.0, 0.87], abs=1e-6)   # the mean of the fingertips


def test_long_branches_do_not_fold(quad):
    tree = _tree(quadruped_draw())
    folded, count = draws.fold_digits(tree, quad.height)
    assert count == 0 and folded.n == tree.n


def test_a_branch_outside_the_skin_goes(quad):
    d = quadruped_draw(extra={"spike": ((0, 0.0, 0.9), "back"), "spike2": ((0, 0.0, 1.2), "spike")},
                       ends={"spike2": (0, 0.0, 1.4)})
    tree = _tree(d)
    cleaned, dropped = draws.drop_outside(tree, quad)
    assert dropped == 1
    assert cleaned.points[:, 2].max() < 1.06
    assert cleaned.joints() == tree.joints() - 2


def test_an_end_poking_out_is_pulled_back_in_but_not_below_the_floor(quad):
    d = quadruped_draw(ends={"head": (0, -1.2, 0.95), "hpaw.L": (0.1, 0.4, -0.1)})
    tree = draws.pull_in(_tree(d), quad)
    tips = tree.points[tree.tip]
    snout = tips[np.argmin(tips[:, 1])]
    assert -0.97 <= snout[1] <= -0.9                          # back inside the head (which ends at -0.95)
    assert quad.off_skin(snout[None])[0] <= 1.5 * quad.size
    assert tips[:, 2].min() >= quad.floor


def test_a_joint_left_without_an_end_gets_one():
    tree = Tree([[0, 0, 0], [0, 0, 1], [1, 0, 0]], [-1, 0, 0], [False, False, True])
    grown = draws.add_missing_ends(tree)
    assert grown.n == 4
    assert grown.points[-1] == pytest.approx([0, 0, 1.3])


def test_clean_runs_every_step(man):
    tree, info = draws.clean(_tree(biped_draw(fingers=True)), man)
    assert info == {"folded": 8, "dropped": 0}
    leaves = [i for i, k in enumerate(tree.children()) if not k]
    assert all(tree.tip[i] for i in leaves)


def test_score_counts_missed_contacts_the_mirror_gap_and_far_body(quad):
    good = draws.score(draws.clean(_tree(quadruped_draw()), quad)[0], quad)
    assert good["missed_contacts"] == 0 and good["contacts"] == 4
    assert good["mirror_gap"] < 0.01
    assert good["far_share"] < 0.05
    assert good["score"] < 1

    no_leg = draws.score(draws.clean(_tree(quadruped_draw(drop=("shoulder.L",))), quad)[0], quad)
    assert no_leg["missed_contacts"] == 1
    assert no_leg["score"] >= 100

    lopsided = quadruped_draw(move={"knee.L": (0.1, 0.3, 0.36), "hock.L": (0.1, 0.45, 0.16),
                                    "elbow.L": (0.1, -0.3, 0.36), "wrist.L": (0.1, -0.45, 0.16)})
    bent = draws.score(draws.clean(_tree(lopsided), quad)[0], quad)
    assert bent["missed_contacts"] == 0
    assert bent["mirror_gap"] > good["mirror_gap"] + 0.05


def test_a_draw_leaving_half_the_body_without_bones_is_marked_far(quad):
    head_only = quadruped_draw(drop=("tail1", "hip.L", "hip.R", "shoulder.L", "shoulder.R"))
    s = draws.score(draws.clean(_tree(head_only), quad)[0], quad)
    assert s["missed_contacts"] == 4
    lone = Tree([[0, -0.6, 0.9], [0, -0.9, 0.95]], [-1, 0], [False, True])
    assert draws.far_share(lone, quad) > draws.FAR_SHARE


def test_best_is_the_lowest_and_the_earlier_on_a_tie():
    assert draws.best([{"score": 5.0}, {"score": 2.0}, {"score": 2.0}]) == 1
    assert draws.best([None, {"score": 3.0}]) == 1
    assert draws.best([None, {"score": None}]) is None
