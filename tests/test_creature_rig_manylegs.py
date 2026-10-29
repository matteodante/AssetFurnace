"""Six or more legs: traced from the body, named Coxa/Femur/Tibia/Tarsus front to back."""

from __future__ import annotations

import numpy as np
import pytest
from creature_rig_shapes import biped, box, mirrored, quadruped, rod, spider

from image_to_3dlab.creature_rig import manylegs
from image_to_3dlab.creature_rig.shell import Shell


@pytest.fixture(scope="module")
def eight():
    return Shell(spider(), 0.605)


def _spider(legs_at, reach: float = 0.72, radius: float = 0.025):
    parts = [box([-0.15, -0.25, 0.25], [0.15, 0.25, 0.45])]
    side = []
    for y in legs_at:
        root, knee, foot = np.array([0.13, y, 0.35]), np.array([0.13 + 0.4 * reach, y * 1.3, 0.58]), \
            np.array([reach, y * 1.6, 0.02])
        side += [rod(root, knee, radius), rod(knee, foot, radius)]
    return parts + mirrored(side)


def test_the_body_and_its_leg_pieces(eight):
    body, _, n = manylegs.split_body(eight.solid, eight.depth)
    assert n == 8
    assert body.sum() > 0.5 * (0.3 * 0.5 * 0.2) / eight.size ** 3
    assert manylegs.leg_pieces(eight) == 8


def test_four_legs_are_not_many(eight):
    assert manylegs.rig(Shell(quadruped(), 1.05)) is None
    assert manylegs.rig(Shell(biped(), 1.2)) is None


def test_a_spider_gets_four_bones_a_leg_numbered_front_to_back(eight):
    bones, notes, facing = manylegs.rig(eight)
    by = {b.name: b for b in bones}
    assert facing == "-Y"
    assert {"Root", "Hips", "Tail1", "Head"} <= set(by)
    assert by["Root"].deform is False and by["Root"].head[2] == pytest.approx(eight.floor)
    for k in range(1, 5):
        for s in "LR":
            chain = [by[f"{seg}{k}.{s}"] for seg in manylegs.SEGMENTS]
            assert chain[1].parent == chain[0].name and chain[0].parent in ("Hips", "Tail1", "Head")
            assert chain[-1].tail[2] < 0.08 * eight.height              # stands on the floor
            assert (chain[0].head[0] > 0) == (s == "L")
            assert all(b.kind == "chain" and b.side == s for b in chain)
    assert by["Coxa1.L"].head[1] < by["Coxa4.L"].head[1]                # 1 is the front pair
    assert by["Head"].tail[1] < by["Tail1"].tail[1]
    assert "legs on the floor" in notes[0]


def test_twin_legs_get_their_joints_at_the_same_share_of_their_length(eight):
    bones, _, _ = manylegs.rig(eight)
    by = {b.name: b for b in bones}
    for k in range(1, 5):
        left = [by[f"{seg}{k}.L"].length for seg in manylegs.SEGMENTS]
        right = [by[f"{seg}{k}.R"].length for seg in manylegs.SEGMENTS]
        assert np.array(left) / sum(left) == pytest.approx(np.array(right) / sum(right), abs=0.03)


def test_legs_joining_the_back_of_the_body_turn_it_round():
    shell = Shell(_spider((0.08, 0.14, 0.2, 0.26)), 0.605)
    bones, notes, facing = manylegs.rig(shell)
    by = {b.name: b for b in bones}
    assert facing == "+Y"
    assert by["Head"].tail[1] > by["Tail1"].tail[1]
    assert by["Coxa1.L"].head[0] < 0                                  # facing +Y, its left is -X
    assert any("+Y" in n for n in notes)


def test_legs_too_thin_for_the_grid_are_traced_on_half_size_cells():
    shell = Shell(_spider((-0.2, -0.07, 0.07, 0.2), reach=0.45, radius=0.002), 0.605)
    result = manylegs.rig(shell)
    assert result is not None
    bones, notes, _ = result
    assert any("half-size" in n for n in notes)
    assert sum(1 for b in bones if b.name.startswith("Tarsus")) >= 6
