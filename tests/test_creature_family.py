"""Tests for `image_to_3dlab/creature_family.py`: which family a skeleton belongs to, and
which bones are legs, wings, tail and head.

Skeletons come from `_creature_skeletons.py`: tiny, synthetic, Y up and facing +Z like a
generated glTF.
"""

import json

import numpy as np
import pytest
from _creature_skeletons import build, chain, flat_snake, humanoid, plant, quadruped, snake, spider, werewolf

from image_to_3dlab import creature_family as cf


def test_humanoid_is_a_plantigrade_biped():
    parents, heads, n = humanoid()
    found = cf.classify(parents, heads)
    assert found["family"] == "humanoid" and found["confidence"] == "high"
    assert {leg["side"] for leg in found["legs"]} == {"L", "R"}
    assert not any(leg["digitigrade"] for leg in found["legs"])
    assert sorted(a["side"] for a in found["arms"]) == ["L", "R"]
    assert found["neck"][-1] == n["spine3"]


def soles(heads, n, legs, heel):
    """Ground vertices under each foot: from `heel` (in foot lengths, minus = behind the
    ankle) to the toe tip, as a grid two centimetres wide."""
    out = []
    for leg in legs:
        ankle, toe = heads[n[f"{leg}2"]], heads[n[f"{leg}3"]]
        for t in np.linspace(heel, 1.0, 12):
            for dx in (-0.01, 0.0, 0.01):
                out.append([ankle[0] + t * (toe[0] - ankle[0]) + dx, toe[1], ankle[2] + t * (toe[2] - ankle[2])])
    return np.array(out)


def test_werewolf_stands_on_its_toes():
    parents, heads, n = werewolf()
    found = cf.classify(parents, heads, points=soles(heads, n, ("legL", "legR"), heel=0.0))
    assert found["family"] == "digitigrade"
    left = next(leg for leg in found["legs"] if leg["side"] == "L")
    assert left["bones"] == [n["legL0"], n["legL1"], n["legL2"], n["legL3"]]
    assert left["digitigrade"] and left["ankle"] == n["legL2"]
    assert found["frame"]["forward"][2] > 0.99  # across the hips, though its feet turn out


def test_a_heel_behind_the_ankle_is_a_human_foot():
    parents, heads, n = werewolf()  # the same joints: only the sole tells them apart
    assert cf.classify(parents, heads, points=soles(heads, n, ("legL", "legR"), heel=-0.6))["family"] == "humanoid"
    assert cf.classify(parents, heads)["family"] == "humanoid"  # no mesh: joints alone are not enough


def test_quadruped_legs_front_and_back_with_a_tail():
    parents, heads, n = quadruped()
    found = cf.classify(parents, heads)
    assert found["family"] == "quadruped" and found["confidence"] == "high"
    ends = sorted((leg["end"], leg["side"]) for leg in found["legs"])
    assert ends == [("back", "L"), ("back", "R"), ("front", "L"), ("front", "R")]
    assert found["tail"] == [n["tail0"], n["tail1"], n["tail2"]]
    assert found["neck"][-1] == n["spine3"]
    assert found["frame"]["forward"][2] > 0.9


def test_quadruped_facing_sideways_is_still_found():
    parents, heads, _ = quadruped()
    turn = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)  # +Z -> +X
    found = cf.classify(parents, {b: turn @ p for b, p in heads.items()})
    assert found["family"] == "quadruped"
    assert found["frame"]["forward"][0] > 0.9


def test_four_legs_and_two_wings_is_winged():
    parents, heads, n = quadruped(wings=True)
    found = cf.classify(parents, heads)
    assert found["family"] == "winged"
    assert sorted(w["side"] for w in found["wings"]) == ["L", "R"]
    assert len(found["legs"]) == 4


@pytest.mark.parametrize("legs", [6, 8])
def test_many_legs(legs):
    parents, heads, _ = spider(legs)
    found = cf.classify(parents, heads)
    assert found["family"] == "multileg" and len(found["legs"]) == legs


def test_a_dragging_tail_is_not_a_leg():
    parents, heads, n = quadruped()
    heads = dict(heads)
    for k, z in enumerate((-0.6, -0.85, -1.1)):
        heads[n[f"tail{k}"]] = np.array([0, 0.0 if k else 0.3, z])
    assert cf.classify(parents, heads)["family"] == "quadruped"


def test_snake_and_plant():
    parents, heads, n = snake()
    found = cf.classify(parents, heads)
    assert found["family"] == "serpent"
    assert found["body"][0] == n["seg0"] and found["body"][-1] == n["seg11"]  # head first
    parents, heads, n = plant()
    found = cf.classify(parents, heads)
    assert found["family"] == "plant"
    assert found["body"][0] == n["stem0"] and found["body"][-1] == n["stem4"]  # base first


def test_choice_overrides_the_guess_but_keeps_the_roles():
    parents, heads, _ = humanoid()
    found = cf.classify(parents, heads, family="digitigrade")
    assert found["family"] == "digitigrade" and found["guessed"] == "humanoid"
    assert found["chosen"] is True and len(found["legs"]) == 2
    with pytest.raises(ValueError):
        cf.classify(parents, heads, family="kraken")


def test_family_json_round_trip(tmp_path):
    parents, heads, _ = werewolf()
    glb = tmp_path / "wolf_rigged.glb"
    path = cf.write_family(glb, cf.classify(parents, heads, family="digitigrade"))
    assert path == tmp_path / "wolf_rigged.family.json"
    assert cf.read_family(glb)["family"] == "digitigrade"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert cf.read_family(tmp_path / "missing_rigged.glb") is None


def test_classify_a_glb(tmp_path):
    import _skinned_glb as fx

    found = cf.classify_glb(fx.write(tmp_path / "strip.glb"))
    assert found["family"] in cf.FAMILIES and found["legs"] == []


@pytest.mark.parametrize("lift", [0.15, 0.06])  # well clear of the ground, and just short of it
def test_a_generated_spider_whose_legs_stop_in_the_air_is_still_a_spider(lift):
    parents, heads, _ = spider(8, lifted=("R1", "L2", "R3"), lift=lift)
    found = cf.classify(parents, heads)
    assert found["family"] == "multileg" and len(found["legs"]) == 8


def test_arms_hanging_in_an_a_pose_are_not_legs():
    parents, heads, _ = humanoid(a_pose=True)
    found = cf.classify(parents, heads)
    assert found["family"] == "humanoid" and len(found["legs"]) == 2 and len(found["arms"]) == 2


def test_a_spider_faces_the_camera_whichever_way_its_feet_lean():
    for splay in (0.25, -0.25):
        parents, heads, _ = spider(8, splay=splay, palps=True)
        found = cf.classify(parents, heads)
        assert found["family"] == "multileg" and found["wings"] == [] and len(found["legs"]) == 8
        assert found["frame"]["forward"][2] > 0.9


def test_a_flat_snake_is_a_serpent_facing_its_head():
    parents, heads, n = flat_snake()
    found = cf.classify(parents, heads)
    assert found["family"] == "serpent" and found["legs"] == []
    assert found["body"][0] == n["seg0"] and found["body"][-1] == n["seg12"]
    assert found["frame"]["forward"][0] < -0.3  # along the body, towards the head at -X


def test_legs_on_one_hub_still_split_front_and_back():
    parents, heads, _ = quadruped(hub=True)
    found = cf.classify(parents, heads)
    assert found["family"] == "quadruped"
    assert sorted((leg["end"], leg["side"]) for leg in found["legs"]) == [
        ("back", "L"), ("back", "R"), ("front", "L"), ("front", "R")]


def test_a_clip_record_carries_the_rigs_licence(tmp_path):
    rig = {"license": {"name": "MIT"}, "rig": {"tool": "SkinTokens"}}
    record = cf.clip_record(rig, "digitigrade-walk", "digitigrade", tool="test")
    assert record["license"] == {"name": "MIT"} and record["rig"] == {"tool": "SkinTokens"}
    assert record["motion"]["clip"] == "digitigrade-walk" and record["motion"]["license"]["name"] == "MIT"
    assert cf.clip_record(None, "x-walk", "x", tool="test")["license"] is None  # unknown, never invented


def test_a_low_slung_spider_is_still_a_spider():
    parents, heads, _ = spider(8)
    low = {b: p * np.array([1, 0.6, 1]) for b, p in heads.items()}  # the same spider, squatting
    found = cf.classify(parents, low)
    assert found["family"] == "multileg" and len(found["legs"]) == 8


def test_a_tail_hanging_just_above_the_ground_is_not_a_leg():
    parents, heads, n = quadruped()
    heads = dict(heads)
    for k, z in enumerate((-0.6, -0.85, -1.1)):
        heads[n[f"tail{k}"]] = np.array([0, (0.0 if k else 0.3) + 0.1, z])
    found = cf.classify(parents, heads)
    assert found["family"] == "quadruped" and len(found["legs"]) == 4


def test_a_clip_record_names_the_clip_not_the_rig(tmp_path):
    clip = tmp_path / "wolf_digitigrade-walk.glb"
    clip.write_bytes(b"clip bytes")
    rig = {"license": {"name": "MIT"}, "output": {"path": "wolf_rigged.glb", "sha256": "0" * 64,
                                                   "classification": "commercial"}}
    record = cf.clip_record(rig, "digitigrade-walk", "digitigrade", tool="test", output=clip)
    from image_to_3dlab.provenance import sha256_file
    assert record["output"]["sha256"] == sha256_file(clip) and record["output"]["path"] == str(clip)
    assert record["output"]["classification"] == "commercial"  # the licence class carries over


def test_toe_walking_legs_with_their_toes_further_forward_are_still_legs():
    parents, heads, n = werewolf()
    heads = dict(heads)
    for leg in ("legL3", "legR3"):
        heads[n[leg]] = heads[n[leg]] + np.array([0, 0, 0.05])
    found = cf.classify(parents, heads, family="digitigrade")
    assert len(found["legs"]) == 2


def test_raised_legs_sharing_a_second_hub_are_each_one_leg():
    rows = [("body", None, (0, 0.35, 0)), ("abdomen", "body", (0, 0.4, -0.35)), ("head", "body", (0, 0.36, 0.2)),
            ("hub", "body", (0.05, 0.36, 0.0))]
    for k in range(4):  # four grounded legs on the left
        z = 0.15 - 0.1 * k
        rows += chain(f"legL{k}", "body", [(0.08, 0.36, z), (0.35, 0.6, z * 1.5), (0.6, 0.25, z * 2), (0.7, 0.0, z * 2.2)])
    for k in range(4):  # four raised legs on the right, three of them from one hub
        z = 0.15 - 0.1 * k
        parent = "hub" if k else "body"
        rows += chain(f"legR{k}", parent, [(-0.08, 0.36, z), (-0.35, 0.6, z * 1.5), (-0.6, 0.3, z * 2),
                                             (-0.7, 0.15, z * 2.2)])
    parents, heads, _ = build(rows)
    found = cf.classify(parents, heads)
    bones = [b for leg in found["legs"] for b in leg["bones"]]
    assert len(bones) == len(set(bones))  # no bone in two legs
    assert len(found["legs"]) <= 8


def test_a_tail_lying_out_to_one_side_is_not_a_leg():
    """The generated dragon's tail rests on the ground curling off to one side."""
    parents, heads, n = quadruped(wings=True)
    heads = dict(heads)
    for k, pos in enumerate(((0.05, 0.45, -0.65), (0.35, 0.12, -0.9), (0.8, 0.0, -1.1))):
        heads[n[f"tail{k}"]] = np.array(pos)
    found = cf.classify(parents, heads)
    assert found["family"] == "winged" and len(found["legs"]) == 4


def test_a_wing_reaching_forward_and_up_is_not_the_head():
    parents, heads, n = quadruped(wings=True)
    heads = dict(heads)
    for k, pos in enumerate(((0.1, 0.75, 0.3), (0.4, 0.95, 0.6), (0.6, 1.05, 0.85), (0.75, 1.0, 1.0))):
        heads[n[f"wingL{k}"]] = np.array(pos)  # the left wing swept forward, above the head
    found = cf.classify(parents, heads)
    assert found["family"] == "winged" and found["neck"][-1] == n["spine3"]
    assert sorted(w["side"] for w in found["wings"]) == ["L", "R"]


def test_a_plant_rooted_at_its_flower_is_still_a_plant():
    """SkinTokens rooted a generated plant at its flower: the stem runs down to a fan of roots."""
    rows = [("head", None, (0, 1.1, 0.2))]
    rows += chain("stem", "head", [(0, 0.9, 0.1), (0, 0.6, 0.05), (0, 0.3, 0.02), (0, 0.06, 0)])
    for k, (x, z) in enumerate(((0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25), (0.18, 0.18))):
        rows += chain(f"root{k}", "stem3", [(x * 0.5, 0.02, z * 0.5), (x, 0.0, z)])
    parents, heads, n = build(rows)
    found = cf.classify(parents, heads)
    assert found["family"] == "plant" and found["legs"] == []
    assert found["body"][0] == n["stem3"] and found["body"][-1] == n["head"]  # base first, up to the flower


def test_a_fan_of_roots_under_a_lone_stem_is_not_a_foot():
    """The real generated plant: rooted at its flower, with small bones beside the stem at
    the top, so the flower is a fork and the stem looks like a leg down to its roots."""
    rows = [("head", None, (0, 1.1, 0.2)), ("petal", "head", (0.05, 1.15, 0.2)), ("lip", "head", (-0.05, 1.12, 0.25))]
    rows += chain("stem", "head", [(0, 0.9, 0.1), (0, 0.6, 0.05), (0, 0.3, 0.02), (0, 0.06, 0)])
    for k, (x, z) in enumerate(((0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25), (0.18, 0.18))):
        rows += chain(f"root{k}", "stem3", [(x * 0.5, 0.02, z * 0.5), (x, 0.0, z)])
    parents, heads, n = build(rows)
    found = cf.classify(parents, heads)
    assert found["family"] == "plant" and found["legs"] == []
    assert found["body"][0] == n["stem3"]

