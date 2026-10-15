"""Tests for `image_to_3dlab/creature_family.py`: which family a skeleton belongs to, and
which bones are legs, wings, tail and head.

Skeletons come from `_creature_skeletons.py`: tiny, synthetic, Y up and facing +Z like a
generated glTF.
"""

import json

import numpy as np
import pytest
from _creature_skeletons import humanoid, plant, quadruped, snake, spider, werewolf

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


def test_a_clip_record_names_the_clip_not_the_rig(tmp_path):
    clip = tmp_path / "wolf_digitigrade-walk.glb"
    clip.write_bytes(b"clip bytes")
    rig = {"license": {"name": "MIT"}, "output": {"path": "wolf_rigged.glb", "sha256": "0" * 64,
                                                   "classification": "commercial"}}
    record = cf.clip_record(rig, "digitigrade-walk", "digitigrade", tool="test", output=clip)
    from image_to_3dlab.provenance import sha256_file
    assert record["output"]["sha256"] == sha256_file(clip) and record["output"]["path"] == str(clip)
    assert record["output"]["classification"] == "commercial"  # the licence class carries over
