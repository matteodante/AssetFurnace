"""Prompt templates per asset kind: the recipe's 3D wording, then the bible's style."""

import pytest

from image_to_3dlab.style import bible as sb
from image_to_3dlab.style import prompts

RECIPES = prompts.load_recipes()


def test_the_recipes_cover_the_five_asset_kinds():
    for kind in sb.KINDS:
        goal = RECIPES["goals"][kind]
        assert goal["label"] and goal["wording"] and goal["tips"]
        assert "plain light grey background" in goal["wording"]


def test_every_trouble_rule_names_known_goals():
    goals = set(RECIPES["goals"])
    for rule in RECIPES["trouble"]:
        assert set(rule["goals"]) <= goals, rule


def test_without_a_bible_the_prompt_is_the_subject_and_the_kind_wording():
    out = prompts.compose("A rusty iron lantern.", "prop", recipes=RECIPES)
    assert out["prompt"] == f"A rusty iron lantern, {RECIPES['goals']['prop']['wording']}"
    assert [p["from"] for p in out["parts"]] == ["subject", "recipe"]


def test_the_bible_adds_its_style_then_its_kind_wording_in_that_order():
    bible = {"style": {"wording": "bold saturated colours"},
             "kinds": {"creature": {"wording": "oversized claws"}}}
    out = prompts.compose("a forest troll", "creature", bible=bible, recipes=RECIPES)
    recipe = RECIPES["goals"]["creature"]["wording"]
    assert out["prompt"] == f"a forest troll, {recipe}, bold saturated colours, oversized claws"
    assert [p["from"] for p in out["parts"]] == ["subject", "recipe", "style", "kind"]


def test_style_references_add_one_sentence_naming_them():
    bible = {"style": {"wording": "bold colours"}, "kinds": {}}
    out = prompts.compose("a barrel", "map_object", bible=bible, recipes=RECIPES, references=2)
    assert out["parts"][-1]["from"] == "references"
    assert out["prompt"].endswith(prompts.REFERENCE_SENTENCE)
    assert prompts.compose("a barrel", "map_object", recipes=RECIPES, references=0)["parts"][-1]["from"] == "recipe"


def test_empty_pieces_are_left_out_and_the_subject_is_required():
    out = prompts.compose("  a cup ,. ", "prop", bible={"style": {"wording": ""}, "kinds": {}}, recipes=RECIPES)
    assert out["prompt"].startswith("a cup, ")
    with pytest.raises(ValueError, match="subject"):
        prompts.compose("   ", "prop", recipes=RECIPES)


def test_an_unknown_kind_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="creature"):
        prompts.compose("a car", "vehicle", recipes=RECIPES)


def test_the_bible_can_be_given_as_a_folder(tmp_path):
    root = sb.create(tmp_path / "b", "b", wording="soft pastel palette")
    out = prompts.compose("a chair", "prop", bible=root, recipes=RECIPES)
    assert out["prompt"].endswith("soft pastel palette")
