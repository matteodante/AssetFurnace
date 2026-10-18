import json
from pathlib import Path

import pytest

from image_to_3dlab import piece_materials as pm


def test_every_material_has_roughness_and_metallic():
    assert set(pm.PBR) == set(pm.MATERIALS)
    assert pm.pbr("metal") == (0.35, 1.0)
    assert all(0.0 <= r <= 1.0 and m in (0.0, 1.0) for r, m in pm.PBR.values())
    with pytest.raises(ValueError, match="one of"):
        pm.pbr("adamantium")


@pytest.mark.parametrize(("text", "found"), [
    ("a steel longsword with a leather grip", "metal"),
    ("a leather hood", "leather"),
    ("a woollen cloak", "cloth"),
    ("an oak shield with iron rim", "wood"),
    ("a crystal staff", "gem"),
    ("a plain round thing", None),
    ("", None),
])
def test_first_material_word_in_the_text(text, found):
    assert pm.material_in(text) == found


def test_garment_nouns_read_as_cloth():
    assert pm.material_in("a hooded robe") == "cloth"


@pytest.mark.parametrize(("text", "found"), [
    ("a golden silk tunic", "cloth"),        # the silk and the tunic say what it is; golden how it looks
    ("a golden tunic", "cloth"),
    ("a metallic blue cloak", "cloth"),
    ("a golden crown", "metal"),             # nothing else to go on
    ("a golden helmet with a silk plume", "metal"),   # the first clause that names a material decides
    ("red and gold plate armour", "metal"),
    ("a golden helmet. It has a silk plume.", "metal"),   # a sentence ends a clause too
])
def test_a_word_for_the_look_gives_way_to_what_the_thing_is(text, found):
    assert pm.material_in(text) == found


def test_an_outfit_clause_with_a_golden_garment_is_cloth():
    outfit = "a golden silk tunic, steel greaves and leather boots"
    assert pm.propose(slot="torso", outfit=outfit) == {
        "material": "cloth", "by": "rule", "reason": "the outfit says 'a golden silk tunic'"}
    assert pm.propose(slot="legs", outfit=outfit)["material"] == "metal"


def test_a_piece_takes_its_own_words_first():
    proposal = pm.propose(slot="helmet", words="a leather hood")
    assert proposal == {"material": "leather", "by": "rule", "reason": "the piece's words say 'leather'"}


def test_a_piece_takes_the_outfit_clause_that_names_it():
    outfit = "crimson steel plate armour, leather boots and a fur-lined cloak"
    assert pm.propose(slot="boots", outfit=outfit)["material"] == "leather"
    assert pm.propose(slot="torso", outfit=outfit)["material"] == "metal"
    assert pm.propose(slot="helmet", outfit=outfit)["material"] == "metal"   # no clause names it: default


def test_a_piece_without_words_takes_its_slot_default():
    proposal = pm.propose(slot="shield")
    assert proposal["material"] == "wood"
    assert proposal["reason"] == "default for a shield"


def test_a_single_asset_takes_its_prompt_or_its_kind_default():
    assert pm.propose(kind="prop", words="a stone well with a wooden roof") == {
        "material": "stone", "by": "rule", "reason": "the prompt says 'stone'"}
    assert pm.propose(kind="creature")["material"] == "skin"
    assert pm.propose(kind="prop")["material"] == "wood"
    assert pm.propose()["material"] == "cloth"


def test_unknown_slot_or_kind_is_refused():
    with pytest.raises(ValueError, match="slot"):
        pm.propose(slot="tail")
    with pytest.raises(ValueError, match="kind"):
        pm.propose(kind="vehicle")


def test_check_refuses_words_outside_the_vocabulary():
    assert pm.check("Metal ") == "metal"
    with pytest.raises(ValueError, match="one of"):
        pm.check("steel")


def _set_folder(tmp_path, slots, pieces, outfit="crimson plate armour and leather boots"):
    folder = tmp_path / "sets" / "red" / "human"
    folder.mkdir(parents=True)
    record = {"set": "red", "slots": slots, "pieces": pieces,
              "concept": {"prompt": f"Add {outfit} to the character in image 1.", "outfit": outfit}}
    (folder / "red.parts.json").write_text(json.dumps(record))
    return folder


def test_set_materials_proposes_corrects_and_keeps_what_was_asked(tmp_path):
    folder = _set_folder(
        tmp_path,
        slots={"helmet": {"route": "C", "split": False, "prompt": "a leather hood"},
               "torso": {"route": "C", "split": False, "prompt": ""},
               "boots": {"route": "C", "split": False, "prompt": "", "material": "cloth"},
               "weapon": {"route": "C", "split": False, "prompt": ""}},
        pieces={"helmet": {}, "torso": {}, "boots": {},
                "weapon": {"prompt": "a bone club, a single standalone game asset", "held": True}})
    got = pm.set_materials(folder)
    assert got["helmet"]["material"] == "leather" and got["helmet"]["by"] == "rule"
    assert got["torso"]["material"] == "metal"
    assert got["boots"] == {"material": "cloth", "by": "asked", "reason": "chosen before the run"}
    assert got["weapon"]["material"] == "bone"

    pm.save_corrections(folder, {"torso": "leather"})
    got = pm.set_materials(folder)
    assert got["torso"] == {"material": "leather", "by": "person", "reason": "corrected by a person"}
    saved = json.loads((folder / pm.CORRECTIONS).read_text())
    assert saved == {"schema": pm.SCHEMA, "pieces": {"torso": "leather"}}


def test_save_corrections_checks_slots_and_words_and_can_clear(tmp_path):
    folder = _set_folder(tmp_path, slots={"helmet": {"prompt": ""}}, pieces={"helmet": {}})
    with pytest.raises(ValueError, match="not a piece of this set"):
        pm.save_corrections(folder, {"boots": "leather"})
    with pytest.raises(ValueError, match="one of"):
        pm.save_corrections(folder, {"helmet": "steel"})
    pm.save_corrections(folder, {"helmet": "cloth"})
    pm.save_corrections(folder, {"helmet": None})
    assert pm.set_materials(folder)["helmet"]["by"] == "rule"


def test_piece_of_a_fitted_file():
    assert pm.piece_of("gloves_L.glb") == "gloves"
    assert pm.piece_of("shoulders_R") == "shoulders"
    assert pm.piece_of("helmet.glb") == "helmet"
    assert pm.piece_of("body_masked.glb") is None


def test_prompt_for_follows_the_inputs_to_the_words(tmp_path):
    picture = tmp_path / "concept.png"
    (tmp_path / "concept.provenance.json").write_text(json.dumps({"prompt": "a leather saddle"}))
    model = tmp_path / "run" / "saddle.glb"
    model.parent.mkdir()
    (model.parent / "saddle.provenance.json").write_text(json.dumps({"input": {"path": str(picture)}}))
    assert pm.prompt_for(model) == "a leather saddle"
    assert pm.prompt_for(tmp_path / "nothing.glb") == ""
    (model.parent / "saddle.provenance.json").write_text("not json")
    assert pm.prompt_for(model) == ""


def test_outfit_words_are_read_from_the_concept_as_saved():
    here = Path("sets/red/human")
    assert pm.outfit_words({"concept": {"prompt": "Add a hood to the character in image 1.",
                                        "outfit": "red plate"}}, here) == "red plate"
    assert pm.outfit_words({"concept": {"given": True, "outfit": ""}}, here) == ""   # a concept given ready-made
    assert pm.outfit_words({"concept": None}, here) == ""                           # weapons and shields alone


def test_a_set_saved_before_the_outfit_words_says_how_to_migrate_it(tmp_path):
    folder = _set_folder(tmp_path, slots={"boots": {"prompt": ""}}, pieces={"boots": {}})
    path = folder / "red.parts.json"
    record = json.loads(path.read_text())
    del record["concept"]["outfit"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="migrate_set_outfits.py") as refused:
        pm.set_materials(folder)
    assert str(refused.value).endswith(f'migrate_set_outfits.py "{folder}"` once to write them in')
