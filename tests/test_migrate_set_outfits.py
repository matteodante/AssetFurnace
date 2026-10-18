"""scripts/migrate_set_outfits.py: the outfit's words written once into sets made before they were saved."""

import importlib.util
import json
from pathlib import Path

import pytest

from image_to_3dlab import piece_materials as pm

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("migrate_set_outfits", REPO / "scripts" / "migrate_set_outfits.py")
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def _old_set(output: Path, name: str, concept: dict | None) -> Path:
    folder = output / "sets" / name / "human"
    folder.mkdir(parents=True)
    record = {"set": name, "slots": {"torso": {"prompt": ""}, "boots": {"prompt": ""}},
              "pieces": {"torso": {}, "boots": {}}, "concept": concept}
    (folder / f"{name}.parts.json").write_text(json.dumps(record, indent=1) + "\n")
    return folder


def test_old_sets_get_the_words_their_prompt_held_and_a_second_run_changes_nothing(tmp_path):
    drawn = _old_set(tmp_path, "red", {"prompt": "Add red plate armour and leather boots to the character in "
                                                 "image 1, fitted closely to the body.", "seed": 3})
    given = _old_set(tmp_path, "given", {"given": True})
    alone = _old_set(tmp_path, "arms", None)
    with_words = _old_set(tmp_path, "new", {"prompt": "Add x to the character in image 1.", "outfit": "steel plate"})
    (drawn / "run_state.json").write_text(json.dumps({"waiting": "picks", "concept": "c.png", "concept_info": {
        "prompt": "Dress the character in image 1 as a monk.", "seed": 3}}))

    assert len(migration.migrate([tmp_path / "sets"], dry_run=True)) == 3
    assert "outfit" not in json.loads((drawn / "red.parts.json").read_text())["concept"]   # a dry run writes nothing
    changed = migration.migrate([tmp_path / "sets"])
    assert {(p.parent.parent.name, p.name, words) for p, words in changed} == {
        ("red", "red.parts.json", "red plate armour and leather boots"),
        ("given", "given.parts.json", ""),
        ("red", "run_state.json", "Dress the character in image 1 as a monk."),   # another form: taken whole
    }
    assert json.loads((drawn / "red.parts.json").read_text())["concept"]["seed"] == 3   # the rest kept
    assert json.loads((alone / "arms.parts.json").read_text())["concept"] is None
    assert json.loads((with_words / "new.parts.json").read_text())["concept"]["outfit"] == "steel plate"
    assert migration.migrate([tmp_path / "sets"]) == []
    assert not list(tmp_path.rglob("*.tmp"))

    materials = pm.set_materials(drawn)
    assert materials["boots"]["material"] == "leather" and materials["torso"]["material"] == "metal"
    assert pm.set_materials(given)["boots"]["reason"] == "default for boots"


def test_sets_kept_elsewhere_and_a_run_waiting_on_its_pictures_are_found(tmp_path, capsys):
    """`run --root` and `refit --out` put sets anywhere; a run waiting for picks has no parts yet."""
    elsewhere = _old_set(tmp_path / "mine", "red", {"prompt": "Add a leather hood to the character in image 1."})
    waiting = tmp_path / "mine" / "sets" / "blue" / "human"
    waiting.mkdir(parents=True)
    (waiting / "run_state.json").write_text(json.dumps({"waiting": "picks", "concept": "c.png", "concept_info": {
        "prompt": "Add steel greaves to the character in image 1.", "seed": 3}}))
    (waiting / "other_state.json").write_text("{}")
    assert migration.main([str(elsewhere)]) == 0
    assert json.loads((elsewhere / "red.parts.json").read_text())["concept"]["outfit"] == "a leather hood"
    assert migration.main([str(tmp_path / "mine")]) == 0
    assert json.loads((waiting / "run_state.json").read_text())["concept_info"]["outfit"] == "steel greaves"
    assert "1 record(s) changed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        migration.main([str(tmp_path / "nowhere")])
