"""The style bible: a self-contained folder of approved and rejected pictures with notes."""

import json

import pytest
from PIL import Image

from image_to_3dlab.style import bible as sb


def picture(path, colour=(200, 40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path)
    return path


@pytest.fixture
def root(tmp_path):
    return sb.create(tmp_path / "my-game-style", "my-game", description="Hand-painted.",
                     wording="bold saturated colours")


def test_a_new_bible_is_a_folder_with_a_valid_bible_json(root):
    data = json.loads((root / "bible.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["name"] == "my-game"
    assert data["style"]["wording"] == "bold saturated colours"
    assert data["style"]["criteria"]  # generic starting questions for the judge
    assert data["images"] == [] and data["rejected"] == []
    assert sb.load(root)["name"] == "my-game"


def test_create_refuses_an_existing_bible_and_a_bad_name(tmp_path, root):
    with pytest.raises(sb.BibleError, match="already"):
        sb.create(root, "again")
    with pytest.raises(sb.BibleError, match="name"):
        sb.create(tmp_path / "x", "Bad Name!")


def test_adding_copies_the_picture_and_its_provenance_with_the_note(tmp_path, root):
    source = picture(tmp_path / "out" / "knight.png")
    source.with_suffix(".provenance.json").write_text('{"license": {"name": "x"}}', encoding="utf-8")
    entry = sb.add(root, source, "Colours are right.", kind="character", source_label="images/knight.png")
    assert entry["file"].startswith("images/") and entry["file"].endswith("-knight.png")
    assert entry["sha256"] == sb.sha256_file(source)
    assert entry["kind"] == "character" and entry["note"] == "Colours are right."
    assert entry["source"] == "images/knight.png" and entry["added"]
    copied = root / entry["file"]
    assert copied.read_bytes() == source.read_bytes()
    assert copied.with_suffix(".provenance.json").is_file()
    assert sb.load(root)["images"] == [entry]


def test_a_rejected_picture_goes_to_its_own_list_and_folder(tmp_path, root):
    entry = sb.add(root, picture(tmp_path / "wolf.png"), "Grey and photographic.", rejected=True)
    assert entry["file"].startswith("rejected/")
    data = sb.load(root)
    assert data["rejected"] == [entry] and data["images"] == []


def test_the_note_is_required_and_the_kind_must_be_known(tmp_path, root):
    source = picture(tmp_path / "a.png")
    with pytest.raises(sb.BibleError, match="note"):
        sb.add(root, source, "   ")
    with pytest.raises(sb.BibleError, match="kind"):
        sb.add(root, source, "ok", kind="vehicle")


def test_the_same_picture_cannot_be_added_twice_even_to_the_other_list(tmp_path, root):
    source = picture(tmp_path / "a.png")
    sb.add(root, source, "good")
    with pytest.raises(sb.BibleError, match="already"):
        sb.add(root, source, "bad", rejected=True)


def test_only_real_pictures_are_accepted(tmp_path, root):
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"not a picture")
    with pytest.raises(sb.BibleError, match="picture"):
        sb.add(root, fake, "note")
    text = tmp_path / "notes.txt"
    text.write_text("hi")
    with pytest.raises(sb.BibleError, match="picture"):
        sb.add(root, text, "note")


def test_remove_deletes_the_entry_the_file_and_its_provenance(tmp_path, root):
    source = picture(tmp_path / "a.png")
    source.with_suffix(".provenance.json").write_text("{}", encoding="utf-8")
    entry = sb.add(root, source, "good")
    removed = sb.remove(root, entry["sha256"][:8])
    assert removed == entry
    assert not (root / entry["file"]).exists()
    assert not (root / entry["file"]).with_suffix(".provenance.json").exists()
    assert sb.load(root)["images"] == []
    with pytest.raises(sb.BibleError, match="no picture"):
        sb.remove(root, entry["sha256"])


def test_an_ambiguous_short_hash_is_refused(tmp_path, root, monkeypatch):
    a = sb.add(root, picture(tmp_path / "a.png", (1, 2, 3)), "a")
    b = sb.add(root, picture(tmp_path / "b.png", (4, 5, 6)), "b")
    common = ""
    for x, y in zip(a["sha256"], b["sha256"]):
        if x != y:
            break
        common += x
    with pytest.raises(sb.BibleError):
        sb.remove(root, common)  # an empty or shared prefix names both


def test_set_note_changes_only_the_note(tmp_path, root):
    entry = sb.add(root, picture(tmp_path / "a.png"), "old")
    changed = sb.set_note(root, entry["sha256"], "new")
    assert changed == {**entry, "note": "new"}
    with pytest.raises(sb.BibleError, match="note"):
        sb.set_note(root, entry["sha256"], "")


def test_problems_lists_missing_and_changed_files(tmp_path, root):
    a = sb.add(root, picture(tmp_path / "a.png", (1, 1, 1)), "a")
    b = sb.add(root, picture(tmp_path / "b.png", (2, 2, 2)), "b")
    assert sb.problems(root) == []
    (root / a["file"]).unlink()
    picture(root / b["file"], (9, 9, 9))
    found = sb.problems(root)
    assert any(a["file"] in p and "missing" in p for p in found)
    assert any(b["file"] in p and "changed" in p for p in found)


def test_load_refuses_a_broken_or_unsafe_bible_json(tmp_path, root):
    path = root / "bible.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, "schema_version": 2}), encoding="utf-8")
    with pytest.raises(sb.BibleError, match="schema"):
        sb.load(root)
    escaping = {**data, "images": [{"file": "../outside.png", "sha256": "0" * 64, "note": "x",
                                    "added": "2026-10-16T00:00:00+00:00"}]}
    path.write_text(json.dumps(escaping), encoding="utf-8")
    with pytest.raises(sb.BibleError, match="inside"):
        sb.load(root)
    with pytest.raises(sb.BibleError, match="no bible"):
        sb.load(tmp_path / "nowhere")


def test_writes_are_atomic_and_leave_no_temporary_files(tmp_path, root):
    sb.add(root, picture(tmp_path / "a.png"), "a")
    assert sorted(p.name for p in root.iterdir()) == ["bible.json", "images", "rejected"]
