"""Choosing 1-3 approved pictures from the bible as style references for generation."""

import pytest
from PIL import Image

from image_to_3dlab.style import bible as sb
from image_to_3dlab.style import refs


def add(root, tmp_path, name, colour, kind=None, rejected=False):
    path = tmp_path / "src" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)
    return sb.add(root, path, name, kind=kind, rejected=rejected)


@pytest.fixture
def root(tmp_path):
    root = sb.create(tmp_path / "bible", "b")
    add(root, tmp_path, "knight", (1, 0, 0), kind="character")
    add(root, tmp_path, "troll", (2, 0, 0), kind="creature")
    add(root, tmp_path, "any", (3, 0, 0))
    add(root, tmp_path, "wolf", (4, 0, 0), kind="creature")
    add(root, tmp_path, "bad", (5, 0, 0), kind="creature", rejected=True)
    return root


def names(paths):
    return [p.name.split("-", 1)[1] for p in paths]


def test_the_same_kind_comes_first_then_pictures_for_any_kind_then_the_rest(root):
    assert names(refs.pick(root, "creature", k=3)) == ["troll.png", "wolf.png", "any.png"]
    assert names(refs.pick(root, "character", k=3)) == ["knight.png", "any.png", "troll.png"]


def test_without_a_kind_the_oldest_approved_come_first(root):
    assert names(refs.pick(root, None, k=2)) == ["knight.png", "troll.png"]


def test_rejected_pictures_are_never_references(root):
    assert "bad.png" not in names(refs.pick(root, "creature", k=3))


def test_typicality_scores_order_pictures_within_a_group(root):
    data = sb.load(root)
    wolf = next(e for e in data["images"] if e["file"].endswith("wolf.png"))
    picked = refs.pick(root, "creature", k=1, typicality={wolf["sha256"]: 0.9})
    assert names(picked) == ["wolf.png"]


def test_missing_files_are_skipped_and_k_is_bounded(root):
    data = sb.load(root)
    (root / data["images"][1]["file"]).unlink()  # troll
    assert names(refs.pick(root, "creature", k=2)) == ["wolf.png", "any.png"]
    for bad in (0, 4):
        with pytest.raises(ValueError, match="1 and 3"):
            refs.pick(root, "creature", k=bad)


def test_an_empty_bible_gives_no_references(tmp_path):
    assert refs.pick(sb.create(tmp_path / "e", "e"), "prop", k=3) == []


def test_a_picture_changed_since_it_was_approved_is_not_a_reference(root):
    """Codex review 1: the hash is what was approved; other bytes in the same file are not."""
    data = sb.load(root)
    troll = root / data["images"][1]["file"]
    Image.new("RGB", (8, 8), (9, 9, 9)).save(troll)
    assert names(refs.pick(root, "creature", k=2)) == ["wolf.png", "any.png"]
