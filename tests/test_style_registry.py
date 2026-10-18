"""The list of bibles AssetFurnace knows about, one per project, and which one is active."""

import pytest

from image_to_3dlab.style import bible as sb
from image_to_3dlab.style import registry


def test_an_empty_registry_lists_nothing(tmp_path):
    assert registry.bibles(tmp_path) == []
    assert registry.active(tmp_path) is None


def test_registering_a_bible_inside_output_stores_a_relative_path(tmp_path):
    root = sb.create(tmp_path / "styles" / "my-game", "my-game")
    registry.register(tmp_path, root)
    raw = (tmp_path / "styles" / "bibles.json").read_text(encoding="utf-8")
    assert '"styles/my-game"' in raw.replace("\\\\", "/")
    assert registry.bibles(tmp_path) == [{"name": "my-game", "path": root.resolve()}]


def test_an_external_bible_keeps_its_absolute_path(tmp_path):
    output = tmp_path / "output"
    root = sb.create(tmp_path / "game-repo" / "art" / "style", "game")
    registry.register(output, root)
    assert registry.bibles(output) == [{"name": "game", "path": root.resolve()}]


def test_registering_twice_or_a_clashing_name_is_refused(tmp_path):
    one = sb.create(tmp_path / "a", "same")
    registry.register(tmp_path, one)
    with pytest.raises(sb.BibleError, match="already"):
        registry.register(tmp_path, one)
    two = sb.create(tmp_path / "b", "same")
    with pytest.raises(sb.BibleError, match="same"):
        registry.register(tmp_path, two)


def test_only_a_real_bible_can_be_registered(tmp_path):
    with pytest.raises(sb.BibleError, match="no bible"):
        registry.register(tmp_path, tmp_path / "empty")


def test_the_active_bible_is_chosen_by_name_and_forgotten_on_unregister(tmp_path):
    root = sb.create(tmp_path / "a", "first")
    registry.register(tmp_path, root)
    registry.set_active(tmp_path, "first")
    assert registry.active(tmp_path) == {"name": "first", "path": root.resolve()}
    with pytest.raises(sb.BibleError, match="no bible"):
        registry.set_active(tmp_path, "other")
    registry.unregister(tmp_path, "first")
    assert registry.active(tmp_path) is None and registry.bibles(tmp_path) == []
    assert (root / "bible.json").is_file()  # unregistering never deletes the folder


def test_find_takes_a_name_or_a_folder(tmp_path):
    root = sb.create(tmp_path / "a", "first")
    registry.register(tmp_path, root)
    assert registry.find(tmp_path, "first") == root.resolve()
    assert registry.find(tmp_path, str(root)) == root.resolve()
    with pytest.raises(sb.BibleError, match="no bible"):
        registry.find(tmp_path, "unknown")


def test_registry_edits_wait_for_another_process(tmp_path, monkeypatch):
    """Codex review 2: two processes editing bibles.json must not drop each other's change."""
    root = sb.create(tmp_path / "a", "first")
    monkeypatch.setattr(sb, "LOCK_WAIT_SECONDS", 0.3)
    (tmp_path / "styles").mkdir(exist_ok=True)
    (tmp_path / "styles" / registry.LOCK_FILE).write_text("1", encoding="utf-8")
    with pytest.raises(sb.BibleError, match="busy"):
        registry.register(tmp_path, root)
    (tmp_path / "styles" / registry.LOCK_FILE).unlink()
    registry.register(tmp_path, root)
    assert not (tmp_path / "styles" / registry.LOCK_FILE).exists()


def test_the_studio_names_registered_bibles_only_never_a_folder(tmp_path, monkeypatch):
    """Code read 18/10/2026: a request's `bible` could be any folder holding a bible.json,
    whose pictures then went to sd-cli. Over HTTP only a registered name or "-" is taken."""
    import image_api

    from image_to_3dlab.style import apply

    root = sb.create(tmp_path / "anywhere" / "style", "loose")
    assert registry.find(tmp_path, str(root)) == root.resolve()   # the command line keeps its folder
    with pytest.raises(sb.BibleError, match="registered"):
        registry.find(tmp_path, str(root), allow_path=False)
    monkeypatch.setattr(apply, "OUTPUT", tmp_path)
    with pytest.raises(sb.BibleError, match="registered"):
        image_api.with_style("a helmet", {"bible": str(root)})
    registry.register(tmp_path, root)
    prompt, extra = image_api.with_style("a helmet", {"bible": "loose"})
    assert extra["style"]


def test_the_concept_command_line_keeps_a_bible_folder(tmp_path, monkeypatch):
    """Codex review: the HTTP restriction reached scripts/reference_concept.py, which builds
    with the same function; only the HTTP route refuses a folder."""
    import concept_api

    from image_to_3dlab.style import apply

    root = sb.create(tmp_path / "anywhere" / "style", "loose")
    seen = []
    monkeypatch.setattr(apply, "resolve", lambda *a, **k: seen.append(k.get("allow_path")) or None)
    monkeypatch.setattr(concept_api.requests, "plan", lambda *a, **k: {})
    concept_api.build(tmp_path, {"mode": concept_api.prompts.MODES[0], "bible": str(root)})
    concept_api.build(tmp_path, {"mode": concept_api.prompts.MODES[0], "bible": str(root)}, allow_path=False)
    assert seen == [True, False]
