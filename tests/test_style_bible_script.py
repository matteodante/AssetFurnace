"""scripts/style_bible.py: the style bible from a terminal."""

from __future__ import annotations

import json

import style_bible
from PIL import Image

from image_to_3dlab.style import bible as sb


def picture(path, colour=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)
    return path


def run(*args):
    return style_bible.main([str(a) for a in args])


def test_new_add_reject_and_list(tmp_path, capsys):
    output = tmp_path / "output"
    assert run("--output", output, "new", "my-game", "--wording", "bold colours") == 0
    root = output / "styles" / "my-game"
    assert sb.load(root)["style"]["wording"] == "bold colours"
    assert run("--output", output, "add", "my-game", picture(tmp_path / "k.png"), "--note", "right colours",
               "--kind", "character") == 0
    assert run("--output", output, "reject", "my-game", picture(tmp_path / "w.png", (1, 1, 1)),
               "--note", "too grey") == 0
    capsys.readouterr()
    assert run("--output", output, "list", "my-game", "--json") == 0
    listed = json.loads(capsys.readouterr().out)
    assert [e["note"] for e in listed["images"]] == ["right colours"]
    assert [e["note"] for e in listed["rejected"]] == ["too grey"]
    assert listed["problems"] == []


def test_a_folder_outside_output_can_be_registered_and_made_active(tmp_path, capsys):
    output = tmp_path / "output"
    external = tmp_path / "game" / "style"
    assert run("--output", output, "new", "game", "--at", external) == 0
    assert run("--output", output, "use", "game") == 0
    capsys.readouterr()
    assert run("--output", output, "bibles") == 0
    out = capsys.readouterr().out
    assert "game" in out and "active" in out
    other = sb.create(tmp_path / "other", "other")
    assert run("--output", output, "register", other) == 0


def test_the_active_bible_is_the_default(tmp_path, capsys):
    output = tmp_path / "output"
    run("--output", output, "new", "g")
    run("--output", output, "use", "g")
    assert run("--output", output, "add", picture(tmp_path / "a.png"), "--note", "fine") == 2  # no bible named
    capsys.readouterr()
    assert run("--output", output, "add", "-", picture(tmp_path / "a.png"), "--note", "fine") == 0


def test_prompt_and_refs_print_what_generation_would_use(tmp_path, capsys):
    output = tmp_path / "output"
    run("--output", output, "new", "g", "--wording", "saturated colours")
    run("--output", output, "add", "g", picture(tmp_path / "a.png"), "--note", "a", "--kind", "creature")
    capsys.readouterr()
    assert run("--output", output, "prompt", "g", "creature", "a forest troll", "--references", "1") == 0
    out = capsys.readouterr().out
    assert out.startswith("a forest troll, ") and "saturated colours" in out
    assert run("--output", output, "refs", "g", "creature", "--k", "1") == 0
    assert capsys.readouterr().out.strip().endswith("-a.png")


def test_remove_and_note_by_short_hash(tmp_path, capsys):
    output = tmp_path / "output"
    run("--output", output, "new", "g")
    source = picture(tmp_path / "a.png")
    run("--output", output, "add", "g", source, "--note", "a")
    short = sb.sha256_file(source)[:8]
    assert run("--output", output, "note", "g", short, "better") == 0
    assert sb.load(output / "styles" / "g")["images"][0]["note"] == "better"
    assert run("--output", output, "remove", "g", short) == 0
    assert sb.load(output / "styles" / "g")["images"] == []


def test_errors_are_one_line_and_exit_2(tmp_path, capsys):
    output = tmp_path / "output"
    assert run("--output", output, "add", "nope", picture(tmp_path / "a.png"), "--note", "x") == 2
    err = capsys.readouterr().err
    assert "no bible" in err and "Traceback" not in err
