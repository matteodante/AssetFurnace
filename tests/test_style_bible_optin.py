"""The style bible, opt-in on every route that draws (gap 6 of
docs/plans/2026-10-17-step-parity-design.md): text to image, a concept from a reference and
characters in pieces, from the studio, the command line and batches. Off unless asked;
"-" is the active bible, a name a registered one. Nothing draws here."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from batch_fakes import install_fake_generator, tiny_png

from image_to_3dlab import batch, step_catalog
from image_to_3dlab.style import apply
from image_to_3dlab.style import bible as sb
from image_to_3dlab.style import registry

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "viewer"))
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


@pytest.fixture
def output(tmp_path, monkeypatch):
    """An output folder with an active bible: style words, words for props, one reference."""
    out = tmp_path / "output"
    root = sb.create(out / "styles" / "auroconts", "auroconts", wording="hand-painted, warm palette")
    data = json.loads((root / "bible.json").read_text())
    data["kinds"] = {"prop": {"wording": "chunky silhouettes"}}
    (root / "bible.json").write_text(json.dumps(data))
    sb.add(root, tiny_png(tmp_path / "ref.png", 3), "the original game's barrel", kind="prop", reference=True)
    registry.register(out, root)
    registry.set_active(out, "auroconts")
    monkeypatch.setattr(apply, "OUTPUT", out)
    return out


def test_no_bible_unless_asked(output):
    assert apply.resolve(None, "prop") is None and apply.resolve("", "prop") is None


def test_the_active_bible_gives_its_words_and_references(output):
    style = apply.resolve("-", "prop", references=3)
    assert style.name == "auroconts" and style.words == "hand-painted, warm palette, chunky silhouettes"
    assert len(style.references) == 1 and style.references[0].name.endswith(".png")
    assert apply.styled("a barrel", style) == "a barrel, hand-painted, warm palette, chunky silhouettes"
    record = apply.record(style)
    assert record["bible"] == "auroconts" and len(record["bible_sha256"]) == 64 and record["words"] == style.words
    assert apply.resolve("auroconts", "creature").words == "hand-painted, warm palette"   # no creature words


def test_an_unknown_bible_is_refused(output):
    with pytest.raises(sb.BibleError):
        apply.resolve("nope", "prop")
    registry.set_active(output, None)
    with pytest.raises(sb.BibleError, match="no bible is active"):
        apply.resolve("-", "prop")


def test_kinds_are_the_bibles_own():
    assert apply.bible_kind("equipment") == "equipment_part" and apply.bible_kind("character") == "character"
    assert apply.bible_kind("set") == "prop" and apply.bible_kind("picture") is None


# ------------------------------------------------------------------ text to image

def test_the_command_line_draws_with_the_bible(output, monkeypatch, capsys):
    import text_image as cli

    from image_to_3dlab import text_image

    seen = {}

    def generate(prompt, settings, path, **kwargs):
        seen.update(prompt=prompt, provenance=kwargs.get("extra_provenance"))
        return tiny_png(path)

    monkeypatch.setattr(text_image, "generate", generate)
    assert cli.main(["a barrel", "--bible", "-", "--kind", "prop", "--out", str(output / "x")]) == 0
    assert seen["prompt"] == "a barrel, hand-painted, warm palette, chunky silhouettes"
    assert seen["provenance"]["style"]["bible"] == "auroconts"


def test_the_studio_picture_job_takes_the_bible(output):
    import image_api

    prompt, extra = image_api.with_style("a barrel", {"bible": "-", "kind": "prop"})
    assert prompt.endswith("chunky silhouettes") and extra == {"style": apply.record(apply.resolve("-", "prop"))}
    assert image_api.with_style("a barrel", {}) == ("a barrel", {})


def test_the_chain_sends_the_bible_to_the_picture_step(tmp_path):
    import activity_api
    from test_viewer_activity_api import _runner

    from viewer.generate_api import parse_multipart

    body, ctype = activity_api.multipart({"title": "x", "steps": json.dumps(["picture"]), "prompt": "a crate",
                                          "want": "prop", "bible": "-"}, {})
    chain = activity_api.chain_from_form(parse_multipart(ctype, body))
    assert chain.bible == "-"
    runner, jobs = _runner(tmp_path)
    runner.submit(chain, background=False)
    sent = json.loads(next(b for m, p, b in jobs.calls if p == "/api/image"))
    assert sent["bible"] == "-" and sent["kind"] == "prop"


@needs_node
def test_create_sends_the_bible_when_ticked():
    url = (REPO / "viewer" / "studio" / "create.js").as_uri()
    program = (f"import * as m from {json.dumps(url)};\n"
               "const on = m.planForm({ steps: ['picture'], bible: '-' }); const off = m.planForm({ steps: ['picture'] });\n"
               "console.log(JSON.stringify([on.get('bible'), off.get('bible')]));")
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True,
                            text=True, encoding="utf-8")
    assert json.loads(result.stdout) == ["-", None]


def test_the_studio_says_which_bible_is_active(output):
    import style_api

    info = style_api.info(output)
    assert info["active"] == "auroconts" and info["bibles"] == ["auroconts"]


# ------------------------------------------------------------------------- concept

def test_a_concept_takes_the_bibles_words_and_style_pictures(output, tmp_path):
    from image_to_3dlab.concept import requests

    shape = tiny_png(tmp_path / "shape.png")
    style = apply.resolve("-", "prop", references=3)
    plan = requests.plan("redraw", [shape], [], {"subject": "barrel", "colours": "brown", "style": "inked"},
                         bible=style)
    assert "inked" in plan["prompt"] and "chunky silhouettes" in plan["prompt"]
    assert [role for _, role in plan["references"]] == ["shape", "style"]
    assert plan["style_record"]["bible"] == "auroconts"


def test_the_studios_concept_request_names_the_bible(output, tmp_path):
    import concept_api

    tiny_png(output / "images" / "wolf.png")
    plan = concept_api.build(output, {"mode": "redraw", "subject": "wolf", "colours": "grey",
                                      "shape": ["images/wolf.png"], "bible": "-", "kind": "creature"})
    assert "hand-painted" in plan["prompt"] and plan["style_record"]["bible"] == "auroconts"


def test_the_concept_command_line_takes_the_bible():
    import reference_concept

    args = reference_concept.parser().parse_args(["make", "redraw", "--bible", "-", "--kind", "prop"])
    assert args.bible == "-" and args.kind == "prop"
    assert reference_concept.request(args, 3)["bible"] == "-"


# -------------------------------------------------------------------------- pieces

def test_a_set_of_pieces_takes_the_bible(output):
    import slot_parts

    from image_to_3dlab.slot_parts import runs

    assert slot_parts.style_with_bible("crimson", "-") == "crimson, hand-painted, warm palette"
    assert slot_parts.style_with_bible("crimson", None) == "crimson"
    args = slot_parts.build_parser().parse_args(["run", "--body", "b.json", "--set", "red", "--bible", "-"])
    assert args.bible == "-"
    assert "--bible" in runs.bible_args({"bible": "-"}) and runs.bible_args({}) == []


# -------------------------------------------------------------------------- batches

def test_a_batch_picture_is_drawn_with_the_bible(output, tmp_path, monkeypatch):
    from image_to_3dlab.batch_steps import picture as picture_step

    seen = []

    def generate(prompt, settings, path, **kwargs):
        seen.append((prompt, kwargs.get("extra_provenance")))
        tiny_png(path)
        path.with_suffix(".provenance.json").write_text("{}")
        return path

    monkeypatch.setattr(picture_step.text_image, "generate", generate)
    monkeypatch.setattr(picture_step, "readiness", list)
    install_fake_generator(monkeypatch)
    spec_path = tmp_path / "lot.json"
    spec_path.write_text(json.dumps({"schema_version": 1, "name": "lot", "output": {"directory": "out"},
                                     "rows": [{"id": "SM_Barrel", "prompt": "a barrel", "backend": "fake", "steps": ["picture", "generate"],
                                               "picture": {"bible": "-"}}]}))
    assert batch.run_batch(batch.load_batch(spec_path)).exit_code == 0
    ((prompt, provenance),) = seen
    assert prompt == "a barrel, hand-painted, warm palette, chunky silhouettes"
    assert provenance["style"]["bible"] == "auroconts"


def test_the_bible_is_a_setting_of_every_drawing_step():
    for step in ("picture", "concept"):
        param = step_catalog.get(step).param("bible")
        assert param is not None and param.cli_flag == "--bible"
