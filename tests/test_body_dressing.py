"""The user's decision A (17/10/2026): the user decides how a base body is dressed under its
pieces: a bodysuit (leotard), simple clothes (shirt, breeches and cloth shoes, as in 4Story or
WoW, so the gaps between armour pieces show a garment), minimal underwear, or words of their
own. The choice is made when a body is registered, on every route, kept in body.json, and the
outfit concept keeps it. Nothing is chosen for them; bodies registered before keep the
bodysuit they were made in."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from image_to_3dlab.concept import prompts
from image_to_3dlab.slot_parts import body as body_mod
from image_to_3dlab.slot_parts import runs

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "viewer"))
import slot_parts as cli  # noqa: E402

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def test_a_dressing_is_one_of_the_choices_or_words_of_ones_own():
    assert body_mod.dressing("clothes") == {"kind": "clothes", "words": ""}
    assert body_mod.dressing("other", " a linen tunic ") == {"kind": "other", "words": "a linen tunic"}
    for kind, words in (("armour", ""), ("other", ""), (None, "")):
        with pytest.raises(ValueError, match="dressed"):
            body_mod.dressing(kind, words)


def test_the_concept_keeps_the_bodys_own_clothes():
    clothes = body_mod.dressing_words({"dressing": {"kind": "clothes", "words": "dark blue"}})
    assert "shirt" in clothes and "breeches" in clothes and "dark blue" in clothes
    assert "bodysuit" not in prompts.add_piece("red plate armour", wearing=clothes)
    assert clothes in prompts.add_piece("red plate armour", wearing=clothes)
    # a body registered before the choice existed was made in a bodysuit, and keeps it
    assert body_mod.dressing_words({}) == "bodysuit"
    assert prompts.add_piece("red plate") == prompts.add_piece("red plate", wearing="bodysuit")


def test_the_command_line_asks_how_the_body_is_dressed():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):   # no default: the user chooses
        parser.parse_args(["body", "elf", "--image", "a.png", "--rigged", "r.glb", "--views", "v"])
    args = parser.parse_args(["body", "elf", "--image", "a.png", "--rigged", "r.glb", "--views", "v",
                              "--dressing", "underwear"])
    assert args.dressing == "underwear" and args.dressing_words == ""


def _registered(monkeypatch):
    seen = {}

    def register(name, image, rigged, views=None, matte=None, labels=None, root=None, clips=None, dressed=None):
        seen.update(name=name, dressing=dressed)
        path = Path(root) / name / "body.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": body_mod.SCHEMA_VERSION, "name": name, "dressing": dressed}))
        return path

    monkeypatch.setattr(body_mod, "register", register)
    return seen


def test_the_studio_and_agents_register_with_the_dressing(tmp_path, monkeypatch):
    seen = _registered(monkeypatch)
    for path in ("a.png", "m.glb", "r.glb"):
        (tmp_path / path).write_bytes(b"x")
    runs.register_body(tmp_path, "elf", tmp_path / "a.png", tmp_path / "m.glb", tmp_path / "r.glb",
                       dressing="clothes", dressing_words="green")
    assert seen["dressing"] == {"kind": "clothes", "words": "green"}
    with pytest.raises(ValueError, match="dressed"):
        runs.register_body(tmp_path, "elf", tmp_path / "a.png", tmp_path / "m.glb", tmp_path / "r.glb")

    from image_to_3dlab import mcp_server

    monkeypatch.setattr(mcp_server, "OUTPUT", tmp_path)
    answer = mcp_server.pieces_add_body("orc", str(tmp_path / "a.png"), str(tmp_path / "m.glb"),
                                        str(tmp_path / "r.glb"), dressing="underwear")
    assert answer["name"] == "orc" and seen["dressing"] == {"kind": "underwear", "words": ""}
    odd = mcp_server.pieces_add_body("orc", str(tmp_path / "a.png"), str(tmp_path / "m.glb"),
                                     str(tmp_path / "r.glb"), dressing=["clothes"])
    assert "problem" in odd   # an answer, not an exception

    import slot_parts_api

    output = tmp_path / "output"
    for path in ("a.png", "m.glb", "r.glb"):
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        (output / path).write_bytes(b"x")
    slot_parts_api.register_body(output, {"name": "dwarf", "picture": "a.png", "model": "m.glb", "rigged": "r.glb",
                                          "dressing": "other", "dressing_words": "a leather apron"})
    assert seen["dressing"] == {"kind": "other", "words": "a leather apron"}


@needs_node
def test_the_studio_form_asks_how_the_body_is_dressed():
    url = (REPO / "viewer" / "studio" / "slot-parts.js").as_uri()
    program = (f"import * as m from {json.dumps(url)};\n"
               "const asset = { picture: 'a.png', model: 'm.glb', rigged: 'r.glb', clips: [] };\n"
               "const out = { choices: m.DRESSINGS.map(([v]) => v) };\n"
               "try { m.bodyPayload(asset, 'elf', {}, ''); } catch (e) { out.missing = e.message; }\n"
               "out.payload = m.bodyPayload(asset, 'elf', {}, 'clothes', 'green');\n"
               "console.log(JSON.stringify(out));")
    out = json.loads(subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True, encoding="utf-8").stdout)
    assert out["choices"] == ["bodysuit", "clothes", "underwear", "other"]
    assert "dressed" in out["missing"]
    assert out["payload"]["dressing"] == "clothes" and out["payload"]["dressing_words"] == "green"
