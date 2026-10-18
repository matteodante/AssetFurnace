"""Text to image as Qwen-Image 2.1 Turbo's card says, on every route (handed over by the audit,
17/10/2026): the batches' picture step, Create in the studio and a set's weapon and shield
pictures draw at the card's presets (2048 for text alone), with its transparent background
and prompt rewriting; the step catalogue says so, and the Unreal export's two_sided is in it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from image_to_3dlab import step_catalog, text_image
from image_to_3dlab.batch_steps import picture as picture_step

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _param(step, name):
    return next(p for p in step_catalog.CATALOG[step].params if p.name == name)


def test_the_catalogue_offers_the_cards_sizes_and_its_two_switches():
    width, height = _param("picture", "width"), _param("picture", "height")
    assert width.default == height.default == 2048
    assert width.range[1] == height.range[1] == text_image.MAX_SIZE
    assert _param("picture", "transparent").default is True and _param("picture", "rewrite").default is True
    assert "above 1" in _param("picture", "negative_prompt").help
    assert _param("export_unreal", "two_sided").type == "bool"


def test_the_picture_step_draws_as_the_card_says_unless_asked_otherwise(tmp_path, monkeypatch):
    from tests.test_batch_steps_export_checks import _context, _row

    seen = []
    monkeypatch.setattr(text_image, "generate",
                        lambda prompt, settings, path, **k: seen.append(settings) or path.write_bytes(b"png") or path)
    from dataclasses import replace

    ctx = replace(_context(tmp_path, _row(tmp_path, prompt="a barrel", images=()), "picture", {}), candidate_key="k")
    picture_step.run(ctx)
    assert {k: seen[-1][k] for k in ("width", "height", "transparent", "rewrite")} == {
        "width": 2048, "height": 2048, "transparent": True, "rewrite": True}
    ctx = replace(_context(tmp_path / "b", _row(tmp_path, prompt="a barrel", images=()), "picture",
                           {"width": 1024, "height": 1024, "transparent": False}), candidate_key="k2")
    picture_step.run(ctx)
    assert seen[-1]["width"] == 1024 and seen[-1]["transparent"] is False and seen[-1]["rewrite"] is True


def test_the_command_line_turns_the_switches_off(monkeypatch):
    import text_image as cli

    seen = {}
    monkeypatch.setattr(cli, "make", lambda prompt, out, settings, count, **k: seen.update(settings) or [])
    assert cli.main(["a barrel", "--opaque", "--no-rewrite"]) == 0
    assert seen["transparent"] is False and seen["rewrite"] is False


def test_a_sets_weapon_picture_is_drawn_as_text_to_image(monkeypatch):
    import slot_parts as cli

    sent = {}

    def prepare(prompt, settings, **k):
        sent.update(settings)
        return "REWRITTEN " + prompt, {}

    monkeypatch.setattr(text_image, "prepare_prompt", prepare)
    monkeypatch.setattr(text_image, "resolve_weights", lambda: {"diffusion_model": Path("d"), "vae": Path("v"),
                                                                "llm": Path("l")})
    command = cli.text_command("a longsword", Path("s1.png"), 3)
    assert command[command.index("-W") + 1] == "2048" and sent["transparent"] is True
    assert any("REWRITTEN a longsword" in part for part in command)


def test_create_offers_the_cards_presets_with_2048_first():
    if shutil.which("node") is None:
        pytest.skip("Node is required to execute browser ES modules")
    url = (REPO / "viewer" / "studio" / "create.js").as_uri()
    program = f"""import * as m from {json.dumps(url)};
      console.log(JSON.stringify({{presets: m.SIZE_PRESETS, plain: m.pictureSettings({{}}, () => 0),
        wide: m.pictureSettings({{size: '16:9'}}, () => 0), odd: m.pictureSettings({{size: 'huge'}}, () => 0)}}));"""
    out = json.loads(subprocess.run(["node", "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True).stdout)
    card = {name: [w, h] for name, w, h, _ in out["presets"] if name in text_image.CARD_SIZES}
    assert card == {name: list(size) for name, size in text_image.CARD_SIZES.items()}
    assert out["presets"][0][0] == "1:1" and (out["plain"]["width"], out["plain"]["height"]) == (2048, 2048)
    assert (out["wide"]["width"], out["wide"]["height"]) == (2752, 1536) and out["odd"]["width"] == 2048


def test_a_chosen_size_reaches_the_picture_settings():
    # Codex on #113: the menu's change handler made "4:3" a number (NaN), so every choice drew 2048x2048
    if shutil.which("node") is None:
        pytest.skip("Node is required to execute browser ES modules")
    url = (REPO / "viewer" / "studio" / "create.js").as_uri()
    program = f"""import * as m from {json.dumps(url)};
      const size = m.picValue('size', {{type: 'select-one', value: '4:3'}});
      console.log(JSON.stringify({{size, settings: m.pictureSettings({{size}}, () => 0),
        draft: m.pictureSettings({{size: m.picValue('size', {{type: 'select-one', value: 'draft'}})}}, () => 0),
        seed: m.picValue('seed', {{type: 'number', value: '12'}}),
        fresh: m.picValue('newEachTime', {{type: 'checkbox', checked: false}})}}));"""
    out = json.loads(subprocess.run(["node", "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True).stdout)
    assert out["size"] == "4:3" and (out["settings"]["width"], out["settings"]["height"]) == (2400, 1792)
    assert (out["draft"]["width"], out["draft"]["height"]) == (1024, 1024)
    assert out["seed"] == 12 and out["fresh"] is False
    assert "steps" not in out["settings"]   # the Turbo's schedule runs 8 steps; a number here did nothing
