"""A weapon's picture turned grip down before it becomes 3D (slot_parts/orient.py): the picture
model drew the first real set's sword point down in three seeds out of three, and one of two
again on 17/10/2026 despite the canonical prompt. The picture judge is a stand-in."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_3dlab.slot_parts import orient, pieces

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import slot_parts  # noqa: E402


def test_the_weapon_picture_asks_for_the_grip_at_the_bottom():
    assert "grip at the bottom" in pieces.held_prompt("weapon", "a longsword")
    assert "grip at the bottom" not in pieces.held_prompt("shield", "a round shield")


def _sword(path: Path, point_down: bool) -> Path:
    picture = np.full((64, 32, 3), 217, np.uint8)
    picture[(0 if point_down else 40):(24 if point_down else 64), 12:20] = 40   # the dark grip: up when point down
    Image.fromarray(picture).save(path)
    return path


def test_a_weapon_drawn_point_down_is_turned_grip_down(tmp_path):
    down, up_ = _sword(tmp_path / "s1.png", point_down=True), _sword(tmp_path / "s2.png", point_down=False)
    before = np.asarray(Image.open(down)).copy()

    def ask(picture, question):   # the judge: the grip is the dark part
        rows = np.asarray(Image.open(picture))[:, :, 0] < 100
        top = rows[:32].sum() > rows[32:].sum()
        return 0.95 if ("top" in question) == top else 0.02

    turned = orient.orient_held([down, up_], ask)
    assert turned == {down: True, up_: False}
    # turned half a turn, never mirrored (code read 18/10/2026: a mirror makes a scimitar its own mirror image)
    assert (np.asarray(Image.open(down)) == before[::-1, ::-1]).all()
    assert json.loads(down.with_name("s1.png.orient.json").read_text())["handle"] == "top"


def test_without_the_judge_the_pictures_are_kept_and_it_is_said(tmp_path, capsys):
    picture = _sword(tmp_path / "s1.png", point_down=True)
    before = picture.read_bytes()
    assert orient.orient_held([picture], None) == {picture: None}
    assert picture.read_bytes() == before and "picture judge" in capsys.readouterr().out


def test_a_picture_already_checked_is_not_asked_again(tmp_path):
    picture = _sword(tmp_path / "s1.png", point_down=True)
    orient.orient_held([picture], lambda p, q: 0.95 if "top" in q else 0.02)
    asked = []
    orient.orient_held([picture], lambda p, q: asked.append(q) or 0.5)
    assert asked == []


def test_the_studio_names_no_game():
    text = (Path(__file__).resolve().parents[1] / "viewer" / "studio" / "slot-parts.js").read_text(encoding="utf-8")
    assert "Story" not in text
