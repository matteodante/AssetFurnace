"""A weapon's or shield's grip (docs/plans/2026-10-17-standalone-weapons.md): found by the picture
judge (Qwen3-VL grounding, a box on its 0-1000 grid) on the canonical picture, brought onto the
3D through the Pixal3D camera, and corrected by a person on any route. The judge is a stand-in."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab.slot_parts import grip
from image_to_3dlab.slot_parts.camera import Camera

CAM = Camera(fov=0.8, dist=3.0, size=1000)


def _sword(path: Path) -> Path:
    """A blade from y = 0.1 to 1.0 and a narrow handle from y = -0.3 to 0.0, upright in glTF."""
    blade = trimesh.creation.box(extents=(0.12, 0.9, 0.02))
    blade.apply_translation((0, 0.55, 0))
    handle = trimesh.creation.box(extents=(0.04, 0.3, 0.04))
    handle.apply_translation((0, -0.15, 0))
    trimesh.util.concatenate([blade, handle]).export(path)
    return path


def _picture(path: Path) -> Path:
    Image.new("RGB", (1000, 1000), (220, 220, 220)).save(path)
    return path


@pytest.mark.parametrize("slot, words, wanted", [
    ("weapon", "a heavy battle axe", "axe"), ("weapon", "an elven longbow", "bow"),
    ("weapon", "a one-handed sword", "sword"), ("weapon", "a war hammer", "hammer"),
    ("weapon", "a flanged mace", "mace"), ("weapon", "a short dagger", "dagger"),
    ("weapon", "a strange relic", "weapon"), ("shield", "a kite shield", "shield"),
])
def test_the_class_comes_from_the_slot_and_the_words(slot, words, wanted):
    assert grip.weapon_class(slot, words) == wanted


@pytest.mark.parametrize("words, wanted", [
    ("a mage's elbow-length gauntlet relic", "weapon"), ("a rainbow crystal", "weapon"),
    ("a waxed leather cudgel", "weapon"), ("a balance of power", "weapon"), ("a spiked collar", "weapon"),
    ("a prod of iron", "weapon"), ("a grimace mask", "weapon"),
    ("a crossbow", "bow"), ("twin longbows", "bow"), ("a battle-axe", "axe"), ("a greataxe", "axe"),
    ("a warhammer", "hammer"), ("a quarterstaff", "staff"), ("a greatsword", "sword"),
    ("a longspear", "spear"), ("a shortspear", "spear"), ("a sledgehammer", "hammer"), ("a greatclub", "mace"),
    ("two staffs", "staff"), ("a morning star", "mace"), ("throwing knives", "dagger"), ("a pickaxe", "axe"),
])
def test_a_class_is_named_by_whole_words(words, wanted):
    """Code read 18/10/2026: "elbow" and "rainbow" read as a bow, "waxed" as an axe."""
    assert grip.weapon_class("weapon", words) == wanted


def test_the_judges_box_is_read_on_its_0_to_1000_grid():
    text = '```json\n[{"bbox_2d": [472, 767, 519, 945], "label": "grip"}]\n```'
    assert grip.parse_box(text, (2048, 1024)) == pytest.approx((966.656, 785.408, 1062.912, 967.68))
    assert grip.parse_box("I cannot see a grip", (100, 100)) is None


def test_the_box_lands_on_the_handle_and_the_axis_points_to_the_tip(tmp_path):
    points = np.asarray(trimesh.load(_sword(tmp_path / "t.glb"), force="mesh").vertices)
    handle = CAM.project(np.array([[0, -0.3, 0], [0, 0.0, 0]]))
    box = (handle[:, 0].min() - 15, handle[:, 1].min() - 10, handle[:, 0].max() + 15, handle[:, 1].max() + 10)
    point, axis = grip.from_box(points, CAM, box)
    assert point == pytest.approx([0, -0.15, 0], abs=0.03)
    assert axis == pytest.approx([0, 1, 0], abs=0.05)


def test_a_shield_is_held_at_the_centre_of_its_back():
    disc = trimesh.creation.cylinder(radius=0.4, height=0.06)   # along z: its face to the camera
    disc.apply_translation((0.1, 0.2, 0.0))
    point, axis = grip.shield_grip(np.asarray(disc.vertices))
    assert point == pytest.approx([0.1, 0.2, -0.03], abs=1e-6) and axis == pytest.approx([0, 1, 0])


def test_the_grip_is_found_by_the_judge_and_recorded(tmp_path):
    model, picture = _sword(tmp_path / "weapon.glb"), _picture(tmp_path / "s1.png")
    handle = CAM.project(np.array([[0, -0.3, 0], [0, 0.0, 0]])) / 1000 * 1000

    def ask_box(picture, phrase):
        assert "grip" in phrase or "handle" in phrase
        return f'[{{"bbox_2d": [{handle[:, 0].min() - 15:.0f}, {handle[:, 1].min():.0f}, ' \
               f'{handle[:, 0].max() + 15:.0f}, {handle[:, 1].max():.0f}], "label": "grip"}}]'

    record = grip.find(model, picture, CAM, slot="weapon", words="a longsword", ask_box=ask_box)
    assert record["by"] == "judge" and record["class"] == "sword" and record["size_cm"] == grip.CLASSES["sword"].size_cm
    assert record["point"] == pytest.approx([0, -0.15, 0], abs=0.03)
    saved = grip.write(record, tmp_path / "grip.json")
    assert json.loads(saved.read_text())["camera"] == CAM.to_dict()


def test_without_the_judge_there_is_no_grip_and_it_is_said(tmp_path, capsys):
    record = grip.find(_sword(tmp_path / "w.glb"), _picture(tmp_path / "s1.png"), CAM, slot="weapon", words="",
                       ask_box=None)
    assert record["by"] == "none" and record["point"] is None
    assert "grip" in capsys.readouterr().out


def test_a_person_moves_the_grip_and_changes_the_class(tmp_path):
    model, picture = _sword(tmp_path / "weapon.glb"), _picture(tmp_path / "s1.png")
    record = grip.find(model, picture, CAM, slot="weapon", words="a longsword", ask_box=None)
    at = CAM.project(np.array([[0, 0.5, 0]]))[0] / 1000   # on the blade, as a fraction of the picture
    moved = grip.correct(record, at=tuple(at), weapon_class="axe")
    assert moved["by"] == "person" and moved["class"] == "axe"
    assert moved["size_cm"] == grip.CLASSES["axe"].size_cm   # the class's size, unless one is given
    assert moved["point"][1] == pytest.approx(0.5, abs=0.06)
    assert grip.correct(moved, size_cm=91)["size_cm"] == 91
    with pytest.raises(ValueError, match="class"):
        grip.correct(record, weapon_class="banana")
    with pytest.raises(ValueError, match="picture"):
        grip.correct(record, at=(1.4, 0.5))


def test_a_long_handle_is_held_near_its_end_for_an_axe_and_in_the_middle_for_a_staff(tmp_path):
    haft = trimesh.creation.box(extents=(0.04, 1.0, 0.04))   # a haft from y = -0.5 to 0.5
    head = trimesh.creation.box(extents=(0.4, 0.2, 0.04))
    head.apply_translation((0, 0.55, 0))
    points, _ = trimesh.sample.sample_surface(trimesh.util.concatenate([haft, head]), 20000, seed=0)
    ends = CAM.project(np.array([[0, -0.5, 0], [0, 0.5, 0]]))
    box = (ends[:, 0].min() - 10, ends[:, 1].min(), ends[:, 0].max() + 10, ends[:, 1].max())   # the judge's: the haft
    axe, _ = grip.from_box(points, CAM, box, hold=grip.CLASSES["axe"].hold)
    staff, _ = grip.from_box(points, CAM, box, hold=grip.CLASSES["staff"].hold)
    assert axe[1] == pytest.approx(-0.3, abs=0.06) and staff[1] == pytest.approx(0.0, abs=0.06)


def test_a_new_class_moves_the_grip_by_its_own_rule(tmp_path):
    """Code read 18/10/2026: a class changed by a person changed only the size; a staff made an
    axe kept the grip in the middle of the haft, a weapon made a shield kept its front grip."""
    haft = trimesh.creation.box(extents=(0.04, 1.0, 0.04))
    head = trimesh.creation.box(extents=(0.4, 0.2, 0.04))
    head.apply_translation((0, 0.55, 0))
    model = tmp_path / "haft.glb"
    trimesh.util.concatenate([haft, head]).export(model)
    picture = _picture(tmp_path / "s1.png")
    ends = CAM.project(np.array([[0, -0.5, 0], [0, 0.5, 0]]))
    text = (f'[{{"bbox_2d": [{ends[:, 0].min() - 10:.0f}, {ends[:, 1].min():.0f}, '
            f'{ends[:, 0].max() + 10:.0f}, {ends[:, 1].max():.0f}]}}]')
    staff = grip.find(model, picture, CAM, slot="weapon", words="a wizard's staff", ask_box=lambda p, q: text)
    assert staff["class"] == "staff" and staff["point"][1] == pytest.approx(0.0, abs=0.06)
    axe = grip.correct(staff, weapon_class="axe")
    assert axe["point"][1] == pytest.approx(-0.3, abs=0.06) and axe["size_cm"] == grip.CLASSES["axe"].size_cm
    shield = grip.correct(axe, weapon_class="shield")
    assert shield["axis"] == [0.0, 1.0, 0.0] and shield["point"][2] == pytest.approx(-0.02, abs=0.005)
    placed = grip.correct(staff, at=(0.5, 0.5))
    assert grip.correct(placed, weapon_class="axe")["point"] == placed["point"]   # the person's point stays
    assert grip.correct(shield, weapon_class="sword")["point"] is not None       # the judge's box is kept


def test_a_grip_corrected_before_the_judges_box_was_kept_survives_a_class_change(tmp_path):
    """Codex review: a record saved by a person before `placed` and `judge_box` existed lost its
    grip when only its class changed."""
    model = _sword(tmp_path / "weapon.glb")
    picture = _picture(tmp_path / "s1.png")
    old = {"slot": "weapon", "class": "staff", "size_cm": 170, "picture": str(picture), "model": str(model),
           "camera": CAM.to_dict(), "box": [490, 600, 510, 620], "point": [0, -0.1, 0], "axis": [0, 1, 0],
           "by": "person"}
    corrected = grip.correct(old, weapon_class="axe")
    assert corrected["point"] == [0, -0.1, 0] and corrected["axis"] == [0, 1, 0]
