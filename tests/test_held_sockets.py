"""The user's decision C (17/10/2026): the weapon and shield of a set are static meshes on a
socket of the hand in Unreal (hand_r, hand_l), not skinned pieces. The grip is set by the
socket and by the weapon standing handle down, which the picture judge checks and turns when
the picture model drew it the other way (it drew the first real set's sword point down in
three seeds out of three). The socket is made at import, from the skeleton Unreal imported,
so it cannot disagree with it. Blender, Unreal and the judge are stand-ins here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import unreal_profile as up
from image_to_3dlab.slot_parts import pieces

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import export_unreal  # noqa: E402
import slot_parts  # noqa: E402

from tests.batch_fakes import tiny_glb  # noqa: E402


def test_the_weapon_picture_asks_for_the_handle_at_the_bottom():
    assert "handle at the bottom" in pieces.held_prompt("weapon", "a longsword")
    assert "handle at the bottom" not in pieces.held_prompt("shield", "a round shield")


def _sword(path: Path, point_down: bool) -> Path:
    picture = np.full((64, 32, 3), 217, np.uint8)
    picture[(0 if point_down else 40):(24 if point_down else 64), 12:20] = 40   # the dark grip: up when point down
    Image.fromarray(picture).save(path)
    return path


def test_a_weapon_drawn_point_down_is_turned_handle_down(tmp_path):
    down, up_ = _sword(tmp_path / "s1.png", point_down=True), _sword(tmp_path / "s2.png", point_down=False)
    before = np.asarray(Image.open(down)).copy()

    def ask(picture, question):   # the judge: the grip is the dark part
        rows = np.asarray(Image.open(picture))[:, :, 0] < 100
        top = rows[:32].sum() > rows[32:].sum()
        return 0.95 if ("top" in question) == top else 0.02

    turned = slot_parts.orient_held([down, up_], ask)
    assert turned == {down: True, up_: False}
    assert (np.asarray(Image.open(down)) == before[::-1]).all()                 # flipped top to bottom
    assert json.loads(down.with_name("s1.png.orient.json").read_text())["handle"] == "top"


def test_without_the_judge_the_pictures_are_kept_and_it_is_said(tmp_path, capsys):
    picture = _sword(tmp_path / "s1.png", point_down=True)
    before = picture.read_bytes()
    assert slot_parts.orient_held([picture], None) == {picture: None}
    assert picture.read_bytes() == before and "picture judge" in capsys.readouterr().out


def test_a_picture_already_checked_is_not_asked_again(tmp_path):
    picture = _sword(tmp_path / "s1.png", point_down=True)
    slot_parts.orient_held([picture], lambda p, q: 0.95 if "top" in q else 0.02)
    asked = []
    slot_parts.orient_held([picture], lambda p, q: asked.append(q) or 0.5)
    assert asked == []


# ---------------------------------------------------------------------- export

def test_a_held_piece_is_a_static_mesh_on_a_hand_socket(tmp_path):
    weapon, body = tiny_glb(tmp_path / "weapon.glb"), tiny_glb(tmp_path / "body_masked.glb")
    p = export_unreal.plan(weapon, kind="prop", size_cm=180, body=body, body_name="HumanMale",
                           socket={"name": "SOCKET_Weapon", "bone": "hand_r"}, name="HumanMaleWeapon",
                           out_dir=tmp_path / "out")
    assert p["job"]["held"] == {"name": "SOCKET_Weapon", "bone": "hand_r"} and p["job"]["body"].endswith("body_masked.glb")
    assert p["names"]["mesh"].startswith("SM_")   # static, not skinned
    side = up.sidecar(kind="prop", names=p["names"], glb="x.glb", size_cm=180, measure="height", lods=1,
                      collision=None, humanoid=False, skeleton_from="SK_HumanMale", animations=[], tint=None,
                      problems=[], socket=p["job"]["held"])
    assert side["skinned"] is False and side["socket"] == {"name": "SOCKET_Weapon", "bone": "hand_r",
                                                            "on": "SK_HumanMale"}
    assert side["import"]["reuse_skeleton"] is False   # a static mesh has no skeleton of its own
    with pytest.raises(ValueError, match="socket"):
        export_unreal.plan(weapon, kind="prop", size_cm=180, socket={"name": "SOCKET_Weapon", "bone": "hand_r"},
                           out_dir=tmp_path / "o2")   # a socket needs the body it is on


def test_unreal_checks_the_socket_not_the_floor():
    expected = {"kind": "prop", "problems": [], "orm": False, "skinned": False, "collision": None, "mesh": "SM_W", "lods": 1,
                "socket": {"name": "SOCKET_Weapon", "bone": "hand_r", "on": "SK_HumanMale"},
                "expect": {"size_cm": 180, "measure": "height", "base_z_cm": 0.0}}
    good = up.evaluate_unreal({"bounds_cm": [5, 5, 80], "min_z": 90.0, "names": ["SM_W"], "lods": 1,
                               "socket": {"made": True, "bone_found": True, "grip_inside": True,
                                          "grip_gap_cm": 0.0}}, expected)
    names = {r["check"]: r["ok"] for r in good}
    assert "size" not in names and "base" not in names and names["socket"] is True
    bad = up.evaluate_unreal({"bounds_cm": [5, 5, 80], "min_z": 90.0, "names": ["SM_W"], "lods": 1,
                              "socket": {"made": True, "bone_found": True, "grip_inside": False,
                                         "grip_gap_cm": 37.0}}, expected)
    assert {r["check"]: r["ok"] for r in bad}["socket"] is False


def test_a_set_exports_its_weapon_and_shield_on_sockets(tmp_path):
    from tests.test_slot_parts_cli import _body

    body_json, *_ = _body(tmp_path)
    fit = tmp_path / "sets" / "red" / "stick" / "fit"
    fit.mkdir(parents=True)
    for name in ("body_masked.glb", "torso.glb", "weapon.glb", "shield.glb"):
        tiny_glb(fit / name)
    (fit / "fit.json").write_text(json.dumps({"files": ["body_masked.glb", "torso.glb", "weapon.glb", "shield.glb"]}))
    calls = []

    def exporter(source, **kwargs):
        calls.append((Path(source).name, kwargs))
        return {"glb": str(source), "problems": []}

    path = slot_parts.unreal_set(fit, body_json, tmp_path / "unreal", size_cm=180, exporter=exporter)
    by_file = dict(calls)
    assert by_file["torso.glb"]["kind"] == "equipment" and "socket" not in by_file["torso.glb"]
    assert by_file["weapon.glb"]["kind"] == "prop" and by_file["weapon.glb"]["socket"] == {
        "name": "SOCKET_Weapon", "bone": "hand_r"}
    assert by_file["shield.glb"]["socket"] == {"name": "SOCKET_Shield", "bone": "hand_l"}
    summary = json.loads(path.read_text())
    assert summary["sockets"] == {"weapon": {"name": "SOCKET_Weapon", "bone": "hand_r"},
                                  "shield": {"name": "SOCKET_Shield", "bone": "hand_l"}}
    assert any("SOCKET_Weapon" in line for line in summary["how"])


@pytest.mark.skipif(__import__("image_to_3dlab.blender", fromlist=["find_blender"]).find_blender() is None,
                    reason="needs Blender")
def test_real_blender_keeps_a_held_piece_where_the_body_puts_it(tmp_path):
    # The fit leaves the weapon skinned to the body's skeleton, in the body's space: as a static
    # mesh it must move exactly as the body does (scale, turn, ground), or the socket misses it.
    from image_to_3dlab.glb_turn import parse_glb
    from tests.test_export_unreal import _skinned_glb

    body, held = tmp_path / "body.glb", tmp_path / "held.glb"
    body.write_bytes(_skinned_glb())
    held.write_bytes(_skinned_glb())
    whole = export_unreal.export_unreal(body, kind="creature", size_cm=150, out_dir=tmp_path / "b", log=lambda m: None)
    piece = export_unreal.export_unreal(held, kind="prop", size_cm=150, body=body, name="Sword",
                                        socket={"name": "SOCKET_Weapon", "bone": "hand_r"},
                                        out_dir=tmp_path / "w", log=lambda m: None)
    boxes = [json.loads(Path(r["sidecar"]).read_text(encoding="utf-8"))["blender"]["box_after_m"] for r in (whole, piece)]
    assert np.allclose(boxes[0], boxes[1], atol=1e-4)
    document, _ = parse_glb(Path(piece["glb"]).read_bytes())
    assert not document.get("skins") and "SM_Sword" in [n["name"] for n in document["nodes"]]
