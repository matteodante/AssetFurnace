"""The slot_parts command: body registration refusals and a whole cuts run on a stick figure."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_to_3dlab.creature_family import classify
from image_to_3dlab.slot_parts import body, camera
from tests._creature_skeletons import humanoid

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("slot_parts_cli", SCRIPTS / "slot_parts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load()


def test_slot_list_parses_flags():
    assert cli.parse_slots("helmet, shoulders:split,gloves:B0,") == {
        "helmet": {}, "shoulders": {"split": True}, "gloves": {"route": "B0"}}


def _body(tmp_path, size=400):
    """The humanoid fixture (1.7 units tall) seen by a camera that fills a `size` picture."""
    parents, heads, names = humanoid()
    centred = {b: h - np.array([0, 0.85, 0]) for b, h in heads.items()}
    cam = camera.Camera(fov=0.8, dist=2.8, size=size)
    rec = body.record("stick", image=tmp_path / "body.png", image_size=(size, size), rigged=tmp_path / "r.glb",
                      views=tmp_path / "v", cam=cam, parents=parents, heads=centred, found=classify(parents, centred))
    return body.write(rec, tmp_path / "bodies"), cam, centred, names


def _concept(tmp_path, cam, heads, size=400):
    """A matted concept: fat lines along the bones, so every slot has pixels."""
    rgba = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba)
    parents = humanoid()[0]
    for bone, parent in parents.items():
        if parent is None:
            continue
        (x0, y0), (x1, y1) = cam.project(np.stack([heads[parent], heads[bone]]))
        draw.line([(x0, y0), (x1, y1)], fill=(180, 30, 30, 255), width=18)
    path = tmp_path / "concept.png"
    rgba.save(path)
    return path


def test_cuts_writes_one_flattened_picture_per_slot(tmp_path):
    body_json, cam, heads, _ = _body(tmp_path)
    concept = _concept(tmp_path, cam, heads)
    chosen = cli.slots.choose(["helmet", "torso", "gloves", "boots", "weapon"])
    result = cli.make_cuts(body_json, concept, chosen, tmp_path / "cuts", size=256)
    assert result["subject_from"] == "alpha"
    assert set(result["slots"]) == {"helmet", "torso", "gloves", "boots"}
    assert "weapon" in result["skipped"]
    for entry in result["slots"].values():
        canvas = Image.open(tmp_path / "cuts" / entry["picture"])
        assert canvas.size == (256, 256) and canvas.mode == "RGB"
        assert Image.open(tmp_path / "cuts" / entry["cut"]).mode == "RGBA"
    gloves = result["slots"]["gloves"]["box"]
    assert gloves[2] < 200  # the character's right hand: the image-left one
    saved = json.loads((tmp_path / "cuts" / "cuts.json").read_text())
    assert saved["slots"]["torso"]["placement"]["size"] == 256
    labels = np.asarray(Image.open(tmp_path / "cuts" / "slots.png"))
    assert labels.shape == (400, 400)


def test_cuts_refuses_a_concept_framed_differently(tmp_path):
    body_json, *_ = _body(tmp_path)
    wide = tmp_path / "wide.png"
    Image.new("RGB", (800, 400), "white").save(wide)
    with pytest.raises(ValueError, match="framing"):
        cli.make_cuts(body_json, wide, cli.slots.choose(["torso"]), tmp_path / "out")


def test_an_unmatted_concept_goes_through_the_cut_out(tmp_path):
    body_json, cam, heads, _ = _body(tmp_path)
    matted = Image.open(_concept(tmp_path, cam, heads))
    flat = tmp_path / "flat.png"
    Image.new("RGB", matted.size, (217, 217, 217)).save(flat)
    calls = []

    def fake_cut_out(picture):
        calls.append(picture.size)
        return matted, "fake-matte"

    result = cli.make_cuts(body_json, flat, cli.slots.choose(["torso"]), tmp_path / "out", 128, cut_out=fake_cut_out)
    assert calls == [(400, 400)] and result["subject_from"] == "fake-matte"


def test_body_needs_the_pixal3d_cut_out(tmp_path, capsys):
    Image.new("RGB", (8, 8)).save(tmp_path / "body.png")
    code = cli.main(["body", "b", "--image", str(tmp_path / "body.png"), "--rigged", str(tmp_path / "r.glb"),
                     "--views", str(tmp_path / "v"), "--root", str(tmp_path), "--dressing", "bodysuit"])
    assert code == 1 and "__matted.png" in capsys.readouterr().out


def test_body_refuses_a_rig_that_is_not_a_humanoid(tmp_path, capsys):
    from tests._skinned_glb import write as write_skinned_glb

    Image.new("RGB", (8, 8)).save(tmp_path / "body.png")
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(tmp_path / "body__matted.png")
    rig = write_skinned_glb(tmp_path / "strip.glb")
    code = cli.main(["body", "b", "--image", str(tmp_path / "body.png"), "--rigged", str(rig),
                     "--views", str(tmp_path / "v"), "--root", str(tmp_path), "--dressing", "bodysuit"])
    assert code == 1 and "humanoid" in capsys.readouterr().out
