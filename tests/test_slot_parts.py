"""Slot parts, first part: the slot catalogue, bone roles, Pixal3D's camera, the slot map
and the cut-outs that route C completes into one clean picture per piece."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab.creature_family import classify
from image_to_3dlab.slot_parts import body, camera, cuts, masks, skeleton, slots
from tests._creature_skeletons import humanoid, quadruped

# ------------------------------------------------------------------ the catalogue


def test_choose_keeps_the_catalogue_order_and_fills_defaults():
    chosen = slots.choose({"boots": {}, "helmet": {"prompt": "a closed visor"}})
    assert list(chosen) == ["helmet", "boots"]
    assert chosen["helmet"] == {"route": "C", "split": False, "prompt": "a closed visor"}
    assert chosen["boots"]["route"] == "C"


def test_choose_accepts_a_plain_list():
    assert list(slots.choose(["torso", "gloves"])) == ["torso", "gloves"]


@pytest.mark.parametrize("raw, words", [
    ({"cloak": {}}, "unknown slot"),
    ({"helmet": {"route": "A"}}, "route"),
    ({"helmet": {"split": True}}, "split"),
    ({}, "at least one"),
])
def test_choose_refuses_what_it_cannot_make(raw, words):
    with pytest.raises(ValueError, match=words):
        slots.choose(raw)


def test_shoulders_may_be_one_piece_or_two():
    assert slots.choose({"shoulders": {"split": True}})["shoulders"]["split"] is True


def test_every_drawn_slot_has_a_label_and_a_face_budget():
    for name in slots.DRAWN:
        assert slots.LABELS[slots.label(name)] == name
        assert 0 < slots.CATALOGUE[name].faces <= 8000  # the equipment_part profile's budget
    assert slots.LABELS[0] == "background" and slots.LABELS[-1] == "uncovered"


# ------------------------------------------------------------------ bone roles


def _found(skeleton_rows):
    parents, heads, names = skeleton_rows
    return parents, heads, names, classify(parents, heads)


def test_roles_of_a_humanoid_from_its_tree_alone():
    _, _, names, found = _found(humanoid())
    roles = skeleton.bone_roles(found)
    assert roles[names["hips"]][0] == "hips"
    assert roles[names["spine2"]][0] == "neck" and roles[names["spine3"]][0] == "head"
    assert roles[names["armL0"]] == ("upperarm", 0, "L")  # a three-bone arm has no clavicle
    assert roles[names["armR2"]][0] == "hand"
    assert roles[names["legL0"]][0] == "thigh" and roles[names["legL3"]][0] == "foot"


def test_a_long_arm_starts_with_a_clavicle_and_ends_in_hand_bones():
    assert skeleton.arm_roles(7) == ["clavicle", "upperarm", "forearm", "hand", "hand", "hand", "hand"]
    assert skeleton.arm_roles(2) == ["upperarm", "forearm"]


def test_only_a_humanoid_carries_slots():
    parents, heads, _ = quadruped()
    with pytest.raises(ValueError, match="humanoid"):
        skeleton.bone_roles(classify(parents, heads))


@pytest.mark.parametrize("role, t, slot", [
    ("head", 0.5, "helmet"), ("spine", 0.1, "torso"), ("clavicle", 0.9, "shoulders"),
    ("upperarm", 0.3, "shoulders"), ("upperarm", 0.6, "uncovered"), ("hand", 0.0, "gloves"),
    ("thigh", 1.0, "legs"), ("shin", 0.6, "legs"), ("shin", 0.8, "boots"), ("foot", 0.0, "boots"),
])
def test_slot_follows_the_bone_and_where_along_it(role, t, slot):
    assert skeleton.slot_for(role, t) == slot


def test_a_bone_runs_to_its_child_and_the_hips_to_the_spine():
    parents, heads, names, found = _found(humanoid())
    segs = {s.bone: s for s in skeleton.joint_segments(parents, heads, skeleton.bone_roles(found))}
    hips = segs[names["hips"]]
    assert np.allclose(hips.end, heads[names["spine0"]])  # not the average with the thighs
    shin = segs[names["legL1"]]
    assert np.allclose(shin.end, heads[names["legL2"]])
    hand = segs[names["armL2"]]  # a leaf: a point
    assert np.allclose(hand.head, hand.end)


# ------------------------------------------------------------------ camera


def _views(tmp_path, size=100, box=(30, 10, 70, 90), fov=0.349, dist=2.8):
    folder = tmp_path / "body.svviews"
    folder.mkdir()
    alpha = np.zeros((size, size, 4), np.uint8)
    x0, y0, x1, y1 = box
    alpha[y0:y1 + 1, x0:x1 + 1] = 255
    Image.fromarray(alpha).save(folder / "input.png")
    matrix = np.eye(4)
    matrix[2, 3] = dist
    (folder / "transforms.json").write_text(json.dumps({
        "camera_angle_x": fov, "frames": [{"file_path": "input.png", "transform_matrix": matrix.tolist()}]}))
    return folder


def test_the_origin_lands_in_the_middle_and_plus_x_to_the_right():
    cam = camera.Camera(fov=0.349, dist=2.8, size=100)
    (u, v), (ur, _), (_, vu) = cam.project(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]))
    assert (u, v) == pytest.approx((50, 50))
    assert ur > 50 and vu < 50  # image x = +X, image up = +Y


def test_nearer_points_spread_further_from_the_centre():
    cam = camera.Camera(fov=0.349, dist=2.8, size=100)
    near, far = cam.project(np.array([[0.1, 0, 0.5], [0.1, 0, -0.5]]))
    assert near[0] > far[0]


def test_camera_maps_pixal3d_input_onto_the_source_picture(tmp_path):
    views = _views(tmp_path)  # subject box 40 x 80 in a 100 px input
    source = np.zeros((400, 400), bool)
    source[100:261, 150:231] = True  # the same subject, twice the size, elsewhere
    cam = camera.from_views(views, source)
    assert cam.scale == pytest.approx(2.0, rel=0.02)
    assert cam.to_source(np.array([[30.0, 10.0]]))[0] == pytest.approx((150, 100), abs=1.5)
    assert camera.Camera.from_dict(cam.to_dict()) == cam


def test_alpha_box_of_nothing_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        camera.alpha_box(np.zeros((4, 4), bool))


# ------------------------------------------------------------------ slot map


def _segments():
    # a stick figure in pixels: neck up to the head, a spine, one arm to the image-left, a leg
    S = skeleton.Segment
    return [
        S("neck", "neck", "", np.array([50.0, 30.0]), np.array([50.0, 10.0])),
        S("spine", "spine", "", np.array([50.0, 60.0]), np.array([50.0, 30.0])),
        S("upper", "upperarm", "R", np.array([45.0, 32.0]), np.array([25.0, 32.0])),
        S("fore", "forearm", "R", np.array([25.0, 32.0]), np.array([5.0, 32.0])),
        S("thigh", "thigh", "R", np.array([45.0, 60.0]), np.array([45.0, 80.0])),
        S("shin", "shin", "R", np.array([45.0, 80.0]), np.array([45.0, 99.0])),
    ]


def test_each_pixel_takes_the_slot_of_its_nearest_bone():
    fg = np.ones((100, 100), bool)
    labels = masks.slot_labels(fg, _segments())

    def at(x, y):
        return slots.LABELS[labels[y, x]]

    assert at(50, 12) == "helmet"
    assert at(50, 50) == "torso"
    assert at(8, 33) == "gloves"
    assert at(45, 97) == "boots"
    assert at(45, 65) == "legs"


def test_background_stays_background_and_unchosen_slots_stay_body():
    fg = np.zeros((100, 100), bool)
    fg[5:, 40:60] = True
    labels = masks.slot_labels(fg, _segments(), chosen={"torso"})
    names = {slots.LABELS[i] for i in np.unique(labels)}
    assert names == {"background", "torso", "uncovered"}
    assert labels[0, 0] == 0


def test_overlay_is_a_picture_of_the_same_size():
    labels = np.zeros((20, 30), np.uint8)
    labels[5:10, 5:10] = slots.label("helmet")
    out = masks.overlay(np.zeros((20, 30, 3), np.uint8), labels)
    assert out.shape == (20, 30, 3) and out[7, 7].any()


# ------------------------------------------------------------------ cut-outs


def _two_gloves():
    rgb = np.full((60, 100, 3), 200, np.uint8)
    labels = np.zeros((60, 100), np.uint8)
    labels[20:30, 5:25] = slots.label("gloves")   # image-left: the character's right hand
    labels[20:30, 75:95] = slots.label("gloves")
    labels[10:50, 40:60] = slots.label("torso")
    rgb[20:30, 5:25] = (255, 0, 0)
    return rgb, labels


def test_a_paired_slot_keeps_the_characters_right_side():
    rgb, labels = _two_gloves()
    piece, box = cuts.cut_slot(rgb, labels, "gloves")
    assert box == (5, 20, 24, 29)
    assert piece.shape == (10, 20, 4)
    assert (piece[..., :3][piece[..., 3] > 0] == (255, 0, 0)).all()


def test_a_single_slot_keeps_all_of_it():
    rgb, labels = _two_gloves()
    _, box = cuts.cut_slot(rgb, labels, "torso")
    assert box == (40, 10, 59, 49)


def test_a_missing_slot_is_an_error():
    rgb, labels = _two_gloves()
    with pytest.raises(ValueError, match="boots"):
        cuts.cut_slot(rgb, labels, "boots")


def test_flatten_centres_the_piece_on_grey_and_remembers_how():
    piece = np.zeros((10, 40, 4), np.uint8)
    piece[..., 0] = 255
    piece[..., 3] = 255
    canvas, place = cuts.flatten(piece, (100, 50, 139, 59), size=200)
    assert canvas.shape == (200, 200, 3)
    assert tuple(canvas[0, 0]) == cuts.GREY
    assert place.scale == pytest.approx(0.73 * 200 / 40)
    assert tuple(canvas[100, 100]) == (255, 0, 0)


def test_a_box_in_the_canvas_maps_back_to_the_concept():
    piece = np.full((10, 40, 4), 255, np.uint8)
    canvas, place = cuts.flatten(piece, (100, 50, 139, 59), size=200)
    ys, xs = np.where((canvas == (255, 255, 255)).all(-1))
    back = place.to_concept((xs.min(), ys.min(), xs.max(), ys.max()))
    assert back == pytest.approx((100, 50, 139, 59), abs=1.0)
    # a piece the edit grew by a quarter on each side comes back a quarter wider
    grown = place.to_concept((xs.min() - 0.25 * (xs.max() - xs.min()), ys.min(), xs.max(), ys.max()))
    assert grown[0] == pytest.approx(100 - 0.25 * 39, abs=1.0)
    assert cuts.Placement.from_dict(place.to_dict()) == place


# ------------------------------------------------------------------ the base body


def test_a_body_record_round_trips_and_projects_into_any_concept_size(tmp_path):
    parents, heads, names, found = _found(humanoid())
    cam = camera.Camera(fov=0.349, dist=2.8, size=100, scale=4.0, offset=(0.0, 0.0))
    record = body.record("knight", image=tmp_path / "body.png", image_size=(400, 400), rigged=tmp_path / "rig.glb",
                         views=tmp_path / "body.svviews", cam=cam, parents=parents, heads=heads, found=found,
                         labels={"race": "human"})
    path = body.write(record, tmp_path / "bodies")
    loaded = body.load(path)
    assert loaded["name"] == "knight" and loaded["labels"] == {"race": "human"}
    small = body.segments_px(loaded, (400, 400))
    big = body.segments_px(loaded, (800, 800))
    hips_small = next(s for s in small if s.bone == names["hips"])
    hips_big = next(s for s in big if s.bone == names["hips"])
    assert hips_big.head == pytest.approx(2 * hips_small.head)


def test_a_body_name_must_be_a_plain_folder_name(tmp_path):
    with pytest.raises(ValueError, match="name"):
        body.write({"name": "../escape"}, tmp_path)
