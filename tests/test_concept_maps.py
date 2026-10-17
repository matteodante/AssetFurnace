"""The input pictures of a concept: renders and depth from a GLB, a pose map from a rig, lineart."""

import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab.blender import find_blender
from image_to_3dlab.concept import maps


def rgba(path, size=32, value=200, box=(8, 8, 24, 24)):
    image = np.zeros((size, size, 4), np.uint8)
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1, :3] = value
    image[y0:y1, x0:x1, 3] = 255
    Image.fromarray(image).save(path)
    return path


def test_flatten_puts_the_subject_on_plain_grey(tmp_path):
    out = maps.flatten(rgba(tmp_path / "front_colour.png"), tmp_path / "flat.png")
    pixels = np.asarray(Image.open(out))
    assert pixels.shape == (32, 32, 3)
    assert tuple(pixels[0, 0]) == maps.GREY and tuple(pixels[16, 16]) == (200, 200, 200)


def test_depth_is_stretched_over_the_subject_and_the_background_is_black(tmp_path):
    image = np.zeros((32, 32, 4), np.uint8)
    image[8:24, 8:24, 3] = 255
    image[8:24, 8:24, :3] = np.linspace(100, 120, 16).astype(np.uint8)[None, :, None]  # a shallow ramp
    Image.fromarray(image).save(tmp_path / "depth.png")
    out = np.asarray(Image.open(maps.depth_on_black(tmp_path / "depth.png", tmp_path / "black.png")))
    assert out[0, 0].tolist() == [0, 0, 0]
    subject = out[8:24, 8:24, 0]
    assert subject.min() <= maps.DEPTH_FAR + 2 and subject.max() >= 250  # the full range, not 100-120
    assert subject.min() > 0  # the farthest part of the subject still reads apart from the background


def test_depth_needs_a_subject(tmp_path):
    Image.new("RGBA", (8, 8)).save(tmp_path / "empty.png")
    with pytest.raises(ValueError, match="nothing"):
        maps.depth_on_black(tmp_path / "empty.png", tmp_path / "out.png")


def test_lineart_draws_dark_lines_on_white_at_the_asked_size(tmp_path):
    picture = np.full((64, 64, 3), 220, np.uint8)
    picture[16:48, 16:48] = (230, 120, 20)  # orange on light grey: a colour edge more than a light one
    Image.fromarray(picture).save(tmp_path / "robot.png")
    out = np.asarray(Image.open(maps.lineart(tmp_path / "robot.png", tmp_path / "lines.png", size=128)))
    assert out.shape == (128, 128)
    assert out[2, 2] == 255 and out[64, 64] == 255  # flat areas stay white
    assert out[30:34, :].min() < 64  # the square's top edge is a line


def bones_ue(scale=1.0):
    """A front view of a T-posed UE-named rig, in pixels (the character's left on the viewer's right)."""
    def bone(head, tail):
        return {"head": [head[0] * scale, head[1] * scale], "tail": [tail[0] * scale, tail[1] * scale]}
    out = {"neck_01": bone((50, 30), (50, 25)), "head": bone((50, 25), (50, 10))}
    for side, sign in (("l", 1), ("r", -1)):
        out[f"upperarm_{side}"] = bone((50 + 8 * sign, 32), (50 + 20 * sign, 32))
        out[f"lowerarm_{side}"] = bone((50 + 20 * sign, 32), (50 + 32 * sign, 32))
        out[f"hand_{side}"] = bone((50 + 32 * sign, 32), (50 + 36 * sign, 32))
        out[f"thigh_{side}"] = bone((50 + 5 * sign, 55), (50 + 6 * sign, 72))
        out[f"calf_{side}"] = bone((50 + 6 * sign, 72), (50 + 6 * sign, 90))
        out[f"foot_{side}"] = bone((50 + 6 * sign, 90), (50 + 6 * sign, 95))
    return out


def test_pose_points_read_a_ue_rig_by_name_with_openpose_sides():
    points = maps.pose_points(bones_ue())
    assert set(points) == set(maps.KEYPOINTS)
    assert points["neck"] == pytest.approx((50, 30))
    assert points["l_shoulder"] == pytest.approx((58, 32)) and points["r_shoulder"] == pytest.approx((42, 32))
    assert points["l_wrist"] == pytest.approx((82, 32)) and points["r_ankle"] == pytest.approx((44, 90))
    assert 10 < points["nose"][1] < 25 and points["r_eye"][0] < points["nose"][0] < points["l_eye"][0]


def test_pose_points_read_a_soma_rig_too():
    renames = {"neck_01": "Neck1", "head": "Head"}
    for ue, soma in (("upperarm", "Arm"), ("lowerarm", "ForeArm"), ("hand", "Hand"), ("thigh", "Leg"),
                     ("calf", "Shin"), ("foot", "Foot")):
        renames[f"{ue}_l"], renames[f"{ue}_r"] = f"Left{soma}", f"Right{soma}"
    soma = {renames[name]: bone for name, bone in bones_ue().items()}
    assert maps.pose_points(soma) == maps.pose_points(bones_ue())


def test_a_rig_without_named_joints_is_refused_with_what_is_missing():
    with pytest.raises(ValueError, match="upperarm_l"):
        maps.pose_points({"bone_0": {"head": [0, 0], "tail": [0, 1]}})


def test_a_rig_with_numbered_bones_can_be_read_with_its_roles():
    numbered = {f"bone_{i}": bone for i, bone in enumerate(bones_ue().values())}
    roles = {joint: f"bone_{list(bones_ue()).index(name)}" for joint, name in maps.NAMINGS[0].items()}
    assert maps.pose_points(numbered, naming=roles) == maps.pose_points(bones_ue())
    with pytest.raises(ValueError, match="l_wrist"):
        maps.pose_points(numbered, naming={k: v for k, v in roles.items() if k != "l_wrist"})


def test_the_pose_map_is_drawn_on_black_with_coloured_limbs(tmp_path):
    out = maps.draw_pose(maps.pose_points(bones_ue(scale=10.0)), 1024, tmp_path / "pose.png")
    pixels = np.asarray(Image.open(out).convert("RGB"))
    assert pixels.shape == (1024, 1024, 3) and pixels[5, 5].tolist() == [0, 0, 0]
    assert pixels[320, 700].max() > 100  # along the left forearm


def test_the_render_command_runs_blender_headless_on_our_script(tmp_path):
    command = maps.render_command("blender.exe", tmp_path / "wolf.glb", tmp_path / "out", ("front", "side"), 512)
    assert command[:4] == ["blender.exe", "-b", "--factory-startup", "-P"]
    assert command[4].endswith("blender_concept_views.py")
    assert command[5:] == ["--", str(tmp_path / "wolf.glb"), str(tmp_path / "out"), "front,side", "512"]
    with pytest.raises(ValueError, match="view"):
        maps.render_command("blender.exe", tmp_path / "w.glb", tmp_path, ("three_quarter",), 512)


def test_render_says_how_to_get_blender_when_there_is_none(tmp_path):
    with pytest.raises(RuntimeError, match="Blender"):
        maps.render(tmp_path / "w.glb", tmp_path / "out", blender=None, find=lambda: None)


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_blender_renders_colour_depth_and_bones_of_a_glb(tmp_path):
    blender = find_blender()
    glb = tmp_path / "rigged.glb"
    subprocess.run([str(blender), "-b", "--factory-startup", "--python-expr",
                    "import bpy\n"
                    "for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)\n"
                    "bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))\n"
                    "bpy.ops.object.armature_add(location=(0, 0, 0))\n"
                    "bpy.context.object.data.bones[0].name = 'head'\n"
                    f"bpy.ops.export_scene.gltf(filepath=r'{glb}')\n"],
                   check=True, capture_output=True)
    files = maps.render(glb, tmp_path / "out", views=("front", "side"), size=64, blender=blender)
    for view in ("front", "side"):
        colour = np.asarray(Image.open(files[view]["colour"]))
        depth = np.asarray(Image.open(files[view]["depth"]))
        assert colour.shape == (64, 64, 4) and colour[32, 32, 3] == 255 and colour[1, 1, 3] == 0
        assert depth[32, 32, :3].max() > 0
    bones = json.loads(files["bones"].read_text())
    assert "head" in bones["front"]


def test_lineart_of_a_wide_picture_keeps_its_proportions(tmp_path):
    """A round shield in an 800x400 picture stays round: the picture is fitted, not stretched."""
    import cv2

    wide = np.full((40, 80, 3), 230, np.uint8)
    cv2.circle(wide, (40, 20), 15, (40, 40, 40), -1)
    Image.fromarray(wide).save(tmp_path / "shield.png")
    lines = np.asarray(Image.open(maps.lineart(tmp_path / "shield.png", tmp_path / "lines.png", size=160)))
    assert lines.shape == (160, 160)
    rows, cols = np.nonzero(lines < 128)
    width, height = cols.max() - cols.min(), rows.max() - rows.min()
    assert abs(width / height - 1) < 0.15, (width, height)
    assert lines[5, 80] == 255 and lines[155, 80] == 255  # the padding is empty paper
