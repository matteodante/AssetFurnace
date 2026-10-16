"""Turning a text-to-image command into an edit: the vision file and the reference pictures."""

import pytest
from PIL import Image

from image_to_3dlab.concept import edit


def picture(path, colour=(1, 2, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)
    return path


def test_the_vision_file_is_found_under_any_snapshot(tmp_path):
    hub = tmp_path / "hub"
    assert edit.find_vision(hub) is None
    target = hub / edit.VISION_REPO / "snapshots" / "abc" / edit.VISION_FILE
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert edit.find_vision(hub) == target


def test_references_go_right_after_the_binary_in_the_order_given(tmp_path):
    shape, style = picture(tmp_path / "shape.png"), picture(tmp_path / "style.png")
    base = ["sd-cli", "--diffusion-model", "m.gguf", "-p", "a prompt", "-o", "out.png"]
    command = edit.with_references(base, [shape, style], tmp_path / "vision.gguf")
    assert command[:6] == ["sd-cli", "--llm_vision", str(tmp_path / "vision.gguf"),
                           "-r", str(shape.resolve()), "-r"]
    assert command[6] == str(style.resolve())
    assert command[7:] == base[1:]
    assert base == ["sd-cli", "--diffusion-model", "m.gguf", "-p", "a prompt", "-o", "out.png"]


def test_an_edit_needs_at_least_one_picture_and_not_too_many(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        edit.with_references(["sd-cli"], [], tmp_path / "v.gguf")
    many = [picture(tmp_path / f"{i}.png") for i in range(edit.MAX_REFERENCES + 1)]
    with pytest.raises(ValueError, match=str(edit.MAX_REFERENCES)):
        edit.with_references(["sd-cli"], many, tmp_path / "v.gguf")


def test_a_missing_or_unreadable_picture_is_refused_before_the_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        edit.with_references(["sd-cli"], [tmp_path / "gone.png"], tmp_path / "v.gguf")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a picture")
    with pytest.raises(ValueError, match="broken.png"):
        edit.with_references(["sd-cli"], [broken], tmp_path / "v.gguf")
