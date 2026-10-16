"""The `<image>.concept.json` record: which pictures shaped a concept, and how."""

import hashlib
import json

import pytest
from PIL import Image

from image_to_3dlab.concept import sidecar


def picture(path, colour=(9, 8, 7)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)
    return path


def test_the_record_keeps_each_reference_by_name_hash_and_role(tmp_path):
    image = picture(tmp_path / "out" / "wolf.png")
    shape = picture(tmp_path / "refs" / "depth.png", (1, 1, 1))
    style = picture(tmp_path / "refs" / "painting.png", (2, 2, 2))
    command = ["sd-cli", "--llm_vision", "C:/hub/mmproj.gguf", "--diffusion-model", "C:/hub/turbo-Q8_0.gguf",
               "--llm", "C:/hub/enc.gguf", "--vae", "C:/hub/vae.safetensors", "-p", "x"]
    path = sidecar.write(image, mode="redraw", prompt="the prompt", seed=11, command=command,
                         references=[(shape, "shape"), (style, "style")],
                         inputs={"structure": "depth", "source": "wolf.glb"}, iou=0.93)
    assert path == tmp_path / "out" / "wolf.png.concept.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mode"] == "redraw" and data["prompt"] == "the prompt" and data["seed"] == 11
    assert data["references"] == [
        {"name": "depth.png", "role": "shape", "sha256": hashlib.sha256(shape.read_bytes()).hexdigest()},
        {"name": "painting.png", "role": "style", "sha256": hashlib.sha256(style.read_bytes()).hexdigest()},
    ]
    assert data["inputs"] == {"structure": "depth", "source": "wolf.glb"}
    assert data["models"] == {"diffusion_model": "turbo-Q8_0.gguf", "llm": "enc.gguf",
                              "vae": "vae.safetensors", "llm_vision": "mmproj.gguf"}
    assert data["iou"] == 0.93
    assert data["image"] == {"name": "wolf.png", "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}


def test_no_folder_of_the_machine_goes_in_the_record(tmp_path):
    image = picture(tmp_path / "out" / "a.png")
    ref = picture(tmp_path / "private" / "ref.png")
    path = sidecar.write(image, mode="add_piece", prompt="p", seed=1, command=["sd-cli"],
                         references=[(ref, "shape")])
    text = path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
    assert json.loads(text)["iou"] is None


def test_an_unknown_role_or_mode_is_refused(tmp_path):
    image = picture(tmp_path / "a.png")
    with pytest.raises(ValueError, match="role"):
        sidecar.write(image, mode="redraw", prompt="p", seed=1, command=[], references=[(image, "colour")])
    with pytest.raises(ValueError, match="mode"):
        sidecar.write(image, mode="inpaint", prompt="p", seed=1, command=[], references=[(image, "shape")])


def test_reading_back_a_record_and_a_missing_one(tmp_path):
    image = picture(tmp_path / "a.png")
    assert sidecar.read(image) is None
    sidecar.write(image, mode="dress", prompt="p", seed=3, command=[], references=[(image, "shape")])
    assert sidecar.read(image)["seed"] == 3
