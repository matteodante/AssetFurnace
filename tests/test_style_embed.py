"""Embedding plumbing: preprocessing, the cache, and the CSD checkpoint layout. No real weights."""

import argparse

import numpy as np
import pytest
import torch
from PIL import Image

from image_to_3dlab.style import embed
from image_to_3dlab.style.clip_vit import CSDStyle, csd_state_dict


class Counting(embed.Embedder):
    key = "fake"

    def __init__(self):
        self.calls = 0

    def embed(self, images):
        self.calls += len(images)
        rows = np.array([np.asarray(i, dtype=float).mean(axis=(0, 1)) + 1 for i in images])
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def picture(path, colour, mode="RGB"):
    Image.new(mode, (32, 24), colour).save(path)
    return path


def test_vectors_are_cached_by_hash_and_only_new_pictures_are_embedded(tmp_path):
    a = picture(tmp_path / "a.png", (200, 0, 0))
    b = picture(tmp_path / "b.png", (0, 200, 0))
    cache = tmp_path / ".cache" / "embeddings-fake.npz"
    fake = Counting()
    first = embed.embed_paths(fake, [a, b], cache=cache)
    assert first.shape == (2, 3) and fake.calls == 2 and cache.is_file()
    again = embed.embed_paths(fake, [b, a, picture(tmp_path / "c.png", (0, 0, 200))], cache=cache)
    assert fake.calls == 3  # only c
    np.testing.assert_allclose(again[0], first[1])


def test_a_cut_out_picture_is_put_back_on_the_concept_grey(tmp_path):
    clear = picture(tmp_path / "clear.png", (255, 0, 0, 0), mode="RGBA")
    assert embed.open_rgb(clear).getpixel((0, 0)) == embed.BACKGROUND


def test_clip_preprocessing_crops_the_centre_square_and_normalises():
    tensor = embed.clip_tensor(Image.new("RGB", (448, 300), (255, 255, 255)))
    assert tuple(tensor.shape) == (3, 224, 224)
    expected = (1 - embed.CLIP_MEAN[0]) / embed.CLIP_STD[0]
    assert float(tensor[0, 100, 100]) == pytest.approx(expected, rel=1e-4)


def test_the_csd_checkpoint_keys_load_into_the_style_model():
    small = {"resolution": 28, "patch": 14, "width": 64, "layers": 2, "heads": 4}
    source = CSDStyle(**small)
    torch.nn.init.normal_(source.last_layer_style)
    state = {f"module.{k}": v for k, v in source.state_dict().items()}
    state["module.last_layer_content"] = torch.zeros(64, 768)  # in the checkpoint, not used
    checkpoint = {"model_state_dict": state, "args": argparse.Namespace(arch="vit_large"), "iter": 1}
    target = CSDStyle(**small)
    target.load_state_dict(csd_state_dict(checkpoint), strict=True)
    out = target(torch.randn(2, 3, 28, 28))
    assert tuple(out.shape) == (2, 768)
    np.testing.assert_allclose(out.norm(dim=1).detach().numpy(), 1.0, rtol=1e-5)


def test_an_unknown_embedding_is_refused():
    with pytest.raises(ValueError, match="csd"):
        embed.load_embedder("dinov3")
