"""Tests for `image_to_3dlab/twocam/reader.py`: heatmap peaks, view geometry, and that
reading never downloads weights or accepts another rig's reader."""

import numpy as np
import pytest
import torch

from image_to_3dlab.twocam import reader
from image_to_3dlab.twocam.layout import Layout


def test_a_heatmap_peak_is_found_to_a_fraction_of_a_pixel():
    yy, xx = np.mgrid[0:reader.HM, 0:reader.HM]
    true = [(40.3, 70.6), (5.8, 100.2)]
    hm = np.stack([np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2)) for x, y in true])
    uv, conf = reader.decode(torch.tensor(hm[None], dtype=torch.float32))
    np.testing.assert_allclose(uv[0].numpy(), true, atol=0.15)
    assert (conf > 0.9).all()


def test_peaks_map_back_to_the_view_they_were_read_from():
    geometry = reader.square_geometry(720, 640)  # a 640-wide, 720-tall view
    assert geometry == (40, 0, 720)
    centre = np.array([reader.HM / 2, reader.HM / 2])
    np.testing.assert_allclose(reader.heatmap_to_view(centre, geometry), [320, 360])


def test_the_backbone_is_built_from_its_pinned_shape_without_downloading(monkeypatch):
    tf = pytest.importorskip("transformers")

    def no_download(*_args, **_kwargs):
        raise AssertionError("reading must not download weights")

    monkeypatch.setattr(tf.AutoModel, "from_pretrained", no_download)
    backbone = reader.build_backbone(pretrained=False)
    assert backbone.config.patch_size == 14
    # 37 x 37 patches at the native 518 px, plus the class token
    assert backbone.embeddings.position_embeddings.shape[1] == 37 * 37 + 1
    assert backbone.config.num_register_tokens == 4


def test_a_file_that_is_not_a_reader_is_refused(tmp_path):
    path = tmp_path / "x.pt"
    torch.save({"weights": torch.zeros(2)}, path)
    with pytest.raises(reader.ReaderError, match="not a joint reader"):
        reader.load_reader(path, "cpu")


def test_a_reader_trained_on_another_rig_is_refused(monkeypatch, tmp_path):
    frames = np.zeros((3, 720, 1280, 3), np.uint8)
    monkeypatch.setattr(reader, "read_frames", lambda _path: (frames, 24.0))
    monkeypatch.setattr(reader, "load_reader", lambda _path, _device: (None, ["a", "b"]))
    with pytest.raises(reader.ReaderError, match="another rig"):
        reader.read_video(tmp_path / "v.mp4", tmp_path / "r.pt", None, Layout(), ["b", "a"])


def test_a_clip_with_no_frame_rate_asks_for_one(monkeypatch, tmp_path):
    monkeypatch.setattr(reader, "read_frames", lambda _path: (np.zeros((3, 8, 8, 3), np.uint8), 0.0))
    with pytest.raises(reader.ReaderError, match="--fps"):
        reader.read_video(tmp_path / "v.mp4", tmp_path / "r.pt", None, Layout(), ["a"])
