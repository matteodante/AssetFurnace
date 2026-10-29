"""Tests for `image_to_3dlab/twocam/train.py`: the training samples and targets the reader
learns from, on a tiny made-up dataset. The network itself is not trained here (it needs
Meta's pretrained weights); `scripts/twocam_mocap.py train` is run by hand."""

import json

import numpy as np
import pytest
import torch
from twocam_toy import toy_rig

from image_to_3dlab.twocam import train
from image_to_3dlab.twocam.layout import Layout
from image_to_3dlab.twocam.reader import HM, SIZE, decode


def _dataset(folder, info, count=60, scale=0.25):
    cv2 = pytest.importorskip("cv2")
    layout = Layout.from_rig_info(info)
    rest = np.array([k["rest"] for k in info["keypoints"]])
    front, side = layout.project(rest)
    for i in range(count):
        for view, w in (("front", layout.w_front), ("side", layout.w_side)):
            img = np.zeros((round(layout.h * scale), round(w * scale), 4), np.uint8)
            img[40:80, 30:60] = (200, 100, 50, 255)
            cv2.imwrite(str(folder / f"{i:05d}_{view}.png"), img)
        (folder / f"{i:05d}_meta.json").write_text(json.dumps(
            {"front_uv": front.tolist(), "side_uv": side.tolist(), "scale": scale}), encoding="utf-8")


def test_every_25th_sample_is_held_out(tmp_path):
    info = toy_rig()
    _dataset(tmp_path, info, count=60)
    tr, val = train.split(tmp_path)
    assert [m.name[:5] for m in val] == ["00000", "00025", "00050"]
    assert len(tr) == 57


def test_a_dataset_for_another_rig_is_refused(tmp_path):
    info = toy_rig()
    _dataset(tmp_path, info, count=60)
    other = toy_rig()
    other["keypoints"] = other["keypoints"][:-1]
    with pytest.raises(train.TrainError, match="another rig"):
        train.check_dataset(tmp_path, other)
    with pytest.raises(train.TrainError, match="render the dataset"):
        train.check_dataset(tmp_path / "empty", info)


def test_a_held_out_sample_keeps_its_joints_where_the_picture_put_them(tmp_path):
    info = toy_rig()
    _dataset(tmp_path, info, count=30)
    _, val = train.split(tmp_path)
    views = train.Views(val, Layout.from_rig_info(info), train=False)
    x, kp, geo = views[1]  # the side view of sample 0
    assert x.shape == (3, SIZE, SIZE) and kp.shape == (len(info["keypoints"]), 2)
    ox, oy, side, scale = geo.tolist()
    back = (kp.numpy() * side / SIZE - [ox, oy]) / scale
    meta = json.loads(val[0].read_text(encoding="utf-8"))
    np.testing.assert_allclose(back, meta["side_uv"], atol=1e-3)


def test_training_samples_are_jittered_but_stay_well_formed(tmp_path):
    info = toy_rig()
    _dataset(tmp_path, info, count=30)
    tr, _ = train.split(tmp_path)
    x, kp, _ = train.Views(tr, Layout.from_rig_info(info), train=True)[0]
    assert x.shape == (3, SIZE, SIZE) and float(x.min()) >= 0 and float(x.max()) <= 1
    assert torch.isfinite(kp).all()


def test_the_target_heatmap_peaks_where_decoding_finds_the_joint():
    kp = torch.tensor([[[100.0, 300.0], [250.5, 40.25]]])
    uv, _ = decode(train.heatmaps(kp))
    np.testing.assert_allclose((uv * SIZE / HM).numpy(), kp.numpy(), atol=0.6)
