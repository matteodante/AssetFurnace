"""Pixal3D runs the official recipe (TencentARC/Pixal3D inference.py, master, read 2026-10-17):
12 sampling steps in every flow, and the camera's field of view estimated from the picture
with MoGe-2 (Ruicheng/moge-2-vitl). Before, 8 steps on a patched build and a fixed 20 degrees.
The user's decision: where the official recipe says otherwise, follow it."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import pixal3d_generate as pg  # noqa: E402

from image_to_3dlab import moge_fov  # noqa: E402


def test_the_steps_are_the_official_twelve_unless_asked(tmp_path):
    steps, note = pg.resolve_steps(None, tmp_path / "flow_runner.cpp", tmp_path / "cli")
    assert steps == 12 and "official" in note
    assert pg.run_env(steps, {})  == {}   # nothing overrides trellis-cli's own 12


def test_the_field_of_view_is_the_official_formula():
    # inference.py: camera_angle_x = 2 * atan(width / (2 * fx)), fx = intrinsics[0, 0] * width
    assert moge_fov.camera_angle_x(fx_normalized=0.5) == pytest.approx(2 * math.atan(1.0))
    assert moge_fov.camera_angle_x(fx_normalized=1 / (2 * math.tan(math.radians(10)))) == \
        pytest.approx(math.radians(20))


def test_the_estimate_runs_the_worker_and_holds_the_gpu(tmp_path, monkeypatch):
    seen = {}

    def run(command, **kwargs):
        seen["command"], seen["timeout"] = command, kwargs.get("timeout")
        seen["crop"] = Image.open(command[2]).size   # the cropped picture, while it exists
        return subprocess.CompletedProcess(command, 0, json.dumps({"fx_normalized": 0.5}), "")

    import contextlib

    @contextlib.contextmanager
    def lock(session, **kw):
        seen["lock"] = session
        yield

    monkeypatch.setattr(moge_fov, "installed", lambda: True)
    monkeypatch.setattr(moge_fov, "on_gpu", lambda: True)
    picture = tmp_path / "a.png"
    Image.new("RGBA", (400, 200), (0, 0, 0, 0)).save(picture)
    fov = moge_fov.estimate(picture, run=run, lock=lock)
    assert fov == pytest.approx(math.pi / 2) and seen["lock"] == "moge-2 fov"
    assert seen["command"][1].endswith("moge_worker.py") and seen["timeout"] == moge_fov.TIMEOUT
    assert not Path(seen["command"][2]).exists()   # the crop is cleaned up


def test_the_estimate_is_on_the_recipes_crop(tmp_path):
    """As preprocess_image: the alpha's box, a square 1.1 times its size, on black."""
    picture = tmp_path / "sword.png"
    rgba = np.zeros((1000, 800, 4), np.uint8)
    rgba[100:900, 380:420] = (200, 200, 200, 255)          # a tall thin subject
    Image.fromarray(rgba).save(picture)
    crop = moge_fov.preprocessed(picture, tmp_path / "crop.png")
    with Image.open(crop) as opened:
        assert opened.mode == "RGB" and opened.size[0] == opened.size[1] == pytest.approx(int(799 * 1.1), abs=2)
        assert opened.getpixel((0, 0)) == (0, 0, 0)          # the background is black


def test_an_answer_trellis_cli_could_not_take_is_not_used(tmp_path, monkeypatch):
    monkeypatch.setattr(moge_fov, "installed", lambda: True)
    monkeypatch.setattr(moge_fov, "on_gpu", lambda: False)
    picture = tmp_path / "a.png"
    Image.new("RGBA", (64, 64), (255, 255, 255, 255)).save(picture)
    wide = lambda command, **kw: subprocess.CompletedProcess(command, 0, json.dumps({"fx_normalized": 0.01}), "")
    assert moge_fov.estimate(picture, run=wide) is None

    def hung(command, **kw):
        raise subprocess.TimeoutExpired(command, kw["timeout"])

    assert moge_fov.estimate(picture, run=hung) is None


def test_without_moge_the_old_gauge_is_used_and_said(monkeypatch, capsys):
    monkeypatch.setattr(moge_fov, "installed", lambda: False)
    fov, source = pg.choose_fov(None, Path("a.png"))
    assert fov == pg.DEFAULT_FOV and source == "gauge 20 degrees (MoGe-2 not installed)"
    assert "bootstrap_pixal3d.py --moge" in capsys.readouterr().out
    assert pg.choose_fov(0.25, Path("a.png")) == (0.25, "asked")


def test_with_moge_the_estimate_is_the_cameras(monkeypatch):
    monkeypatch.setattr(moge_fov, "installed", lambda: True)
    monkeypatch.setattr(moge_fov, "estimate", lambda image, **kw: 0.31)
    assert pg.choose_fov(None, Path("a.png")) == (0.31, "moge-2")


def test_the_command_carries_the_official_values(monkeypatch, tmp_path):
    command = pg.build_command(tmp_path / "a.png", tmp_path / "a.glb", 1024, 42, 0.31, tmp_path, tmp_path / "cli",
                               True, pg.DEFAULT_GSS, None)
    assert command[command.index("--fov") + 1] == "0.31"
    assert pg.FAST_STEPS != pg.DEFAULT_STEPS == 12


def test_moge_is_pinned_and_set_up_with_pixal3d():
    assert moge_fov.REPO_ID == "Ruicheng/moge-2-vitl" and len(moge_fov.REVISION) == 40
    assert moge_fov.CODE_COMMIT.startswith("07444410f1e3")   # MoGe 2.0.0, before MoGe-3
    import bootstrap_pixal3d

    flags = {f for a in bootstrap_pixal3d.build_parser()._actions for f in a.option_strings}
    assert "--moge" in flags
    sys.path.insert(0, str(REPO / "viewer"))
    import backend_catalog

    moge = backend_catalog.BY_ID["moge"]   # optional: a card of its own, Pixal3D is ready without it
    assert moge.kind == "tool" and moge.install == "scripts/bootstrap_pixal3d.py --moge"
    assert not any("MoGe" in w.label for w in backend_catalog.BY_ID["pixal3d"].weights)


def test_moge_gets_the_pytorch_for_the_machine():
    import bootstrap_pixal3d

    from image_to_3dlab import host

    assert bootstrap_pixal3d.torch_index(host.NVIDIA).endswith("/cu128")
    assert bootstrap_pixal3d.torch_index(host.APPLE) is None
    assert bootstrap_pixal3d.torch_index("other").endswith("/cpu")
    install = bootstrap_pixal3d.moge_commands(platform=host.NVIDIA)[2]
    assert install[install.index("--index-url") + 1].endswith("/cu128")
    swap = bootstrap_pixal3d.moge_commands(platform=host.NVIDIA, wrong_torch=True)
    assert swap[2][-4:] == ["uninstall", "-y", "torch", "torchvision"]   # a CPU build pip would keep
