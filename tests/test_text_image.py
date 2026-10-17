"""Text to image outside the studio: `image_to_3dlab.text_image.generate` and
`scripts/text_image.py`, with a stand-in sd-cli (a Python script that writes a PNG)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from batch_fakes import tiny_png

from image_to_3dlab import text_image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import text_image as cli  # noqa: E402

WEIGHTS = {"diffusion_model": Path("/w/dit.gguf"), "llm": Path("/w/llm.gguf"), "vae": Path("/w/vae.safetensors")}


@pytest.fixture
def fake_sdcli(tmp_path, monkeypatch):
    """sd-cli that prints progress, records what it was asked and writes a 1x1 PNG; the
    GPU lock in a test folder; the weights found."""
    script = tmp_path / "fake_sd_cli.py"
    script.write_text(
        "import sys, json, pathlib, struct, zlib\n"
        "args = sys.argv[1:]\n"
        "out = pathlib.Path(args[args.index('-o') + 1])\n"
        f"lock = pathlib.Path({str(tmp_path / 'gpu.lock')!r})\n"
        "print('|=====>    | 4/8 - 0.5s/it', flush=True)\n"
        "pathlib.Path(str(out) + '.args.json').write_text(json.dumps({'args': args, 'locked': lock.exists()}))\n"
        "if 'fail' in args[args.index('-p') + 1]:\n"
        "    print('out of memory', flush=True); sys.exit(3)\n"
        "out.write_bytes(pathlib.Path(sys.argv[0]).with_name('one.png').read_bytes())\n")
    tiny_png(tmp_path / "one.png")
    monkeypatch.setattr(text_image, "GPU_LOCK", tmp_path / "gpu.lock")
    monkeypatch.setattr(text_image, "resolve_weights", lambda: WEIGHTS)
    real = text_image.build_command

    def command(prompt, settings, output_path, weights, binary=None):
        built = real(prompt, settings, output_path, weights, binary=Path("sd-cli"))
        return [sys.executable, str(script), *built[1:]]

    monkeypatch.setattr(text_image, "build_command", command)
    return tmp_path


def _asked(picture: Path) -> dict:
    return json.loads(Path(str(picture) + ".args.json").read_text())


def test_generate_writes_the_picture_and_its_licence_under_the_gpu_lock(fake_sdcli):
    picture = text_image.generate("a wooden barrel", {"seed": 5, "width": 700}, fake_sdcli / "out" / "barrel.png")
    asked = _asked(picture)
    assert asked["locked"] is True and not (fake_sdcli / "gpu.lock").exists()   # held, then let go
    args = asked["args"]
    assert args[args.index("--seed") + 1] == "5" and args[args.index("-W") + 1] == "704"
    record = json.loads(picture.with_suffix(".provenance.json").read_text())
    assert record["prompt"] == "a wooden barrel" and record["settings"]["seed"] == 5
    assert record["license"]["name"] == text_image.LICENSE_NAME


def test_a_failed_picture_says_why_and_lets_the_gpu_go(fake_sdcli):
    with pytest.raises(RuntimeError, match="code 3: out of memory"):
        text_image.generate("fail please", {}, fake_sdcli / "out" / "x.png")
    assert not (fake_sdcli / "gpu.lock").exists()


def test_an_empty_prompt_is_refused_before_the_gpu(fake_sdcli):
    with pytest.raises(ValueError, match="empty"):
        text_image.generate("  ", {}, fake_sdcli / "x.png")


def test_the_command_line_makes_several_to_pick_from(fake_sdcli, capsys):
    out = fake_sdcli / "pictures"
    assert cli.main(["a wolf, side view", "--seed", "7", "--count", "3", "--out", str(out), "--json"]) == 0
    pictures = [Path(p) for p in json.loads(capsys.readouterr().out)["pictures"]]
    assert [p.name.rsplit("_s", 1)[1] for p in pictures] == ["7.png", "8.png", "9.png"]
    assert all(p.is_file() and p.with_suffix(".provenance.json").is_file() for p in pictures)
    assert [_asked(p)["args"][_asked(p)["args"].index("--seed") + 1] for p in pictures] == ["7", "8", "9"]


def test_the_command_line_reports_a_failure(fake_sdcli, capsys):
    assert cli.main(["fail now", "--out", str(fake_sdcli / "x")]) == 1
    assert "out of memory" in capsys.readouterr().err
    assert cli.main(["a crate", "--count", "20"]) == 1


def test_missing_weights_are_named_not_downloaded(monkeypatch, tmp_path, capsys):
    def missing():
        raise text_image.MissingWeights(["qwen-image-2.1-turbo-Q8_0.gguf"])

    monkeypatch.setattr(text_image, "resolve_weights", missing)
    assert cli.main(["a crate", "--out", str(tmp_path)]) == 1
    assert "qwen-image-2.1-turbo-Q8_0.gguf" in capsys.readouterr().err


def test_the_studio_shares_the_engines_functions():
    # the same settings, command runner and licence record; that the studio's job also
    # takes the GPU lock is tested on its own run (test_image_api_extras.py)
    sys.path.insert(0, str(REPO / "viewer"))
    import image_api

    assert image_api.clean_settings is text_image.clean_settings
    assert image_api.run_sdcli is text_image.run_sdcli and image_api.provenance is text_image.provenance
