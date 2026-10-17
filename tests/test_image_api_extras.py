"""The hook other steps use to run an image job with more arguments: references, a name, a record."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

from test_image_api import WEIGHTS, api


def fake_sd_cli(tmp_path: Path) -> list[str]:
    """A stand-in for sd-cli that keeps its arguments and writes a small PNG where -o says."""
    script = tmp_path / "fake_sd_cli.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        f"Path({str(tmp_path / 'argv.json')!r}).write_text(json.dumps(args))\n"
        "from PIL import Image\n"
        "Image.new('RGB', (8, 8), (1, 2, 3)).save(args[args.index('-o') + 1])\n"
        "print('generate_image completed in 1.00s', flush=True)\n"
    )
    # One executable, as sd-cli is, so arguments put after the binary reach the script.
    if os.name == "nt":
        binary = tmp_path / "sd-cli.cmd"
        binary.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        binary = tmp_path / "sd-cli"
        binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return [str(binary)]


def test_extra_arguments_go_right_after_the_binary(tmp_path, monkeypatch):
    fake = fake_sd_cli(tmp_path)
    monkeypatch.setattr(api, "build_command", lambda prompt, settings, out, weights: [*fake, "-p", prompt, "-o", str(out)])
    manager = api.ImageJobManager(output_root=tmp_path / "images")
    job = manager.create("a fox", api.clean_settings({}), extra_args=["--llm_vision", "v.gguf", "-r", "ref.png"])
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status == "done", job.error
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[:4] == ["--llm_vision", "v.gguf", "-r", "ref.png"]
    assert argv[4:6] == ["-p", "a fox"]


def test_a_plain_job_runs_the_command_unchanged(tmp_path, monkeypatch):
    fake = fake_sd_cli(tmp_path)
    monkeypatch.setattr(api, "build_command", lambda prompt, settings, out, weights: [*fake, "-p", prompt, "-o", str(out)])
    manager = api.ImageJobManager(output_root=tmp_path / "images")
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    assert json.loads((tmp_path / "argv.json").read_text())[:2] == ["-p", "a fox"]


def test_extra_provenance_is_added_to_the_sidecar_without_replacing_the_licence(tmp_path, monkeypatch):
    fake = fake_sd_cli(tmp_path)
    monkeypatch.setattr(api, "build_command", lambda prompt, settings, out, weights: [*fake, "-o", str(out)])
    manager = api.ImageJobManager(output_root=tmp_path / "images")
    job = manager.create("a fox", api.clean_settings({}),
                         extra_provenance={"concept": {"mode": "redraw"}, "license": "overwritten?"})
    api.run_job(job, manager, weights=WEIGHTS)
    sidecar = json.loads(job.output_path.with_suffix(".provenance.json").read_text())
    assert sidecar["concept"] == {"mode": "redraw"}
    assert sidecar["license"]["name"] == api.LICENSE_NAME


def test_a_job_can_be_named_instead_of_slugging_its_prompt(tmp_path):
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("Image 1 is a depth map of a wolf", api.clean_settings({}), name="Wolf redrawn!")
    assert job.directory.name.startswith("wolf-redrawn__")
    assert job.output_path.name.startswith("wolf-redrawn__")


# ---- the studio's pictures take the shared GPU lock, like every other GPU job ---------

def lock_checking_sd_cli(tmp_path: Path, lock: Path) -> list[str]:
    """A stand-in sd-cli that only succeeds while the GPU lock names this server."""
    script = tmp_path / "locked_sd_cli.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"lock = Path({str(lock)!r})\n"
        "args = sys.argv[1:]\n"
        "if not lock.exists() or 'studio-image' not in lock.read_text():\n"
        "    print('no GPU lock held', flush=True); sys.exit(9)\n"
        "from PIL import Image\n"
        "Image.new('RGB', (8, 8)).save(args[args.index('-o') + 1])\n")
    return [sys.executable, str(script)]


def test_a_studio_picture_holds_the_gpu_lock_while_sd_cli_runs(tmp_path, monkeypatch):
    from image_to_3dlab import text_image

    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(text_image, "GPU_LOCK", lock)
    fake = lock_checking_sd_cli(tmp_path, lock)
    monkeypatch.setattr(api, "build_command", lambda prompt, settings, out, weights: [*fake, "-o", str(out)])
    manager = api.ImageJobManager(output_root=tmp_path / "images")
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status == "done", job.error
    assert not lock.exists()


def test_a_studio_picture_waits_for_a_busy_gpu_and_can_be_cancelled_meanwhile(tmp_path, monkeypatch):
    import threading
    import time

    from image_to_3dlab import text_image

    lock = tmp_path / "gpu.lock"
    lock.write_text(f"{os.getpid()} pixal3d", encoding="utf-8")   # a live holder: this process
    monkeypatch.setattr(text_image, "GPU_LOCK", lock)
    monkeypatch.setattr(api, "GPU_POLL", 0.05)
    fake = lock_checking_sd_cli(tmp_path, lock)
    monkeypatch.setattr(api, "build_command", lambda prompt, settings, out, weights: [*fake, "-o", str(out)])
    manager = api.ImageJobManager(output_root=tmp_path / "images")
    job = manager.create("a fox", api.clean_settings({}))
    worker = threading.Thread(target=api.run_job, args=(job, manager), kwargs={"weights": WEIGHTS})
    worker.start()
    deadline = time.monotonic() + 10
    while not any("Waiting for the GPU" in str(e.get("message", "")) for e in job.events):
        assert time.monotonic() < deadline, job.events
        time.sleep(0.02)
    assert job.status == "queued" and "pixal3d" in job.events[-1]["message"]
    assert manager.cancel(job.id)
    worker.join(10)
    assert job.status == "cancelled"
    assert lock.read_text(encoding="utf-8") == f"{os.getpid()} pixal3d"   # theirs, untouched
    assert not job.output_path.exists()
