"""Tests for `image_to_3dlab/twocam/video.py` (filming with MiniMax H3 through stable-diffusion.cpp)
and `image_to_3dlab/twocam/prompt.py` (the words the video model gets). No GPU, no weights."""

from pathlib import Path

import pytest

from image_to_3dlab.twocam import prompt, video

CREATURE = {"noun": "grey wolf", "describe": "pale grey fur", "count": "Exactly four legs.",
            "colour_code": "front-left leg pink, front-right leg blue, hind-left leg yellow, hind-right leg cyan"}


@pytest.mark.parametrize("seconds, frames", [(0.1, 5), (2.0, 56), (3.0, 73), (5.0, 124)])
def test_frames_follow_the_models_17k_plus_5_grid(seconds, frames):
    n = video.frames_for(seconds)
    assert n == frames and (n - 5) % 17 == 0 and n >= min(round(seconds * 24), 5)


def test_sizes_round_to_multiples_of_32():
    assert video.size_for(1280, 720) == (1280, 704)
    assert video.size_for(1024, 576) == (1024, 576)


def test_the_film_command_starts_from_the_reference_frame_without_sound():
    weights = {"diffusion_model": Path("dit.gguf"), "llm": Path("te.gguf"), "vae": Path("vae.safetensors")}
    cmd = video.film_command(Path("sd-cli"), weights, "a prompt", Path("ref.png"), Path("take.webm"),
                             seconds=3, seed=7)
    assert cmd[:3] == ["sd-cli", "-M", "vid_gen"]
    assert cmd[cmd.index("-i") + 1] == "ref.png"
    assert cmd[cmd.index("--video-frames") + 1] == "73"
    assert cmd[cmd.index("-s") + 1] == "7"
    assert "--audio-vae" not in cmd and "--offload-to-cpu" in cmd
    assert cmd[cmd.index("--sampling-method") + 1] == "res_multistep"


def test_the_film_command_refuses_a_container_sd_cli_cannot_write():
    with pytest.raises(video.VideoError, match=".webm"):
        video.film_command(Path("sd-cli"), {"diffusion_model": "a", "llm": "b", "vae": "c"}, "p",
                           Path("ref.png"), Path("take.mp4"))


def test_cached_weights_are_found_through_the_snapshot(tmp_path):
    w = video.WEIGHTS[2]  # one with a folder in its file name
    folder = tmp_path / ("models--" + w.repo.replace("/", "--")) / "snapshots" / "abc123" / Path(w.filename).parent
    folder.mkdir(parents=True)
    (folder / Path(w.filename).name).write_bytes(b"vae")
    assert video.find_weight(w, tmp_path).read_bytes() == b"vae"
    assert video.find_weight(video.WEIGHTS[0], tmp_path) is None


def test_the_plan_says_what_is_missing_how_big_and_under_which_licence(tmp_path):
    plan = video.weights_plan(tmp_path)
    assert not plan["ready"]
    assert plan["download_gb"] == pytest.approx(sum(w.gigabytes for w in video.WEIGHTS), abs=0.1)
    assert "Community License" in plan["license"] and "EU" in plan["license_note"]


def test_filming_without_the_weights_never_starts_a_download(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "hub_root", lambda env=None: tmp_path)

    def no_download(*_a, **_k):
        raise AssertionError("film must not download")

    monkeypatch.setattr(video, "download", no_download)
    with pytest.raises(video.VideoError, match="not downloaded"):
        video.film(Path("sd-cli"), "p", Path("ref.png"), tmp_path / "t.webm")


def test_the_prompt_fixes_everything_but_the_action():
    a = prompt.make_prompt(CREATURE, "it rears up and howls", end="standing on all four legs")
    b = prompt.make_prompt(CREATURE, "it pounces forward")
    assert "front-left leg pink" in a and "Exactly four legs." in a
    assert "Action: it rears up and howls. In both views it ends standing on all four legs." in a
    assert "In both views it ends" not in b
    assert a.split("Action:")[0] == b.split("Action:")[0]
    assert "no motion blur, dust, smoke, sparks" in a


def test_a_prompt_needs_an_action_and_a_prepared_creature():
    with pytest.raises(prompt.PromptError):
        prompt.make_prompt(CREATURE, "  ")
    with pytest.raises(prompt.PromptError, match="prepare"):
        prompt.make_prompt({"noun": "wolf"}, "it howls")
