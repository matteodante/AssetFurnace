"""Qwen-Image 2.1 as its model card and README say to run it (user decision, 2026-10-17).

- the resolution presets ("Use the same resolution presets as Qwen-Image-2.1"), 2048x2048
  by default for text to image;
- the transparent-background prompt ("This is an RGBA image with transparency. ...");
- the official prompt rewriting: QwenLM's t2i system prompt and sampling, run locally on
  the Qwen3-VL-8B already installed as the text encoder, with llama.cpp. Missing, the run
  says so and carries on with the prompt as written.
Runs that draw from reference pictures keep 1024 and their prompt: 2048 with references
does not fit a 16 GB card (measured), and the rewriter is for text to image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from image_to_3dlab import prompt_rewrite as pr
from image_to_3dlab import text_image

REPO = Path(__file__).resolve().parents[1]


# ---- the card's sizes ---------------------------------------------------------------

def test_the_presets_are_the_cards():
    assert text_image.CARD_SIZES == {
        "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
        "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)}


@pytest.mark.parametrize("size", sorted(set(text_image.CARD_SIZES.values())))
def test_every_preset_survives_the_settings_check(size):
    settings = text_image.clean_settings({"width": size[0], "height": size[1]})
    assert (settings["width"], settings["height"]) == size


def test_text_to_image_defaults_are_the_cards_and_the_shared_engine_keeps_1024():
    assert (text_image.T2I_DEFAULTS["width"], text_image.T2I_DEFAULTS["height"]) == (2048, 2048)
    assert text_image.T2I_DEFAULTS["transparent"] is True and text_image.T2I_DEFAULTS["rewrite"] is True
    # reference runs (concepts, pieces) and the batches start from the engine's defaults
    assert (text_image.DEFAULTS["width"], text_image.DEFAULTS["height"]) == (text_image.REFERENCE_SIZE,) * 2 == (1024, 1024)
    assert text_image.DEFAULTS["transparent"] is False and text_image.DEFAULTS["rewrite"] is False


@pytest.mark.parametrize("given, expected", [(True, True), ("true", True), ("1", True), (False, False),
                                             ("false", False), ("0", False), (None, False), ("yes please", False)])
def test_switches_read_only_a_clear_yes(given, expected):
    assert text_image.clean_settings({"transparent": given, "rewrite": given})["transparent"] is expected


# ---- the transparent-background prompt ----------------------------------------------

def test_the_rgba_prompt_is_the_official_template():
    assert text_image.rgba_prompt("a wolf.") == (
        "This is an RGBA image with transparency. a wolf. The image has alpha channel and the "
        "background is transparent.")
    assert text_image.rgba_prompt(text_image.rgba_prompt("a wolf")) == text_image.rgba_prompt("a wolf")


# ---- the rewriter -------------------------------------------------------------------

ANSWER = ('Thinking it over.\n{"rewritten_prompt": "A red and gold helmet, alone, centred.", '
          '"wh_ratio": "1:1"} [end of text]\n')


def test_the_answer_is_read_from_the_last_json_object():
    assert pr.parse_answer(ANSWER) == {"rewritten_prompt": "A red and gold helmet, alone, centred.",
                                       "wh_ratio": "1:1"}
    assert pr.parse_answer('{"a": 1} then {"rewritten_prompt": "x {y}", "wh_ratio": ""}')["rewritten_prompt"] == "x {y}"
    assert pr.parse_answer("no json here") is None
    assert pr.parse_answer('{"wh_ratio": "1:1"}') is None


def _installed(tmp_path, monkeypatch):
    system = tmp_path / "system_prompt_t2i.txt"
    system.write_text("# Image Prompt Rewriting Expert\n", encoding="utf-8")
    binary = tmp_path / "llama-completion.exe"
    binary.write_bytes(b"")
    monkeypatch.setattr(pr, "system_prompt_path", lambda: system)
    monkeypatch.setattr(pr, "llama_completion", lambda: binary)
    return system, binary


def test_the_rewriter_runs_the_official_sampling_on_the_local_model_holding_the_gpu(tmp_path, monkeypatch):
    system, binary = _installed(tmp_path, monkeypatch)
    calls, locks = [], []

    class Done:
        returncode, stdout, stderr = 0, ANSWER, ""

    def run(command, **kwargs):
        calls.append(command)
        return Done()

    def lock(session, path=None):
        import contextlib
        locks.append(session)
        return contextlib.nullcontext()

    found = pr.rewrite("a helmet", model=Path("qwen3vl.gguf"), run=run, lock=lock)
    assert found["prompt"] == "A red and gold helmet, alone, centred." and found["wh_ratio"] == "1:1"
    command = calls[0]
    assert command[0] == str(binary) and command[command.index("-m") + 1] == "qwen3vl.gguf"
    assert command[command.index("-sysf") + 1] == str(system) and command[command.index("-p") + 1] == "a helmet"
    for flag, value in (("--temp", "1.0"), ("--top-p", "0.95"), ("--top-k", "20"), ("--min-p", "0.0"),
                        ("--presence-penalty", "1.5")):
        assert command[command.index(flag) + 1] == value, flag
    assert locks == ["prompt-rewrite"]
    assert found["system_prompt"]["sha256"] == pr.SYSTEM_PROMPT_SHA256


@pytest.mark.parametrize("missing", ["system", "binary"])
def test_without_the_rewriter_the_run_says_so_and_carries_on(tmp_path, monkeypatch, missing):
    _installed(tmp_path, monkeypatch)
    if missing == "system":
        monkeypatch.setattr(pr, "system_prompt_path", lambda: None)
    else:
        monkeypatch.setattr(pr, "llama_completion", lambda: None)
    said = []
    assert pr.rewrite("a helmet", model=Path("m.gguf"), log=said.append,
                      run=lambda *a, **k: pytest.fail("ran")) is None
    assert said and "without rewriting" in said[0]


def test_a_rewriter_that_fails_or_answers_nothing_usable_is_skipped(tmp_path, monkeypatch):
    _installed(tmp_path, monkeypatch)

    class Broken:
        returncode, stdout, stderr = 1, "", "out of memory"

    said = []
    assert pr.rewrite("a helmet", model=Path("m.gguf"), log=said.append, run=lambda *a, **k: Broken(),
                      lock=lambda *a, **k: __import__("contextlib").nullcontext()) is None
    assert "out of memory" in said[0]


# ---- the whole prompt, as sd-cli gets it ---------------------------------------------

def test_text_to_image_rewrites_then_asks_for_transparency(monkeypatch):
    seen = []
    monkeypatch.setattr(pr, "rewrite", lambda brief, **kw: seen.append(brief) or
                        {"prompt": "A helmet, alone.", "wh_ratio": "1:1", "seconds": 6.1})
    settings = text_image.clean_settings(text_image.T2I_DEFAULTS)
    sent, notes = text_image.prepare_prompt("a helmet", settings, weights={"llm": Path("m.gguf")})
    # the brief already says transparent, so the rewriter keeps the subject alone
    assert seen == [text_image.rgba_prompt("a helmet")]
    assert sent == text_image.rgba_prompt("A helmet, alone.")
    assert notes["prompt_rewrite"]["applied"] is True and notes["prompt_rewrite"]["wh_ratio"] == "1:1"


def test_a_run_with_references_keeps_its_prompt(monkeypatch):
    monkeypatch.setattr(pr, "rewrite", lambda *a, **k: pytest.fail("rewrote an edit"))
    settings = text_image.clean_settings({**text_image.T2I_DEFAULTS})
    sent, notes = text_image.prepare_prompt("the knight", settings, references=True, weights={"llm": Path("m")})
    assert sent == "the knight" and notes == {}


def test_without_a_rewriter_the_prompt_is_still_transparent(monkeypatch):
    monkeypatch.setattr(pr, "rewrite", lambda *a, **k: None)
    settings = text_image.clean_settings(text_image.T2I_DEFAULTS)
    sent, notes = text_image.prepare_prompt("a wolf", settings, weights={"llm": Path("m")})
    assert sent == text_image.rgba_prompt("a wolf")
    assert notes["prompt_rewrite"]["applied"] is False


def test_the_record_keeps_what_was_asked_and_what_was_sent(tmp_path):
    settings = text_image.clean_settings(text_image.T2I_DEFAULTS)
    record = text_image.provenance("a wolf", settings, 1.0, tmp_path / "w.png",
                                   prompt_sent=text_image.rgba_prompt("a wolf"))
    assert record["prompt"] == "a wolf" and record["prompt_sent"].startswith("This is an RGBA image")
    assert record["settings"]["transparent"] is True
    assert "card" in record["settings_source"]


# ---- the studio ---------------------------------------------------------------------

def test_the_studio_draws_at_the_cards_size_and_transparent_unless_asked_otherwise():
    sys.path.insert(0, str(REPO / "viewer"))
    import image_api

    assert image_api.t2i_settings({})["width"] == 2048
    assert image_api.t2i_settings({})["transparent"] is True
    assert image_api.t2i_settings({"width": 1024, "transparent": False}) == {
        **image_api.t2i_settings({}), "width": 1024, "transparent": False}


def test_the_system_prompt_comes_with_the_qwen_setup():
    sys.path.insert(0, str(REPO / "scripts"))
    import bootstrap_qwen_image as boot

    assert pr.SYSTEM_PROMPT_URL in boot.announcement(build=False)
    assert pr.SYSTEM_PROMPT_COMMIT in pr.SYSTEM_PROMPT_URL and "latest" not in pr.SYSTEM_PROMPT_URL


def test_a_downloaded_system_prompt_with_another_checksum_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "SYSTEM_PROMPT_FILE", tmp_path / "pe" / "system_prompt_t2i.txt")
    with pytest.raises(SystemExit, match="checksum"):
        pr.install_system_prompt(fetch=lambda url: b"something else")
    assert not (tmp_path / "pe" / "system_prompt_t2i.txt").exists()
    good = b"the prompt"
    monkeypatch.setattr(pr, "SYSTEM_PROMPT_SHA256", __import__("hashlib").sha256(good).hexdigest())
    assert pr.install_system_prompt(fetch=lambda url: good).read_bytes() == good


# ---- what a fresh install has (code read, 18/10/2026) ----------------------------------

def test_the_rewriter_says_what_is_missing_and_where_to_install_it(monkeypatch, tmp_path):
    """Rewriting is on by default; with only Qwen-Image installed llama.cpp was missing and
    only the job's log said so. The studio now reads what is missing and where to get it."""
    monkeypatch.setattr(pr, "system_prompt_path", lambda: tmp_path / "system.txt")
    monkeypatch.setattr(pr, "llama_completion", lambda: None)
    assert pr.missing() == "llama.cpp"
    assert pr.state() == {"ready": False, "missing": "llama.cpp", "setup": pr.SETUP_HINT}
    assert "Qwen-Image" in pr.SETUP_HINT
    monkeypatch.setattr(pr, "llama_completion", lambda: tmp_path / "llama-completion")
    assert pr.state() == {"ready": True, "missing": None, "setup": None}


def test_a_picture_drawn_without_the_rewriter_records_what_to_install(monkeypatch):
    monkeypatch.setattr(pr, "rewrite", lambda *a, **k: None)
    monkeypatch.setattr(pr, "missing", lambda: "llama.cpp")
    settings = text_image.clean_settings(text_image.T2I_DEFAULTS)
    _, notes = text_image.prepare_prompt("a wolf", settings, weights={"llm": Path("m")})
    assert "llama.cpp" in notes["prompt_rewrite"]["reason"] and "Qwen-Image" in notes["prompt_rewrite"]["reason"]


def test_the_record_names_the_backend_this_machine_drew_with(monkeypatch):
    """Code read 18/10/2026: every picture's record said "Metal", on Windows and Linux too."""
    sys.path.insert(0, str(REPO / "scripts"))
    import bootstrap_qwen_image

    assert set(text_image.RUNTIMES) == set(bootstrap_qwen_image.BUILDS)
    for target, build in bootstrap_qwen_image.BUILDS.items():
        assert text_image.RUNTIMES[target].split()[0] in build.route
    assert text_image.runtime("windows-nvidia") == "stable-diffusion.cpp (CUDA 12)"
    from image_to_3dlab import host

    monkeypatch.setattr(host, "build_target", lambda: "linux-nvidia")
    record = text_image.provenance("a wolf", text_image.clean_settings({}), 1.0, REPO / "missing.png")
    assert record["model"]["runtime"] == "stable-diffusion.cpp (Vulkan)"


def test_a_cancelled_picture_can_stop_the_rewriter(tmp_path, monkeypatch):
    """Code read 18/10/2026: Cancel was looked at only while the picture waited for the GPU; the
    rewriter (6-18 s, up to 10 minutes) ran on. Its process now goes to the job, which stops it."""
    import subprocess
    from contextlib import nullcontext

    monkeypatch.setattr(pr, "system_prompt_path", lambda: tmp_path / "system.txt")
    monkeypatch.setattr(pr, "llama_completion", lambda: tmp_path / "llama-completion")
    seen = []

    class Process:
        returncode = 0

        def communicate(self, timeout=None):
            return '{"rewritten_prompt": "a wolf, painted", "wh_ratio": "1:1"}', ""

    def popen(command, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL
        return Process()

    found = pr.rewrite("a wolf", model=tmp_path / "m.gguf", lock=lambda *a, **k: nullcontext(),
                       on_process=seen.append, popen=popen)
    assert found["prompt"] == "a wolf, painted" and isinstance(seen[0], Process)
