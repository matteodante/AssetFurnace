"""Animate from video (viewer/twocam_api.py): what each stage runs, what it refuses, how its
progress reads, and where its results land. The stages are scripts/twocam_mocap.py; here they
are stood in for by tiny Python scripts that print what the real one prints."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import assets_api
import twocam_api as tc

RUN = "wolf__rig__20261029-120000"
PLAN = {"engine": "MiniMax H3 (stable-diffusion.cpp, local)", "license": "MiniMax H3 Community License Agreement",
        "license_url": "https://example.test/LICENSE", "license_note": "users in the USA, the EU, the UK and "
        "South Korea apply first", "download_gb": 29.7, "missing": ["a (11.4 GB)"], "files": {}}


@pytest.fixture
def output(tmp_path):
    root = tmp_path / "output"
    rigged = root / "animate" / RUN / "wolf_rigged.glb"
    rigged.parent.mkdir(parents=True)
    rigged.write_bytes(b"glTF")
    return root


@pytest.fixture
def manager(output, monkeypatch):
    monkeypatch.setattr(tc, "video_plan", lambda: {**PLAN, "ready": True, "missing": [], "download_gb": 0})
    monkeypatch.setattr(tc, "engine_ready", lambda: True)
    monkeypatch.setattr(tc, "dino_cached", lambda root=None: False)
    counted = []
    monkeypatch.setattr(tc.usage_api, "count_done", counted.append)
    made = tc.TwocamJobManager(output)
    made.counted = counted
    return made


def rig(output) -> Path:
    return (output / "animate" / RUN / "wolf_rigged.glb").resolve()


def prepare(output, *, reader=False, taught=False) -> tc.Creature:
    c = tc.Creature(rig(output))
    c.folder.mkdir(parents=True, exist_ok=True)
    (c.file("creature.json")).write_text(json.dumps({"noun": "grey wolf", "prep": {"length_m": 1.6}}))
    for name in ("rig_info.json", "rigged.blend", "ref.png", "preview.png"):
        c.file(name).write_bytes(b"x")
    if reader:
        c.file("reader.pt").write_bytes(b"pt")
    if taught:
        tc.write_json(c.file("reader.json"), {"held_out_error_px": {"front": 4.0, "side": 6.0}})
    return c


def take(c: tc.Creature, move="wolf_rears", seed=1, check=None) -> Path:
    folder = c.clips / move
    folder.mkdir(parents=True, exist_ok=True)
    clip = folder / f"take{seed}_s{seed}.webm"
    clip.write_bytes(b"webm")
    clip.with_suffix(".provenance.json").write_text(json.dumps({"made_with": PLAN["engine"], "license": PLAN["license"]}))
    if check is not None:
        tc.write_json(clip.with_suffix(".check.json"), {"seed": seed, "check": check})
    return clip


def option(command, name):
    return command[command.index(name) + 1]


# ------------------------------------------------------------------------------ input


def test_only_a_rigged_model_from_the_animate_tab_is_accepted(output):
    assert tc.resolve_rig(f"animate/{RUN}/wolf_rigged.glb", output) == rig(output)
    (output / "animate" / RUN / "wolf_walk.glb").write_bytes(b"x")
    for bad in ("", f"animate/{RUN}/wolf_walk.glb", f"animate/{RUN}/../{RUN}/missing_rigged.glb",
                f"animate/{RUN}/../../secret_rigged.glb", "finish/x/x_rigged.glb", str(rig(output).parent / "nope_rigged.glb")):
        with pytest.raises(ValueError, match="not a rigged model"):
            tc.resolve_rig(bad, output)


def test_a_move_is_named_from_its_words_and_never_looks_like_a_take():
    assert tc.move_slug("The wolf rears up and howls, then drops back") == "wolf_rears_up_howls"
    assert tc.move_slug("a b c") == "move"
    assert tc.move_slug("It jumps to x") == "jumps"
    assert len(tc.move_slug("x" * 80)) == 40
    for words in ("it leaps over a fence then lands on b", "rolls to z", "spins a"):
        assert not assets_api.TAKE_SUFFIX.search(tc.move_slug(words))   # the Library would hide it


def test_more_takes_of_the_same_move_share_its_folder_and_never_reuse_a_seed(tmp_path):
    first = tc.move_folder(tmp_path, "it howls", "")
    first.mkdir()
    tc.write_json(first / "move.json", {"action": "it howls", "end": ""})
    (first / "take1_s1.webm").write_bytes(b"")
    (first / "take2_s2.webm").write_bytes(b"")
    assert tc.move_folder(tmp_path, "it howls", "") == first
    assert tc.next_seed(first) == 3
    assert tc.move_folder(tmp_path, "it howls", "lying down").name == "howls_2"   # other words, own folder
    assert tc.next_seed(tmp_path / "new") == 1


@pytest.mark.parametrize("kind, body, error", [
    ("prepare", {"length_m": "abc", "noun": "wolf"}, "metres"),
    ("prepare", {"length_m": 0, "noun": "wolf"}, "between"),
    ("prepare", {"length_m": 1000, "noun": "wolf"}, "between"),
    ("prepare", {"length_m": 1.6, "noun": "  "}, "what it is"),
    ("prepare", {"length_m": 1.6, "noun": "--yaw-deg"}, "start"),
    ("film", {"action": ""}, "what it does"),
    ("film", {"action": "-x jumps"}, "start"),
    ("film", {"action": "it jumps", "takes": 0}, "takes"),
    ("film", {"action": "it jumps", "takes": 4}, "takes"),
    ("film", {"action": "it jumps", "end": "x" * 201}, "under 200"),
    ("dance", {}, "no such stage"),
])
def test_wrong_input_is_refused_before_anything_runs(manager, output, kind, body, error):
    prepare(output)
    with pytest.raises(ValueError, match=error):
        manager.start(kind, {"rig": f"animate/{RUN}/wolf_rigged.glb", **body})
    assert manager.active is None


# ----------------------------------------------------------------------------- commands


def test_prepare_runs_the_script_on_the_rig_with_its_licence(manager, output):
    job = manager.start("prepare", {"rig": f"animate/{RUN}/wolf_rigged.glb", "length_m": "1.6", "noun": " grey  wolf "})
    command, = job.commands
    c = tc.Creature(rig(output))
    assert command[:5] == [sys.executable, str(tc.SCRIPT), "prepare", str(c.rigged), str(c.folder)]
    assert option(command, "--length-m") == "1.6" and "--noun=grey wolf" in command
    assert option(command, "--source-model") == str(c.rigged)    # the model's licence carries over


def test_teach_renders_the_poses_then_teaches_and_says_yes_only_to_dinov2(manager, output):
    with pytest.raises(ValueError, match="prepare"):
        manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"})
    c = prepare(output, reader=True, taught=True)
    job = manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"})
    dataset, train = job.commands
    assert dataset[2:6] == ["dataset", str(c.file("rigged.blend")), str(c.file("rig_info.json")), str(c.file("dataset"))]
    assert train[2:6] == ["train", str(c.file("dataset")), str(c.file("rig_info.json")), str(c.file("reader.pt"))]
    assert "--yes" in train and "--yes" not in dataset
    assert not c.file("reader.json").exists()    # teaching again: not taught until it ends


def test_film_never_downloads_the_video_model_without_its_own_button(manager, output, monkeypatch):
    prepare(output)
    monkeypatch.setattr(tc, "video_plan", lambda: {**PLAN, "ready": False})
    body = {"rig": f"animate/{RUN}/wolf_rigged.glb", "action": "it howls", "takes": 2}
    for download in (None, False, "yes", 1):
        with pytest.raises(ValueError, match="29.7 GB"):
            manager.start("film", {**body, "download": download})
    job = manager.start("film", {**body, "download": True})
    command, = job.commands
    assert command[-2:] == ["--download", "--yes"]


def test_film_reads_its_takes_only_with_a_reader_and_continues_the_seeds(manager, output):
    c = prepare(output)
    body = {"rig": f"animate/{RUN}/wolf_rigged.glb", "action": "The wolf rears up", "end": "on all fours", "takes": 2}
    command, = manager.start("film", body).commands
    assert command[2:5] == ["film", str(c.folder), "The wolf rears up"]
    assert option(command, "--takes") == "2" and option(command, "--seed") == "1" and "--end=on all fours" in command
    assert "--reader" not in command and "--download" not in command and "--yes" not in command
    manager.active = None
    take(c, "wolf_rears_up", seed=1)
    take(c, "wolf_rears_up", seed=2)
    c.file("reader.pt").write_bytes(b"pt")
    command, = manager.start("film", body).commands
    assert option(command, "--seed") == "3" and option(command, "--reader") == str(c.file("reader.pt"))


def test_film_needs_the_engine_installed(manager, output, monkeypatch):
    prepare(output)
    monkeypatch.setattr(tc, "engine_ready", lambda: False)
    with pytest.raises(ValueError, match="Setup"):
        manager.start("film", {"rig": f"animate/{RUN}/wolf_rigged.glb", "action": "it howls"})


def test_capture_uses_the_take_and_forces_a_bad_one_only_from_fit_anyway(manager, output):
    c = prepare(output)
    clip = take(c, "wolf_rears", seed=2)
    body = {"rig": f"animate/{RUN}/wolf_rigged.glb", "take": f"animate/{RUN}/twocam/clips/wolf_rears/take2_s2.webm"}
    with pytest.raises(ValueError, match="teach the reader"):
        manager.start("capture", body)
    c.file("reader.pt").write_bytes(b"pt")
    command, = manager.start("capture", body).commands
    assert command[2:8] == ["capture", str(clip), str(c.file("reader.pt")), str(c.file("rig_info.json")),
                            str(c.file("rigged.blend")), str(c.captures / "wolf_rears")]
    assert option(command, "--ref") == str(c.file("ref.png"))
    assert "--name=wolf_wolf_rears" in command       # lands as <stem>_<move>.glb
    assert f"--made-with={PLAN['engine']}" in command and f"--video-license={PLAN['license']}" in command
    assert "--force" not in command
    manager.active = None
    for force in ("yes", 1, None):
        assert "--force" not in manager.start("capture", {**body, "force": force}).commands[0]
        manager.active = None
    assert manager.start("capture", {**body, "force": True}).commands[0][-1] == "--force"


def test_capture_takes_only_this_creatures_takes(manager, output):
    c = prepare(output, reader=True)
    take(c)
    stray = output / "animate" / RUN / "twocam" / "take1_s1.webm"
    stray.write_bytes(b"")
    for bad in (f"animate/{RUN}/twocam/take1_s1.webm", f"animate/{RUN}/twocam/clips/wolf_rears/../../take1_s1.webm",
                f"animate/{RUN}/twocam/clips/wolf_rears/move.json", f"animate/{RUN}/wolf_rigged.glb"):
        with pytest.raises(ValueError, match="not a take"):
            manager.start("capture", {"rig": f"animate/{RUN}/wolf_rigged.glb", "take": bad})


def test_one_stage_at_a_time_for_every_creature(manager, output):
    prepare(output, reader=True)
    job = manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"})
    with pytest.raises(RuntimeError, match="wait for it"):
        manager.start("film", {"rig": f"animate/{RUN}/wolf_rigged.glb", "action": "it howls"})
    job.status = "done"
    manager.finish(job)
    assert manager.start("film", {"rig": f"animate/{RUN}/wolf_rigged.glb", "action": "it howls"})


def test_prepare_again_sets_the_old_creature_aside_and_deletes_nothing(manager, output):
    c = prepare(output, reader=True)
    take(c)
    manager.start("prepare", {"rig": f"animate/{RUN}/wolf_rigged.glb", "length_m": 2, "noun": "wolf"})
    earlier, = (c.root / "earlier").iterdir()
    assert (earlier / "creature" / "reader.pt").is_file() and (earlier / "clips" / "wolf_rears" / "take1_s1.webm").is_file()
    assert c.folder.is_dir() and not c.file("reader.pt").exists() and not c.clips.exists()


def test_prepare_again_before_anything_was_learned_just_redoes_it(manager, output):
    c = prepare(output)
    manager.start("prepare", {"rig": f"animate/{RUN}/wolf_rigged.glb", "length_m": 2, "noun": "wolf"})
    assert not (c.root / "earlier").exists() and c.file("creature.json").is_file()


# ----------------------------------------------------------------------------- progress


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_teaching_reads_two_phases_from_the_scripts_own_lines():
    clock = Clock()
    p = tc.TeachProgress(epochs=20, clock=clock)
    first = p.feed("1100/2200 poses rendered, 10 s")     # a resumed render starts half-way
    assert first["phase"] == "dataset" and first["overall_pct"] == 17.5 and first["stage_eta_seconds"] is None
    clock.now += 100
    half = p.feed("1650/2200 poses rendered, 110 s")
    assert half["overall_pct"] == 26.2 and half["stage_eta_seconds"] == 100.0 and "1,650 of 2,200" in half["message"]
    assert p.feed("1600/2200 poses rendered, 120 s")["overall_pct"] == 26.2     # never backwards
    assert p.feed("2200/2200 poses rendered, 200 s")["overall_pct"] == 35.0
    assert p.begin(1)["phase"] == "train" and p.begin(0) is None
    assert p.begin(1)["message"] == "Getting the reader ready"
    assert "DINOv2 (346 MB)" in tc.TeachProgress(fetches_dino=True).begin(1)["message"]
    assert p.feed("epoch 0 step 50 loss 12.3000 30s")["overall_pct"] == 35.0    # how long a round is: unknown yet
    end = p.feed("epoch 0: held-out error front 4.00 px, side 6.00 px")
    assert end["overall_pct"] == 38.2 and "round 2 of 20" in end["message"] and "5.0 pixels" in end["message"]
    later = p.feed("epoch 1 step 75 loss 3.1 60s")     # half-way through the second round
    assert 38.2 < later["overall_pct"] < 41.5
    assert p.feed("epoch 19: held-out error front 2.00 px, side 2.00 px")["overall_pct"] == 100.0
    assert p.feed("Some other line") is None and p.ticks is False


def test_filming_counts_takes_and_shows_a_first_download():
    clock = Clock()
    p = tc.FilmProgress(2, clock=clock)
    said = p.feed("downloading leejet/MiniMax-H3-GGUF/minimax_h3_fl2va_pruned-Q4_K_M.gguf (11.4 GB)")
    assert said["phase"] == "download" and "minimax_h3_fl2va_pruned-Q4_K_M.gguf (11.4 GB)" in said["message"]
    assert p.feed("minimax_h3_fl2va_pruned-Q4_K_M.gguf:  45%|####5     | 5.1G/11.4G")["overall_pct"] == 45
    clock.now += 5
    assert p.tick() is None                       # still downloading: the bar is the file's
    clock.now += 11
    filming = p.tick()      # the take's clock started at the last download line, not at the job's start
    assert filming["message"] == "Filming take 1 of 2" and filming["overall_pct"] == round(100 * 16 / tc.TAKE_SECONDS / 2, 1)
    clock.now += tc.TAKE_SECONDS / 2 - 16
    assert p.tick()["overall_pct"] == 25.0
    one = p.feed("[INFO ] stable-diffusion.cpp:4120 - generate_video completed in 371.21s")
    assert one["takes_done"] == 1 and one["overall_pct"] == 50.0 and one["message"] == "Filming take 2 of 2"
    clock.now += tc.TAKE_SECONDS * 3
    assert p.tick()["overall_pct"] == 97.5        # a slow take never reads as done
    assert p.feed("generate_video completed in 380.0s")["message"] == "Judging the takes"
    assert p.tick() is None


def test_quick_stages_guess_from_the_clock_and_stop_short_of_done():
    clock = Clock()
    p = tc.Progress(60, "Reading", clock=clock)
    clock.now += 30
    assert p.tick()["overall_pct"] == 50
    clock.now += 600
    assert p.tick()["overall_pct"] == 95


def test_a_failure_is_told_in_the_scripts_own_words():
    assert tc.failure(["epoch 1", "error: Blender failed (exit 1):", "Traceback", "boom"]) == \
        "Blender failed (exit 1):\nTraceback\nboom"
    assert tc.failure(["nothing useful"]) is None
    said = ("this clip is not worth fitting: the views disagree by 9.0% of the creature's height for a typical "
            "joint. Generate it again, or pass --force")
    words = tc.bad_clip_words(said)
    assert "disagree by 9.0%" in words and "fit this one anyway" in words and "--force" not in words
    assert tc.bad_clip_words("no reader") is None


def test_the_summary_is_read_even_with_chatter_before_it(tmp_path):
    path = tmp_path / "out.json"
    path.write_text('{\n "a": 1\n}\n')
    assert tc.summary_of(path) == {"a": 1}
    path.write_text('warning: something\n{\n "a": 2\n}\n')
    assert tc.summary_of(path) == {"a": 2}


# ---------------------------------------------------------------------------- running


def stand_in(job: tc.TwocamJob, script: str) -> None:
    """Swap the real script for a few lines of Python that print what it prints."""
    job.commands = [[sys.executable, "-c", script]]


def test_prepare_finishes_with_its_preview_and_a_record_of_what_it_said(manager, output):
    job = manager.start("prepare", {"rig": f"animate/{RUN}/wolf_rigged.glb", "length_m": 1.6, "noun": "wolf"})
    c = job.creature
    stand_in(job, f"""
import json, sys, pathlib
folder = pathlib.Path(r'{c.folder}')
for name in ('rig_info.json', 'rigged.blend', 'ref.png', 'preview.png'):
    (folder / name).write_bytes(b'x')
(folder / 'creature.json').write_text(json.dumps({{'noun': 'wolf', 'prep': {{'length_m': 1.6}}}}))
print('rendering', file=sys.stderr)
print(json.dumps({{'creature': str(folder), 'limbs': ['leg_L'], 'licence': 'MIT'}}, indent=1))
""")
    tc.run_job(job, manager)
    assert job.status == "done", job.events
    done = job.events[-1]
    assert done["phase"] == "done" and done["preview"] == f"animate/{RUN}/twocam/creature/preview.png"
    assert json.loads(c.file("prepared.json").read_text())["licence"] == "MIT"
    assert "rendering" in c.log.read_text() and not c.pid.exists()
    assert manager.counted == ["creature"] and manager.active is None
    state = tc.creature_state(rig(output), manager)
    assert state["prepared"]["noun"] == "wolf" and state["prepared"]["licence"] == "MIT"
    assert state["job"]["status"] == "done" and state["job"]["kind"] == "prepare"


def test_teach_marks_the_reader_taught_only_when_both_parts_end(manager, output):
    c = prepare(output)
    job = manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"})
    render = ("import sys, json; print('2200/2200 poses rendered, 9 s', file=sys.stderr);"
              " print(json.dumps({'poses': 2200}))")
    teach = (f"import sys, json, pathlib; pathlib.Path(r'{c.file('reader.pt')}').write_bytes(b'pt');"
             " print('epoch 19: held-out error front 3.00 px, side 5.00 px', file=sys.stderr);"
             " print(json.dumps({'reader': 'r', 'held_out_error_px': {'front': 3.0, 'side': 5.0}}))")
    job.commands = [[sys.executable, "-c", render], [sys.executable, "-c", teach]]
    tc.run_job(job, manager)
    assert job.status == "done"
    assert [e.get("phase") for e in job.events if e.get("phase") in ("dataset", "train")] == ["dataset", "train", "train"]
    state = tc.creature_state(rig(output), manager)
    assert state["reader"] == "taught" and state["reader_info"]["held_out_error_px"] == {"front": 3.0, "side": 5.0}


def test_film_keeps_each_takes_verdict_beside_it(manager, output):
    c = prepare(output, reader=True)
    job = manager.start("film", {"rig": f"animate/{RUN}/wolf_rigged.glb", "action": "it howls", "takes": 1})
    clip = job.move / "take1_s1.webm"
    check = {"level": "bad", "reasons": ["the feet move differently"], "height_gap_fraction": 0.09}
    stand_in(job, f"""
import json, pathlib
pathlib.Path(r'{clip}').write_bytes(b'webm')
print(json.dumps({{'takes': [{{'clip': r'{clip}', 'seed': 1, 'check': {json.dumps(check)}}}], 'best': r'{clip}'}}))
""")
    tc.run_job(job, manager)
    assert job.status == "done"
    move, = tc.creature_state(rig(output), manager)["moves"]
    assert move["action"] == "it howls" and move["takes"] == [
        {"clip": f"animate/{RUN}/twocam/clips/howls/take1_s1.webm", "seed": 1, "check": check, "number": 1}]
    assert manager.counted == ["video"] and c.prepared()


def test_a_captured_move_lands_in_the_rig_folder_where_the_library_lists_it(manager, output):
    c = prepare(output, reader=True)
    clip = take(c, "wolf_rears", seed=1)
    job = manager.start("capture", {"rig": f"animate/{RUN}/wolf_rigged.glb", "take": tc.rel(clip, output)})
    made = c.captures / "wolf_rears" / "wolf_wolf_rears.glb"
    stand_in(job, f"""
import json, pathlib
made = pathlib.Path(r'{made}')
made.parent.mkdir(parents=True, exist_ok=True)
made.write_bytes(b'glTF')
made.with_suffix('.provenance.json').write_text(json.dumps({{'license': 'MIT'}}))
print(json.dumps({{'read': {{}}, 'fit': {{'verdict': {{'level': 'good'}}, 'reproj_px': 1.5}}, 'apply': {{'glb': str(made)}}}}))
""")
    tc.run_job(job, manager)
    assert job.status == "done", job.events
    glb = c.run / "wolf_wolf_rears.glb"
    assert glb.is_file() and json.loads(glb.with_suffix(".provenance.json").read_text())["license"] == "MIT"
    done = job.events[-1]
    assert done["clip"] == "wolf_rears" and done["path"] == f"animate/{RUN}/wolf_wolf_rears.glb"
    wolf, = assets_api._rigs(output)
    assert [x["name"] for x in wolf["clips"]] == ["wolf_rears"]     # a move, beside the presets
    captured, = tc.creature_state(rig(output), manager)["captured"]
    assert captured["take"] == tc.rel(clip, output) and captured["verdict"] == {"level": "good"}
    assert manager.counted == ["animation"]


def test_a_take_too_bad_to_fit_offers_fit_anyway(manager, output):
    c = prepare(output, reader=True)
    clip = take(c)
    job = manager.start("capture", {"rig": f"animate/{RUN}/wolf_rigged.glb", "take": tc.rel(clip, output)})
    stand_in(job, "import sys; print('error: this clip is not worth fitting: the feet move differently. "
                  "Generate it again, or pass --force', file=sys.stderr); sys.exit(1)")
    tc.run_job(job, manager)
    assert job.status == "error"
    failed = job.events[-1]
    assert failed["bad_clip"] is True and failed["take"] == tc.rel(clip, output)
    assert "the feet move differently" in failed["message"] and "anyway" in failed["message"]
    assert manager.counted == [] and not c.pid.exists()


def test_any_other_failure_says_the_scripts_last_words(manager, output):
    prepare(output)
    job = manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"})
    stand_in(job, "import sys; print('error: no prepared rig at x: run prepare first', file=sys.stderr); sys.exit(1)")
    tc.run_job(job, manager)
    assert job.status == "error" and job.events[-1]["message"] == "no prepared rig at x: run prepare first"
    assert job.events[-1]["bad_clip"] is False
    assert tc.creature_state(rig(output), manager)["reader"] == "none"


def test_a_long_stage_can_be_cancelled_and_leaves_no_pid_file(manager, output):
    prepare(output)
    job = manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"})
    stand_in(job, "import sys, time; print('10/2200 poses rendered, 10 s', file=sys.stderr, flush=True); time.sleep(120)")
    thread = threading.Thread(target=tc.run_job, args=(job, manager))
    thread.start()
    deadline = time.monotonic() + 30
    while not (job.creature.pid.exists() and job.log_lines) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.creature.pid.exists()      # a server that dies now is cleaned up after on restart
    tc.cancel_job(job)
    thread.join(30)
    assert not thread.is_alive() and job.status == "cancelled"
    assert not job.creature.pid.exists() and manager.active is None
    with pytest.raises(RuntimeError, match="already"):
        tc.cancel_job(job)


# ------------------------------------------------------------------------------- state


def test_a_new_rig_starts_with_nothing_done(manager, output, monkeypatch):
    monkeypatch.setattr(tc, "video_plan", lambda: {**PLAN, "ready": False})
    state = tc.creature_state(rig(output), manager)
    assert state["prepared"] is None and state["reader"] == "none" and state["moves"] == [] and state["job"] is None
    assert state["video_model"]["ready"] is False and state["video_model"]["download_gb"] == 29.7
    assert "South Korea" in state["video_model"]["license_note"]
    assert state["dino"] == {**tc.DINO, "cached": False} and state["poses"] == {"done": 0, "count": 2200}


def test_the_state_tells_an_unfinished_reader_from_a_taught_one(manager, output):
    prepare(output, reader=True)
    assert tc.creature_state(rig(output), manager)["reader"] == "unfinished"
    prepare(output, reader=True, taught=True)
    assert tc.creature_state(rig(output), manager)["reader"] == "taught"


def test_a_capture_whose_glb_is_gone_is_not_listed(manager, output):
    c = prepare(output, reader=True)
    tc.write_json(c.captures / "wolf_rears" / "capture.json", {"take": "x", "file": f"animate/{RUN}/wolf_wolf_rears.glb"})
    assert tc.creature_state(rig(output), manager)["captured"] == []


def test_dinov2_is_found_in_the_hugging_face_cache(tmp_path):
    assert tc.dino_cached(tmp_path) is False
    snapshot = tmp_path / "models--facebook--dinov2-with-registers-base" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"")
    assert tc.dino_cached(tmp_path) is True


def test_the_named_download_is_the_one_the_reader_uses():
    pytest.importorskip("torch")
    from image_to_3dlab.twocam import reader

    assert tc.DINO["id"] == reader.BACKBONE_ID
    assert f"{tc.DINO['megabytes']} MB" in (REPO / "scripts" / "twocam_mocap.py").read_text(encoding="utf-8")


def test_every_stage_is_counted_under_a_kind_the_usage_counts_know():
    # count_done refuses an unknown kind, and it runs as a stage finishes
    assert set(tc.USAGE) == set(tc.KINDS) and set(tc.USAGE.values()) <= tc.usage_api.KINDS
    assert {tc.usage_api.STARTS[("api", "twocam", kind)] for kind in tc.KINDS} == set(tc.USAGE.values())


# ------------------------------------------------------------------- after the second review


def test_teach_passes_no_yes_when_dinov2_is_already_here(manager, output, monkeypatch):
    """--yes only when the button named the download, checked the way train checks."""
    prepare(output)
    monkeypatch.setattr(tc, "dino_cached", lambda root=None: True)
    _, train = manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"}).commands
    assert "--yes" not in train


def test_the_page_and_train_ask_the_same_question_about_dinov2(tmp_path):
    from image_to_3dlab.twocam import video

    snap = tmp_path / ("models--" + video.DINO_ID.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert tc.dino_cached(tmp_path) is False
    (snap / "pytorch_model.bin").write_bytes(b"w")  # the older file name counts too
    assert tc.dino_cached(tmp_path) is True and video.dino_cached(tmp_path) is True


def test_a_move_called_rigged_is_not_saved_over_the_rig():
    assert tc.move_slug("Rigged.") == "move_rigged"
    assert tc.move_slug("rigged and ready") == "move_rigged_ready"
    assert tc.move_slug("the wolf howls") == "wolf_howls"


def test_a_cancel_before_the_first_process_starts_runs_nothing(manager, output):
    prepare(output, reader=True, taught=True)
    job = manager.start("teach", {"rig": f"animate/{RUN}/wolf_rigged.glb"})
    tc.cancel_job(job)
    tc.run_job(job, manager)
    assert job.status == "cancelled" and job.process is None


def test_a_creature_facing_the_wrong_way_is_prepared_again_half_a_turn_round(manager, output):
    c = prepare(output)
    with pytest.raises(ValueError, match="prepare the creature first"):
        manager.start("prepare", {"rig": f"animate/{RUN}/wolf_rigged.glb", "length_m": 1.6, "noun": "wolf",
                                  "turn": True})
    tc.write_json(c.file("prepared.json"), {"turned_deg": 270.0})
    command, = manager.start("prepare", {"rig": f"animate/{RUN}/wolf_rigged.glb", "length_m": 1.6,
                                         "noun": "wolf", "turn": True}).commands
    assert "--yaw-deg=90" in command


def test_every_prepare_warning_reaches_the_page(manager, output):
    c = prepare(output)
    tc.write_json(c.file("prepared.json"), {"noun": "wolf", "warnings": ["no limbs", "bone_9 has no mirror"]})
    assert tc.creature_state(rig(output), manager)["prepared"]["warning"] == "no limbs; bone_9 has no mirror"
