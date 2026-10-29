"""The studio's Animate from video card (viewer/studio/twocam.js), run under Node against the shipped
file, and the Animate step that hands a non-humanoid to it (viewer/studio/steps.js).

What matters: a humanoid keeps its preset moves; anything else gets the four stages; every
download is named with its size and licence before the click that starts it; a take shows its
verdict, and a bad one is fitted only through "Fit anyway"; nothing typed reaches the page unescaped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(module: str, body: str):
    url = (REPO / "viewer" / "studio" / module).as_uri()
    program = f"import * as m from {json.dumps(url)};\nconst url = (r) => '/output/' + r;\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


WOLF = {"name": "grey wolf · v2", "kind": "character", "rigged": "animate/w/wolf_rigged.glb", "clips": [],
        "fits_moves": False}
PLAN = {"ready": True, "engine": "MiniMax H3 (stable-diffusion.cpp, local)", "license": "MiniMax H3 Community License Agreement",
        "license_url": "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
        "license_note": "MiniMax asks users in the USA, the EU, the UK and South Korea to apply with the form first",
        "download_gb": 0, "missing": []}
FRESH = {"rig": "animate/w/wolf_rigged.glb", "stem": "wolf", "prepared": None, "reader": "none", "reader_info": None,
         "poses": {"done": 0, "count": 2200}, "dino": {"name": "DINOv2", "license": "Apache-2.0", "megabytes": 346, "cached": False},
         "video_model": PLAN, "engine_ready": True, "moves": [], "captured": [], "job": None, "busy": None, "blender": True}
PREPARED = {**FRESH, "prepared": {"noun": "grey wolf", "length_m": 1.6, "preview": "animate/w/twocam/creature/preview.png",
                                  "updated": 7, "warning": None}}
GOOD = {"level": "good", "reasons": []}
BAD = {"level": "bad", "reasons": ["the views disagree by 9.0% of the creature's height"]}


def state(**changes):
    return {**PREPARED, **changes}


def html(s, extra=""):
    return _run("twocam.js", f"""
      const asset = {json.dumps(WOLF)};
      console.log(JSON.stringify(m.twocamHtml({json.dumps(s)}, asset, url{extra})));""")


def test_a_humanoid_keeps_its_presets_and_anything_else_moves_from_a_video():
    out = _run("steps.js", f"""
      const ctx = {{ presets: [{{ id: 'walk', label: 'Walk' }}], engines: [] }};
      const wolf = {json.dumps(WOLF)};
      console.log(JSON.stringify({{
        wolf: m.stepBody('animated', wolf, ctx),
        knight: m.stepBody('animated', {{ ...wolf, fits_moves: true }}, ctx),
        unknown: m.stepBody('animated', {{ ...wolf, fits_moves: null }}, ctx),
        rig: m.stepBody('rigged', {{ ...wolf, rigged: null, model: 'm/w.glb' }}, ctx),
      }}));""")
    assert "data-twocam" in out["wolf"] and "data-preset" not in out["wolf"]
    assert "humanoids only" not in out["wolf"] and "animate in Blender" not in out["wolf"]
    assert "data-preset" in out["knight"] and "data-twocam" not in out["knight"]
    assert "data-preset" in out["unknown"]          # not judged yet: the presets, as before
    assert "from a video" in out["rig"]


def test_a_new_creature_starts_at_prepare_named_after_itself():
    out = html(FRESH)
    assert out.index('data-stage="prepare"') < out.index('data-stage="teach"') < out.index('data-stage="film"') \
        < out.index('data-stage="capture"')
    assert 'value="grey wolf"' in out                      # the title, without its version
    assert "nose to tail, in metres" in out and 'data-tc-act="prepare"' in out
    assert 'data-tc-act="teach"' not in out and 'data-tc-act="film"' not in out
    assert 'class="tc-stage later" data-stage="teach"' in out


def test_the_preview_is_shown_to_check_its_sides():
    out = html(state(prepared={**PREPARED["prepared"], "warning": "no left/right limbs were found"}))
    assert '<img src="/output/animate/w/twocam/creature/preview.png?v=7"' in out
    assert "left joints are pink" in out and "no left/right limbs were found" in out
    assert "Prepare again" in out and "data-starts-over" not in out    # nothing learned yet to set aside
    out = html(state(reader="taught"))
    assert "data-starts-over" in out and "earlier" in out


def test_teaching_names_dinov2_its_size_and_licence_until_it_is_here():
    first = html(PREPARED)
    assert "Download DINOv2 (346 MB) and teach the reader" in first and "Apache-2.0" in first
    assert "about an hour" in first and "2,200 practice poses" in first
    cached = html(state(dino={**FRESH["dino"], "cached": True}))
    assert ">Teach the reader</button>" in cached and "Download DINOv2" not in cached
    resumed = html(state(reader="unfinished", poses={"done": 812, "count": 2200}))
    assert "812 practice poses already rendered are kept" in resumed


def test_a_running_stage_shows_its_bar_and_can_be_cancelled():
    job = {"job_id": "a" * 32, "kind": "teach", "status": "running", "log_tail": "epoch 3 step 650 loss 1.2",
           "last_event": {"overall_pct": 52.5, "message": "Teaching the reader: round 4 of 20"}}
    out = html(state(job=job))
    assert f'data-tc-job="{"a" * 32}"' in out and "width:52.5%" in out and "round 4 of 20" in out
    assert "data-tc-cancel" in out and "close this tab" in out
    assert 'data-tc-act="teach"' not in out             # its own button gives way to the bar


def test_the_video_model_is_downloaded_only_from_its_own_button():
    missing = html(state(reader="taught", video_model={**PLAN, "ready": False, "download_gb": 29.7}))
    assert "about 29.7 GB" in missing and "MiniMax H3 Community License Agreement" in missing
    assert "South Korea" in missing and PLAN["license_url"] in missing
    assert 'data-tc-act="film" data-download>Download the video model (29.7 GB) and film' in missing
    assert 'data-tc-act="film">Film</button>' not in missing
    ready = html(state(reader="taught"))
    assert 'data-tc-act="film">Film</button>' in ready and "data-download" not in ready
    assert "What does it do?" in ready and "How does it end?" in ready and "3 takes · about 20 min" in ready
    no_engine = html(state(engine_ready=False))
    assert "stable-diffusion.cpp" in no_engine and "setup.html" in no_engine and "data-tc-act=\"film\"" not in no_engine


def test_each_take_plays_with_its_verdict_and_a_bad_one_fits_only_anyway():
    moves = [{"slug": "wolf_rears", "action": "It rears up", "end": "on all fours", "created": "2",
              "takes": [{"clip": "animate/w/twocam/clips/wolf_rears/take1_s1.webm", "seed": 1, "number": 1, "check": GOOD},
                        {"clip": "animate/w/twocam/clips/wolf_rears/take2_s2.webm", "seed": 2, "number": 2, "check": BAD}]}]
    out = html(state(reader="taught", moves=moves))
    assert '<video src="/output/animate/w/twocam/clips/wolf_rears/take1_s1.webm" controls' in out
    assert "Good: the views agree and the rig can follow" in out and "Bad: the move will likely look wrong" in out and "9.0%" in out
    assert 'data-tc-capture="animate/w/twocam/clips/wolf_rears/take1_s1.webm">Use this take' in out
    assert 'data-tc-capture="animate/w/twocam/clips/wolf_rears/take2_s2.webm" data-force>Fit anyway' in out
    assert 'class="tc-stage now" data-stage="capture"' in out
    used = html(state(reader="taught", moves=moves,
                      captured=[{"move": "wolf_rears", "take": moves[0]["takes"][0]["clip"], "warnings": []}]))
    assert "in use ✓" in used and "Use it again" in used and 'class="tc-stage done" data-stage="capture"' in used
    unread = html(state(moves=moves))
    assert "data-tc-capture" not in unread                      # no reader yet: nothing to capture with
    filming = {"job_id": "b" * 32, "kind": "film", "status": "running", "log_tail": "", "last_event": {}, "move": "wolf_howls"}
    busy = html(state(reader="taught", job=filming,
                      moves=[{"slug": "wolf_howls", "action": "It howls", "end": "", "created": "3", "takes": []}, *moves]))
    assert "data-tc-capture" not in busy                        # one stage at a time
    assert "take1_s1.webm" in busy and "Filming…" in busy       # takes land under the bar as they come
    assert "What does it do?" not in busy and f'data-tc-job="{"b" * 32}"' in busy


def test_a_failed_capture_says_why_and_offers_fit_anyway_once():
    job = {"job_id": "c" * 32, "kind": "capture", "status": "error", "log_tail": "error: ...",
           "last_event": {"message": "The two views in this take disagree too much to fit", "bad_clip": True,
                          "take": "animate/w/twocam/clips/wolf_rears/take1_s1.webm"}}
    out = html(state(reader="taught", job=job))
    assert "disagree too much" in out and "data-tc-copy" in out and "data-tc-dismiss" in out
    assert 'data-tc-capture="animate/w/twocam/clips/wolf_rears/take1_s1.webm" data-force>Fit anyway' in out
    gone = html(state(reader="taught", job=job), f", {{ dismissed: new Set(['{'c' * 32}']) }}")
    assert "disagree too much" not in gone


def test_typed_words_never_reach_the_page_unescaped():
    moves = [{"slug": "x", "action": "<img src=x onerror=alert(1)>", "end": "\"><b>", "created": "1", "takes": []}]
    out = html(state(moves=moves, prepared={**PREPARED["prepared"], "noun": "<script>"}))
    assert "<img src=x" not in out and "&lt;img src=x" in out and "<script>" not in out and '"><b>' not in out


def test_stage_states_and_the_default_noun():
    out = _run("twocam.js", f"""
      const fresh = {json.dumps(FRESH)}, prepared = {json.dumps(PREPARED)};
      const filmed = {{ ...prepared, reader: 'unfinished', moves: [{{ takes: [{{}}] }}] }};
      console.log(JSON.stringify({{
        fresh: m.stageStates(fresh), prepared: m.stageStates(prepared), filmed: m.stageStates(filmed),
        nouns: [m.defaultNoun({{ name: 'grey wolf · v12' }}), m.defaultNoun({{ name: 'dragon' }}), m.defaultNoun(null)],
        running: [m.runningJob({{ job: {{ status: 'running' }} }})?.status ?? null, m.runningJob({{ job: {{ status: 'error' }} }})],
      }}));""")
    assert out["fresh"] == {"prepare": "now", "teach": "later", "film": "later", "capture": "later"}
    assert out["prepared"] == {"prepare": "done", "teach": "now", "film": "now", "capture": "later"}
    assert out["filmed"] == {"prepare": "done", "teach": "now", "film": "done", "capture": "now"}
    assert out["nouns"] == ["grey wolf", "dragon", ""]
    assert out["running"] == ["running", None]


def test_a_prepared_creature_can_be_turned_round_from_the_page():
    out = html(state())
    assert "It faces the wrong way: turn it round" in out and "data-turn" in out
    assert "data-turn" not in html(FRESH), "nothing to turn before it is prepared"
