"""The Draw-from-a-reference sheet's pure parts under Node: maps, problems, the request, seeds."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def run(body: str):
    url = (REPO / "viewer" / "studio" / "reference-concept.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


@needs_node
def test_each_way_prepares_the_maps_the_trial_used():
    out = run("""console.log(JSON.stringify({
      redrawModel: m.mapsFor('redraw', 'wolf/wolf.glb'),
      redrawPicture: m.mapsFor('redraw', 'images/robot.png'),
      dressBody: m.mapsFor('dress', 'body/body.glb', {guide: 'body'}),
      dressPose: m.mapsFor('dress', 'body/body.glb', {guide: 'pose'}),
      piece: m.mapsFor('add_piece', 'images/body.png'),
      backWithModel: m.mapsFor('other_view', 'images/front.png', {view: 'back', model: 'wolf/wolf.glb'}),
      sideAlone: m.mapsFor('other_view', 'images/front.png', {view: 'side'}),
      nothing: m.mapsFor('redraw', '')}));""")
    assert out["redrawModel"] == [{"source": "wolf/wolf.glb", "kind": "depth", "view": "front"}]
    assert out["redrawPicture"][0]["kind"] == "lineart"
    assert out["dressBody"][0]["kind"] == "colour" and out["dressPose"][0]["kind"] == "pose"
    assert out["piece"][0]["kind"] == "picture"
    assert out["backWithModel"] == [{"source": "images/front.png", "kind": "picture", "view": "front"},
                                    {"source": "wolf/wolf.glb", "kind": "depth", "view": "back"}]
    assert len(out["sideAlone"]) == 1 and out["nothing"] == []


@needs_node
def test_what_is_missing_is_said_in_words():
    out = run("""console.log(JSON.stringify([
      m.problemsOf({way: 'redraw', source: '', subject: '', colours: ''}),
      m.problemsOf({way: 'redraw', source: 'wolf.glb', subject: 'wolf', colours: 'grey'}),
      m.problemsOf({way: 'other_view', source: 'wolf.glb'}),
      m.problemsOf({way: 'dress', source: 'images/body.png', guide: 'pose', outfit: 'a ranger'})]));""")
    assert len(out[0]) == 3 and "What it is is needed." in out[0]
    assert out[1] == []
    assert out[2] == ["Another view starts from the front picture of the creature."]
    assert out[3] == ["A pose map is drawn from a rigged model, not a picture."]


@needs_node
def test_the_request_matches_what_the_server_builds_from():
    out = run("""console.log(JSON.stringify([
      m.requestFor({way: 'redraw', source: 'wolf.glb', subject: 'wolf', colours: 'grey', style: ' painted ', styleRefs: ['images/s.png']},
                   ['concept/maps/a/front_depth_black.png'], 7),
      m.requestFor({way: 'other_view', source: 'images/front.png', view: 'back'}, ['images/front.png', 'concept/maps/a/back_depth_black.png'], 8),
      m.requestFor({way: 'add_piece', source: 'images/body.png', piece: 'a helmet'}, ['images/body.png'], 9)]));""")
    assert out[0] == {"mode": "redraw", "shape": ["concept/maps/a/front_depth_black.png"], "style_refs": ["images/s.png"],
                      "seed": 7, "style": "painted", "subject": "wolf", "colours": "grey", "parts": "", "structure": "depth"}
    assert out[1]["view"] == "back" and out[1]["depth"] is True
    assert out[2] == {"mode": "add_piece", "shape": ["images/body.png"], "style_refs": [], "seed": 9, "piece": "a helmet"}


@needs_node
def test_tries_take_consecutive_seeds_and_stay_between_one_and_four():
    out = run("""console.log(JSON.stringify([m.seedsFor(3, () => 0.5), m.seedsFor(9, () => 0).length, m.seedsFor(0, () => 0).length]));""")
    assert out[0][1] == out[0][0] + 1 and out[0][2] == out[0][0] + 2
    assert out[1:] == [4, 1]


@needs_node
def test_the_iou_line_and_the_library_sources():
    out = run("""console.log(JSON.stringify([
      m.iouText({iou: 0.934}, false), m.iouText(null, true), m.iouText({iou: null}, false),
      m.sourcesFrom([{name: 'Wolf', picture: 'images/w.png', model: 'w/m.glb', rigged: 'w/r.glb'}, {name: 'Robot', picture: 'images/r.png'}])]));""")
    assert out[:3] == ["outline IoU 0.93", "measuring the outline…", "no outline to compare"]
    assert out[3] == [{"path": "w/r.glb", "label": "Wolf (model)"}, {"path": "images/w.png", "label": "Wolf (picture)"},
                      {"path": "images/r.png", "label": "Robot (picture)"}]
