"""The studio's Review pieces: the pure model under Node, and how the page is wired."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STUDIO = REPO / "viewer" / "studio"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(body: str):
    url = (STUDIO / "review-model.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    done = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                          capture_output=True, text=True, encoding="utf-8")
    return json.loads(done.stdout)


QUEUE = {"items": [
    {"key": "a", "state": "sent_back", "status": "fail", "summary": {"fail": ["mesh.holes"], "warn": []},
     "notes": [{"note": "fix the holes"}]},
    {"key": "b", "state": "to_review", "status": "warn", "summary": {"fail": [], "warn": ["silhouette", "uv.overlap"]}},
], "approved": [{"key": "c", "state": "approved", "status": "pass", "summary": {"fail": [], "warn": []}}]}


@needs_node
def test_the_queue_reads_to_review_then_sent_back_then_approved():
    out = _run(f"""const q = {json.dumps(QUEUE)};
      console.log(JSON.stringify({{
        sections: m.queueSections(q).map((s) => [s.id, s.items.map((i) => i.key)]),
        count: m.waitingCount(q), lines: q.items.map(m.summaryLine).concat(m.summaryLine(q.approved[0])),
        unchecked: [m.summaryLine({{status: null}}), m.statusLabel(null), m.statusLabel('fail')] }}));""")
    assert out["sections"] == [["to_review", ["b"]], ["sent_back", ["a"]], ["approved", ["c"]]]
    assert out["count"] == 1
    assert out["lines"] == ["1 fail", "2 warn", "All checks pass"]
    assert out["unchecked"] == ["Not checked yet", "Not checked", "Fail"]


@needs_node
def test_checks_list_failures_first_and_skips_last():
    report = {"results": [{"check": "mesh.holes", "status": "pass", "message": "ok"},
                          {"check": "budget.faces", "status": "fail", "message": "too many"},
                          {"check": "silhouette", "status": "warn", "message": "off",
                           "details": {"views": {"front": 0.8, "top": 0.7},
                                       "overlay_urls": {"top": "/o/t.png", "front": "/o/f.png"}}}],
              "skipped": [{"check": "gltf.valid", "reason": "not installed"}]}
    out = _run(f"""const r = {json.dumps(report)};
      console.log(JSON.stringify({{ rows: m.checkRows(r).map((x) => x.status + ':' + x.check),
        shots: m.overlays(r).map((s) => [s.view, s.url, s.score]) }}));""")
    assert out["rows"] == ["fail:budget.faces", "warn:silhouette", "pass:mesh.holes", "skip:gltf.valid"]
    assert out["shots"] == [["front", "/o/f.png", 0.8], ["top", "/o/t.png", 0.7]]


@needs_node
def test_a_send_back_needs_a_note_and_the_form_builds_the_request():
    out = _run("""
      const assets = [{id:'k1', name:'knight', rigged:'animate/k_rigged.glb', finished:'finish/k.glb', picture:'images/k.png'},
                      {id:'p1', name:'picture only', picture:'images/p.png'},
                      {id:'h1', name:'hidden', model:'x.glb', hidden:true}];
      const choices = m.checkChoices(assets);
      console.log(JSON.stringify({
        approve: m.canDecide('approved', ''), empty: m.canDecide('sent_back', '  '), noted: m.canDecide('sent_back', 'redo'),
        bogus: m.canDecide('maybe', 'x'), choices: choices.map((c) => [c.id, c.file]),
        body: m.checksRequest(choices[0], {reference: 'images/k.png', profile: 'creature', height: '1.8'}),
        bare: m.checksRequest(choices[0], {height: 'tall'}) }));""")
    assert out["approve"] and out["noted"]
    assert not out["empty"] and not out["bogus"]
    assert out["choices"] == [["k1", "animate/k_rigged.glb"]]
    assert out["body"] == {"asset": "animate/k_rigged.glb", "key": "k1", "profile": "creature",
                           "reference": "images/k.png", "height": 1.8}
    assert out["bare"] == {"asset": "animate/k_rigged.glb", "key": "k1", "profile": "prop"}


def test_review_is_wired_into_the_page_and_puts_a_sheet_away_first():
    html = (REPO / "viewer" / "studio.html").read_text(encoding="utf-8")
    app = (STUDIO / "app.js").read_text(encoding="utf-8")
    assert 'id="reviewBtn"' in html and 'id="reviewCount"' in html
    start = app.index("async function showReview(")
    body = app[start:app.index("\n}", start)]
    assert "closeSheet(" in body
    assert "turntable: true" in body
    viewer = (STUDIO / "viewer.js").read_text(encoding="utf-8")
    assert "setTurntable" in viewer.split("return {")[-1]


def test_review_changes_carry_the_studio_header():
    review = (STUDIO / "review.js").read_text(encoding="utf-8")
    assert "'X-AssetFurnace-Request': 'review'" in review
