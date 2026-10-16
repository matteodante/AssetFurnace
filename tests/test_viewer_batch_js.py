"""The Batches page's pure logic (viewer/studio/batch.js), run under Node against the
shipped file, plus the hooks that put the page in the studio."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(body: str):
    url = (REPO / "viewer" / "studio" / "batch.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


@needs_node
def test_a_batch_reads_as_one_line():
    out = _run("""console.log(JSON.stringify([
      m.summaryLine({ total: 10, counts: { done: 7, failed: 1, queued: 1, running: 1 }, active: 4242 }),
      m.summaryLine({ total: 3, counts: { done: 3 }, active: null }),
      m.summaryLine({ total: 2, counts: { queued: 1, running: 1 }, active: null }),
    ]));""")
    assert out == ["7 of 10 done · 1 failed · 2 to go · running now",
                   "3 of 3 done",
                   "0 of 2 done · 2 to go · stopped"]


@needs_node
def test_the_buttons_follow_the_state():
    out = _run("""console.log(JSON.stringify([
      m.batchActions({ counts: { done: 1, running: 1 }, active: 7 }),
      m.batchActions({ counts: { done: 1, running: 1, failed: 2 }, active: null }),
      m.batchActions({ counts: { done: 3 }, active: null }),
    ]));""")
    assert out == [{"stop": True, "resume": False, "retry": False},
                   {"stop": False, "resume": True, "retry": True},
                   {"stop": False, "resume": False, "retry": False}]


@needs_node
def test_rows_and_reviews_in_words():
    out = _run("""console.log(JSON.stringify({
      statuses: ['queued', 'running', 'done', 'failed', 'odd'].map(m.statusLabel),
      reviews: [null, { event: 'queued' }, { event: 'approved' },
                { event: 'sent_back', note: 'the tail clips' }].map(m.reviewLabel),
    }));""")
    assert out["statuses"] == ["Waiting", "Running", "Done", "Failed", "odd"]
    assert out["reviews"] == ["", "In review", "Approved", "Sent back: the tail clips"]


@needs_node
def test_what_the_form_sends():
    out = _run("""console.log(JSON.stringify([
      m.submitPayload({ path: ' C:/lots/a.json ', useCase: 'game', commercial: true }),
      m.submitPayload({ file: { name: 'a.csv', text: 'id' }, useCase: 'game',
                        distribution: 'private', commercial: true, checkOnly: true }),
      m.submitPayload({ file: { name: 'a.json', text: '{}' }, useCase: 'game' }),
      m.isSpecFile('lot.CSV'), m.isSpecFile('lot.txt'),
    ]));""")
    assert out[0] == {"path": "C:/lots/a.json"}
    assert out[1] == {"filename": "a.csv", "content": "id", "use_case": "game",
                      "distribution": "private", "commercial": True, "check_only": True}
    assert out[2] == {"filename": "a.json", "content": "{}"}
    assert out[3] is True and out[4] is False


def test_the_studio_has_a_batches_button_wired_to_the_page():
    html = (REPO / "viewer" / "studio.html").read_text(encoding="utf-8")
    bar = html.split('<header class="bar">')[1].split("</header>")[0]
    assert re.search(r'id="batchesBtn"[^>]*>Batches<', bar)
    app = (REPO / "viewer" / "studio" / "app.js").read_text(encoding="utf-8")
    assert "from './batch.js'" in app and "$('batchesBtn')" in app
    page = (REPO / "viewer" / "studio" / "batch.js").read_text(encoding="utf-8")
    assert "closeSheet(" in page   # a sheet over the viewer puts the previous one away
    assert "'X-AssetFurnace-Request': 'batch'" in page
