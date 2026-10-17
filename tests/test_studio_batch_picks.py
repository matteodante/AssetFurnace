"""The Batches page picks a waiting row's candidate and reruns a row's step, with the MCP
server's own pick_candidate and rerun_step (docs/plans/2026-10-17-step-parity-design.md,
section 7). The picture model is a stand-in; no runner process starts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from batch_fakes import FakeFinish, fake_step, install_fake_generator, tiny_png
from test_viewer_batch_api import launched, output, request, server  # noqa: F401 - fixtures

from image_to_3dlab import batch
from image_to_3dlab.batch_steps import picture as picture_step

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


@pytest.fixture
def waiting(output, monkeypatch):
    """A batch with a row waiting for a pick among three pictures, and a finished one."""
    def generate(prompt, settings, output_path, **kwargs):
        tiny_png(output_path, seed=settings["seed"])
        output_path.with_suffix(".provenance.json").write_text("{}")
        return output_path

    monkeypatch.setattr(picture_step.text_image, "generate", generate)
    monkeypatch.setattr(picture_step, "readiness", list)
    install_fake_generator(monkeypatch)
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    spec_path = output.parent / "words.json"
    spec_path.write_text(json.dumps({
        "schema_version": 1, "name": "words", "defaults": {"backend": "fake", "steps": ["picture", "generate"]},
        "rows": [{"id": "SM_Barrel", "prompt": "a barrel", "picture": {"seed": 5, "count": 3}},
                 {"id": "SM_Crate", "prompt": "a crate"}]}))
    batch.run_batch(batch.load_batch(spec_path))
    return spec_path


def test_a_waiting_rows_pictures_are_shown_from_the_output_folder(server, waiting):
    status, body = request(server, "GET", "/api/batches/words")
    assert status == 200
    row = next(r for r in body["rows"] if r["id"] == "SM_Barrel")
    assert row["status"] == "waiting" and row["waiting"]["step"] == "picture"
    pictures = [c["picture"] for c in row["waiting"]["candidates"]]
    assert pictures == [f"batches/words/rows/SM_Barrel/picture.candidates/s{s}.png" for s in (5, 6, 7)]
    assert [c["choice"] for c in row["waiting"]["candidates"]] == [5, 6, 7]


def test_picking_on_the_page_records_the_pick_and_resumes(server, waiting, launched):
    status, body = request(server, "POST", "/api/batches/words/pick", {"row": "SM_Barrel", "step": "picture",
                                                                      "choice": 6})
    assert status == 200 and body["picked"] is True and body["resumed"]["started"] is True
    assert batch.load_batch(waiting).rows[0].options["picture"]["pick"] == 6
    status, body = request(server, "POST", "/api/batches/words/pick", {"row": "SM_Barrel", "step": "picture",
                                                                      "choice": 99})
    assert status == 422 and "5, 6, 7" in body["error"]


def test_rerunning_a_rows_step_from_the_page(server, waiting, launched):
    status, body = request(server, "POST", "/api/batches/words/rerun", {"row": "SM_Crate", "step": "generate"})
    assert status == 202 and body["started"] is True
    state = json.loads((waiting.parent / "output" / "batches" / "words" / batch.STATE_FILE).read_text())
    assert state["rows"]["SM_Crate"]["steps"]["generate"]["status"] == "queued"


@pytest.mark.parametrize("data", [{}, {"row": "SM_Barrel"}, {"row": "SM_Barrel", "step": "picture", "choice": "six"}])
def test_odd_picks_are_refused(server, waiting, data):
    status, _ = request(server, "POST", "/api/batches/words/pick", data)
    assert status == 422


def test_picks_need_the_studio_page(server, waiting):
    status, _ = request(server, "POST", "/api/batches/words/pick", {"row": "SM_Barrel", "step": "picture", "choice": 6},
                        headers={"Origin": "https://evil.example"})
    assert status == 403


def _run(body: str):
    url = (REPO / "viewer" / "studio" / "batch.js").as_uri()
    result = subprocess.run([NODE, "--input-type=module", "--eval", f"import * as m from {json.dumps(url)};\n{body}"],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


@needs_node
def test_the_page_shows_the_candidates_and_a_waiting_batch_reads_so():
    out = _run("""
      const row = { id: 'SM_Barrel', status: 'waiting', waiting: { step: 'picture', candidates: [
        { choice: 5, picture: 'batches/w/rows/SM_Barrel/picture.candidates/s5.png' }, { choice: 6, picture: 'x/s6.png' }] } };
      console.log(JSON.stringify({ html: m.candidatesHtml(row, '/output/'), label: m.statusLabel('waiting'),
        line: m.summaryLine({ total: 2, counts: { done: 1, waiting: 1 } }) }));""")
    assert 'src="/output/batches/w/rows/SM_Barrel/picture.candidates/s5.png"' in out["html"]
    assert 'data-pick="5"' in out["html"] and 'data-row="SM_Barrel"' in out["html"] and 'data-step="picture"' in out["html"]
    assert out["label"] == "Pick one" and out["line"] == "1 of 2 done · 1 to pick"


def test_a_row_that_only_draws_needs_no_model(output, monkeypatch):
    """Text to image alone, for an agent: run_steps(["picture"]) is a row with no asset."""
    from image_to_3dlab import mcp_server

    def generate(prompt, settings, output_path, **kwargs):
        tiny_png(output_path, seed=settings["seed"])
        output_path.with_suffix(".provenance.json").write_text("{}")
        return output_path

    monkeypatch.setattr(picture_step.text_image, "generate", generate)
    monkeypatch.setattr(picture_step, "readiness", list)
    spec_path = output.parent / "only.json"
    spec_path.write_text(json.dumps({"schema_version": 1, "name": "only",
                                     "rows": [{"id": "barrel_art", "prompt": "a barrel", "steps": ["picture"]}]}))
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0, report.failed
    (row,) = mcp_server.list_outputs(str(spec_path))["rows"]
    assert row["asset"] is None and row["picture"].endswith("barrel_art.png") and Path(row["picture"]).is_file()
