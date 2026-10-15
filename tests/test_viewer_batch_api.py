"""The studio's Batches page, through the real HTTP handler, and batch rows in the Library.

Batches run in-process with the fake generator; starting one from the page is tested with
the launcher replaced, so no runner process starts here."""

from __future__ import annotations

import functools
import http.client
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from batch_fakes import (
    FakeFinish,
    FakeGenerator,
    fake_step,
    install_fake_generator,
    tiny_png,
)

from image_to_3dlab import batch, mcp_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import assets_api  # noqa: E402
import batch_api  # noqa: E402
import generate_api  # noqa: E402


@pytest.fixture
def output(tmp_path, monkeypatch):
    root = tmp_path / "output"
    (root / "batches").mkdir(parents=True)
    monkeypatch.setattr(batch, "DEFAULT_OUTPUT", root / "batches")
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="row02"))
    monkeypatch.setitem(batch.STEPS, "finish", fake_step(FakeFinish().run))
    return root


def _spec(root: Path, rows: int = 3, name: str = "lot") -> Path:
    refs = root.parent / "refs"
    spec = {"schema_version": 1, "name": name, "use_case": "game",
            "defaults": {"backend": "fake", "kind": "creature", "parameters": {"seed": 1}},
            "rows": []}
    for index in range(1, rows + 1):
        tiny_png(refs / f"row{index:02d}.png", index)
        spec["rows"].append({"id": f"SK_row{index:02d}",
                             "images": [str(refs / f"row{index:02d}.png")]})
    path = root.parent / f"{name}.json"
    path.write_text(json.dumps(spec))
    return path


@pytest.fixture
def ran(output):
    """A batch of three rows under output/batches, the second one failed."""
    spec = batch.load_batch(_spec(output))
    batch.run_batch(spec)
    return spec


@pytest.fixture
def server(output, monkeypatch):
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", output)
    handler = functools.partial(generate_api.Handler, directory=str(output.parent))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.fixture
def launched(monkeypatch):
    calls = []

    def launch(command, log_path):
        calls.append(command)
        return os.getpid()

    monkeypatch.setattr(mcp_server, "launch", launch)
    return calls


def request(httpd, method, path, data=None, headers=None):
    sent = {"Origin": f"http://127.0.0.1:{httpd.server_port}",
            "X-AssetFurnace-Request": "batch"}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        sent["Content-Type"] = "application/json"
    sent.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=sent)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        conn.close()


# --------------------------------------------------------------------------- reading

def test_the_page_lists_batches_with_their_counts(server, ran):
    status, body = request(server, "GET", "/api/batches")
    assert status == 200
    (listed,) = body["batches"]
    assert listed["name"] == "lot" and listed["total"] == 3
    assert listed["counts"] == {"done": 2, "failed": 1} and listed["active"] is None


def test_a_batch_shows_each_row_with_its_reason_and_review_state(server, ran):
    status, body = request(server, "GET", "/api/batches/lot")
    assert status == 200 and body["name"] == "lot"
    rows = {row["id"]: row for row in body["rows"]}
    assert rows["SK_row02"]["status"] == "failed"
    assert "fake failure" in rows["SK_row02"]["reason"]
    done = rows["SK_row01"]
    assert done["kind"] == "creature"
    assert done["file"] == "batches/lot/rows/SK_row01/finish/SK_row01.glb"
    assert done["review"]["event"] == "queued"
    assert rows["SK_row02"]["file"] is None and rows["SK_row02"]["review"] is None


def test_a_sent_back_row_shows_the_note(server, ran, output):
    with (output / batch.REVIEW_LOG).open("a") as log:
        log.write(json.dumps({"event": "sent_back", "asset_key": "SK_row01",
                              "note": "the tail clips"}) + "\n")
    _, body = request(server, "GET", "/api/batches/lot")
    row = next(row for row in body["rows"] if row["id"] == "SK_row01")
    assert row["review"] == {"event": "sent_back", "asset_key": "SK_row01",
                             "note": "the tail clips"}


def test_an_unknown_batch_is_404(server, output):
    assert request(server, "GET", "/api/batches/nope")[0] == 404
    assert request(server, "GET", "/api/batches/..%2F..%2Fetc")[0] == 404


# ------------------------------------------------------------------------- the Library

def test_finished_rows_are_library_assets(ran, output):
    assets = assets_api.list_assets(output)
    by_name = {asset["name"]: asset for asset in assets}
    assert set(by_name) == {"SK row01", "SK row03"}
    asset = by_name["SK row01"]
    assert asset["stage"] == "finished" and asset["kind"] == "model"
    assert asset["finished"] == "batches/lot/rows/SK_row01/finish/SK_row01.glb"
    assert asset["model"].startswith("batches/lot/rows/SK_row01/generate/clear/")
    assert asset["picture"] == "batches/lot/rows/SK_row01/input/row01.png"


def test_a_batch_asset_can_be_hidden_and_renamed(ran, output):
    asset = assets_api.list_assets(output)[0]
    assets_api.set_hidden(output, asset["id"], True)
    assert assets_api.set_name(output, asset["id"], "Spider") == "Spider"
    again = next(a for a in assets_api.list_assets(output) if a["id"] == asset["id"])
    assert again["hidden"] and again["name"] == "Spider"


def test_no_batches_folder_no_batch_assets(tmp_path):
    assert batch_api.library_nodes(tmp_path) == {"picture": [], "model": [], "finished": []}


# ------------------------------------------------------------------------- starting

def test_check_a_spec_by_its_path_without_starting_it(server, output, launched):
    spec = _spec(output, rows=1, name="fresh")
    status, body = request(server, "POST", "/api/batches",
                           {"path": str(spec), "check_only": True})
    assert status == 200 and body["ok"] is True and body["rows"] == 1
    assert launched == []


def test_start_a_spec_by_its_path(server, output, launched):
    spec = _spec(output, rows=1, name="fresh")
    status, body = request(server, "POST", "/api/batches", {"path": str(spec)})
    assert status == 202 and body["started"] is True and body["name"] == "fresh"
    assert launched[0][3] == str(spec.resolve())


def test_an_uploaded_csv_is_kept_and_started_with_its_intent(server, output, launched):
    image = tiny_png(output.parent / "refs" / "crate.png")
    content = f"id,images,backend\nSM_Crate,{image},fake\n"
    status, body = request(server, "POST", "/api/batches", {
        "filename": "crates.csv", "content": content, "use_case": "game",
        "distribution": "private", "commercial": True})
    assert status == 202, body
    kept = output / "batches" / batch_api.UPLOADS / "crates.csv"
    assert kept.read_text() == content
    assert launched[0][3] == str(kept) and "--commercial" in launched[0]


def test_a_bad_spec_comes_back_with_its_problems(server, output, launched):
    status, body = request(server, "POST", "/api/batches", {
        "filename": "bad.json", "content": json.dumps({"schema_version": 1, "rows": []})})
    assert status == 422 and body["started"] is False
    assert any("no rows" in problem for problem in body["problems"])
    assert launched == []


@pytest.mark.parametrize("payload", [
    {"filename": "../escape.json", "content": "{}"},
    {"filename": "spec.txt", "content": "{}"},
    {"path": "C:/nowhere/spec.json"},
    {},
])
def test_odd_submissions_are_refused(server, output, launched, payload):
    status, _ = request(server, "POST", "/api/batches", payload)
    assert status == 422 and launched == []


def test_resume_retry_and_stop(server, ran, launched, monkeypatch):
    status, body = request(server, "POST", "/api/batches/lot/resume", {"retry_failed": True})
    assert status == 202 and body["started"] is True
    assert "--retry-failed" in launched[0]
    (ran.directory / batch.LOCK_FILE).write_text(str(os.getpid()))
    stopped = []
    monkeypatch.setattr(mcp_server, "terminate_group", stopped.append)
    status, body = request(server, "POST", "/api/batches/lot/cancel", {})
    assert status == 200 and body["cancelled"] is True and stopped == [os.getpid()]


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"}, {"X-AssetFurnace-Request": ""},
    {"Host": "evil.example"}, {"Sec-Fetch-Site": "cross-site"},
])
def test_cross_site_requests_never_start_a_batch(server, output, launched, headers):
    spec = _spec(output, rows=1, name="fresh")
    status, _ = request(server, "POST", "/api/batches", {"path": str(spec)}, headers)
    assert status == 403 and launched == []


def test_a_batch_that_is_starting_is_listed_before_its_state_exists(server, output):
    """The runner takes a few seconds to write its state; the page must see it meanwhile,
    or it stops following a batch it has just started."""
    starting = output / "batches" / "fresh"
    starting.mkdir()
    (starting / batch.LOCK_FILE).write_text(str(os.getpid()))
    _, body = request(server, "GET", "/api/batches")
    (listed,) = body["batches"]
    assert listed["name"] == "fresh" and listed["active"] == os.getpid()
    assert listed["counts"] == {} and listed["total"] == 0
