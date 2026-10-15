"""The studio's review queue through the real HTTP handler: check, look, send back, approve."""

from __future__ import annotations

import functools
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import _check_fixtures as fx
import generate_api
import mark_asset
import pytest
import review_api


@pytest.fixture
def server(tmp_path, monkeypatch):
    output = tmp_path / "output"
    run = output / "barrel__pixal3d__20261015-100000"
    run.mkdir(parents=True)
    glb = fx.clean(run / f"{run.name}.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD, suffix=".json")
    fx.stretched(run / "reference.glb", x=1.15)
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", output)
    monkeypatch.setattr(review_api, "REPO", tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, output, f"{run.name}/{run.name}.glb"
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def call(server, method, path, body=None, headers=None):
    httpd = server[0]
    sent = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "review"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        sent["Content-Type"] = "application/json"
    sent.update(headers or {})
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=120)
    connection.request(method, path, body=data, headers={k: v for k, v in sent.items() if v is not None})
    response = connection.getresponse()
    payload = json.loads(response.read() or b"{}")
    connection.close()
    return response.status, payload


def test_the_whole_review_round_trip(server):
    _, output, glb = server
    status, item = call(server, "POST", "/api/review/checks", {
        "asset": glb, "key": "barrel01", "reference": glb.rsplit("/", 1)[0] + "/reference.glb",
        "profile": "prop", "height": 1.8})
    assert status == 200, item
    assert item["state"] == "to_review"
    assert item["report"]["status"] in ("pass", "warn")
    silhouette = next(r for r in item["report"]["results"] if r["check"] == "silhouette")
    assert set(silhouette["details"]["overlay_urls"]) == {"front", "side", "three_quarter", "top"}
    assert all(u.startswith("/output/") for u in silhouette["details"]["overlay_urls"].values())
    assert item["asset_url"].startswith("/output/")

    status, queue = call(server, "GET", "/api/review/queue")
    assert status == 200 and [i["key"] for i in queue["items"]] == ["barrel01"]

    status, problem = call(server, "POST", "/api/review/item/barrel01/decision", {"decision": "sent_back", "note": ""})
    assert status == 400 and "note" in problem["error"]

    status, item = call(server, "POST", "/api/review/item/barrel01/decision",
                        {"decision": "sent_back", "note": "lid too flat"})
    assert status == 200 and item["state"] == "sent_back"
    status, queue = call(server, "GET", "/api/review/queue")
    assert queue["items"][0]["notes"][-1]["note"] == "lid too flat"

    status, item = call(server, "POST", "/api/review/item/barrel01/decision", {"decision": "approved"})
    assert status == 200 and item["state"] == "approved"
    status, queue = call(server, "GET", "/api/review/queue")
    assert queue["items"] == [] and queue["approved"][0]["key"] == "barrel01"
    verdicts = mark_asset.load_register(output / "verdicts.jsonl")
    assert [v["verdict"] for v in verdicts] == ["bad", "good"]


def test_changes_need_the_studio_page(server):
    _, _, glb = server
    status, _ = call(server, "POST", "/api/review/checks", {"asset": glb}, {"X-AssetFurnace-Request": None})
    assert status == 403
    status, _ = call(server, "POST", "/api/review/checks", {"asset": glb}, {"Origin": "http://evil.example"})
    assert status == 403


@pytest.mark.parametrize("asset", ["../secret.glb", "C:/Windows/x.glb", "/etc/passwd", "missing.glb"])
def test_only_files_under_output_are_checked(server, asset):
    status, payload = call(server, "POST", "/api/review/checks", {"asset": asset})
    assert status == 400, payload


def test_an_unknown_asset_is_404(server):
    assert call(server, "GET", "/api/review/item/nobody")[0] == 404
    assert call(server, "POST", "/api/review/item/nobody/decision", {"decision": "approved"})[0] == 404
