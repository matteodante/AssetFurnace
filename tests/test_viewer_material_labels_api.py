"""The labels endpoints through the real handler: bound to the version under review."""

from __future__ import annotations

import functools
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from _check_fixtures import clean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import generate_api

from image_to_3dlab.checks import review

POINT = {"uv": [0.25, 0.5], "material": 0, "label": "leather"}


@pytest.fixture
def output(tmp_path, monkeypatch):
    root = tmp_path / "output"
    (root / "labels").mkdir(parents=True)
    glb = clean(root / "labels" / "mob.glb")
    review.enqueue_for_review(glb, "mob_01", source="material-labels", kind="creature",
                              log=root / "review-log.jsonl")
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", root)
    return root


@pytest.fixture
def server(output):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(output.parent)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def request(httpd, method, path, data=None, headers=None):
    defaults = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "material-labels"}
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        defaults["Content-Type"] = "application/json"
    defaults.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=data, headers=defaults)
        response = conn.getresponse()
        return response.status, json.loads(response.read() or b"null")
    finally:
        conn.close()


def sha(output):
    return review.fold(review.load(output / "review-log.jsonl"))["mob_01"]["asset_sha256"]


def test_the_vocabulary_and_progress_of_the_label_set(server, output):
    status, body = request(server, "GET", "/api/material-labels")
    assert status == 200
    assert "leather" in body["materials"] and body["looks"] == ["right", "wrong"]
    assert body["progress"] == {"total": 1, "complete": 0, "started": 0, "missing": 1}


def test_labels_are_saved_for_the_version_under_review(server, output):
    status, body = request(server, "POST", "/api/material-labels/item/mob_01",
                           {"asset_sha256": sha(output), "baked_light": False, "materials_look": "right",
                            "points": [POINT]})
    assert status == 200, body
    assert body["labels"]["points"] == [POINT] and body["complete"] is True
    status, body = request(server, "GET", "/api/material-labels/item/mob_01")
    assert status == 200 and body["labels"]["baked_light"] is False
    assert request(server, "GET", "/api/material-labels")[1]["progress"]["complete"] == 1


def test_labels_for_a_version_no_longer_under_review_are_refused(server, output):
    status, body = request(server, "POST", "/api/material-labels/item/mob_01",
                           {"asset_sha256": "0" * 64, "baked_light": True})
    assert status == 409 and "version" in body["error"]


def test_an_unknown_asset_and_bad_labels_are_refused(server, output):
    assert request(server, "GET", "/api/material-labels/item/nobody")[0] == 404
    status, body = request(server, "POST", "/api/material-labels/item/mob_01",
                           {"asset_sha256": sha(output), "materials_look": "maybe"})
    assert status == 400 and body["error"]


@pytest.mark.parametrize("headers", [{"X-AssetFurnace-Request": ""}, {"Origin": "https://evil.example"}])
def test_saving_needs_the_studio_page(server, output, headers):
    status, _ = request(server, "POST", "/api/material-labels/item/mob_01",
                        {"asset_sha256": sha(output), "baked_light": True}, headers)
    assert status == 403


def test_items_outside_the_label_set_cannot_be_labelled(server, output):
    """Their labels would never be counted or exported, so they are refused."""
    other = clean(output / "labels" / "barrel.glb")
    review.enqueue_for_review(other, "barrel", source="batch:props", kind="prop", log=output / "review-log.jsonl")
    item_sha = review.fold(review.load(output / "review-log.jsonl"))["barrel"]["asset_sha256"]
    assert request(server, "GET", "/api/material-labels/item/barrel")[0] == 404
    status, _ = request(server, "POST", "/api/material-labels/item/barrel", {"asset_sha256": item_sha, "baked_light": True})
    assert status == 404
