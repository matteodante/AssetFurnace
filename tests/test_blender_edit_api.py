"""Exercise the real HTTP handler, including static aliases, retries and launch failures."""

import functools
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_blender_versions import library as library_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import blender_edit_api as api
import blender_versions as bv
import generate_api

library = library_fixture


@pytest.fixture
def server(library, monkeypatch):
    output, _, _, _ = library
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", output)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(output.parent)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, output
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def request(server, method, path, data=None, headers=None):
    httpd, _ = server
    defaults = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "blender-edit"}
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        defaults["Content-Type"] = "application/json"
    defaults.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    try:
        conn.request(method, path, body=data, headers=defaults)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_http_create_import_retry_and_library(server, library):
    _, source, asset, _ = library
    status, body = request(server, "POST", "/api/blender-edit/sessions", {
        "asset_id": asset["id"], "file": "knight/knight.glb", "request_id": "a" * 32})
    assert status == 201, body
    path = "/api/blender-edit/sessions/" + json.loads(body)["id"] + "/import"
    for _ in range(2):
        status, body = request(server, "POST", path, source.read_bytes(), {"Content-Type": "model/gltf-binary", "X-Filename": "edit.glb"})
        assert status == 201, body
    status, body = request(server, "GET", "/api/assets")
    assets = json.loads(body)["assets"]
    assert status == 200 and len(assets) == 1 and len(assets[0]["blender_versions"]) == 1


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"}, {"X-AssetFurnace-Request": ""},
    {"Host": "evil.example"}, {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"},
])
def test_cross_site_and_rebinding_never_create_an_edit(server, headers):
    status, _ = request(server, "POST", "/api/blender-edit/sessions", {}, headers)
    assert status == 403


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("alias", ["", "/../sessions", "/%2e%2e/sessions"])
def test_private_snapshot_is_not_served_under_static_aliases(server, library, method, alias):
    output, _, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "b" * 32)
    path = f"/{output.name}/blender-edits/sessions{alias}/{session['id']}/input/source.glb"
    status, _ = request(server, method, path)
    assert status == 404


def test_oversized_upload_is_rejected_without_reading_body(server):
    status, _ = request(server, "POST", "/api/blender-edit/sessions/" + "c" * 32 + "/import", b"",
                        {"Content-Type": "model/gltf-binary", "Content-Length": str(api.MAX_BYTES + 1)})
    assert status == 422


def test_launch_is_shell_free_idempotent_and_uses_snapshot(library):
    output, _, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "d" * 32)
    calls = []
    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(poll=lambda: None)
    try:
        api.launch(output, session["id"], blender=Path("Blender with spaces.exe"), popen=popen)
        assert api.launch(output, session["id"], blender=Path("Blender.exe"), popen=popen)["already_open"]
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert "--disable-autoexec" in args and "--background" not in args
        assert "shell" not in kwargs
        assert args[args.index("--source") + 1].endswith("input\\source.glb") or args[args.index("--source") + 1].endswith("input/source.glb")
    finally:
        api.PROCESSES.pop((str(output), session["id"]), None)


def test_launch_failure_is_visible_and_snapshot_survives(library):
    output, _, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "e" * 32)
    def fail(*args, **kwargs):
        raise OSError("unavailable")
    with pytest.raises(ValueError, match="could not start"):
        api.launch(output, session["id"], blender=Path("missing"), popen=fail)
    assert api.status(output, session["id"])["launch"]["state"] == "error"
    assert bv.checked_snapshot(output, session).is_file()


def test_remote_server_does_not_offer_local_desktop_launch(monkeypatch):
    monkeypatch.setattr(api, "find_blender", lambda: Path("blender"))
    handler = SimpleNamespace(headers={"Host": "server.example"}, client_address=("10.0.0.8", 9999))
    assert api.desktop_reason(handler)


def test_server_restart_can_read_pending_sessions(library):
    output, _, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "f" * 32)
    assert bv.list_sessions(output, asset["id"])[0]["id"] == session["id"]
    assert api.status(output, session["id"])["launch"]["state"] == "not_opened"


def test_downloaded_editing_copy_uses_captured_snapshot(server, library):
    output, source, asset, _ = library
    original = source.read_bytes()
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "1" * 32)
    source.write_bytes(b"changed")
    status, body = request(server, "GET", f"/api/blender-edit/sessions/{session['id']}/source.glb")
    assert status == 200 and body == original


def test_public_filename_cannot_alias_private_snapshot(server, library):
    output, _, asset, _ = library
    session = bv.create_session(output, asset["id"], "knight/knight.glb", "2" * 32)
    alias = output / "blender-edits/versions" / ("2" * 32) / "model.glb"
    alias.parent.mkdir(parents=True)
    try:
        alias.symlink_to(bv.session_path(output, session["id"]) / "input/source.glb")
    except OSError:
        pytest.skip("symlink creation not available on this host")
    status, _ = request(server, "GET", f"/{output.name}/blender-edits/versions/{session['id']}/model.glb")
    assert status == 404
