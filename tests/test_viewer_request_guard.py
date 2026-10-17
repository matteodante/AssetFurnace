"""The Studio server refuses requests that a web page could forge.

A page open in the browser can POST a plain form to http://127.0.0.1:8777 without any
preflight, and DNS rebinding lets it read the answers. These tests drive the real handler
over HTTP: cross-origin and rebinding requests stop before any endpoint runs, and the
Studio's own same-origin requests still work.
"""

from __future__ import annotations

import functools
import http.client
import json
import shutil
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import activity_api  # noqa: E402
import generate_api  # noqa: E402

NODE = shutil.which("node")

# every mutating endpoint the old Studio API answers itself, with the call it would make
POSTS = [
    "/api/setup/qwen-image/remove",
    "/api/setup/qwen-image/download",
    "/api/setup/run",
    "/api/hf/sign-in",
    "/api/rig/rebind",
    "/api/image",
    "/api/generate",
    "/api/finish",
    "/api/props",
    "/api/chains",
    "/api/assets/some-id/name",
    "/api/assets/some-id/hidden",
    "/api/blender/install",
    "/api/gltfpack/install",
    "/api/trellis/input-advice",
    "/api/animate/rig",
]


@pytest.fixture
def server(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "AGENTS.md").write_text("private notes", encoding="utf-8")
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", output)
    monkeypatch.setattr(generate_api, "ACCESS_TOKEN", None)
    reached = []

    # any endpoint that runs records itself; a refused request must leave this empty
    def record(self, *args, **kwargs):
        reached.append(self.path)
        self._send_json(299, {"reached": self.path})

    for name in ("_backend_download", "_start_setup", "_create_job", "_create_rig_job",
                 "_create_finish_job", "_create_props_job", "_create_image_job",
                 "_start_blender_install", "_start_gltfpack_install", "_trellis_input_advice",
                 "_create_animate_rig"):
        monkeypatch.setattr(generate_api.Handler, name, record)
    monkeypatch.setattr(generate_api, "hf_sign_in_response", lambda payload: reached.append("sign-in") or (299, {}))
    monkeypatch.setattr(generate_api, "set_asset_name", lambda *a: reached.append("name") or "x")
    monkeypatch.setattr(generate_api, "set_asset_hidden", lambda *a: reached.append("hidden"))
    monkeypatch.setattr(generate_api, "chain_from_form", lambda form: reached.append("chain") or (_ for _ in ()).throw(ValueError("x")))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, reached
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def call(server, method, path, headers=None, body=b"{}"):
    httpd, _ = server
    sent = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "studio",
            "Content-Type": "application/json"}
    sent.update(headers or {})
    sent = {k: v for k, v in sent.items() if v is not None}
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=body if method == "POST" else None, headers=sent)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.mark.parametrize("path", POSTS)
def test_a_cross_origin_form_post_is_refused_before_the_endpoint_runs(server, path):
    status, _, _ = call(server, "POST", path, {"Origin": "http://evil.example", "Content-Type": "text/plain",
                                               "X-AssetFurnace-Request": None})
    assert status == 403
    assert server[1] == []


@pytest.mark.parametrize("headers", [
    {"Origin": "http://evil.example"},
    {"Origin": None},
    {"Origin": "null"},
    {"X-AssetFurnace-Request": None},
    {"Sec-Fetch-Site": "cross-site"},
    {"Host": "evil.example"},
])
@pytest.mark.parametrize("path", ["/api/setup/qwen-image/remove", "/api/hf/sign-in", "/api/image"])
def test_every_missing_or_foreign_header_is_refused(server, path, headers):
    status, _, _ = call(server, "POST", path, headers)
    assert status == 403
    assert server[1] == []


@pytest.mark.parametrize("path", ["/AGENTS.md", "/api/catalog", "/api/assets", "/viewer/studio.html"])
def test_a_dns_rebinding_host_cannot_read_the_repository_or_the_api(server, path):
    status, _, body = call(server, "GET", path, {"Host": "evil.example:8777", "Origin": None})
    assert status == 403
    assert b"private notes" not in body


def test_head_is_refused_for_a_rebinding_host_too(server):
    status, _, _ = call(server, "HEAD", "/AGENTS.md", {"Host": "evil.example", "Origin": None})
    assert status == 403


@pytest.mark.parametrize("path", POSTS)
def test_the_studios_own_same_origin_requests_still_reach_the_endpoint(server, path):
    status, _, _ = call(server, "POST", path)
    assert status != 403
    assert server[1], path


def test_a_same_origin_page_load_still_works(server):
    status, _, body = call(server, "GET", "/AGENTS.md", {"Origin": None})
    assert status == 200 and body == b"private notes"
    # a link followed from another site is a plain navigation, not a forged request
    status, _, _ = call(server, "GET", "/AGENTS.md", {"Origin": None, "Sec-Fetch-Site": "cross-site"})
    assert status == 200


def test_a_cross_site_api_read_is_refused(server):
    status, _, _ = call(server, "GET", "/api/catalog", {"Origin": "http://evil.example"})
    assert status == 403


# ---- a server on a network address needs the token it printed -----------------------

@pytest.fixture
def guarded(server, monkeypatch):
    monkeypatch.setattr(generate_api, "ACCESS_TOKEN", "s3cret-token")
    return server


def test_with_a_token_nothing_is_served_without_it(guarded):
    status, _, body = call(guarded, "GET", "/AGENTS.md", {"Origin": None})
    assert status == 401 and b"private notes" not in body
    status, _, _ = call(guarded, "POST", "/api/hf/sign-in")
    assert status == 401
    assert guarded[1] == []
    status, _, _ = call(guarded, "GET", "/AGENTS.md", {"Origin": None, "Cookie": "assetfurnace_token=wrong"})
    assert status == 401


def test_the_printed_link_sets_a_cookie_and_drops_the_token_from_the_address(guarded):
    status, headers, _ = call(guarded, "GET", "/viewer/studio.html?token=s3cret-token&x=1", {"Origin": None})
    assert status == 302
    assert headers["Location"] == "/viewer/studio.html?x=1"
    cookie = headers["Set-Cookie"]
    assert "assetfurnace_token=s3cret-token" in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    status, _, body = call(guarded, "GET", "/AGENTS.md", {"Origin": None, "Cookie": "assetfurnace_token=s3cret-token"})
    assert status == 200 and body == b"private notes"


def test_the_cookie_alone_does_not_excuse_a_cross_origin_post(guarded):
    status, _, _ = call(guarded, "POST", "/api/hf/sign-in", {"Cookie": "assetfurnace_token=s3cret-token",
                                                             "Origin": "http://evil.example"})
    assert status == 403
    status, _, _ = call(guarded, "POST", "/api/hf/sign-in", {"Cookie": "assetfurnace_token=s3cret-token"})
    assert status == 299


def test_a_script_with_the_bearer_token_needs_no_browser_headers(guarded):
    status, _, _ = call(guarded, "POST", "/api/hf/sign-in", {"Authorization": "Bearer s3cret-token",
                                                             "Origin": None, "X-AssetFurnace-Request": None})
    assert status == 299
    status, _, _ = call(guarded, "POST", "/api/hf/sign-in", {"Authorization": "Bearer nope", "Origin": None})
    assert status == 401


def test_behind_a_proxy_the_forwarded_host_is_the_pages_origin(guarded, monkeypatch):
    # RunPod's proxy may hand the server its own Host; the page's origin is the public one
    httpd, _ = guarded
    monkeypatch.setattr(httpd, "server_address", ("0.0.0.0", httpd.server_port))
    status, _, _ = call(guarded, "POST", "/api/hf/sign-in", {
        "Cookie": "assetfurnace_token=s3cret-token", "Origin": "https://pod-8777.proxy.runpod.net",
        "X-Forwarded-Host": "pod-8777.proxy.runpod.net"})
    assert status == 299


# ---- the server's own callers ---------------------------------------------------------

def test_the_plan_runner_calls_the_server_as_the_studio_does(server, monkeypatch):
    httpd, reached = server
    request = activity_api.local_request(f"http://127.0.0.1:{httpd.server_port}")
    status, _ = request("POST", "/api/image", b"{}", "application/json")
    assert status == 299 and reached


def test_the_plan_runner_carries_the_token(guarded):
    httpd, reached = guarded
    request = activity_api.local_request(f"http://127.0.0.1:{httpd.server_port}", token="s3cret-token")
    status, _ = request("POST", "/api/image", b"{}", "application/json")
    assert status == 299 and reached


def test_the_studio_pages_load_the_request_marker_before_their_modules():
    for page in ("studio.html", "setup.html"):
        html = (REPO / "viewer" / page).read_text(encoding="utf-8")
        marker = html.index('<script src="./request-header.js"></script>')
        assert marker < html.index('<script type="module"'), page


@pytest.mark.skipif(NODE is None, reason="Node runs the browser script")
def test_the_request_marker_tags_same_origin_writes_only():
    script = (REPO / "viewer" / "request-header.js").read_text(encoding="utf-8")
    program = """
      const sent = [];
      globalThis.window = globalThis;
      globalThis.location = new URL('http://127.0.0.1:8777/viewer/studio.html');
      globalThis.fetch = async (input, init = {}) => {
        sent.push({ url: String(input instanceof Request ? input.url : input),
                    header: new Headers(init.headers || (input instanceof Request ? input.headers : undefined))
                      .get('X-AssetFurnace-Request') });
      };
      eval(SCRIPT);
      await fetch('/api/image', { method: 'POST', body: '{}' });
      await fetch('/api/catalog');
      await fetch('https://huggingface.co/x', { method: 'POST' });
      await fetch('/api/batch', { method: 'POST', headers: { 'X-AssetFurnace-Request': 'batch' } });
      await fetch(new Request('http://127.0.0.1:8777/api/chains', { method: 'POST', body: 'x' }));
      console.log(JSON.stringify(sent.map(s => s.header)));
    """.replace("SCRIPT", json.dumps(script))
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                            capture_output=True, text=True, encoding="utf-8")
    assert json.loads(result.stdout) == ["studio", None, None, "batch", "studio"]


# ---- a refused request still gets its answer ---------------------------------------------

@pytest.mark.parametrize("path", ["/api/blender-edit/sessions", "/api/review/decide", "/api/material-labels",
                                  "/api/concept/jobs", "/api/batch", "/api/unreal-export/jobs"])
def test_a_refused_post_with_a_body_gets_its_403_not_a_reset(server, path):
    """Closing on unread request bytes makes Windows reset the connection, so the client saw
    ConnectionAbortedError instead of the reason (a flaky test on 2026-10-17)."""
    body = b"x" * 32 * 1024
    for _ in range(10):
        status, _, _ = call(server, "POST", path, {"Origin": "http://evil.example"}, body=body)
        assert status == 403
