"""The studio's materials endpoints, through the real HTTP handler.

The step itself is tested in test_materials.py; here only what the studio relies on:
where a run lands, what the status says, and that nothing outside output/ is touched.
"""

import functools
import http.client
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from test_materials import banded_albedo, make_glb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import assets_api
import generate_api
import materials_api as api


@pytest.fixture
def output(tmp_path, monkeypatch):
    root = tmp_path / "output"
    (root / "finish" / "knight").mkdir(parents=True)
    make_glb(root / "finish" / "knight" / "knight_18k.glb", banded_albedo())
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", root)
    api.JOBS.clear()
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
    defaults = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "materials"}
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        defaults["Content-Type"] = "application/json"
    defaults.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=10)
    try:
        conn.request(method, path, body=data, headers=defaults)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


FILE = "finish/knight/knight_18k.glb"


def test_run_dir_is_stable_per_source_and_stays_in_output(output):
    first = api.run_dir(output, FILE)
    assert first == api.run_dir(output, FILE)
    assert first.parent == output / "materials"
    assert first != api.run_dir(output, "finish/other/knight_18k.glb")


@pytest.mark.parametrize("bad", ["../secret.glb", "/etc/passwd", "finish/knight/missing.glb",
                                 "finish/knight", "finish/knight/knight_18k.png", ""])
def test_only_existing_glbs_under_output_are_accepted(output, bad):
    with pytest.raises(ValueError):
        api.source_path(output, bad)


def test_status_before_any_run_says_so(server):
    status, body = request(server, "GET", f"/api/materials?file={FILE}")
    assert status == 200, body
    payload = json.loads(body)
    assert payload["run"] is None
    assert payload["job"]["state"] == "idle"


def test_a_run_makes_textures_the_studio_can_load(server, output):
    status, body = request(server, "POST", "/api/materials/run", {"file": FILE})
    assert status == 202, body
    deadline = time.time() + 120
    while True:
        payload = json.loads(request(server, "GET", f"/api/materials?file={FILE}")[1])
        if payload["job"]["state"] != "running" or time.time() > deadline:
            break
        time.sleep(0.5)
    assert payload["job"]["state"] == "done", payload["job"]
    run = payload["run"]
    assert [z["channel"] for z in run["zones"]] == ["R", "G", "B"]
    assert {c["check"] for c in run["checks"]} >= {"tint_zones", "baked_light_direction"}
    for key in ("tint_mask", "orm", "normal"):
        status, _ = request(server, "GET", "/" + output.name + "/" + run["files"][key])
        assert status == 200, key
    assert run["files"]["glb"].endswith(".glb")


def test_a_run_needs_the_studio_header(server):
    status, _ = request(server, "POST", "/api/materials/run", {"file": FILE}, {"X-AssetFurnace-Request": ""})
    assert status == 403
    status, _ = request(server, "POST", "/api/materials/run", {"file": FILE}, {"Origin": "https://evil.example"})
    assert status == 403


def test_a_bad_file_is_refused_with_a_reason(server):
    status, body = request(server, "POST", "/api/materials/run", {"file": "../outside.glb"})
    assert status == 422
    assert json.loads(body)["error"]


class FakeProcess:
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


def test_one_run_at_a_time_per_file(output):
    api.JOBS[FILE] = {"process": FakeProcess(None), "started": time.time()}
    with pytest.raises(RuntimeError):
        api.start(output, FILE)


@pytest.mark.parametrize("code, state", [(None, "running"), (0, "done"), (2, "sent_back"), (1, "failed")])
def test_job_state_follows_the_exit_code(output, code, state):
    api.JOBS[FILE] = {"process": FakeProcess(code), "started": time.time()}
    assert api.job_state(output, FILE)["state"] == state


def test_the_library_does_not_list_material_runs_as_new_models(output):
    run = api.run_dir(output, FILE)
    run.mkdir(parents=True)
    (run / "knight_18k.glb").write_bytes((output / FILE).read_bytes())
    assert "materials" in assets_api.NOT_GENERATED
