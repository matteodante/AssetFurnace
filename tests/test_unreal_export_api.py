"""Studio's Export for Unreal endpoints: requests are checked, exports run, files come back.

The export itself is faked (it is `scripts/export_unreal.py`, tested on its own); the
HTTP part runs through the real server handler, as the Blender edit tests do.
"""

from __future__ import annotations

import functools
import http.client
import io
import json
import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import generate_api
import pytest
import unreal_export_api as api


@pytest.fixture
def output(tmp_path):
    root = tmp_path / "output"
    (root / "animate" / "w").mkdir(parents=True)
    for name in ("w_rigged.glb", "w_idle.glb", "helm.glb"):
        (root / "animate" / "w" / name).write_bytes(b"glb")
    (root / "props" / "p").mkdir(parents=True)
    (root / "props" / "p" / "barrel_LOD0.glb").write_bytes(b"glb")
    (root / "notes.txt").write_text("x")
    api.JOBS.clear()
    yield root
    api.JOBS.clear()


@pytest.mark.parametrize("relative", ["../secret.glb", "/etc/passwd", "notes.txt", "props/p/none.glb", "", None,
                                      "props\\p\\barrel_LOD0.glb"])
def test_only_library_models_inside_output_are_accepted(output, relative):
    with pytest.raises(ValueError):
        api.resolve(output, relative)


def test_a_creature_request_brings_its_moves(output):
    request = api.parse(output, {"file": "animate/w/w_rigged.glb", "kind": "creature", "size_cm": "180",
                                 "measure": "height", "animations": ["animate/w/w_idle.glb"], "check": 1})
    assert request["source"] == (output / "animate/w/w_rigged.glb").resolve()
    assert [p.name for p in request["animations"]] == ["w_idle.glb"]
    assert request["size_cm"] == 180.0 and request["check"] is True


def test_moves_are_dropped_for_anything_but_a_creature(output):
    request = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60,
                                 "measure": "longest", "animations": ["animate/w/w_idle.glb"]})
    assert request["animations"] == []


@pytest.mark.parametrize("data, message", [
    ({"kind": "prop", "size_cm": 10}, "Library"),
    ({"file": "props/p/barrel_LOD0.glb", "kind": "car", "size_cm": 10}, "object"),
    ({"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": "big"}, "centimetres"),
    ({"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 0}, "centimetres"),
    ({"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 5, "measure": "width"}, "height"),
    ({"file": "animate/w/helm.glb", "kind": "equipment", "size_cm": 180}, "Library"),
])
def test_bad_requests_are_explained(output, data, message):
    with pytest.raises(ValueError, match=message):
        api.parse(output, data)


def _fake_export(output):
    def export(source, *, out_dir, log, **kwargs):
        out_dir.mkdir(parents=True)
        (out_dir / "SM_Barrel.glb").write_bytes(b"glb")
        (out_dir / "SM_Barrel.unreal.json").write_text("{}")
        return {"ok": True, "out_dir": str(out_dir), "problems": [], "seen": kwargs}
    return export


def test_a_job_exports_then_checks_when_asked(output):
    request = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60, "check": True})
    job = api.start(output, request, runner=lambda: None)
    checked = []

    def check(folders):
        checked.append(folders)
        (folders[0] / "unreal_check.json").write_text(json.dumps({"ok": True, "results": []}))
        return True

    api.run_job(output, job["id"], request, export=_fake_export(output), check=check)
    view = api.status(output, job["id"])
    assert view["state"] == "done" and view["message"] == "Ready for Unreal."
    assert {f["name"] for f in view["files"]} == {"SM_Barrel.glb", "SM_Barrel.unreal.json", "unreal_check.json"}
    assert all(f["path"].startswith("unreal/") for f in view["files"])
    assert view["unreal"]["ok"] is True and len(checked) == 1
    name, data = api.zip_bytes(output, job["id"])
    assert name.endswith(".zip")
    assert sorted(Path(n).name for n in zipfile.ZipFile(io.BytesIO(data)).namelist()) == [
        "SM_Barrel.glb", "SM_Barrel.unreal.json", "unreal_check.json"]


def test_a_failed_export_says_why(output):
    request = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60})
    job = api.start(output, request, runner=lambda: None)

    def export(*args, **kwargs):
        raise RuntimeError("Blender could not reshape the model: no mesh")

    api.run_job(output, job["id"], request, export=export)
    view = api.status(output, job["id"])
    assert view["state"] == "error" and "no mesh" in view["message"]
    with pytest.raises(KeyError):
        api.zip_bytes(output, job["id"])


def test_one_export_at_a_time(output):
    request = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60})
    api.start(output, request, runner=lambda: None)
    with pytest.raises(RuntimeError):
        api.start(output, request, runner=lambda: None)


# --- through the real server --------------------------------------------------------

@pytest.fixture
def server(output, monkeypatch):
    monkeypatch.setattr(generate_api, "OUTPUT_ROOT", output)
    # Never the real export here: a request that slipped through must not start Blender.
    monkeypatch.setattr(api, "run_job", lambda *args, **kwargs: None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(generate_api.Handler, directory=str(output.parent)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def _request(httpd, method, path, data=None, headers=None):
    defaults = {"Origin": f"http://127.0.0.1:{httpd.server_port}", "X-AssetFurnace-Request": "unreal-export"}
    if data is not None:
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


REAL_RUN_JOB = api.run_job


def test_http_capabilities_and_an_export_round_trip(server, output, monkeypatch):
    finished = threading.Event()
    original = REAL_RUN_JOB

    def run(out, job_id, request, **kwargs):
        original(out, job_id, request, export=_fake_export(out))
        finished.set()

    monkeypatch.setattr(api, "run_job", run)
    status, body = _request(server, "GET", "/api/unreal-export")
    assert status == 200 and set(json.loads(body)) >= {"blender", "unreal", "kinds", "can_reveal"}
    status, body = _request(server, "POST", "/api/unreal-export/jobs",
                            {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60})
    assert status == 202, body
    job_id = json.loads(body)["id"]
    assert finished.wait(5)
    status, body = _request(server, "GET", f"/api/unreal-export/jobs/{job_id}")
    assert status == 200 and json.loads(body)["state"] == "done"
    status, body = _request(server, "GET", f"/api/unreal-export/jobs/{job_id}/zip")
    assert status == 200 and body[:2] == b"PK"


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"}, {"X-AssetFurnace-Request": "blender-edit"},
    {"Host": "evil.example"}, {"Sec-Fetch-Site": "cross-site"},
])
def test_cross_site_requests_never_start_an_export(server, headers):
    status, _ = _request(server, "POST", "/api/unreal-export/jobs",
                         {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60}, headers)
    assert status == 403
    assert api.JOBS == {}


def test_an_unknown_job_is_a_404(server):
    status, _ = _request(server, "GET", "/api/unreal-export/jobs/nope")
    assert status == 404


def test_the_unreal_check_is_refused_without_unreal(server, monkeypatch):
    monkeypatch.setattr(api, "capabilities", lambda handler: {"unreal": False})
    status, body = _request(server, "POST", "/api/unreal-export/jobs",
                            {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60, "check": True})
    assert status == 422 and b"Unreal" in body


# --- review round 1 ----------------------------------------------------------------------

def test_an_unreal_check_that_times_out_ends_the_job(output):
    import subprocess

    request = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60, "check": True})
    job = api.start(output, request, runner=lambda: None)

    def check(folders):
        raise subprocess.TimeoutExpired("UnrealEditor", 1800)

    api.run_job(output, job["id"], request, export=_fake_export(output), check=check)
    view = api.status(output, job["id"])
    assert view["state"] == "error" and "too long" in view["message"]
    api.start(output, request, runner=lambda: None)  # the next export is not blocked


def test_an_unexpected_failure_still_ends_the_job(output):
    request = api.parse(output, {"file": "props/p/barrel_LOD0.glb", "kind": "prop", "size_cm": 60})
    job = api.start(output, request, runner=lambda: None)

    def export(*args, **kwargs):
        raise KeyError("boom")

    api.run_job(output, job["id"], request, export=export)
    assert api.status(output, job["id"])["state"] == "error"


@pytest.mark.parametrize("editor, version", [
    (Path("UnrealEditor"), None),
    (Path("/opt/ue/UnrealEditor"), None),
    (Path("C:/Epic/UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe"), "UE_5.8"),
])
def test_capabilities_name_any_unreal_path_without_failing(monkeypatch, editor, version):
    from types import SimpleNamespace

    monkeypatch.setattr(api.up, "find_unreal_editor", lambda: editor)
    monkeypatch.setattr(api, "local_desktop", lambda handler: True)
    result = api.capabilities(SimpleNamespace())
    assert result["unreal"] is True and result["unreal_version"] == version
