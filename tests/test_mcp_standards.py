"""The MCP server against the MCP specification 2025-11-25, through the official SDK's own
in-memory client: tool annotations, typed results, produced files as resource links,
progress while waiting on a batch, paths held to the repository and the client's roots,
the server's name and instructions, and the optional Streamable HTTP transport."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import mcp_types as types  # noqa: E402
from mcp.client.client import Client  # noqa: E402

from image_to_3dlab import mcp_server  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

READ_ONLY = {"validate_batch", "batch_status", "list_outputs", "list_steps", "describe_steps", "review_queue",
             "review_item", "pieces_list", "pieces_status", "wait_batch"}


def _talk(work, **client):
    async def go():   # the 2025-11-25 handshake (initialize), as clients speak it today
        async with Client(mcp_server.build_server(), mode="legacy", **client) as session:
            return await work(session)

    return asyncio.run(go())


def _tools():
    async def work(session):
        return (await session.list_tools()).tools

    return {tool.name: tool for tool in _talk(work)}


# ------------------------------------------------------------------ 1. annotations

def test_every_tool_says_what_it_does_to_the_machine():
    tools = _tools()
    assert set(tools) == {tool.__name__ for tool in mcp_server.TOOLS}
    for name, tool in tools.items():
        hints = tool.annotations
        assert hints is not None and hints.title, name
        assert hints.read_only_hint is (name in READ_ONLY), name
        assert hints.open_world_hint is False, name   # this machine only
        if not hints.read_only_hint:
            assert hints.destructive_hint is not None and hints.idempotent_hint is not None, name
    # what stops or replaces work says so; what only adds does not
    assert tools["cancel_batch"].annotations.destructive_hint is True
    assert tools["rerun_step"].annotations.destructive_hint is True
    assert tools["pieces_fit"].annotations.destructive_hint is True
    assert tools["review_decide"].annotations.destructive_hint is False
    assert tools["pieces_materials"].annotations.idempotent_hint is True


# ---------------------------------------------------------------- 2. typed results

def test_every_tool_publishes_the_shape_of_its_result():
    for name, tool in _tools().items():
        schema = tool.output_schema
        assert schema and schema.get("type") == "object" and schema.get("properties"), name


def test_a_result_is_structured_content_and_its_text():
    async def work(session):
        return await session.call_tool("list_steps", {})

    result = _talk(work)
    assert not result.is_error
    assert result.structured_content == json.loads(json.dumps(mcp_server.list_steps()))
    assert json.loads(result.content[0].text) == result.structured_content


# ------------------------------------------------------------ 3. files as resource links

def test_files_a_tool_hands_back_are_resource_links(tmp_path, monkeypatch):
    made = tmp_path / "repo" / "output" / "lot"
    made.mkdir(parents=True)
    monkeypatch.setattr(mcp_server, "REPO", tmp_path / "repo")
    monkeypatch.setattr(mcp_server, "OUTPUT", tmp_path / "repo" / "output")
    glb, picture, mine = made / "SM_Cat.glb", made / "cat.png", tmp_path / "mine.png"
    for path in (glb, picture, mine):
        path.write_bytes(b"x")
    monkeypatch.setattr(mcp_server, "list_outputs", lambda batch_ref, row_id=None: {
        "name": "lot", "directory": str(made),
        "rows": [{"id": "SM_Cat", "asset": str(glb), "references": [str(picture), str(mine)],
                  "files": [str(glb), str(made / "gone.json")]}]})

    async def work(session):
        return await session.call_tool("list_outputs", {"batch_ref": str(made)})

    result = _talk(work)
    links = {Path(types_path(link.uri)).name: link for link in result.content if link.type == "resource_link"}
    # once each, only files that exist, and only those resources/read hands out (not mine.png)
    assert set(links) == {"SM_Cat.glb", "cat.png"}
    assert links["SM_Cat.glb"].mime_type == "model/gltf-binary" and links["cat.png"].mime_type == "image/png"
    assert links["SM_Cat.glb"].name == "SM_Cat.glb"


def types_path(uri) -> str:
    from urllib.parse import unquote, urlparse

    parsed = urlparse(str(uri))
    path = unquote(parsed.path)
    return path[1:] if len(path) > 2 and path[2] == ":" else path   # /C:/x -> C:/x


# ---------------------------------------------------------------------- 4. progress

def _statuses(*counts):
    seen = iter(counts)

    def status(batch_ref):
        done, total, active = next(seen)
        rows = [{"id": f"r{i}", "status": "done" if i < done else "running" if i == done else "queued"}
                for i in range(total)]
        return {"name": "lot", "directory": batch_ref, "active": active,
                "counts": {"done": done, "queued": total - done}, "rows": rows}

    return status


def test_waiting_on_a_batch_reports_its_progress(monkeypatch):
    monkeypatch.setattr(mcp_server, "batch_status", _statuses((1, 3, 42), (2, 3, 42), (3, 3, None)))
    seen = []
    result = mcp_server.wait_batch("lot", timeout_s=60, every_s=0, report=lambda *a: seen.append(a),
                                   sleep=lambda s: None)
    assert result["finished"] is True and result["timed_out"] is False
    assert [(p, t) for p, t, _ in seen] == [(1, 3), (2, 3), (3, 3)]
    assert "2 of 3" in seen[1][2]


def test_waiting_stops_at_its_timeout_and_says_so(monkeypatch):
    monkeypatch.setattr(mcp_server, "batch_status", _statuses(*[(1, 3, 42)] * 50))
    clock = iter(range(0, 1000, 10))
    result = mcp_server.wait_batch("lot", timeout_s=30, every_s=10, sleep=lambda s: None, now=lambda: next(clock))
    assert result["finished"] is False and result["timed_out"] is True


def test_progress_reaches_the_client(monkeypatch):
    monkeypatch.setattr(mcp_server, "batch_status", _statuses((1, 2, 42), (2, 2, None)))
    monkeypatch.setattr(mcp_server, "_allowed", lambda path, roots: True)
    monkeypatch.setattr(mcp_server, "WAIT_SLEEP", lambda s: None)
    seen = []

    async def progress(value, total, message):
        seen.append((value, total))

    async def work(session):
        return await session.call_tool("wait_batch", {"batch_ref": "lot", "every_s": 0},
                                       progress_callback=progress)

    result = _talk(work)
    assert result.structured_content["finished"] is True
    assert seen == [(1, 2), (2, 2)]


# ------------------------------------------------------------------------ 5. paths

def test_a_path_outside_the_repository_and_the_clients_roots_is_refused(tmp_path):
    spec = tmp_path / "lot.json"
    spec.write_text("{}")

    async def work(session):
        return await session.call_tool("validate_batch", {"spec": str(spec)})

    refused = _talk(work)
    assert refused.is_error and "roots" in refused.content[0].text and str(spec) in refused.content[0].text

    async def roots(context):
        return types.ListRootsResult(roots=[types.Root(uri=tmp_path.resolve().as_uri(), name="mine")])

    allowed = _talk(work, list_roots_callback=roots)
    assert not allowed.is_error   # the spec itself is checked now (it is empty, so it has problems)
    assert allowed.structured_content["ok"] is False


def test_paths_inside_lists_maps_and_step_options_are_held_too(tmp_path):
    outside = str(tmp_path / "x.glb")
    for tool, arguments in (("run_steps", {"name": "a", "steps": ["checks"], "images": [outside]}),
                            ("run_steps", {"name": "a", "steps": ["export_unreal"], "asset": str(REPO / "README.md"),
                                           "options": {"export_unreal": {"body": outside}}}),
                            ("run_steps", {"name": "a", "steps": ["finish"], "replace": {"finish": outside}}),
                            ("pieces_add_body", {"name": "b", "picture": outside, "model": outside, "rigged": outside,
                                                 "dressing": "bodysuit"})):
        async def work(session, tool=tool, arguments=arguments):
            return await session.call_tool(tool, arguments)

        result = _talk(work)
        assert result.is_error and outside in result.content[0].text, (tool, arguments)


def test_the_repository_and_its_output_are_always_allowed():
    assert mcp_server._allowed(REPO / "output" / "batches" / "x.json", [])
    assert mcp_server._allowed(REPO / "README.md", [])
    assert not mcp_server._allowed(REPO.parent / "elsewhere.json", [])
    assert mcp_server._allowed(REPO.parent / "elsewhere.json", [REPO.parent])


# ------------------------------------------------------------- 6. name, instructions, docs

def test_the_server_is_called_assetfurnace_and_says_what_it_can_do():
    server = mcp_server.build_server()
    assert server.name == "assetfurnace"
    for words in ("batch", "review", "pieces", "describe_steps", "wait_batch", "roots"):
        assert words in server.instructions, words


def test_the_docs_list_every_tool_by_feature():
    docs = (REPO / "docs" / "batches.md").read_text(encoding="utf-8")
    section = docs[docs.index("## The MCP server"):]
    for tool in mcp_server.TOOLS:
        assert f"`{tool.__name__}" in section, tool.__name__
    for heading in ("### Batches", "### Steps", "### Review", "### Characters in pieces"):
        assert heading in section, heading


# ------------------------------------------------------------------ 7. Streamable HTTP

def _http(token="s3cret"):
    from starlette.testclient import TestClient

    return TestClient(mcp_server.http_app(token=token, port=8799), base_url="http://127.0.0.1:8799")


INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                         "clientInfo": {"name": "test", "version": "1"}}}
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_http_asks_for_the_token():
    with _http() as client:
        assert client.post("/mcp", json=INITIALIZE, headers=HEADERS).status_code == 401
        wrong = client.post("/mcp", json=INITIALIZE, headers={**HEADERS, "Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        right = client.post("/mcp", json=INITIALIZE, headers={**HEADERS, "Authorization": "Bearer s3cret"})
        assert right.status_code == 200


def test_http_refuses_another_host_name():   # DNS rebinding: a web page cannot reach it by another name
    with _http() as client:
        answer = client.post("/mcp", json=INITIALIZE,
                             headers={**HEADERS, "Authorization": "Bearer s3cret", "Host": "evil.example:8799"})
        assert answer.status_code in (403, 421)


def test_http_listens_only_on_this_machine(capsys):
    assert mcp_server.main(["--http", "--host", "0.0.0.0"]) == 2
    assert "127.0.0.1" in capsys.readouterr().err


def test_the_http_token_comes_from_the_environment_or_is_made_once():
    assert mcp_server.http_token({"ASSETFURNACE_MCP_TOKEN": "abc"}) == ("abc", False)
    made, new = mcp_server.http_token({})
    assert new is True and len(made) >= 32


# ----------------------------------------------------------- review round (Codex, #109)

@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository of its own, so specs and outputs can be inside and outside it."""
    root = tmp_path / "repo"
    (root / "output").mkdir(parents=True)
    monkeypatch.setattr(mcp_server, "REPO", root)
    monkeypatch.setattr(mcp_server, "OUTPUT", root / "output")
    return root


def test_a_spec_cannot_send_its_batch_outside_through_its_output_folder(repo, tmp_path):
    from batch_fakes import tiny_glb

    spec = repo / "output" / "lot.json"
    asset = tiny_glb(repo / "output" / "a.glb")
    spec.write_text(json.dumps({"schema_version": 1, "name": "lot", "use_case": "showcase", "distribution": "private",
                                "output": {"directory": str(tmp_path / "elsewhere")},
                                "rows": [{"id": "SM_A", "asset": str(asset), "steps": ["checks"]}]}))
    outside = str(tmp_path / "elsewhere")

    for tool in ("validate_batch", "submit_batch"):
        async def work(session, tool=tool):
            return await session.call_tool(tool, {"spec": str(spec)})

        result = _talk(work)
        said = result.content[0].text if result.is_error else " ".join(result.structured_content["problems"])
        assert outside in said, tool
        assert result.is_error or not (result.structured_content.get("ok") or result.structured_content.get("started"))

    async def status(session):
        return await session.call_tool("batch_status", {"batch_ref": str(spec)})

    assert _talk(status).is_error


def test_replacements_in_lists_and_in_options_are_held_too(repo, tmp_path):
    outside = str(tmp_path / "walk.glb")
    asset = repo / "output" / "a.glb"
    asset.write_bytes(b"glb")
    for arguments in ({"replace": {"animate": [outside]}}, {"options": {"replace": {"finish": outside}}}):
        async def work(session, arguments=arguments):
            return await session.call_tool("run_steps", {"name": "a", "steps": ["finish", "animate"],
                                                         "asset": str(asset), **arguments})

        result = _talk(work)
        assert result.is_error and outside in result.content[0].text, arguments


def test_a_link_a_tool_gave_can_be_read_and_nothing_else(repo, tmp_path, monkeypatch):
    made = repo / "output" / "SM_Cat.glb"
    made.write_bytes(b"the model")
    secret = tmp_path / "secret.txt"
    secret.write_text("no")
    monkeypatch.setattr(mcp_server, "list_outputs", lambda batch_ref, row_id=None: {"rows": [{"asset": str(made)}]})

    async def work(session):
        result = await session.call_tool("list_outputs", {"batch_ref": str(repo / "output")})
        link = next(item for item in result.content if item.type == "resource_link")
        read = await session.read_resource(str(link.uri))
        try:
            await session.read_resource(secret.resolve().as_uri())
            leaked = True
        except Exception:  # noqa: BLE001 - refused, however the SDK raises it
            leaked = False
        return read, leaked

    read, leaked = _talk(work)
    import base64

    content = read.contents[0]
    assert base64.b64decode(content.blob) == b"the model" and leaked is False


def test_waiting_never_sleeps_past_its_timeout(monkeypatch):
    monkeypatch.setattr(mcp_server, "batch_status", _statuses(*[(1, 3, 42)] * 50))
    slept, clock = [], [0.0]

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    result = mcp_server.wait_batch("lot", timeout_s=30, every_s=3600, sleep=sleep, now=lambda: clock[0])
    assert result["timed_out"] is True and sum(slept) <= 30
