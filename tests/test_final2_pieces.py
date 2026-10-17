"""Second final review, characters in pieces and MCP: one start at a time, a concept drawn
again when the body's picture changed, weapons alone in mode 2 from the studio, and links
to files on a network share read with resources/read."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

from image_to_3dlab import mcp_server
from image_to_3dlab.slot_parts import runs
from tests.test_pieces_picks import _with_picture
from tests.test_slot_parts_cli import _body, cli
from tests.test_slot_parts_run import _fakes

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- one start at a time

class _Process:
    pid = os.getpid()   # alive: the run looks running once it is recorded

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


def _command(output, payload):
    return ["slot_parts"], output / "sets" / str(payload["set"]) / "run.log"


def test_a_second_start_meanwhile_is_refused_not_started_too(tmp_path, monkeypatch):
    # the studio and an agent start a set at the same moment: the second must not pass the
    # running check while the first is launching, nor overwrite its record
    monkeypatch.setitem(runs.COMMANDS, "run", _command)
    monkeypatch.setattr(runs, "_identity", lambda pid: None)
    started, refused = [], []

    def popen(*args, **kwargs):
        if not started:
            started.append("first")
            with pytest.raises(ValueError, match="already being made|being started") as caught:
                runs.start(tmp_path, "run", {"set": "blue", "body": "stick"}, popen=popen)
            refused.append(str(caught.value))
        return _Process()

    runs.start(tmp_path, "run", {"set": "red", "body": "stick"}, popen=popen)
    assert started == ["first"] and refused
    assert json.loads((tmp_path / "sets" / runs.RUN_FILE).read_text())["set"] == "red"
    assert not (tmp_path / "sets" / runs.STARTING).exists()   # let go once recorded


def test_a_start_that_failed_lets_the_next_one_go(tmp_path, monkeypatch):
    monkeypatch.setitem(runs.COMMANDS, "run", _command)
    monkeypatch.setattr(runs, "_identity", lambda pid: None)

    def broken(*args, **kwargs):
        raise OSError("no python")

    with pytest.raises(OSError):
        runs.start(tmp_path, "run", {"set": "red", "body": "stick"}, popen=broken)
    assert not (tmp_path / "sets" / runs.STARTING).exists()
    runs.start(tmp_path, "run", {"set": "red", "body": "stick"}, popen=lambda *a, **k: _Process())


def test_a_reservation_left_by_a_dead_server_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setitem(runs.COMMANDS, "run", _command)
    monkeypatch.setattr(runs, "_identity", lambda pid: None)
    monkeypatch.setattr(runs.processes, "process_alive", lambda pid: pid == os.getpid())
    (tmp_path / "sets").mkdir()
    (tmp_path / "sets" / runs.STARTING).write_text("4242 gone", encoding="utf-8")
    runs.start(tmp_path, "run", {"set": "red", "body": "stick"}, popen=lambda *a, **k: _Process())
    assert not (tmp_path / "sets" / runs.STARTING).exists()


# ------------------------------------------------- the concept and the body's picture

def test_a_changed_body_picture_draws_the_outfit_again(tmp_path):
    body_json, source = _with_picture(tmp_path)
    calls = []
    edit, _, run = _fakes(source, calls)
    folder = tmp_path / "sets" / "red" / "stick" / "concept"

    def draw():
        cli.make_concept(body_json, "red plate", folder, [3], run=run, edit=edit)
        return len(calls)

    assert draw() == 1
    assert draw() == 1                      # same words, same body: kept
    picture = json.loads(body_json.read_text())["image"]
    Image.new("RGB", Image.open(picture).size, (40, 90, 200)).save(picture)   # the body registered again
    assert draw() == 2


# ------------------------------------------------------- weapons alone in mode 2

def test_the_studio_makes_weapons_alone_in_mode_2_without_a_concept(tmp_path):
    sys.path.insert(0, str(REPO / "viewer"))
    import slot_parts_api as api

    _body(tmp_path)
    command, _ = api.run_command(tmp_path, {"mode": 2, "body": "stick", "set": "arms",
                                            "slots": {"weapon": {"on": True}}, "weapon": "a longsword"})
    assert "--weapon" in command and "--concept" not in command
    with pytest.raises(ValueError, match="a file is missing"):   # drawn pieces still need it
        api.run_command(tmp_path, {"mode": 2, "body": "stick", "set": "red", "slots": {"torso": {}}})


# ------------------------------------------------------- links to a network share

def test_a_link_to_a_file_on_a_network_share_is_read_through_the_check(tmp_path, monkeypatch):
    import asyncio

    from mcp.client.client import Client

    asked = []

    def readable(path, roots):
        asked.append(Path(path))
        return Path(path) == Path(tmp_path / "local.glb")

    (tmp_path / "local.glb").write_bytes(b"local")
    monkeypatch.setattr(mcp_server, "_readable", readable)

    async def work(session):
        try:
            await session.read_resource(Path(r"\\server\share\model.glb").as_uri()
                                        if os.name == "nt" else "file://server/share/model.glb")
            shared = "read"
        except Exception as exc:  # noqa: BLE001 - refused, however the SDK raises it
            shared = str(exc)
        local = await session.read_resource("file://localhost/" + (tmp_path / "local.glb").as_posix().lstrip("/"))
        return shared, local

    async def go():
        async with Client(mcp_server.build_server(), mode="legacy") as session:
            return await work(session)

    shared, local = asyncio.run(go())
    assert Path("//server/share/model.glb") in asked          # checked as the share's file
    assert shared != "read"                                    # refused by the check
    assert local.contents[0].blob                              # file://localhost/ is this machine
