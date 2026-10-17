"""The Characters in pieces page: its pure logic under Node against the shipped file, the
server's endpoints with a stand-in run, and the hooks that put it in the studio."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import slot_parts_api

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _run(body: str):
    url = (REPO / "viewer" / "studio" / "slot-parts.js").as_uri()
    program = f"import * as m from {json.dumps(url)};\n{body}"
    result = subprocess.run([NODE, "--input-type=module", "--eval", program],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


@needs_node
def test_three_modes_and_slot_names_in_words():
    out = _run("console.log(JSON.stringify([m.MODES.map((x) => x.title), ['helmet', 'shield', 'odd'].map(m.slotLabel)]));")
    assert out == [["One piece", "Dressed, split into pieces", "Bare body + equipment"], ["Helmet", "Shield", "odd"]]


@needs_node
def test_the_body_choices_come_from_the_list_and_say_when_there_is_none():
    out = _run("console.log(JSON.stringify([m.bodyOptions([]), m.bodyOptions([{ name: 'human', labels: { sex: 'male' } }])]));")
    assert out == ['<option value="">none yet</option>', '<option value="human">human (male)</option>']


@needs_node
def test_the_run_request_follows_the_mode():
    out = _run("""
      const slots = { helmet: { on: true, route: 'C', prompt: ' a visor ' }, shoulders: { on: true, route: 'B0', split: true },
                      torso: { on: false }, weapon: { on: true, prompt: '' } };
      const three = m.runPayload({ mode: 3, body: 'human', set: 'red-1', outfit: 'red plate', slots });
      const two = m.runPayload({ mode: 2, body: 'human', set: 'red-2', concept: 'images/k.png', slots: { gloves: { on: true } } });
      const errors = [{ mode: 1 }, { mode: 3, set: 'x', slots }, { mode: 3, body: 'h', set: '../x', slots },
                      { mode: 3, body: 'h', set: 'x', slots: {} }, { mode: 3, body: 'h', set: 'x', slots },
                      { mode: 2, body: 'h', set: 'x', slots }].map((s) => { try { m.runPayload(s); return null; } catch (e) { return e.message; } });
      console.log(JSON.stringify({ three, two, errors }));""")
    assert out["three"] == {"mode": 3, "body": "human", "set": "red-1", "seeds": "3", "outfit": "red plate",
                            "slots": {"helmet": {"route": "C", "split": False, "prompt": "a visor"},
                                      "shoulders": {"route": "B0", "split": True, "prompt": ""}},
                            "weapon": "a one-handed sword"}
    assert out["two"]["concept"] == "images/k.png" and "outfit" not in out["two"]
    errors = out["errors"]
    assert "+ Create" in errors[0] and "base body" in errors[1] and "name" in errors[2]
    assert "at least one" in errors[3] and "outfit" in errors[4] and "concept" in errors[5]


@needs_node
def test_body_candidates_and_set_lines():
    out = _run("""
      const assets = [{ id: 'a', kind: 'character', rigged: 'r.glb', picture: 'p.png', model: 'm.glb', family: 'humanoid' },
                      { id: 'b', kind: 'character', rigged: 'r.glb', picture: 'p.png', model: 'm.glb', family: 'quadruped' },
                      { id: 'c', kind: 'model', picture: 'p.png', model: 'm.glb' }];
      console.log(JSON.stringify({
        ids: m.bodyCandidates(assets).map((a) => a.id),
        line: m.setLine({ pieces: ['helmet', 'torso'], covered: 0.734, statuses: { helmet: 'pass', torso: 'warn' } }),
        body: m.bodyPayload(assets[0], 'human-male', { race: 'human' }),
      }));""")
    assert out["ids"] == ["a"]
    assert out["line"] == "2 pieces · body covered 73% · 1 to look at"
    assert out["body"] == {"name": "human-male", "picture": "p.png", "model": "m.glb", "rigged": "r.glb", "clips": [],
                           "labels": {"race": "human"}}


def _body_json(output: Path) -> Path:
    from tests.test_slot_parts_cli import _body

    path, *_ = _body(output.parent)
    folder = output / "bodies" / "stick"
    folder.mkdir(parents=True)
    shutil.copy(path, folder / "body.json")
    return folder / "body.json"


def test_run_command_for_each_mode(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)
    (output / "images").mkdir()
    (output / "images" / "k.png").write_bytes(b"png")
    command, log = slot_parts_api.run_command(output, {
        "mode": 3, "body": "stick", "set": "red", "outfit": "red plate", "weapon": "a longsword", "seeds": "3,7",
        "slots": {"helmet": {"route": "C"}, "shoulders": {"split": True}}})
    assert command[2:4] == ["run", "--body"] and "--outfit" in command and "--weapon" in command
    assert command[command.index("--seeds") + 1] == "3,7"
    spec = json.loads(Path(command[command.index("--slots-json") + 1]).read_text())
    assert spec["shoulders"]["split"] is True and log.name == "run.log"
    command, _ = slot_parts_api.run_command(output, {"mode": 2, "body": "stick", "set": "given", "concept": "images/k.png",
                                                     "slots": {"torso": {}}})
    assert command[command.index("--concept") + 1].endswith("k.png")
    for bad, words in (({"mode": 1}, "Create"), ({"mode": 3, "body": "nobody", "set": "x"}, "base body"),
                       ({"mode": 3, "body": "stick", "set": "x", "outfit": "o"}, "at least one"),
                       ({"mode": 2, "body": "stick", "set": "x", "concept": "../../etc/passwd", "slots": {"torso": {}}},
                        "output folder")):
        with pytest.raises(ValueError, match=words):
            slot_parts_api.run_command(output, bad)


def test_one_run_at_a_time_and_its_status(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)

    class Process:
        def __init__(self):
            self.code = None

        def poll(self):
            return self.code

        pid = 4242

        def terminate(self):
            self.code = -15

        def wait(self, timeout=None):
            return self.code

    started, stopped = [], []

    def popen(command, **kwargs):
        started.append(command)
        assert kwargs.keys() & {"creationflags", "start_new_session"}  # its own process group
        kwargs["stdout"].write("cuts: 3 slots\n")
        kwargs["stdout"].flush()
        return Process()

    slot_parts_api._JOB.clear()
    payload = {"mode": 3, "body": "stick", "set": "red", "outfit": "red", "slots": {"helmet": {}}}
    status = slot_parts_api.start_run(output, payload, popen=popen)
    assert status["running"] and status["set"] == "red"
    with pytest.raises(ValueError, match="already"):
        slot_parts_api.start_run(output, payload, popen=popen)
    assert slot_parts_api.job_status(output)["log"] == ["cuts: 3 slots"]
    slot_parts_api.cancel_run(stop=stopped.append)
    assert stopped == [4242]  # the whole tree, not only the script
    assert slot_parts_api.job_status(output)["running"] is False
    slot_parts_api._JOB.clear()


def test_sets_and_bodies_are_listed(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)
    folder = output / "sets" / "red" / "stick"
    (folder / "fit").mkdir(parents=True)
    (folder / "fit" / "preview.glb").write_bytes(b"glb")
    (folder / "red.parts.json").write_text(json.dumps({"set": "red", "body": {"name": "stick"}, "mode": "body_and_equipment",
                                                       "pieces": {"helmet": {}}, "fit": {"covered_share": 0.7},
                                                       "checks": {"helmet": {"status": "warn"}}, "created": "2026-10-17"}))
    assert [b["name"] for b in slot_parts_api.list_bodies(output)] == ["stick"]
    sets = slot_parts_api.list_sets(output)
    assert sets[0]["preview"] == "sets/red/stick/fit/preview.glb" and sets[0]["statuses"] == {"helmet": "warn"}
    assert {s["name"] for s in slot_parts_api.catalogue()} >= {"helmet", "weapon"}


def test_the_page_is_in_the_studio():
    html = (REPO / "viewer" / "studio.html").read_text(encoding="utf-8")
    app = (REPO / "viewer" / "studio" / "app.js").read_text(encoding="utf-8")
    server = (REPO / "viewer" / "generate_api.py").read_text(encoding="utf-8")
    assert 'id="piecesBtn"' in html
    assert "import { openSlotParts } from './slot-parts.js';" in app and "$('piecesBtn')" in app
    assert server.count("slot_parts_api.handle(self, OUTPUT_ROOT") == 2
