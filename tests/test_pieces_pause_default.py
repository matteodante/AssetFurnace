"""The user's decision B (17/10/2026): in the studio a set stops after the pictures by default,
the automatic pick already ticked, and one click goes on to 3D; the command line and agents
go straight through unless asked, and can still ask for the pause."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from image_to_3dlab import mcp_server
from image_to_3dlab.slot_parts import runs
from tests.test_viewer_slot_parts import _body_json

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


@pytest.fixture
def output(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    _body_json(out)
    return out


@needs_node
def test_the_studio_form_pauses_after_the_pictures_by_default(output):
    url = (REPO / "viewer" / "studio" / "slot-parts.js").as_uri()
    program = (f"import * as m from {json.dumps(url)};\n"
               "const payload = m.runPayload({ mode: 3, body: 'stick', set: 'red', outfit: 'red plate',"
               " slots: { torso: { on: true } }, ...m.FORM_DEFAULTS });\n"
               "const html = m.pickRows({ torso: { auto: 2, picked: 2, candidates: ["
               "{ seed: 1, picture: 'a.png', score: 0.3 }, { seed: 2, picture: 'b.png', score: 0.8 }] } });\n"
               "console.log(JSON.stringify({ payload, html }));")
    out = json.loads(subprocess.run([NODE, "--input-type=module", "--eval", program], check=True,
                                    capture_output=True, text=True, encoding="utf-8").stdout)
    command, _ = runs.run_command(output, out["payload"])
    assert "--stop-at-pictures" in command                       # the studio stops after the pictures
    assert 'value="2" checked' in out["html"] and "best" in out["html"]   # the automatic pick is ticked


def test_the_command_line_and_agents_go_straight_on_unless_asked(output):
    straight, _ = runs.run_command(output, {"mode": 3, "body": "stick", "set": "red", "outfit": "red plate",
                                            "slots": {"torso": {}}, "seeds": "1,2,3"})
    paused, _ = runs.run_command(output, {"mode": 3, "body": "stick", "set": "red", "outfit": "red plate",
                                          "slots": {"torso": {}}, "seeds": "1,2,3", "choose": True})
    assert "--stop-at-pictures" not in straight and "--stop-at-pictures" in paused
    import inspect

    assert inspect.signature(mcp_server.pieces_run).parameters["choose"].default is False
