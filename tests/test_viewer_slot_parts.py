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
def test_the_picks_panel_ticks_the_pick_marks_the_best_and_flags_a_doubt():
    out = _run("""
      const picks = { boots: { auto: 2, picked: 1, doubt: true, candidates: [
        { seed: 1, picture: 'sets/r/b/pieces/boots/s1.png', score: 0.41 }, { seed: 2, picture: 'sets/r/b/pieces/boots/s2.png', score: 0.9 }] } };
      console.log(JSON.stringify(m.pickRows(picks)));""")
    assert 'src="/output/sets/r/b/pieces/boots/s1.png"' in out
    assert 'value="1" checked' in out and "seed 2 · 90% · best" in out  # the person's pick ticked, the best marked
    assert "none of these looked like the piece" in out


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
    assert out["three"] == {"mode": 3, "body": "human", "set": "red-1", "seeds": "3", "choose": False, "outfit": "red plate",
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
        body: m.bodyPayload(assets[0], 'human-male', { race: 'human' }, 'clothes'),
      }));""")
    assert out["ids"] == ["a"]
    assert out["line"] == "2 pieces · body covered 73% · 1 to look at"
    assert out["body"] == {"name": "human-male", "picture": "p.png", "model": "m.glb", "rigged": "r.glb", "clips": [],
                           "labels": {"race": "human"}, "dressing": "clothes", "dressing_words": ""}


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


def test_a_run_can_stop_at_the_pictures_and_resume_with_the_picks(tmp_path):
    from image_to_3dlab.slot_parts import candidates

    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)
    payload = {"mode": 3, "body": "stick", "set": "red", "outfit": "red plate", "seeds": "1,2,3", "choose": True,
               "slots": {"boots": {}}}
    command, _ = slot_parts_api.run_command(output, payload)
    assert command[-1] == "--stop-at-pictures"
    assert "--stop-at-pictures" not in slot_parts_api.run_command(output, dict(payload, seeds="3"))[0]  # one picture: no choice
    folder = output / "sets" / "red" / "stick"
    with pytest.raises(ValueError, match="no pictures waiting"):
        slot_parts_api.run_command(output, {"resume": True, "body": "stick", "set": "red", "picks": {"boots": 2}})
    pictures = folder / "pieces" / "boots"
    pictures.mkdir(parents=True)
    scored = {"boots": {"candidates": {s: {"picture": str(pictures / f"s{s}.png"), "score": 0.3 * s, "is_piece": 0.9,
                                           "shape": {"score": 0.5}} for s in (1, 2, 3)},
                        "auto": 3, "picked": 3, "by": "auto", "doubt": False}}
    candidates.write(folder / "pieces", scored, "fake judge")
    (folder / "run_state.json").write_text("{}")
    waiting = slot_parts_api.waiting_picks(output, folder)
    assert waiting["boots"]["auto"] == 3 and waiting["boots"]["candidates"][0]["picture"] == "sets/red/stick/pieces/boots/s1.png"
    command, _ = slot_parts_api.run_command(output, {"resume": True, "body": "stick", "set": "red", "picks": {"boots": 2}})
    assert "--resume" in command and command[command.index("--pick") + 1] == "boots=2"
    with pytest.raises(ValueError, match="not one of its pictures"):
        slot_parts_api.run_command(output, {"resume": True, "body": "stick", "set": "red", "picks": {"boots": 9}})


def _alive(monkeypatch, pids: set) -> set:
    """Fake processes' pids counted as alive by the run record's check, until stopped."""
    from image_to_3dlab.slot_parts import runs

    monkeypatch.setattr(runs.processes, "process_alive", lambda pid: pid in pids)
    monkeypatch.setattr(runs, "_identity", lambda pid: None)
    return pids


def test_one_run_at_a_time_and_its_status(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)
    alive = _alive(monkeypatch, {4242})

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

    payload = {"mode": 3, "body": "stick", "set": "red", "outfit": "red", "slots": {"helmet": {}}}
    status = slot_parts_api.start_run(output, payload, popen=popen)
    assert status["running"] and status["set"] == "red"
    with pytest.raises(ValueError, match="already"):
        slot_parts_api.start_run(output, payload, popen=popen)
    assert slot_parts_api.job_status(output)["log"] == ["cuts: 3 slots"]
    slot_parts_api.cancel_run(output, stop=lambda pid: stopped.append(pid) or alive.discard(pid))
    assert stopped == [4242]  # the whole tree, not only the script
    assert slot_parts_api.job_status(output)["running"] is False


def test_a_run_started_before_the_server_is_seen_and_stopped(tmp_path, monkeypatch):
    """Code read 18/10/2026: the page followed its runs in memory only; after a restart (or a
    run an agent started) it showed nothing and offered no Stop while the set used the GPU."""
    import json as _json

    from image_to_3dlab.slot_parts import runs

    output = tmp_path / "output"
    (output / "sets" / "red" / "stick").mkdir(parents=True)
    log = output / "sets" / "red" / "stick" / "run.log"
    log.write_text("pieces: helmet\n", encoding="utf-8")
    (output / "sets" / runs.RUN_FILE).write_text(_json.dumps(
        {"pid": 5151, "action": "run", "set": "red", "body": "stick", "log": str(log), "folder": str(log.parent),
         "started": 0.0}), encoding="utf-8")
    alive = _alive(monkeypatch, {5151})
    status = slot_parts_api.job_status(output)
    assert status["running"] is True and status["set"] == "red" and status["log"] == ["pieces: helmet"]
    stopped = []
    assert slot_parts_api.cancel_run(output, stop=lambda pid: stopped.append(pid) or alive.discard(pid))["stopped"]
    assert stopped == [5151] and slot_parts_api.job_status(output)["running"] is False


def test_a_failed_run_offers_no_pictures_left_by_an_earlier_one(tmp_path, monkeypatch):
    """Codex review: a run that failed showed the pictures an earlier run left waiting."""
    import json as _json

    from image_to_3dlab.slot_parts import runs

    output = tmp_path / "output"
    folder = output / "sets" / "red" / "stick"
    folder.mkdir(parents=True)
    log = folder / "run.log"
    log.write_text("refused: Blender failed\n", encoding="utf-8")
    record = {"pid": 7171, "action": "run", "set": "red", "body": "stick", "log": str(log), "folder": str(folder),
              "started": 1.0}
    (output / "sets" / runs.RUN_FILE).write_text(_json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(runs, "waiting_picks", lambda output, folder: {"helmet": {"candidates": []}})
    _alive(monkeypatch, set())
    runs.exit_path(record).write_text(_json.dumps({"pid": 7171, "started": 1.0, "returncode": 1}), encoding="utf-8")
    assert slot_parts_api.job_status(output)["picks"] is None
    runs.exit_path(record).write_text(_json.dumps({"pid": 7171, "started": 1.0, "returncode": 0}), encoding="utf-8")
    assert slot_parts_api.job_status(output)["picks"] == {"helmet": {"candidates": []}}


def test_a_runs_exit_code_is_kept_beside_it(tmp_path, monkeypatch):
    import threading

    from image_to_3dlab.slot_parts import runs

    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)
    done = threading.Event()

    class Process:
        pid = 6161

        def wait(self, timeout=None):
            done.wait(5)
            return 0

    alive = _alive(monkeypatch, {6161})
    slot_parts_api.start_run(output, {"mode": 3, "body": "stick", "set": "red", "outfit": "red",
                                      "slots": {"helmet": {}}}, popen=lambda *a, **k: Process())
    alive.discard(6161)
    done.set()
    for _ in range(50):
        if slot_parts_api.job_status(output).get("returncode") == 0:
            break
        threading.Event().wait(0.05)
    assert slot_parts_api.job_status(output)["returncode"] == 0


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


def _made_set(output):
    folder = output / "sets" / "red" / "stick"
    folder.mkdir(parents=True)
    (folder / "red.parts.json").write_text(json.dumps({
        "set": "red", "body": {"name": "stick"}, "slots": {"helmet": {"prompt": "a leather hood"}, "boots": {"prompt": ""}},
        "pieces": {"helmet": {}, "boots": {}}, "concept": {"prompt": "red plate", "outfit": "red plate"}, "created": "2026-10-17"}))
    return folder


def test_sets_list_each_pieces_material_and_a_person_corrects_it(tmp_path):
    output = tmp_path / "output"
    folder = _made_set(output)
    listed = slot_parts_api.list_sets(output)[0]["materials"]
    assert listed["helmet"]["material"] == "leather" and listed["boots"]["material"] == "metal"
    saved = slot_parts_api.save_materials(output, {"set": "red", "body": "stick", "pieces": {"boots": "leather"}})
    assert saved["materials"]["boots"] == {"material": "leather", "by": "person", "reason": "corrected by a person"}
    assert json.loads((folder / "materials.json").read_text())["pieces"] == {"boots": "leather"}
    saved = slot_parts_api.save_materials(output, {"set": "red", "body": "stick", "pieces": {"boots": None}})
    assert saved["materials"]["boots"]["by"] == "rule"
    for bad, words in (({"set": "red", "body": "stick", "pieces": {"boots": "steel"}}, "one of"),
                       ({"set": "red", "body": "stick", "pieces": {"tail": "fur"}}, "not a piece"),
                       ({"set": "../red", "body": "stick", "pieces": {"boots": "fur"}}, "letters"),
                       ({"set": "blue", "body": "stick", "pieces": {"boots": "fur"}}, "not a finished set"),
                       ({"set": "red", "body": "stick", "pieces": {}}, "give the pieces")):
        with pytest.raises(ValueError, match=words):
            slot_parts_api.save_materials(output, bad)


def test_a_material_asked_before_the_run_reaches_slots_json(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    _body_json(output)
    command, _ = slot_parts_api.run_command(output, {
        "mode": 3, "body": "stick", "set": "red", "outfit": "red plate", "weapon": "a mace",
        "slots": {"helmet": {"material": "cloth"}, "weapon": {"material": "wood"}, "shield": {"material": "metal"}}})
    spec = json.loads(Path(command[command.index("--slots-json") + 1]).read_text())
    assert spec["helmet"]["material"] == "cloth" and spec["weapon"] == {"material": "wood"}
    assert "shield" not in spec                    # no shield is being made: its material is dropped


@needs_node
def test_the_run_request_carries_a_chosen_material():
    out = _run("""
      const slots = { helmet: { on: true, route: 'C', prompt: '', material: 'leather' }, boots: { on: true, material: '' },
                      weapon: { on: true, prompt: 'a mace', material: 'wood' } };
      const payload = m.runPayload({ mode: 3, body: 'human', set: 'red-1', outfit: 'red plate', slots });
      console.log(JSON.stringify(payload.slots));""")
    assert out["helmet"]["material"] == "leather" and "material" not in out["boots"]
    assert out["weapon"] == {"material": "wood"}


@needs_node
def test_material_menus_offer_auto_then_the_words_and_tick_the_current_one():
    out = _run("""
      console.log(JSON.stringify([m.materialOptions(['metal', 'cloth'], ''), m.materialOptions(['metal', 'cloth'], 'cloth', null),
        m.materialRows({ helmet: { material: 'leather', by: 'person', reason: 'corrected by a person' } }, ['metal', 'leather'])]));""")
    auto, chosen, rows = out
    assert auto.startswith('<option value="" selected>Auto</option>') and '<option value="metal">metal</option>' in auto
    assert '<option value="cloth" selected>cloth</option>' in chosen and "Auto" not in chosen
    assert 'data-set-material="helmet"' in rows and '<option value="leather" selected>' in rows
    assert "corrected by a person" in rows


@needs_node
def test_a_corrected_material_can_go_back_to_auto():
    """Code read 18/10/2026: a made set's menu had no Auto, so a person's correction could not
    be taken back from the studio (the API's null)."""
    out = _run("""
      const rows = m.materialRows({ helmet: { material: 'leather', by: 'person', reason: 'corrected by a person' },
                                    boots: { material: 'metal', by: 'rule', reason: 'the slot' } }, ['metal', 'leather']);
      const back = m.changedMaterials([{ slot: 'helmet', value: '', was: 'leather' }, { slot: 'boots', value: '', was: '' }]);
      console.log(JSON.stringify([rows, back]));""")
    rows, back = out
    helmet, boots = rows.split("</tr>")[:2]
    assert '<option value="">Auto</option>' in helmet and '<option value="leather" selected>' in helmet
    assert '<option value="" selected>Auto (metal)</option>' in boots and 'data-was=""' in boots
    assert back == {"helmet": ""}   # sent as no correction; the API takes "" as null


@needs_node
def test_saving_materials_sends_only_what_a_person_changed():
    out = _run("""
      console.log(JSON.stringify(m.changedMaterials([{ slot: 'legs', value: 'cloth', was: 'metal' },
                                                     { slot: 'helmet', value: 'leather', was: 'leather' }])));""")
    assert out == {"legs": "cloth"}
