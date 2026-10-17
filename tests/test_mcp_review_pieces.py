"""The MCP tools for the review queue (feature 7) and characters in pieces (feature 4),
called as plain functions. The same library calls as the studio's review sheet and pieces
page: docs/plans/2026-10-17-step-parity-design.md, section 6. No run really starts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import _check_fixtures as fx
import mark_asset
import pytest

from image_to_3dlab import mcp_server
from image_to_3dlab.slot_parts import candidates, runs

# ------------------------------------------------------------------------- review


@pytest.fixture
def checked(tmp_path):
    output = tmp_path / "output"
    run = output / "barrel__pixal3d__20261015-100000"
    run.mkdir(parents=True)
    glb = fx.clean(run / f"{run.name}.glb")
    fx.sidecar(glb, fx.PIXAL3D_RECORD, suffix=".json")
    return glb, output / "review-log.jsonl"


def test_check_review_and_decide_like_the_studio(checked):
    glb, log = checked
    item = mcp_server.check_asset(str(glb), key="barrel01", profile="prop", height=1.8,
                                  set={"budget.faces.max": 50000}, log=str(log))
    assert item["state"] == "to_review" and item["report"]["status"] in ("pass", "warn")
    assert item["report"]["profile"] == "prop" and item["asset"] == str(glb)
    queue = mcp_server.review_queue(log=str(log))
    assert [i["key"] for i in queue["items"]] == ["barrel01"]

    refused = mcp_server.review_decide("barrel01", "sent_back", item["asset_sha256"], note="", log=str(log))
    assert refused["decided"] is False and "note" in refused["problem"]
    sent = mcp_server.review_decide("barrel01", "sent_back", item["asset_sha256"], note="lid too flat",
                                    log=str(log))
    assert sent["decided"] and sent["item"]["state"] == "sent_back"
    assert mcp_server.review_item("barrel01", log=str(log))["notes"][-1]["note"] == "lid too flat"

    fx.scaled(glb, 1.02)   # a new version, checked again, then approved
    again = mcp_server.check_asset(str(glb), key="barrel01", log=str(log))
    stale = mcp_server.review_decide("barrel01", "approved", item["asset_sha256"], log=str(log))
    assert stale["decided"] is False and "newer version" in stale["problem"]
    approved = mcp_server.review_decide("barrel01", "approved", again["asset_sha256"], log=str(log))
    assert approved["decided"] and approved["item"]["state"] == "approved"
    assert mcp_server.review_queue(log=str(log))["approved"][0]["key"] == "barrel01"
    verdicts = mark_asset.load_register(log.parent / "verdicts.jsonl")
    assert [v["verdict"] for v in verdicts] == ["bad", "good"]


def test_bad_check_requests_come_back_as_problems(checked, tmp_path):
    glb, log = checked
    assert "budget.facez" in mcp_server.check_asset(str(glb), set={"budget.facez.max": 1}, log=str(log))["problem"]
    assert "profile" in mcp_server.check_asset(str(glb), profile="spaceship", log=str(log))["problem"]
    assert "does not exist" in mcp_server.check_asset(str(tmp_path / "gone.glb"), log=str(log))["problem"]
    assert mcp_server.review_item("nobody", log=str(log)) == {"problem": "nobody is not in the review queue"}


# --------------------------------------------------------------------------- pieces

@pytest.fixture
def output(tmp_path, monkeypatch):
    from tests.test_slot_parts_cli import _body

    output = tmp_path / "output"
    path, *_ = _body(tmp_path)
    for name in ("stick", "tall"):
        folder = output / "bodies" / name
        folder.mkdir(parents=True)
        record = json.loads(Path(path).read_text())
        record["name"] = name
        (folder / "body.json").write_text(json.dumps(record))
    monkeypatch.setattr(mcp_server, "OUTPUT", output)
    return output


class Process:
    pid = os.getpid()   # alive, as a started run would be

    def __init__(self, command):
        self.command = command

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def started(monkeypatch):
    commands = []

    def popen(command, **kwargs):
        assert kwargs.keys() & {"creationflags", "start_new_session"}   # its own process group
        commands.append(command)
        return Process(command)

    monkeypatch.setattr(mcp_server, "PIECES_POPEN", popen)
    return commands


def test_a_set_is_started_like_the_studio_starts_it(output, started):
    result = mcp_server.pieces_run("stick", "red", outfit="red plate", slots=["helmet", "boots"],
                                   weapon="a longsword", seeds="3,7", choose=True)
    assert result["started"] is True and result["log"].endswith("run.log")
    (command,) = started
    expected, _ = runs.run_command(output, {"mode": 3, "body": "stick", "set": "red", "outfit": "red plate",
                                            "slots": {"helmet": {}, "boots": {}}, "weapon": "a longsword",
                                            "seeds": "3,7", "choose": True})
    assert command == expected and command[-1] == "--stop-at-pictures"
    status = mcp_server.pieces_status()
    assert status["running"] is True and status["set"] == "red"
    # one run at a time, whoever started it
    again = mcp_server.pieces_run("stick", "blue", outfit="blue plate", slots=["helmet"])
    assert again["started"] is False and "already" in again["problems"][0]


def test_an_agent_can_wait_for_the_outfit_and_pick_it(output, started):
    mcp_server.pieces_run("stick", "red", outfit="red plate", slots=["torso"], concept_seeds="3,4",
                          choose_concept=True)
    (command,) = started
    assert "--stop-at-concept" in command and command[command.index("--concept-seeds") + 1] == "3,4"
    folder = output / "sets" / "red" / "stick"
    (folder / "concept").mkdir(parents=True)
    for seed in (3, 4):
        (folder / "concept" / f"s{seed}.png").write_bytes(b"png")
    (folder / "run_state.json").write_text(json.dumps({"waiting": "concept", "concepts": [3, 4]}))
    (output / "sets" / runs.RUN_FILE).write_text(json.dumps({   # the run has ended, waiting
        "pid": 999999, "action": "run", "set": "red", "body": "stick", "log": str(folder / "run.log"),
        "folder": str(folder), "started": 0}))
    waiting = mcp_server.pieces_status()["picks"]
    assert [c["seed"] for c in waiting["concept"]["candidates"]] == [3, 4]
    assert mcp_server.pieces_pick("stick", "red", {"concept": 4})["started"]
    assert started[-1][started[-1].index("--pick") + 1] == "concept=4"


def test_mode_2_takes_a_concept_file(output, started, tmp_path):
    concept = tmp_path / "drawn.png"
    concept.write_bytes(b"png")
    assert mcp_server.pieces_run("stick", "given", mode=2, concept=str(concept), slots=["torso"])["started"]
    assert started[0][started[0].index("--concept") + 1] == str(concept)
    (output / "sets" / runs.RUN_FILE).unlink()   # that one "finished"
    missing = mcp_server.pieces_run("stick", "given2", mode=2, concept=str(tmp_path / "gone.png"), slots=["torso"])
    assert missing["started"] is False and "concept" in missing["problems"][0]


def _waiting(output: Path) -> Path:
    folder = output / "sets" / "red" / "stick"
    pictures = folder / "pieces" / "boots"
    pictures.mkdir(parents=True)
    scored = {"boots": {"candidates": {s: {"picture": str(pictures / f"s{s}.png"), "score": 0.3 * s, "is_piece": 0.9,
                                           "shape": {"score": 0.5}} for s in (1, 2, 3)},
                        "auto": 3, "picked": 3, "by": "auto", "doubt": False}}
    candidates.write(folder / "pieces", scored, "fake judge")
    (folder / "run_state.json").write_text("{}")
    (folder / "slots.json").write_text("{}")
    return folder


def test_waiting_pictures_and_a_pick(output, started):
    folder = _waiting(output)
    (output / "sets" / runs.RUN_FILE).write_text(json.dumps({
        "pid": 999999, "action": "run", "set": "red", "body": "stick", "log": str(folder / "run.log"),
        "folder": str(folder), "started": 0}))
    status = mcp_server.pieces_status()
    assert status["running"] is False
    boots = status["picks"]["boots"]
    assert boots["auto"] == 3 and boots["candidates"][0]["picture"] == str((folder / "pieces" / "boots" / "s1.png").resolve())
    assert mcp_server.pieces_pick("stick", "red", {"boots": 9})["started"] is False
    assert mcp_server.pieces_pick("stick", "red", {"boots": 2})["started"]
    assert "--resume" in started[0] and started[0][started[0].index("--pick") + 1] == "boots=2"


def test_fit_refit_and_unreal_on_a_fitted_set(output, started):
    folder = output / "sets" / "red" / "stick"
    assert "no fitted pieces" in mcp_server.pieces_unreal("stick", "red")["problems"][0]
    (folder / "fit").mkdir(parents=True)
    (folder / "fit" / "fit.json").write_text("{}")
    (folder / "pieces").mkdir()
    (folder / "pieces" / "pieces.json").write_text("{}")
    assert mcp_server.pieces_refit("stick", "red", "stick")["started"] is False   # onto itself
    for action, call in (("refit", lambda: mcp_server.pieces_refit("stick", "red", "tall")),
                         ("unreal", lambda: mcp_server.pieces_unreal("stick", "red", size_cm=175)),
                         ("fit", lambda: mcp_server.pieces_fit("stick", "red"))):
        (output / "sets" / runs.RUN_FILE).unlink(missing_ok=True)   # the previous one "finished"
        assert call()["started"], action
        command = started[-1]
        assert command[2] == action
    refit, unreal, fit = started
    assert refit[refit.index("--out") + 1] == str(output / "sets" / "red" / "tall" / "fit")
    assert unreal[unreal.index("--size-cm") + 1] == "175"
    assert fit[fit.index("--pieces") + 1] == str(folder / "pieces" / "pieces.json")


def test_cancel_stops_the_recorded_run(output, started, monkeypatch):
    stopped = []
    monkeypatch.setattr(mcp_server, "terminate_group", stopped.append)
    assert mcp_server.pieces_cancel()["stopped"] is False
    mcp_server.pieces_run("stick", "red", outfit="red plate", slots=["helmet"])
    assert mcp_server.pieces_cancel() == {"stopped": True, "pid": os.getpid()}
    assert stopped == [os.getpid()]


def test_bodies_sets_and_slots_are_listed(output, tmp_path):
    listed = mcp_server.pieces_list()
    assert [b["name"] for b in listed["bodies"]] == ["stick", "tall"]
    assert {s["name"] for s in listed["slots"]} >= {"helmet", "weapon"} and listed["sets"] == []


def test_a_body_is_registered_from_files_on_this_machine(output, tmp_path, monkeypatch):
    seen = {}

    def register(name, picture, rigged, views, matte, labels, root, clips, dressing=None):
        seen.update(name=name, picture=picture, rigged=rigged, views=views, labels=labels, clips=clips)
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(output / "bodies" / "stick" / "body.json", folder / "body.json")
        record = json.loads((folder / "body.json").read_text())
        record["name"] = name
        (folder / "body.json").write_text(json.dumps(record))
        return folder / "body.json"

    monkeypatch.setattr(runs.body_mod, "register", register)
    files = {name: tmp_path / f"{name}.x" for name in ("pic", "model", "rig", "walk")}
    for path in files.values():
        path.write_bytes(b"x")
    result = mcp_server.pieces_add_body("elf", str(files["pic"]), str(files["model"]), str(files["rig"]),
                                        "clothes", clips={"walk": str(files["walk"])}, labels={"race": "elf"})
    assert result["name"] == "elf"
    assert seen["views"] == files["model"].with_suffix(".svviews") and seen["clips"] == {"walk": files["walk"]}
    assert mcp_server.pieces_add_body("x", str(tmp_path / "gone"), str(files["model"]), str(files["rig"]),
                                      "clothes")[
        "problem"].endswith("does not exist")


def test_the_new_tools_are_served():
    names = {tool.__name__ for tool in mcp_server.TOOLS}
    assert {"review_queue", "review_item", "check_asset", "review_decide", "pieces_list", "pieces_add_body",
            "pieces_run", "pieces_pick", "pieces_status", "pieces_cancel", "pieces_fit", "pieces_refit",
            "pieces_unreal"} <= names


def test_the_studio_waits_for_a_run_an_agent_started(output, started):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
    import slot_parts_api

    slot_parts_api._JOB.clear()
    assert mcp_server.pieces_run("stick", "red", outfit="red plate", slots=["helmet"])["started"]
    with pytest.raises(ValueError, match="already"):
        slot_parts_api.start_run(output, {"mode": 3, "body": "stick", "set": "blue", "outfit": "o",
                                          "slots": {"helmet": {}}}, popen=lambda *a, **k: Process(a))
