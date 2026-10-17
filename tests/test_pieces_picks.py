"""Characters in pieces, the picks left (gap 7 of docs/plans/2026-10-17-step-parity-design.md):
in mode 3 the outfit concept can wait for a pick among several seeds; the weapon and shield
are picked like the drawn pieces; and the studio fits again, carries a set to another body
and exports it for Unreal, as the command line and the MCP server do. sd-cli, Pixal3D and
Blender are stand-ins."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from image_to_3dlab.slot_parts import candidates, runs
from tests.test_slot_parts_cli import _body, _concept, cli
from tests.test_slot_parts_run import _fakes

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is required to execute browser ES modules")


def _with_picture(tmp_path):
    body_json, cam, heads, _ = _body(tmp_path)
    from image_to_3dlab.slot_parts import body

    data = body.load(body_json)
    Image.new("RGB", tuple(data["image_size"]), (217, 217, 217)).save(tmp_path / "body.png")
    data["image"] = str(tmp_path / "body.png")
    body.write(data, tmp_path / "bodies")
    return body_json, _concept(tmp_path, cam, heads)


def _score(record, cuts_dir, folder, seeds):
    return ({"torso": {"candidates": {s: {"score": 0.5} for s in seeds}, "auto": seeds[0], "picked": seeds[0],
                       "by": "auto"}}, "fake judge")


def test_mode_3_waits_for_a_pick_among_the_outfits_and_goes_on_from_it(tmp_path):
    body_json, source = _with_picture(tmp_path)
    calls, jobs = [], []
    edit, generate, run = _fakes(source, calls)
    common = dict(root=tmp_path / "sets", run=run, edit=edit, generate=generate, score=_score)
    waiting = cli.run_set(body_json, "red", "torso", outfit="red plate", concept_seeds=(3, 4, 5),
                          stop_at_concept=True, **common)
    folder = tmp_path / "sets" / "red" / "stick"
    assert waiting == folder / "concept"
    assert sorted(p.name for p in (folder / "concept").glob("s*.png")) == ["s3.png", "s4.png", "s5.png"]
    state = json.loads((folder / "run_state.json").read_text())
    assert state["waiting"] == "concept" and state["concepts"] == [3, 4, 5]
    assert not (folder / "cuts").exists()   # nothing cut yet

    with pytest.raises(ValueError, match="9 is not one of the outfits"):
        cli.run_set(body_json, "red", "torso", resume=True, picks={"concept": 9}, **common)

    def fitter(job):
        jobs.append(job)
        return {"covered_share": 0.5, "files": []}

    drawn_before = sum(1 for c in calls if c[0] == "sd-cli" and "concept" in c[c.index("-o") + 1])
    path = cli.run_set(body_json, "red", "torso", resume=True, picks={"concept": 4}, fitter=fitter,
                       checker=lambda *a: {}, **common)
    parts = json.loads(path.read_text())
    assert parts["concept"]["seed"] == 4 and parts["mode"] == "body_and_equipment"
    assert sum(1 for c in calls if c[0] == "sd-cli" and "concept" in c[c.index("-o") + 1]) == drawn_before
    assert not (folder / "run_state.json").exists()


def test_the_outfit_picked_is_kept_when_the_run_after_it_fails(tmp_path):
    body_json, source = _with_picture(tmp_path)
    calls = []
    edit, generate, run = _fakes(source, calls)
    common = dict(root=tmp_path / "sets", run=run, edit=edit, generate=generate, score=_score)
    cli.run_set(body_json, "red", "torso", outfit="red plate", concept_seeds=(3, 4, 5), stop_at_concept=True,
                **common)

    def broken(job):
        raise RuntimeError("Blender crashed")

    with pytest.raises(RuntimeError):
        cli.run_set(body_json, "red", "torso", resume=True, picks={"concept": 5}, fitter=broken, **common)
    folder = tmp_path / "sets" / "red" / "stick"
    assert json.loads((folder / "run_state.json").read_text())["picked"] == 5
    output = tmp_path   # waiting_picks gives paths relative to the folder above sets/
    assert runs.waiting_picks(output, folder)["concept"]["picked"] == 5
    path = cli.run_set(body_json, "red", "torso", resume=True, fitter=lambda job: {"covered_share": 0.5, "files": []},
                       checker=lambda *a: {}, **common)
    assert json.loads(path.read_text())["concept"]["seed"] == 5   # no pick given: the one picked before


def test_a_weapon_alone_offers_only_its_own_pictures(tmp_path):
    body_json, source = _with_picture(tmp_path)
    calls = []
    edit, generate, run = _fakes(source, calls)
    folder = tmp_path / "sets" / "given" / "stick"
    candidates.write(folder / "pieces", {"boots": {"candidates": {"1": {"picture": "old.png", "score": 0.9}},
                                                   "auto": 1, "picked": 1}}, "old judge")   # an earlier run's
    cli.run_set(body_json, "given", {}, concept=source, weapon="a longsword", seeds=(1, 2), stop_at_pictures=True,
                root=tmp_path / "sets", run=run, edit=edit, generate=generate, score=_score)
    assert set(candidates.load(folder / "pieces")["slots"]) == {"weapon"}


def test_a_weapon_alone_can_be_fitted_again(tmp_path):
    body_json, source = _with_picture(tmp_path)
    folder = tmp_path / "sets" / "given" / "stick"
    (folder / "pieces").mkdir(parents=True)
    pieces_json = folder / "pieces" / "pieces.json"
    pieces_json.write_text(json.dumps({"cuts": str(folder / "cuts" / "cuts.json"), "seeds": [1],
                                       "slots": {"weapon": {"glb": str(folder / "pieces" / "weapon.glb")}}}))
    job = cli.fit_job(pieces_json, body_json, folder / "fit")   # no cuts: nothing was drawn on the body
    assert [p["slot"] for p in job["pieces"]] == ["weapon"]


def test_a_carried_set_is_listed_on_its_new_body(tmp_path, monkeypatch):
    from image_to_3dlab.slot_parts import body as body_mod

    body_json, source = _with_picture(tmp_path)
    record = body_mod.load(body_json)
    record["name"] = "tall"
    tall_json = body_mod.write(record, tmp_path / "bodies")
    folder = tmp_path / "sets" / "red" / "stick"
    (folder / "fit").mkdir(parents=True)
    (folder / "fit" / "fit.json").write_text(json.dumps({"files": ["torso.glb"], "covered_share": 0.5}))
    (folder / "red.parts.json").write_text(json.dumps({"set": "red", "mode": "body_and_equipment",
                                                       "body": {"name": "stick"}, "pieces": {"torso": {}},
                                                       "created": "2026-10-17T10:00:00",
                                                       "fit": {"covered_share": 0.5}, "checks": {"torso": {}}}))
    monkeypatch.setattr(cli, "refit_job", lambda *a: {"out": str(tmp_path / "sets" / "red" / "tall" / "fit")})
    monkeypatch.setattr(cli, "run_fit", lambda job, blender=None: {"pieces": ["torso"], "covered_share": 0.6,
                                                                  "files": ["torso.glb"]})
    assert cli.main(["refit", "--fit", str(folder / "fit"), "--from", str(body_json), "--body", str(tall_json)]) == 0
    carried = json.loads((tmp_path / "sets" / "red" / "tall" / "red.parts.json").read_text())
    assert carried["body"]["name"] == "tall" and carried["carried_from"] == "stick"
    assert carried["fit"]["covered_share"] == 0.6 and carried["checks"] == {}   # not checked on this body yet
    assert [(s["set"], s["body"]) for s in runs.list_sets(tmp_path)] == [("red", "stick"), ("red", "tall")] or \
        sorted((s["set"], s["body"]) for s in runs.list_sets(tmp_path)) == [("red", "stick"), ("red", "tall")]


def test_a_concept_pick_can_lead_to_a_wait_for_the_pieces(tmp_path):
    body_json, source = _with_picture(tmp_path)
    calls = []
    edit, generate, run = _fakes(source, calls)
    common = dict(root=tmp_path / "sets", run=run, edit=edit, generate=generate, score=_score, seeds=(1, 2))
    cli.run_set(body_json, "red", "torso", outfit="red plate", concept_seeds=(3, 4), stop_at_concept=True,
                stop_at_pictures=True, **common)
    waiting = cli.run_set(body_json, "red", "torso", resume=True, picks={"concept": 3}, **common)
    folder = tmp_path / "sets" / "red" / "stick"
    assert waiting == folder / "pieces" / "candidates.json"
    assert json.loads((folder / "run_state.json").read_text())["waiting"] == "picks"


def test_the_weapon_is_picked_like_the_pieces(tmp_path):
    body_json, source = _with_picture(tmp_path)
    calls, jobs = [], []
    edit, generate, run = _fakes(source, calls)
    common = dict(concept=source, seeds=(1, 2, 3), root=tmp_path / "sets", run=run, edit=edit, generate=generate,
                  score=_score)
    cli.run_set(body_json, "given", "torso", weapon="a longsword", stop_at_pictures=True, **common)
    folder = tmp_path / "sets" / "given" / "stick"
    offered = candidates.load(folder / "pieces")["slots"]
    assert set(offered) == {"torso", "weapon"}
    assert sorted(offered["weapon"]["candidates"], key=int) == ["1", "2", "3"] or \
        sorted(map(int, offered["weapon"]["candidates"])) == [1, 2, 3]
    def weapon_drawn():   # the folder, not the path: pytest's folder is named after the test
        return [c for c in calls if c[0] == "sd-cli" and Path(c[c.index("-o") + 1]).parent.name == "weapon"]

    drawn = weapon_drawn()
    assert len(drawn) == 3

    def fitter(job):
        jobs.append(job)
        return {"covered_share": 0.5, "files": []}

    path = cli.run_set(body_json, "given", "torso", resume=True, picks={"weapon": 2}, fitter=fitter,
                       checker=lambda *a: {}, **common)
    parts = json.loads(path.read_text())
    assert parts["pieces"]["weapon"]["seed"] == 2
    assert len(weapon_drawn()) == 3   # not redrawn
    pixal = [c for c in calls if c[0] != "sd-cli" and c[1].endswith("pixal3d_generate.py")]
    assert any(Path(c[2]).name == "s2.png" and Path(c[2]).parent.name == "weapon" for c in pixal)


def test_the_command_line_takes_the_concept_pause():
    args = cli.build_parser().parse_args(["run", "--body", "b.json", "--set", "red", "--stop-at-concept",
                                          "--concept-seeds", "3,4"])
    assert args.stop_at_concept and args.concept_seeds == "3,4"


# ------------------------------------------------------------- studio and agents

@pytest.fixture
def output(tmp_path):
    from tests.test_viewer_slot_parts import _body_json

    out = tmp_path / "output"
    out.mkdir()
    _body_json(out)
    return out


def test_the_studio_and_agents_can_ask_for_the_concept_pause(output):
    command, _ = runs.run_command(output, {"mode": 3, "body": "stick", "set": "red", "outfit": "red plate",
                                           "slots": {"torso": {}}, "concept_seeds": "3,4,5", "choose_concept": True})
    assert "--stop-at-concept" in command and command[command.index("--concept-seeds") + 1] == "3,4,5"
    plain, _ = runs.run_command(output, {"mode": 3, "body": "stick", "set": "red", "outfit": "red plate",
                                         "slots": {"torso": {}}})
    assert "--stop-at-concept" not in plain


def test_the_waiting_outfits_are_offered_and_a_pick_resumes(output):
    folder = output / "sets" / "red" / "stick"
    (folder / "concept").mkdir(parents=True)
    for seed in (3, 4):
        Image.new("RGB", (8, 8)).save(folder / "concept" / f"s{seed}.png")
    (folder / "slots.json").write_text("{}")
    (folder / "run_state.json").write_text(json.dumps({"waiting": "concept", "concepts": [3, 4]}))
    waiting = runs.waiting_picks(output, folder)
    assert [c["picture"] for c in waiting["concept"]["candidates"]] == ["sets/red/stick/concept/s3.png",
                                                                        "sets/red/stick/concept/s4.png"]
    command, _ = runs.resume_command(output, {"body": "stick", "set": "red", "picks": {"concept": 4}})
    assert command[command.index("--pick") + 1] == "concept=4"
    with pytest.raises(ValueError, match="not one of its pictures"):
        runs.resume_command(output, {"body": "stick", "set": "red", "picks": {"concept": 9}})


def test_the_studio_page_fits_refits_and_exports_a_set(output, monkeypatch):
    import slot_parts_api  # noqa: F401 - viewer is on the path from the fixture's module

    folder = output / "sets" / "red" / "stick"
    (folder / "fit").mkdir(parents=True)
    (folder / "fit" / "fit.json").write_text("{}")
    started = []

    class Process:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def popen(command, **kwargs):
        started.append(command)
        return Process()

    slot_parts_api._JOB.clear()
    status = slot_parts_api.start_run(output, {"action": "unreal", "body": "stick", "set": "red", "size_cm": 175},
                                      popen=popen)
    assert status["running"] and started[0][2] == "unreal"
    slot_parts_api._JOB.clear()
    (output / "sets" / runs.RUN_FILE).unlink()
    with pytest.raises(ValueError, match="action"):
        slot_parts_api.start_run(output, {"action": "explode", "body": "stick", "set": "red"}, popen=popen)
    outside = output.parent / "elsewhere.png"
    Image.new("RGB", (8, 8)).save(outside)
    for action in (None, "", False, "run"):   # however "run" is said, the concept must be under output/
        with pytest.raises(ValueError, match="output folder"):
            slot_parts_api.start_run(output, {"action": action, "mode": 2, "body": "stick", "set": "red",
                                              "concept": str(outside), "slots": {"torso": {}}}, popen=popen)


@needs_node
def test_the_pieces_page_shows_outfits_to_pick_and_set_actions():
    url = (REPO / "viewer" / "studio" / "slot-parts.js").as_uri()
    program = (f"import * as m from {json.dumps(url)};\n"
               "const picks = { concept: { auto: 3, picked: 3, candidates: [{ seed: 3, picture: 'sets/r/b/concept/s3.png', score: null }] },"
               " weapon: { auto: 1, picked: 1, candidates: [{ seed: 1, picture: 'sets/r/b/pieces/weapon/s1.png', score: null }] } };\n"
               "const html = m.pickRows(picks);\n"
               "const set = m.setActions({ set: 'red', body: 'stick' }, [{ name: 'stick' }, { name: 'elf' }]);\n"
               "const payload = m.runPayload({ mode: 3, body: 'stick', set: 'red', outfit: 'red plate', slots: { torso: { on: true } }, conceptSeeds: '3,4', chooseConcept: true });\n"
               "const plain = m.runPayload({ mode: 3, body: 'stick', set: 'red', outfit: 'red plate', slots: { torso: { on: true } }, conceptSeeds: '', chooseConcept: true });\n"
               "console.log(JSON.stringify({ html, set, payload, plain }));")
    result = subprocess.run([NODE, "--input-type=module", "--eval", program], check=True, capture_output=True,
                            text=True, encoding="utf-8")
    out = json.loads(result.stdout)
    assert "Outfit" in out["html"] and "NaN" not in out["html"] and "seed 3" in out["html"]
    assert 'data-set-action="unreal"' in out["set"] and 'data-set-action="refit"' in out["set"]
    assert '<option value="elf">' in out["set"] and '<option value="stick">' not in out["set"]
    assert out["payload"]["choose_concept"] is True and out["payload"]["concept_seeds"] == "3,4"
    assert "concept_seeds" not in out["plain"]   # left empty: the outfit takes the first picture seed, as before


# ------------------------------------------------------------- the concept names every chosen piece

def test_the_outfit_names_every_chosen_piece():
    from image_to_3dlab.slot_parts import pieces

    chosen = {"helmet": {}, "torso": {}, "boots": {}, "gloves": {"prompt": "leather bracers"}}
    words = pieces.concept_outfit("crimson and gold plate armour with armoured boots", chosen)
    assert words.startswith("crimson and gold plate armour with armoured boots")
    assert "a helmet" in words and "leather bracers" in words   # the pieces the outfit left out
    assert words.count("boots") == 1   # named already
    assert pieces.concept_outfit("a red robe", {}) == "a red robe"
    # "armour" is the whole outfit, not a breastplate
    assert "a breastplate" in pieces.concept_outfit("leg armour and boots", {"torso": {}, "boots": {}})


def test_the_concept_is_drawn_with_the_chosen_helmet(tmp_path):
    body_json, source = _with_picture(tmp_path)
    calls = []
    edit, generate, run = _fakes(source, calls)
    cli.run_set(body_json, "red", "helmet,torso", outfit="red plate armour", concept_seeds=(3, 4),
                stop_at_concept=True, root=tmp_path / "sets", run=run, edit=edit, generate=generate, score=_score)
    drawn = [c for c in calls if c[0] == "sd-cli" and Path(c[c.index("-o") + 1]).parent.name == "concept"]
    assert drawn and all("a helmet" in c[c.index("-p") + 1] for c in drawn)
