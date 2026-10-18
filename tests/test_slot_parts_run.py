"""`slot_parts.py run`: a whole set on a stick-figure body with sd-cli, Pixal3D and Blender
stood in for; the parts record; weapons and shields made on their own; body clips."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab.slot_parts import body, pieces, sidecar
from tests.test_slot_parts_cli import _body, _concept, cli


def test_held_prompt_is_alone_upright_and_on_grey():
    text = pieces.held_prompt("weapon", style="hand-painted")
    assert text.startswith("a one-handed sword, a single standalone game asset") and "upright" in text
    assert "hand-painted" in text and text.endswith("lighting.")
    assert pieces.held_prompt("shield", "a kite shield").startswith("a kite shield")


def test_the_body_record_keeps_its_clips(tmp_path):
    path, *_ = _body(tmp_path)
    data = body.load(path)
    assert data["clips"] == {}
    data2 = dict(data, clips={"walk": str(tmp_path / "walk.glb")})
    reloaded = body.load(body.write(data2, tmp_path / "bodies"))
    assert reloaded["clips"]["walk"].endswith("walk.glb")


def test_parts_record_names_every_model_and_refuses_an_unknown_mode(tmp_path):
    path, *_ = _body(tmp_path)
    record = body.load(path)
    concept = tmp_path / "c.png"
    Image.new("RGB", (4, 4)).save(concept)
    parts = sidecar.parts_record(mode="body_and_equipment", set_name="red", body=record, body_json=path,
                                 concept=concept, concept_info={"seed": 3}, chosen={"helmet": {}}, pieces={},
                                 fit={"covered_share": 0.7, "files": ["set.glb"], "pieces": {}}, checks={},
                                 matte="birefnet-general-lite")
    ids = [m["id"] for m in parts["models"]]
    assert ids == ["qwen-image-2.1", "pixal3d", "birefnet-general-lite", "skintokens"]
    assert parts["fit"] == {"covered_share": 0.7, "files": ["set.glb"]}
    assert parts["concept"]["sha256"] and parts["concept"]["seed"] == 3
    with pytest.raises(ValueError, match="mode"):
        sidecar.parts_record(mode="whole", set_name="x", body=record, body_json=path, concept=concept,
                             concept_info={}, chosen={}, pieces={}, fit=None, checks={}, matte=None)


def _fakes(concept_source: Path, calls: list):
    def edit(prompt, reference, picture, seed):
        return ["sd-cli", "-r", str(reference), "-o", str(picture), "-p", prompt]

    def generate(prompt, picture, seed):
        return ["sd-cli", "-o", str(picture), "-p", prompt]

    def run(command):
        calls.append(command)
        if command[0] == "sd-cli":
            out = Path(command[command.index("-o") + 1])
            if "-r" not in command:  # a weapon or shield on its own: a white bar on grey
                image = np.full((64, 64, 3), 217, np.uint8)
                image[8:56, 28:36] = 255
                Image.fromarray(image).save(out)
            elif Path(command[command.index("-r") + 1]).name == "body.png":  # the outfit on the body
                shutil.copy(concept_source, out)
            else:  # a piece completed: the cut as it is
                shutil.copy(command[command.index("-r") + 1], out)
        elif command[1].endswith("pixal3d_generate.py"):
            picture, glb = Path(command[2]), Path(command[3])
            rgb = np.asarray(Image.open(picture).convert("RGB"))
            alpha = (np.abs(rgb.astype(int) - 217).sum(-1) > 30).astype(np.uint8) * 255
            Image.fromarray(np.dstack([rgb, alpha])).save(picture.with_name(f"{picture.stem}__matted.png"))
            glb.parent.mkdir(parents=True, exist_ok=True)
            glb.write_bytes(b"glb")
        elif command[1].endswith("retopo_repaint.py"):
            Path(command[4]).write_bytes(b"glb")
    return edit, generate, run


def test_a_whole_set_on_a_stick_figure(tmp_path):
    body_json, cam, heads, _ = _body(tmp_path)
    data = body.load(body_json)
    Image.new("RGB", tuple(data["image_size"]), (217, 217, 217)).save(tmp_path / "body.png")
    data["image"] = str(tmp_path / "body.png")
    body.write(data, tmp_path / "bodies")
    source = _concept(tmp_path, cam, heads)
    calls, jobs = [], []
    edit, generate, run = _fakes(source, calls)

    def fitter(job):
        jobs.append(job)
        return {"pieces": {p["slot"]: {} for p in job["pieces"]}, "covered_share": 0.6, "files": ["set.glb"]}

    path = cli.run_set(body_json, "red", "helmet,torso,boots", outfit="red plate armour", weapon="a longsword",
                       root=tmp_path / "sets", run=run, edit=edit, generate=generate, fitter=fitter,
                       checker=lambda *a: {"helmet": {"status": "pass"}})
    parts = json.loads(path.read_text())
    assert path == tmp_path / "sets" / "red" / "stick" / "red.parts.json"
    assert parts["mode"] == "body_and_equipment" and parts["concept"]["seed"] == 3
    # every chosen piece is drawn: the outfit's words ("red plate armour") name none of them
    assert parts["concept"]["prompt"].startswith("Add red plate armour, with a helmet, a breastplate, boots to the "
                                                 "character in image 1")
    assert set(parts["pieces"]) == {"helmet", "torso", "boots", "weapon"}
    assert parts["pieces"]["weapon"]["held"] is True and "a longsword" in parts["pieces"]["weapon"]["prompt"]
    job = jobs[0]
    assert not any(p["slot"] == "weapon" for p in job["pieces"])   # an asset of its own, not fitted
    assert all("concept_box" in p for p in job["pieces"])
    concept_record = tmp_path / "sets" / "red" / "stick" / "concept" / "s3.png.concept.json"
    assert json.loads(concept_record.read_text())["mode"] == "add_piece"


def test_a_given_concept_is_mode_two_and_needs_no_outfit(tmp_path):
    body_json, cam, heads, _ = _body(tmp_path)
    source = _concept(tmp_path, cam, heads)
    calls = []
    edit, generate, run = _fakes(source, calls)
    path = cli.run_set(body_json, "given", "torso", concept=source, root=tmp_path / "sets", run=run, edit=edit,
                       generate=generate, fitter=lambda job: {"covered_share": 0.5, "files": []},
                       checker=lambda *a: {})
    assert json.loads(path.read_text())["mode"] == "dressed_split"
    assert not any(Path(c[c.index("-o") + 1]).parent.name == "concept" for c in calls if c[0] == "sd-cli")


def test_a_run_can_wait_for_the_person_to_pick_the_pictures(tmp_path):
    body_json, cam, heads, _ = _body(tmp_path)
    source = _concept(tmp_path, cam, heads)
    calls, jobs = [], []
    edit, generate, run = _fakes(source, calls)

    def score(record, cuts_dir, folder, seeds):  # the judge picks seed 2 for the torso
        return ({"torso": {"candidates": {s: {"score": 0.5} for s in seeds}, "auto": 2, "picked": 2, "by": "auto"}},
                "fake judge")

    common = dict(concept=source, seeds=(1, 2), root=tmp_path / "sets", run=run, edit=edit, generate=generate,
                  score=score)
    waiting = cli.run_set(body_json, "given", "torso", stop_at_pictures=True, **common)
    folder = tmp_path / "sets" / "given" / "stick"
    assert waiting == folder / "pieces" / "candidates.json" and (folder / "run_state.json").is_file()
    assert not any(c[1].endswith("pixal3d_generate.py") for c in calls if c[0] != "sd-cli")  # no 3D yet

    def fitter(job):
        jobs.append(job)
        return {"covered_share": 0.5, "files": []}

    path = cli.run_set(body_json, "given", "torso", resume=True, picks={"torso": 1}, fitter=fitter,
                       checker=lambda *a: {}, **common)
    parts = json.loads(path.read_text())
    assert parts["mode"] == "dressed_split" and parts["pieces"]["torso"]["seed"] == 1  # the person's pick
    assert parts["pieces"]["torso"]["picked_by"] == "person" and not (folder / "run_state.json").exists()
    pixal = [c for c in calls if c[0] != "sd-cli" and c[1].endswith("pixal3d_generate.py")]
    assert len(pixal) == 1 and Path(pixal[0][2]).name == "s1.png"
    with pytest.raises(ValueError, match="nothing to resume"):
        cli.run_set(body_json, "given", "torso", resume=True, **common)


def test_a_resumed_run_keeps_the_weapon_and_can_be_resumed_again_after_a_failed_fit(tmp_path):
    """Codex review: the weapon given apart from the slots was lost on resume, and the waiting
    state went before the fit, so a fit that failed could not be resumed."""
    body_json, cam, heads, _ = _body(tmp_path)
    source = _concept(tmp_path, cam, heads)
    calls = []
    edit, generate, run = _fakes(source, calls)

    def score(record, cuts_dir, folder, seeds):
        return ({"torso": {"candidates": {s: {"score": 0.5} for s in seeds}, "auto": 2, "picked": 2, "by": "auto"}},
                "fake judge")

    common = dict(concept=source, root=tmp_path / "sets", run=run, edit=edit, generate=generate, score=score)
    cli.run_set(body_json, "given", "torso", weapon="a longsword", seeds=(1, 2), style="hand-painted",
                stop_at_pictures=True, **common)
    folder = tmp_path / "sets" / "given" / "stick"

    def broken(job):
        raise ValueError("Blender failed")

    with pytest.raises(ValueError, match="Blender failed"):
        cli.run_set(body_json, "given", "torso", resume=True, fitter=broken, checker=lambda *a: {}, **common)
    assert (folder / "run_state.json").is_file()  # still waiting: the pictures and 3D are there

    jobs = []

    def fitter(job):
        jobs.append(job)
        return {"covered_share": 0.5, "files": []}

    path = cli.run_set(body_json, "given", "torso", resume=True, fitter=fitter, checker=lambda *a: {}, **common)
    parts = json.loads(path.read_text())
    assert "weapon" in parts["pieces"] and "a longsword" in parts["pieces"]["weapon"]["prompt"]
    assert "hand-painted" in parts["pieces"]["weapon"]["prompt"]  # the style came back too
    # the weapon is made beside the set, never fitted to the body
    assert not any(p["slot"] == "weapon" for p in jobs[0]["pieces"]) and not (folder / "run_state.json").exists()


def test_run_needs_words_or_a_concept_and_a_plain_set_name(tmp_path):
    body_json, *_ = _body(tmp_path)
    with pytest.raises(ValueError, match="outfit"):
        cli.run_set(body_json, "red", "torso", root=tmp_path)
    with pytest.raises(ValueError, match="set name"):
        cli.run_set(body_json, "../red", "torso", outfit="x", root=tmp_path)


def test_every_fitted_file_gets_a_licence_record_bound_to_it(tmp_path):
    from image_to_3dlab.checks import Asset, run_checks
    from tests._skinned_glb import write as write_skinned_glb

    out = tmp_path / "fit"
    out.mkdir()
    write_skinned_glb(out / "gloves_R.glb")
    write_skinned_glb(out / "body_masked.glb")
    source = tmp_path / "gloves_finished.glb"
    source.write_bytes(b"glb")
    job = {"body": str(tmp_path / "body.json"), "pieces": [{"slot": "gloves", "glb": str(source)}]}
    written = sidecar.fit_provenance(out, job, {"files": ["gloves_R.glb", "body_masked.glb", "missing.glb"]})
    assert [p.name for p in written] == ["gloves_R.provenance.json", "body_masked.provenance.json"]
    record = json.loads(written[0].read_text())
    assert record["source"]["path"] == str(source) and record["source"]["sha256"]
    report = run_checks(Asset(glb=out / "gloves_R.glb"), profile="equipment_part", only=["license.components"])
    # bound and readable; it warns because the pieces come from Qwen pictures (non-commercial)
    assert report["results"][0]["status"] == "warn"
    assert "non-commercial" in report["results"][0]["message"]


def test_a_set_run_is_one_object_and_the_fit_is_steps():
    """Code read 18/10/2026: _finish_set took 18 positional arguments and the Blender fit was one
    function of about 540 lines. A set's run is a SetRun; the fit a Fitter with one method a step."""
    import inspect

    from tests.test_blender_fit_parts import bf

    assert list(inspect.signature(cli._finish_set).parameters) == ["set_run", "made", "held_picks"]
    steps = ("load_body", "load_source_body", "fit_piece", "place_piece", "onto_limb", "shape_and_skin",
             "mask_body", "join_shoulders", "trim_body", "write")
    assert all(callable(getattr(bf.Fitter, step)) for step in steps)
    longest = max(len(inspect.getsource(getattr(bf.Fitter, name)).splitlines())
                  for name, value in vars(bf.Fitter).items() if callable(value))
    assert longest < 90
