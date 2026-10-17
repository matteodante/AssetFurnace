"""Route C pieces: the edit jobs, the 3D commands, the placement back on the concept, and a
whole `pieces` run with sd-cli and Pixal3D stood in for."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab.slot_parts import cuts, pieces
from tests.test_slot_parts_cli import cli


def _cut_record(tmp_path, routes=None):
    cuts_dir = tmp_path / "set" / "cuts"
    cuts_dir.mkdir(parents=True)
    record = {"slots": {}}
    piece = np.full((20, 10, 4), 255, np.uint8)
    for slot, box in (("helmet", (40, 0, 49, 19)), ("boots", (30, 80, 39, 99))):
        canvas, place = cuts.flatten(piece, box, size=64)
        Image.fromarray(canvas).save(cuts_dir / f"{slot}.png")
        record["slots"][slot] = {"route": (routes or {}).get(slot, "C"), "split": False, "prompt": "",
                                 "box": list(box), "placement": place.to_dict(), "picture": f"{slot}.png"}
    (cuts_dir / "cuts.json").write_text(json.dumps(record))
    return cuts_dir / "cuts.json", record


def test_one_edit_per_route_c_slot_and_seed(tmp_path):
    path, record = _cut_record(tmp_path, routes={"boots": "B0"})
    jobs = pieces.picture_jobs(record, path.parent, tmp_path / "pieces", [3, 7])
    assert [(j.slot, j.seed) for j in jobs] == [("helmet", 3), ("helmet", 7)]
    assert jobs[0].picture == tmp_path / "pieces" / "helmet" / "s3.png"
    assert jobs[0].canvas == path.parent / "helmet.png"
    assert "a closed helmet" in jobs[0].prompt and "nothing inside" in jobs[0].prompt


def test_a_persons_words_replace_the_default(tmp_path):
    path, record = _cut_record(tmp_path)
    record["slots"]["helmet"]["prompt"] = "a horned viking helmet"
    jobs = pieces.picture_jobs(record, path.parent, tmp_path, [1], style="hand-painted")
    assert "a horned viking helmet" in jobs[0].prompt and "hand-painted" in jobs[0].prompt


def test_no_seeds_no_jobs(tmp_path):
    _, record = _cut_record(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        pieces.picture_jobs(record, tmp_path, tmp_path, [])


def test_model_commands_run_pixal3d_finish_to_the_budget_then_winding(tmp_path):
    first, finish, winding = pieces.model_commands(tmp_path / "s3.png", tmp_path, "boots", "py", Path("scripts"))
    assert first[1].endswith("pixal3d_generate.py") and first[-1].endswith("boots.glb")
    assert finish[finish.index("--faces") + 1] == "4000"
    assert finish[finish.index("--views") + 1].endswith("boots.svviews") and "--skip-paint" in finish
    assert winding[1].endswith("fix_winding.py") and winding[2].endswith("boots_finished.glb")


def test_the_piece_box_comes_back_onto_the_concept():
    piece = np.full((20, 10, 4), 255, np.uint8)
    canvas, place = cuts.flatten(piece, (40, 0, 49, 19), size=64)
    alpha = (canvas == 255).all(-1)
    assert pieces.concept_box(alpha, place) == pytest.approx([40, 0, 49, 19], abs=1.0)


@pytest.mark.parametrize("text, picks", [("", {}), ("helmet=7, boots=3", {"helmet": 7, "boots": 3})])
def test_picks_parse(text, picks):
    assert pieces.parse_picks(text) == picks


@pytest.mark.parametrize("text", ["helmet", "cape=3", "helmet=x"])
def test_bad_picks_are_refused(text):
    with pytest.raises(ValueError, match="slot=seed"):
        pieces.parse_picks(text)


def _fakes(calls):
    def edit(prompt, canvas, picture, seed):
        return ["sd-cli", "--diffusion-model", "qwen-image-2.1-turbo-Q8_0.gguf", "-r", str(canvas), "-o", str(picture)]

    def run(command):
        calls.append(command)
        if command[0] == "sd-cli":  # the edit: a picture with the piece in the middle
            out = Path(command[command.index("-o") + 1])
            Image.open(command[command.index("-r") + 1]).save(out)
        elif command[1].endswith("pixal3d_generate.py"):  # Pixal3D mattes the picture beside it
            picture, glb = Path(command[2]), Path(command[3])
            rgb = np.asarray(Image.open(picture).convert("RGB"))
            alpha = ((rgb == 255).all(-1) * 255).astype(np.uint8)
            Image.fromarray(np.dstack([rgb, alpha])).save(picture.with_name(f"{picture.stem}__matted.png"))
            glb.parent.mkdir(parents=True, exist_ok=True)
            glb.write_bytes(b"glb")
        elif command[1].endswith("retopo_repaint.py"):
            Path(command[4]).write_bytes(b"glb")
    return edit, run


def test_a_whole_pieces_run(tmp_path):
    path, _ = _cut_record(tmp_path)
    calls = []
    edit, run = _fakes(calls)
    result = cli.make_pieces(path, [3, 7], picks={"boots": 7}, run=run, edit=edit)
    folder = tmp_path / "set" / "pieces"
    assert (folder / "helmet" / "s7.png").is_file()
    assert (folder / "helmet" / "s3.png.concept.json").is_file()
    assert json.loads((folder / "helmet" / "s3.png.concept.json").read_text())["mode"] == "complete_piece"
    assert result["slots"]["helmet"]["seed"] == 3 and result["slots"]["boots"]["seed"] == 7
    assert result["slots"]["boots"]["concept_box"] == pytest.approx([30, 80, 39, 99], abs=1.0)
    assert json.loads((folder / "pieces.json").read_text())["slots"]["helmet"]["glb"].endswith("helmet_finished.glb")
    tools = [Path(c[1]).name if c[0] == sys.executable else c[0] for c in calls]
    assert tools.count("sd-cli") == 4 and tools.count("fix_winding.py") == 2


def test_images_only_stops_before_3d_and_models_only_needs_the_pictures(tmp_path):
    path, _ = _cut_record(tmp_path)
    calls = []
    edit, run = _fakes(calls)
    cli.make_pieces(path, [3], models=False, run=run, edit=edit)
    assert all(c[0] == "sd-cli" for c in calls)
    with pytest.raises(ValueError, match="no picture for seed 9"):
        cli.make_pieces(path, [9], images=False, run=run, edit=edit)


def test_a_pick_for_a_slot_that_was_not_cut_is_refused(tmp_path):
    path, _ = _cut_record(tmp_path)
    with pytest.raises(ValueError, match="gloves"):
        cli.make_pieces(path, [3], picks={"gloves": 3}, run=lambda c: None, edit=lambda *a: [])


def test_fit_job_carries_each_piece_box_and_the_concept_size(tmp_path):
    path, _ = _cut_record(tmp_path)
    concept = tmp_path / "concept.png"
    Image.new("RGB", (512, 512)).save(concept)
    record = json.loads(path.read_text())
    record["concept"] = str(concept)
    path.write_text(json.dumps(record))
    pieces_json = tmp_path / "pieces.json"
    pieces_json.write_text(json.dumps({"cuts": str(path), "slots": {
        "helmet": {"glb": "h.glb", "concept_box": [1, 2, 3, 4]}}}))
    from image_to_3dlab.slot_parts import body as body_mod
    from tests.test_slot_parts_cli import _body

    body_json, *_ = _body(tmp_path)
    data = body_mod.load(body_json)
    data["clips"] = {"walk": str(tmp_path / "walk.glb")}
    body_mod.write(data, tmp_path / "bodies")
    job = cli.fit_job(pieces_json, body_json, tmp_path / "fit", {"offset": 0.003})
    assert job["concept_size"] == [512, 512] and job["offset"] == 0.003
    assert job["clips"] == [str((tmp_path / "walk.glb").resolve())]  # the mask is checked in the body's poses
    assert job["pieces"] == [{"slot": "helmet", "glb": "h.glb", "split": False, "concept_box": [1, 2, 3, 4]}]
    record["slots"]["shoulders"] = {"split": True}  # the person's two shoulder pieces reach the fit
    path.write_text(json.dumps(record))
    pieces_json.write_text(json.dumps({"cuts": str(path), "slots": {"shoulders": {"glb": "s.glb", "concept_box": [1, 2, 3, 4]}}}))
    assert cli.fit_job(pieces_json, body_json, tmp_path / "fit")["pieces"][0]["split"] is True
    pieces_json.write_text(json.dumps({"cuts": str(path), "slots": {}}))
    with pytest.raises(ValueError, match="no pieces"):
        cli.fit_job(pieces_json, body_json, tmp_path / "fit")


def test_run_fit_writes_the_job_and_runs_blender_headless(tmp_path):
    job = {"body": "b.json", "concept_size": [8, 8], "pieces": [], "out": str(tmp_path / "fit")}
    seen = []

    def run(command):
        seen.append(command)
        (tmp_path / "fit" / "fit.json").write_text(json.dumps({"pieces": {}, "covered_share": 0.5}))

    report = cli.run_fit(job, blender=Path("blender.exe"), run=run)
    assert report["covered_share"] == 0.5
    command = seen[0]
    assert command[:4] == ["blender.exe", "-b", "--factory-startup", "--python-exit-code"]
    assert command[-1] == str(tmp_path / "fit" / "job.json")
    assert json.loads((tmp_path / "fit" / "job.json").read_text())["body"] == "b.json"



def test_b0_commands_make_the_whole_concept_once(tmp_path):
    pixal, finish = pieces.b0_commands(tmp_path / "concept.png", tmp_path, "py", Path("scripts"))
    assert pixal[-1].endswith("concept.glb") and finish[finish.index("--faces") + 1] == "40000"
    assert finish[finish.index("--views") + 1].endswith("concept.svviews")


def test_b0_slots_are_cut_from_the_whole_model(tmp_path):
    import json as _json

    path, record = _cut_record(tmp_path, routes={"boots": "B0"})
    concept = tmp_path / "concept.png"
    picture = np.full((100, 100, 3), 217, np.uint8)
    picture[20:80, 40:60] = 255  # the figure, which the stand-in Pixal3D cuts out as white
    Image.fromarray(picture).save(concept)
    labels = np.zeros((100, 100), np.uint8)
    Image.fromarray(labels).save(path.parent / "slots.png")
    record.update(concept=str(concept), body=str(tmp_path / "body.json"), cuts_dir=str(path.parent))
    path.write_text(_json.dumps(record))
    calls = []
    edit, run = _fakes(calls)

    def run_b0(command):
        run(command)
        name = Path(command[1]).name if len(command) > 1 else ""
        if name == "pixal3d_generate.py" and command[2].endswith("concept.png"):
            views = Path(command[3]).with_suffix(".svviews")
            views.mkdir(parents=True, exist_ok=True)
            alpha = np.zeros((50, 50, 4), np.uint8)
            alpha[10:40, 20:30] = 255
            Image.fromarray(alpha).save(views / "input.png")
            matrix = np.eye(4)
            matrix[2, 3] = 2.8
            (views / "transforms.json").write_text(_json.dumps({"camera_angle_x": 0.349, "frames": [
                {"file_path": "input.png", "transform_matrix": matrix.tolist()}]}))
        if command[-2:-1] == ["--"] and command[-1].endswith("job.json"):  # Blender: the split
            job = _json.loads(Path(command[-1]).read_text())
            out = Path(job["out"])
            (out / "boots.glb").write_bytes(b"glb")
            (out / "segment.json").write_text(_json.dumps({"files": {"boots": str(out / "boots.glb")},
                                                          "frame": [-0.4, -0.45, 0.4, 0.45, 0.0], "faces": {}}))

    result = cli.make_pieces(path, [3], run=run_b0, edit=edit)
    boots = result["slots"]["boots"]
    assert boots["route"] == "B0" and boots["frame"] == [-0.4, -0.45, 0.4, 0.45, 0.0]
    assert boots["glb"].endswith("boots.glb") and len(boots["concept_box"]) == 4
    assert result["slots"]["helmet"]["seed"] == 3  # route C slots are made as before
    job = _json.loads((tmp_path / "set" / "pieces" / "b0" / "job.json").read_text())
    assert job["chosen"] == ["boots"] and Path(job["slot_map"]).suffix == ".npy"
