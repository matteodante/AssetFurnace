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
