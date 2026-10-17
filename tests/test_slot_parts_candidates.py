"""Several pictures per piece: scoring them, the automatic pick, the person's pick."""

import numpy as np
import pytest

from image_to_3dlab.slot_parts import candidates


def _box(size, x0, y0, x1, y1):
    mask = np.zeros((size, size), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_the_shape_counts_cover_and_what_lies_well_outside_the_slot():
    slot = _box(100, 40, 40, 60, 60)
    same = candidates.shape_score(slot, slot)
    assert same == {"cover": 1.0, "spill": 0.0, "score": 1.0}
    grown = candidates.shape_score(_box(100, 37, 37, 63, 63), slot)  # completed a little past the cut
    assert grown["spill"] == 0.0 and grown["score"] == 1.0
    elsewhere = candidates.shape_score(_box(100, 0, 0, 20, 20), slot)  # a piece drawn somewhere else
    assert elsewhere["cover"] == 0.0 and elsewhere["score"] == 0.0
    half = candidates.shape_score(_box(100, 40, 40, 50, 60), slot)
    assert half["cover"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="slot mask"):
        candidates.shape_score(np.zeros((10, 10), bool), slot)


def test_the_judge_leads_and_the_shape_only_breaks_ties():
    assert candidates.combined(1.0, 0.9) == 0.9 and candidates.combined(0.0, 0.987) == 0.99
    assert candidates.combined(0.7, None) == 0.7  # no judge: the shape alone
    copied = {"score": 0.99, "shape": {"score": 0.99}}  # a pauldron that kept its chest plate: copies the cut
    alone = {"score": 1.0, "shape": {"score": 0.58}}
    assert candidates.best({2: copied, 3: alone}) == 3
    assert candidates.best({1: {"score": 1.0, "shape": {"score": 0.63}}, 3: {"score": 1.0, "shape": {"score": 0.99}}}) == 3
    assert candidates.best({3: {"score": 0.8}, 1: {"score": 0.8}}) == 1


def test_a_glove_drawn_for_the_boot_slot_loses_to_a_boot(tmp_path, monkeypatch):
    slot = _box(64, 20, 20, 44, 44)
    pictures = {seed: tmp_path / f"s{seed}.png" for seed in (1, 2)}
    answers = {pictures[1]: 0.1, pictures[2]: 0.95}  # both fill the slot alike; only seed 2 is a boot
    monkeypatch.setattr(candidates, "outline", lambda picture, cut_out=None: slot)

    def ask(picture, text):
        assert "one single boot" in text
        return answers[picture]

    result = candidates.score_slot("boots", slot, pictures, ask=ask)
    assert result["auto"] == 2 and result["picked"] == 2 and result["by"] == "auto"
    assert result["candidates"][1]["is_piece"] == 0.1 and result["doubt"] is False
    answers.update({pictures[2]: 0.2})  # none of them is a boot: the pick is flagged
    assert candidates.score_slot("boots", slot, pictures, ask=ask)["doubt"] is True


def test_the_person_can_change_a_pick_to_another_candidate(tmp_path):
    scored = {"boots": {"candidates": {1: {"score": 0.2}, 2: {"score": 0.9}}, "auto": 2, "picked": 2, "by": "auto"}}
    candidates.write(tmp_path, scored, judge="qwen3-vl")
    assert candidates.picks(tmp_path) == {"boots": 2}
    data = candidates.choose(tmp_path, {"boots": 1})
    assert data["slots"]["boots"]["by"] == "person" and candidates.picks(tmp_path) == {"boots": 1}
    with pytest.raises(ValueError, match="not one of"):
        candidates.choose(tmp_path, {"boots": 7})
    with pytest.raises(ValueError, match="no candidates"):
        candidates.choose(tmp_path, {"helmet": 1})


def test_the_question_names_the_piece_of_each_drawn_slot():
    from image_to_3dlab.slot_parts import slots

    for slot in slots.DRAWN:
        assert candidates.NOUNS[slot] in candidates.question(slot) or slot in candidates.QUESTIONS
    assert "chest plate" in candidates.question("shoulders")


def test_the_slot_mask_sits_where_the_canvas_holds_the_cut(tmp_path):
    from PIL import Image

    cut = np.zeros((10, 20, 4), np.uint8)
    cut[..., 3] = 255
    Image.fromarray(cut, "RGBA").save(tmp_path / "cut_boots.png")
    entry = {"cut": "cut_boots.png", "placement": {"box": [0, 0, 20, 10], "scale": 2.0, "pad": [12, 22], "size": 64}}
    mask = candidates.slot_mask(tmp_path, entry)
    assert mask.shape == (64, 64) and mask.sum() == 40 * 20
    ys, xs = np.nonzero(mask)
    assert (xs.min(), ys.min(), xs.max(), ys.max()) == (12, 22, 51, 41)
