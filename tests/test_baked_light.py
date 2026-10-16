"""The learned baked-light check: synthetic pairs, a small head on frozen embeddings, the verdict.

Everything here is pure: views are made up, embeddings are made up. The real render and
SigLIP 2 run in scripts/baked_light_trial.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab import baked_light as bl
from image_to_3dlab import materials as m
from image_to_3dlab import multiview as mv

SIZE = 32


def view_with_light(shade: np.ndarray, albedo: int = 128) -> mv.View:
    """A flat-albedo view whose studio picture is the albedo times `shade` (linear)."""
    colour = np.full((SIZE, SIZE, 3), albedo, np.uint8)
    material = np.zeros((SIZE, SIZE), np.int64)
    material[:4] = -1                                            # a strip of background
    lit_linear = m.srgb_to_linear(colour / 255.0) * shade[..., None]
    lit = np.round(m.linear_to_srgb(lit_linear) * 255).astype(np.uint8)
    return mv.View(0, np.array([0.0, 0.0, -1.0]), np.zeros((SIZE, SIZE, 2)), material,
                   np.tile([0.0, 0.0, 1.0], (SIZE, SIZE, 1)), colour, lit)


def ramp() -> np.ndarray:
    return np.tile(np.linspace(0.25, 1.0, SIZE), (SIZE, 1))


def test_baking_paints_the_lights_shading_into_the_albedo_relative_to_its_median():
    view = view_with_light(ramp())
    baked = bl.bake_views([view], strength=1.0)[0]
    lin = m.luminance(m.srgb_to_linear(baked / 255.0))
    base = m.luminance(m.srgb_to_linear(view.color / 255.0))
    ratio = lin[10] / base[10]
    shown = ramp()[10]
    np.testing.assert_allclose(ratio, shown / np.median(ramp()[4:]), rtol=0.06)


def test_no_strength_means_no_change_and_the_background_is_never_touched():
    view = view_with_light(ramp())
    np.testing.assert_array_equal(bl.bake_views([view], strength=0.0)[0], view.color)
    baked = bl.bake_views([view], strength=0.8)[0]
    np.testing.assert_array_equal(baked[:4], view.color[:4])


def test_a_picture_puts_the_object_on_grey_and_the_normal_as_colour():
    view = view_with_light(ramp())
    picture = np.asarray(bl.picture(view.color, view.material))
    assert tuple(picture[0, 0]) == bl.GREY and tuple(picture[20, 20]) == (128, 128, 128)
    normal = np.asarray(bl.normal_picture(view))
    assert tuple(normal[20, 20]) == (128, 128, 255) and tuple(normal[0, 0]) == bl.GREY


def test_the_head_learns_a_separable_problem_and_survives_a_round_trip():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (40, 5)) + [2, 0, 0, 0, 0], rng.normal(0, 1, (40, 5)) - [2, 0, 0, 0, 0]])
    y = np.r_[np.ones(40), np.zeros(40)]
    head = bl.LogisticHead.fit(x, y, l2=1.0)
    assert bl.roc_auc(y, head.score(x)) > 0.97
    again = bl.LogisticHead.from_json(json.loads(json.dumps(head.to_json())))
    np.testing.assert_allclose(again.score(x), head.score(x))


def test_auc_counts_ties_as_half():
    assert bl.roc_auc(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert bl.roc_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.5, 0.1])) == pytest.approx(0.75)


def test_the_bootstrap_resamples_whole_assets_and_is_repeatable():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.8, 0.3, 0.2, 0.7])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    low = bl.auc_lower_bound(y, s, groups, seed=1)
    assert low == bl.auc_lower_bound(y, s, groups, seed=1)
    assert 0.0 <= low <= bl.roc_auc(y, s)


def test_the_split_never_puts_one_asset_on_two_sides():
    names = [f"asset_{i}" for i in range(40)]
    split = bl.split_assets(names)
    sides = [set(split[k]) for k in ("train", "calibration", "test")]
    assert sum(len(s) for s in sides) == 40
    assert not (sides[0] & sides[1]) and not (sides[0] & sides[2]) and not (sides[1] & sides[2])
    assert bl.split_assets(list(reversed(names))) == split


def test_the_threshold_is_chosen_on_calibration_to_catch_four_in_five():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.5, 0.3, 0.1, 0.05, 0.0])
    threshold = bl.choose_threshold(y, s)
    assert 0.5 < threshold <= 0.6                     # 4 of 5 bad caught, the fewest good flagged
    caught, flagged = bl.rates(y, s, threshold)
    assert caught == pytest.approx(0.8) and flagged == pytest.approx(0.0)


def test_the_verdict_needs_every_number_and_human_labels():
    good = bl.verdict(auc_low=0.93, caught=0.85, flagged=0.05, human=True)
    assert good["admitted"] is True
    assert bl.verdict(auc_low=0.93, caught=0.85, flagged=0.05, human=False)["admitted"] is False
    assert bl.verdict(auc_low=0.88, caught=0.85, flagged=0.05, human=True)["admitted"] is False
    assert bl.verdict(auc_low=0.93, caught=0.75, flagged=0.05, human=True)["admitted"] is False
    assert bl.verdict(auc_low=0.93, caught=0.85, flagged=0.12, human=True)["admitted"] is False


def _trial_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "baked_light_trial.py"
    spec = importlib.util.spec_from_file_location("baked_light_trial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_synthetic_sun_comes_from_above_with_a_strength_in_range_and_is_repeatable():
    trial = _trial_script()
    for name in ("crowbar_01", "dartboard", "metal_jug"):
        for k in range(2):
            travel, strength = trial.light_for(name, k)
            assert travel[2] < -0.4 and np.linalg.norm(travel) == pytest.approx(1.0, abs=1e-5)
            assert 0.35 <= strength <= 1.0
            assert trial.light_for(name, k) == (travel, strength)
    assert trial.light_for("crowbar_01", 0) != trial.light_for("crowbar_01", 1)


def test_the_evaluation_takes_its_threshold_from_calibration_not_from_the_test():
    trial = _trial_script()
    y_cal, s_cal = np.array([1, 1, 1, 1, 1, 0, 0]), np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.0])
    y_te = np.array([1, 0, 1, 0])
    s_te = np.array([0.65, 0.3, 0.55, 0.62])
    result = trial.evaluate(y_cal, s_cal, y_te, s_te, np.array(["a", "a", "b", "b"]), human=False)
    assert result["threshold"] == pytest.approx(0.6)
    assert result["caught"] == pytest.approx(0.5) and result["flagged"] == pytest.approx(0.5)
    assert result["verdict"]["admitted"] is False


def test_qwen_runs_in_the_materials_environment_with_the_gpu_held_only_meanwhile(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from image_to_3dlab import material_models as mm

    monkeypatch.setattr(mm, "require", lambda model, hub=None: tmp_path / "qwen")
    held = []

    @contextmanager
    def lock(session):
        held.append("in")
        yield
        held.append("out")

    def run(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        assert job["task"] == "yes_no" and job["model"] == str(tmp_path / "qwen") and held == ["in"]
        Path(job["result"]).write_text(json.dumps({"answers": [0.9, 0.1]}), encoding="utf-8")

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()

    answers = bl.run_qwen("yes_no", [tmp_path / "a.png", tmp_path / "b.png"], bl.QUESTION, tmp_path / "work",
                          python=tmp_path / "python.exe", run=run, lock=lock)
    assert answers == [0.9, 0.1] and held == ["in", "out"]
