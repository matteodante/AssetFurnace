"""Baked-light removal: light painted into a texture comes back out, the rest is left alone.

Marigold and Blender are replaced by fakes: a "view" that sees the atlas flat, pixel for
texel, and a model that answers with an albedo we choose. What is tested is everything
around them: the gain each pixel implies, carried to its texel, smoothed inside its island,
normalised, applied to the full-resolution texture, and written back with its licence.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_materials import make_glb

from image_to_3dlab import delight as dl
from image_to_3dlab import material_models as mm
from image_to_3dlab import materials as m
from image_to_3dlab import multiview as mv
from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import parse_glb

SIZE = 128
LIT = 150


def painted(shadow_columns: int = 38, darken: float = 0.4) -> np.ndarray:
    """A flat grey albedo with a shadow painted down its left side (in linear light)."""
    image = np.full((SIZE, SIZE, 3), LIT, np.uint8)
    lit = m.srgb_to_linear(LIT / 255.0)
    image[:, :shadow_columns] = np.round(m.linear_to_srgb(lit * darken) * 255).astype(np.uint8)
    return image


def flat_view(index: int, colour: np.ndarray) -> mv.View:
    """A camera that sees the whole atlas head-on, one pixel per texel."""
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    uv = np.dstack([(cols + 0.5) / SIZE, 1.0 - (rows + 0.5) / SIZE])     # Blender's v points up
    return mv.View(index, np.array([0.0, 0.0, -1.0]), uv, np.zeros((SIZE, SIZE), np.int64),
                   np.tile([0.0, 0.0, 1.0], (SIZE, SIZE, 1)), colour)


def fake_render(texture: np.ndarray, count: int = 2):
    def render(asset, folder, *, size, around, lit=False):
        Path(folder).mkdir(parents=True, exist_ok=True)
        views = []
        for index in range(count):
            Image.fromarray(texture).save(Path(folder) / f"view_{index:02d}_color.png")
            if lit:
                Image.fromarray(texture).save(Path(folder) / f"view_{index:02d}_lit.png")
            views.append(flat_view(index, texture))
        return views
    return render


def fake_marigold(albedo_of):
    """An intrinsics runner that answers `albedo_of(colour_linear)` for each picture."""
    def run(images, outs, model, **_):
        for image, out in zip(images, outs):
            colour = m.srgb_to_linear(np.asarray(Image.open(image).convert("RGB")) / 255.0)
            np.savez(out, albedo=albedo_of(colour).astype(np.float16), shading=colour, residual=colour * 0)
        return {"target_properties": {"albedo": {"prediction_space": "linear"}}, "device": "fake",
                "seconds": 0.0, "steps": 4, "resolution": 768, "ensemble": 1}
    return run


def flat_albedo(colour):
    return np.full_like(colour, m.srgb_to_linear(LIT / 255.0))


@pytest.fixture
def shadowed(tmp_path):
    return make_glb(tmp_path / "knight.glb", painted())


def base_colour(path: Path) -> np.ndarray:
    glb = EditGLB(path.read_bytes())
    return next(iter(m.surfaces(glb).values())).image


def test_each_pixel_implies_the_gain_that_would_turn_its_colour_into_the_albedo():
    colour = np.array([[[100, 100, 100], [0, 0, 0]]], np.uint8)
    albedo = m.srgb_to_linear(np.array([[[100, 100, 100], [50, 50, 50]]]) / 255.0) * 2.0
    gain, valid = dl.pixel_gain(colour, albedo)
    assert gain[0, 0] == pytest.approx(np.log(2.0))
    assert valid.tolist() == [[True, False]]                 # black says nothing about light


def test_a_painted_shadow_is_lifted_to_the_lit_colour(shadowed, tmp_path):
    result = dl.delight_asset(shadowed, tmp_path / "out", render=fake_render(painted()),
                              intrinsics=fake_marigold(flat_albedo))
    out = base_colour(result.glb_path)
    assert np.abs(out[:, :30].astype(int) - LIT).max() <= 3       # the shadow, away from its edge
    assert np.abs(out[:, 50:].astype(int) - LIT).max() <= 3       # the lit side, untouched


def test_the_models_overall_brightness_is_not_trusted(shadowed, tmp_path):
    # Marigold says everything is twice as bright: that is exposure, not painted light.
    result = dl.delight_asset(shadowed, tmp_path / "out", render=fake_render(painted()),
                              intrinsics=fake_marigold(lambda colour: colour * 2.0))
    np.testing.assert_allclose(base_colour(result.glb_path).astype(int), painted().astype(int), atol=1)


def test_no_texel_is_changed_by_more_than_the_largest_gain(shadowed, tmp_path):
    def blinding(colour):
        out = colour.copy()
        out[:, :38] *= 100.0
        return out
    result = dl.delight_asset(shadowed, tmp_path / "out", render=fake_render(painted()),
                              intrinsics=fake_marigold(blinding))
    before = m.srgb_to_linear(painted()[:, :30] / 255.0)
    after = m.srgb_to_linear(base_colour(result.glb_path)[:, :30] / 255.0)
    assert (after / before).max() <= dl.MAX_GAIN * 1.02


def test_smoothing_never_mixes_two_uv_islands():
    islands = np.zeros((8, 8), np.int64)
    islands[:, :4], islands[:, 4:] = 1, 2
    values = np.where(islands == 1, 0.0, 1.0)
    smoothed, filled = dl.smooth_within(values, np.ones((8, 8)), islands, sigma=3.0)
    assert filled.all()
    np.testing.assert_allclose(smoothed, values)


def test_the_result_keeps_geometry_and_records_the_model_and_its_restrictions(shadowed, tmp_path):
    result = dl.delight_asset(shadowed, tmp_path / "out", render=fake_render(painted()),
                              intrinsics=fake_marigold(flat_albedo))
    before, _ = parse_glb(shadowed.read_bytes())
    after, _ = parse_glb(result.glb_path.read_bytes())
    assert after["accessors"] == before["accessors"] and after["meshes"] == before["meshes"]
    provenance = json.loads(result.glb_path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    component = next(c for c in provenance["components"] if c["component"].startswith(mm.MARIGOLD_LIGHTING.repo))
    assert mm.MARIGOLD_LIGHTING.revision in component["component"]
    assert component["license"] == mm.MARIGOLD_LIGHTING.license
    assert component["conditions"] == list(mm.OPENRAIL_CONDITIONS)
    assert provenance["input"]["path"] == str(shadowed)
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["admitted"] is False and "labelled" in sidecar["trial"]
    assert (tmp_path / "out" / "T_knight_BaseColor.png").is_file()


def test_alpha_of_the_base_colour_survives(tmp_path):
    rgba = np.dstack([painted(), np.where(np.arange(SIZE) < 64, 0, 255)[:, None].repeat(SIZE, 1)]).astype(np.uint8)
    asset = make_glb(tmp_path / "leaf.glb", rgba)
    result = dl.delight_asset(asset, tmp_path / "out", render=fake_render(painted()),
                              intrinsics=fake_marigold(flat_albedo))
    glb = EditGLB(result.glb_path.read_bytes())
    alpha = m._decode_image(glb, 0, "RGBA")[..., 3]
    np.testing.assert_array_equal(alpha, rgba[..., 3])


def test_writing_into_the_assets_own_folder_is_refused(shadowed):
    with pytest.raises(ValueError, match="overwrite"):
        dl.delight_asset(shadowed, shadowed.parent, render=fake_render(painted()),
                         intrinsics=fake_marigold(flat_albedo))


def test_a_concept_loses_its_painted_shadow_and_keeps_its_cut_out(tmp_path):
    rgba = np.dstack([painted(), np.full((SIZE, SIZE), 255, np.uint8)])
    rgba[:10, :, 3] = 0
    concept = tmp_path / "concept.png"
    Image.fromarray(rgba).save(concept)
    out = dl.delight_image(concept, tmp_path / "concept_albedo.png", intrinsics=fake_marigold(flat_albedo))
    result = np.asarray(Image.open(out))
    np.testing.assert_array_equal(result[..., 3], rgba[..., 3])
    assert np.abs(result[20:, :30, :3].astype(int) - LIT).max() <= 3
    assert json.loads(out.with_suffix(".provenance.json").read_text(encoding="utf-8"))["components"]


def test_the_model_runs_in_its_own_environment_with_the_gpu_held_only_meanwhile(tmp_path):
    held, calls = [], []

    @contextmanager
    def lock(session):
        held.append(True)
        yield
        held.append(False)

    def run(command, **kwargs):
        calls.append((command, list(held)))
        job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        Path(job["result"]).write_text(json.dumps({"device": "cuda"}), encoding="utf-8")

        class Done:
            returncode, stdout, stderr = 0, "", ""
        return Done()

    images = [tmp_path / "a.png"]
    record = dl.run_intrinsics(images, [tmp_path / "a.npz"], mm.MARIGOLD_LIGHTING, folder=tmp_path / "snap",
                               python=tmp_path / "python.exe", run=run, lock=lock)
    command, during = calls[0]
    assert command[:2] == [str(tmp_path / "python.exe"), str(dl.WORKER)]
    assert during == [True] and held == [True, False]
    job = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
    assert job["model"] == str(tmp_path / "snap") and job["images"][0]["path"] == str(images[0])
    assert record["model"]["revision"] == mm.MARIGOLD_LIGHTING.revision


def test_a_failed_model_run_says_why(tmp_path):
    @contextmanager
    def lock(session):
        yield

    def run(command, **kwargs):
        class Failed:
            returncode, stdout, stderr = 1, "", "Traceback...\ntorch.OutOfMemoryError: CUDA out of memory"
        return Failed()

    with pytest.raises(RuntimeError, match="out of memory"):
        dl.run_intrinsics([tmp_path / "a.png"], [tmp_path / "a.npz"], mm.MARIGOLD_LIGHTING,
                          folder=tmp_path, python=tmp_path / "python.exe", run=run, lock=lock)


def test_fine_painted_detail_survives_when_the_model_flattens_it(tmp_path):
    # Marigold reads thin painted trims as shading; only broad light should come out.
    texture = painted()
    texture[:, 60::8] = 230                                   # one-texel bright trims on the lit side
    asset = make_glb(tmp_path / "trim.glb", texture)
    result = dl.delight_asset(asset, tmp_path / "out", render=fake_render(texture),
                              intrinsics=fake_marigold(flat_albedo))
    out = base_colour(result.glb_path).astype(int)
    assert (out[:, 60::8] - out[:, 63::8]).mean() > 0.6 * (230 - LIT)
    assert np.abs(out[:, :30] - LIT).max() <= 6               # the broad shadow still comes out
