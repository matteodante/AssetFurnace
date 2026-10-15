"""The shipped batch steps, `generate` and `finish`, with their workers replaced."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from batch_fakes import install_fake_generator, tiny_glb, tiny_png, write_spec

from image_to_3dlab import batch
from image_to_3dlab.batch_steps import StepContext
from image_to_3dlab.batch_steps import finish as finish_step
from image_to_3dlab.batch_steps import generate as generate_step


@pytest.fixture(autouse=True)
def blender_answers(monkeypatch):
    """The stand-in Blender files here are empty: say they run, as a real one would."""
    monkeypatch.setattr(finish_step, "blender_version", lambda path: (5, 2))


def _context(tmp_path: Path, row: batch.BatchRow, asset=None, sidecar=None, options=None):
    row_dir = tmp_path / "rows" / row.id
    return StepContext(
        row=row, row_dir=row_dir, step_dir=row_dir / "step", asset=asset, sidecar=sidecar,
        intent={"use_case": "game", "distribution": "private", "commercial_intent": True},
        license_policy={"allow_conditional": True}, options=options or {}, log=lambda line: None)


def _row(tmp_path: Path, **fields) -> batch.BatchRow:
    values = {"id": "SM_Cat", "images": (tiny_png(tmp_path / "refs" / "cat.png"),),
              "asset": None, "kind": "prop", "faces": 3000, "profile": None,
              "steps": ("generate", "finish"), "backend": "fake",
              "parameters": {"seed": 2}, "options": {}}
    values.update(fields)
    return batch.BatchRow(**values)


# ------------------------------------------------------------------------- generate

def test_generate_writes_a_row_manifest_and_keeps_inputs_in_the_row(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    side = tiny_png(tmp_path / "refs" / "cat_side.png", 9)
    row = _row(tmp_path, images=(tmp_path / "refs" / "cat.png", side))
    ctx = _context(tmp_path, row)
    result = generate_step.run(ctx)

    manifest = json.loads((ctx.row_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["model"] == {"backend": "fake", "parameters": {"seed": 2}}
    assert Path(manifest["input"]["path"]) == ctx.row_dir / "input" / "cat.png"
    assert manifest["batch"]["references"] == [str(ctx.row_dir / "input" / "cat_side.png")]
    assert manifest["batch"]["faces"] == 3000 and manifest["batch"]["kind"] == "prop"
    assert (ctx.row_dir / "input" / "cat_side.png").is_file()
    assert result.asset.parent == ctx.step_dir / "clear"
    assert result.sidecar == result.asset.with_suffix(".provenance.json")
    assert result.asset in result.files and result.sidecar in result.files
    assert not (ctx.step_dir / ".working").exists()
    assert ctx.step_dir / "fake.log" in result.files          # the generator's log is kept
    assert ctx.row_dir / "input" / "cat.png" in result.files  # and the inputs it used


def test_generate_preflight_reports_each_backend_once(monkeypatch, tmp_path):
    install_fake_generator(monkeypatch, ready=False)
    rows = [_row(tmp_path, id="A"), _row(tmp_path, id="B")]
    assert generate_step.preflight(rows) == ["fake weights missing: run bootstrap_fake.py"]


# --------------------------------------------------------------------------- finish

@pytest.fixture
def generated(tmp_path):
    asset = tmp_path / "gen" / "cat__pixal3d.glb"
    asset.parent.mkdir()
    tiny_glb(asset, "raw")
    sidecar = asset.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "input": {"path": str(tmp_path / "refs" / "cat.png")},
        "output": {"classification": "commercial-conditional"},
        "license": {"name": "MIT + DINOv3", "url": "https://example.invalid"},
    }))
    return asset, sidecar


def test_finish_command_skips_the_repaint_unless_asked(tmp_path, generated):
    asset, sidecar = generated
    ctx = _context(tmp_path, _row(tmp_path), asset, sidecar)
    command = finish_step.command(ctx, Path("blender.exe"))
    assert command[3:6] == [str(asset), str(tmp_path / "refs" / "cat.png"),
                            str(ctx.step_dir / "SM_Cat.glb")]
    assert command[command.index("--faces") + 1] == "3000"
    assert "--skip-paint" in command and "--views" not in command
    assert command[command.index("--blender") + 1] == "blender.exe"
    painted = finish_step.command(_context(tmp_path, _row(tmp_path), asset, sidecar,
                                           {"paint": True, "atlas": 1024}), Path("b"))
    assert "--skip-paint" not in painted
    assert painted[painted.index("--atlas") + 1] == "1024"


def test_finish_uses_the_saved_camera_for_pixel_match(tmp_path, generated):
    asset, sidecar = generated
    views = asset.with_suffix(".svviews")
    views.mkdir()
    (views / "transforms.json").write_text("{}")
    command = finish_step.command(_context(tmp_path, _row(tmp_path), asset, sidecar), Path("b"))
    assert command[command.index("--views") + 1] == str(views)
    off = finish_step.command(_context(tmp_path, _row(tmp_path), asset, sidecar,
                                       {"photo": False}), Path("b"))
    assert "--views" not in off


def test_finish_falls_back_to_the_source_records_image(tmp_path, generated):
    asset, sidecar = generated
    tiny_png(tmp_path / "refs" / "cat.png")
    row = _row(tmp_path, images=(), steps=("finish",), asset=asset)
    command = finish_step.command(_context(tmp_path, row, asset, sidecar), Path("b"))
    assert command[4] == str(tmp_path / "refs" / "cat.png")


def test_finish_preflight_checks_options_blender_and_the_repaint(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_step, "find_blender", lambda: None)
    rows = [_row(tmp_path, options={"finish": {"atlass": 1}})]
    problems = finish_step.preflight(rows)
    assert any("atlass" in problem for problem in problems)
    assert any("Blender" in problem for problem in problems)

    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    monkeypatch.setattr(finish_step, "host_platform", lambda: "nvidia")
    painted = [_row(tmp_path, options={"finish": {"paint": True}})]
    assert any("Apple Silicon" in problem for problem in finish_step.preflight(painted))
    assert finish_step.preflight([_row(tmp_path)]) == []


def test_finish_writes_a_derived_provenance_sidecar(tmp_path, monkeypatch, generated):
    asset, sidecar = generated
    monkeypatch.setattr(finish_step, "find_blender", lambda: Path("blender.exe"))

    def fake_run(command, stdout, **kwargs):
        output = Path(command[5])
        tiny_glb(output, "finished")
        output.with_suffix(".retopo-repaint.json").write_text(json.dumps({"stages": ["x"]}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(finish_step, "run_worker", fake_run)
    ctx = _context(tmp_path, _row(tmp_path), asset, sidecar)
    result = finish_step.run(ctx)
    assert result.ok and result.asset == ctx.step_dir / "SM_Cat.glb"
    record = json.loads(result.sidecar.read_text())
    assert record["license"] == {"name": "MIT + DINOv3", "url": "https://example.invalid"}
    assert record["classification"] == "commercial-conditional"
    assert record["source_record"]["output"]["classification"] == "commercial-conditional"
    assert record["parameters"]["faces"] == 3000 and record["parameters"]["paint"] is False
    assert record["output"]["sha256"] and record["input"]["source"] == "generated"
    assert record["worker_record"] == {"stages": ["x"]}
    assert record["components"] == []
    assert ctx.step_dir / "SM_Cat.retopo-repaint.json" in result.files
    assert ctx.step_dir / "finish.log" in result.files


def test_finish_failure_reports_the_log_tail(tmp_path, monkeypatch, generated):
    asset, sidecar = generated
    monkeypatch.setattr(finish_step, "find_blender", lambda: Path("blender.exe"))

    def fake_run(command, stdout, **kwargs):
        stdout.write("[bake] failed: no UVs\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(finish_step, "run_worker", fake_run)
    result = finish_step.run(_context(tmp_path, _row(tmp_path), asset, sidecar))
    assert not result.ok and "no UVs" in result.reason


def test_the_real_steps_run_inside_a_batch(tmp_path, monkeypatch):
    """generate (fake generator) feeds the real finish step (fake worker) its GLB."""
    install_fake_generator(monkeypatch)
    monkeypatch.setattr(finish_step, "find_blender", lambda: tmp_path / "blender.exe")
    (tmp_path / "blender.exe").write_bytes(b"")
    seen = []

    def fake_run(command, stdout, **kwargs):
        seen.append(Path(command[3]).name)
        Path(command[5]).write_bytes(b"glTFfinished")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(finish_step, "run_worker", fake_run)
    spec_path = write_spec(tmp_path, rows=2)
    report = batch.run_batch(batch.load_batch(spec_path))
    assert report.exit_code == 0
    assert all(name.startswith("row0") and "__fake__" in name for name in seen)



def test_a_repaint_is_in_the_record_and_sets_the_classification(tmp_path, monkeypatch, generated):
    asset, sidecar = generated
    monkeypatch.setattr(finish_step, "find_blender", lambda: Path("blender.exe"))

    def fake_run(command, stdout, **kwargs):
        Path(command[5]).write_bytes(b"glTFpainted")
        steps = Path(command[command.index("--steps-dir") + 1])
        steps.mkdir(parents=True)
        (steps / "2_painted.glb").write_bytes(b"glTF")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(finish_step, "run_worker", fake_run)
    ctx = _context(tmp_path, _row(tmp_path), asset, sidecar, {"paint": True})
    result = finish_step.run(ctx)
    record = json.loads(result.sidecar.read_text())
    assert record["parameters"]["paint"] is True
    assert record["classification"] == "territory-restricted"
    (component,) = record["components"]
    assert "Hunyuan" in component["license"] and component["purpose"] == "repaint"
    assert ctx.step_dir / "steps" / "2_painted.glb" in result.files


def test_the_repaint_install_is_checked_before_the_batch(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    monkeypatch.setattr(finish_step, "host_platform", lambda: finish_step.APPLE)
    monkeypatch.setattr(finish_step, "PAINT_PYTHON", tmp_path / "missing" / "python")
    monkeypatch.setattr(finish_step, "PAINT_WEIGHTS", tmp_path / "missing" / "weights")
    problems = finish_step.preflight([_row(tmp_path, options={"finish": {"paint": True}})])
    assert any("uv sync" in problem for problem in problems)
    assert any("weights" in problem for problem in problems)
    assert finish_step.preflight([_row(tmp_path)]) == []



def test_a_finish_only_row_without_a_source_image_is_caught_before_the_batch(tmp_path,
                                                                            monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    asset = tmp_path / "old" / "crate.glb"
    asset.parent.mkdir()
    tiny_glb(asset, "")
    row = _row(tmp_path, images=(), steps=("finish",), asset=asset)
    assert any("source image" in p for p in finish_step.preflight([row]))
    image = tiny_png(tmp_path / "old" / "crate.png")
    asset.with_suffix(".provenance.json").write_text(json.dumps({"input": {"path": str(image)}}))
    assert finish_step.preflight([row]) == []


def test_a_top_level_classification_is_kept(tmp_path, monkeypatch):
    asset = tmp_path / "props" / "barrel_LOD0.glb"
    asset.parent.mkdir()
    tiny_glb(asset, "")
    sidecar = asset.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({"classification": "commercial-conditional",
                                   "license": {"name": "MIT + DINOv3"}}))
    output = tmp_path / "out.glb"
    tiny_glb(output, "done")
    record = finish_step.provenance_record(
        _context(tmp_path, _row(tmp_path), asset, sidecar), output, None)
    assert record["classification"] == "commercial-conditional"
    painted = finish_step.provenance_record(
        _context(tmp_path, _row(tmp_path), asset, sidecar, {"paint": True}), output, None)
    assert painted["classification"] == "territory-restricted"



def test_finish_option_values_are_checked_before_the_batch(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    row = _row(tmp_path, options={"finish": {"atlas": "large", "photo": "yes",
                                             "roughness": 2.0, "voxel": 0.004}})
    problems = finish_step.preflight([row])
    assert any("atlas" in p for p in problems)
    assert any("photo" in p for p in problems)
    assert any("roughness" in p for p in problems)
    assert not any("voxel" in p for p in problems)


def test_a_half_installed_repaint_is_caught(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    monkeypatch.setattr(finish_step, "host_platform", lambda: finish_step.APPLE)
    python = tmp_path / "venv" / "python"
    python.parent.mkdir()
    python.write_bytes(b"")
    weights = tmp_path / "weights"
    for name in finish_step.PAINT_FILES[:-1]:   # all but the DINOv2 weights
        (weights / name).parent.mkdir(parents=True, exist_ok=True)
        (weights / name).write_bytes(b"x")
    monkeypatch.setattr(finish_step, "PAINT_PYTHON", python)
    monkeypatch.setattr(finish_step, "PAINT_WEIGHTS", weights)
    (problem,) = finish_step.preflight([_row(tmp_path, options={"finish": {"paint": True}})])
    assert "dinov2/model.safetensors" in problem


def test_an_unknown_source_stays_unknown_after_a_repaint(tmp_path):
    asset = tmp_path / "upload.glb"
    tiny_glb(asset, "")
    output = tmp_path / "out.glb"
    tiny_glb(output, "done")
    record = finish_step.provenance_record(
        _context(tmp_path, _row(tmp_path), asset, None, {"paint": True}), output, None)
    assert record["classification"] is None
    assert record["input"]["source"] == "unknown" and record["components"]



def test_an_old_blender_is_caught_before_the_batch(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    monkeypatch.setattr(finish_step, "blender_version", lambda path: (3, 6))
    (problem,) = finish_step.preflight([_row(tmp_path)])
    assert "older than" in problem


def test_the_finish_record_names_the_picture_it_used(tmp_path, generated):
    asset, sidecar = generated
    image = tiny_png(tmp_path / "refs" / "cat.png")
    output = tmp_path / "out.glb"
    tiny_glb(output, "done")
    record = finish_step.provenance_record(_context(tmp_path, _row(tmp_path), asset, sidecar),
                                           output, None)
    assert record["input"]["image"]["path"] == str(image)
    assert record["input"]["image"]["sha256"]



def test_finish_runs_its_worker_through_its_own_wrapper(tmp_path, monkeypatch, generated):
    """A test replaces finish's worker, never the subprocess module the lock's `ps` uses."""
    asset, sidecar = generated

    def no_subprocess(*args, **kwargs):
        raise AssertionError("finish called subprocess.run directly")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(finish_step, "find_blender", lambda: Path("blender.exe"))

    def fake_run(command, stdout, **kwargs):
        Path(command[5]).write_bytes(b"glTFdone")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(finish_step, "run_worker", fake_run)
    assert finish_step.run(_context(tmp_path, _row(tmp_path), asset, sidecar)).ok


def test_a_finish_only_rows_broken_picture_is_caught_before_the_batch(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    broken = tmp_path / "refs" / "broken.png"
    broken.parent.mkdir(exist_ok=True)
    broken.write_bytes(b"not a picture")
    asset = tmp_path / "crate.glb"
    tiny_glb(asset, "")
    row = _row(tmp_path, images=(broken,), steps=("finish",), asset=asset)
    (problem,) = finish_step.preflight([row])
    assert "not a readable picture" in problem


def test_the_staged_pictures_are_finish_files(tmp_path, monkeypatch, generated):
    asset, sidecar = generated
    monkeypatch.setattr(finish_step, "find_blender", lambda: Path("blender.exe"))

    def fake_run(command, stdout, **kwargs):
        Path(command[5]).write_bytes(b"glTFdone")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(finish_step, "run_worker", fake_run)
    ctx = _context(tmp_path, _row(tmp_path), asset, sidecar)
    staged = tiny_png(ctx.row_dir / "input" / "cat.png")
    assert staged in finish_step.run(ctx).files



def test_the_finish_record_carries_this_runs_intent(tmp_path, generated):
    asset, sidecar = generated
    output = tmp_path / "out.glb"
    tiny_glb(output, "done")
    record = finish_step.provenance_record(_context(tmp_path, _row(tmp_path), asset, sidecar),
                                           output, None)
    assert record["intent"] == {"use_case": "game", "distribution": "private",
                                "commercial_intent": True}



def test_pixel_match_views_are_checked_before_the_batch(tmp_path, monkeypatch, generated):
    asset, sidecar = generated
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    tiny_png(tmp_path / "refs" / "cat.png")
    views = asset.with_suffix(".svviews")
    views.mkdir()
    (views / "transforms.json").write_text(json.dumps({"frames": [{"file_path": "missing.png"}]}))
    row = _row(tmp_path, images=(), steps=("finish",), asset=asset)
    (problem,) = finish_step.preflight([row])
    assert "Pixel Match" in problem and "missing.png" in problem
    off = _row(tmp_path, images=(), steps=("finish",), asset=asset,
               options={"finish": {"photo": False}})
    assert finish_step.preflight([off]) == []



def test_pixel_match_cameras_are_read_with_the_real_loader(tmp_path, monkeypatch, generated):
    asset, sidecar = generated
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    tiny_png(tmp_path / "refs" / "cat.png")
    views = asset.with_suffix(".svviews")
    views.mkdir()
    tiny_png(views / "front.png")
    (views / "transforms.json").write_text(json.dumps({"frames": [{"file_path": "front.png"}]}))
    row = _row(tmp_path, images=(), steps=("finish",), asset=asset)
    (problem,) = finish_step.preflight([row])
    assert "cameras cannot be read" in problem


def test_an_old_picture_left_in_input_is_not_a_finish_file(tmp_path, monkeypatch, generated):
    asset, sidecar = generated
    monkeypatch.setattr(finish_step, "find_blender", lambda: Path("blender.exe"))

    def fake_run(command, stdout, **kwargs):
        Path(command[5]).write_bytes(b"glTFdone")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(finish_step, "run_worker", fake_run)
    ctx = _context(tmp_path, _row(tmp_path), asset, sidecar)
    current = tiny_png(ctx.row_dir / "input" / "cat.png")
    old = tiny_png(ctx.row_dir / "input" / "old_concept.png")
    files = finish_step.run(ctx).files
    assert current in files and old not in files



def test_the_repaints_upscaler_weights_are_checked(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    monkeypatch.setattr(finish_step, "host_platform", lambda: finish_step.APPLE)
    python = tmp_path / "venv" / "python"
    python.parent.mkdir()
    python.write_bytes(b"")
    weights = tmp_path / "weights"
    for name in finish_step.PAINT_FILES:
        (weights / name).parent.mkdir(parents=True, exist_ok=True)
        (weights / name).write_bytes(b"x")
    monkeypatch.setattr(finish_step, "PAINT_PYTHON", python)
    monkeypatch.setattr(finish_step, "PAINT_WEIGHTS", weights)
    upscaler = (tmp_path / "rrdbnet_mlx.safetensors", tmp_path / "rrdbnet.npz")
    monkeypatch.setattr(finish_step, "PAINT_UPSCALER", upscaler)
    row = _row(tmp_path, options={"finish": {"paint": True}})
    (problem,) = finish_step.preflight([row])
    assert "realesrgan" in problem
    upscaler[1].write_bytes(b"x")   # the older converted file will do
    assert finish_step.preflight([row]) == []



def test_a_blender_that_does_not_run_is_caught_before_the_batch(tmp_path, monkeypatch):
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"")
    monkeypatch.setattr(finish_step, "find_blender", lambda: blender)
    monkeypatch.setattr(finish_step, "blender_version", lambda path: None)
    (problem,) = finish_step.preflight([_row(tmp_path)])
    assert "did not answer" in problem
