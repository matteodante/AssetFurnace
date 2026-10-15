"""Run manifests through registered generators (Pixal3D), the path batches share with
`pipeline.py --run-manifest`. Pixal3D itself is replaced: no run here touches a GPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from batch_fakes import FakeGenerator, install_fake_generator, tiny_png

from image_to_3dlab import cli, manifest_run


def _manifest(tmp_path: Path, backend: str = "fake", parameters=None, **extra) -> Path:
    tiny_png(tmp_path / "cat.png", 5)
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "schema_version": 1, "use_case": "game", "distribution": "private",
        "commercial_intent": True, "input": {"path": "cat.png"},
        "model": {"backend": backend, "parameters": parameters or {"seed": 1}},
        "output": {"directory": str(tmp_path / "out")}, **extra,
    }))
    return path


def test_pixal3d_command_passes_only_what_was_asked():
    command = manifest_run.pixal3d_command(
        Path("in.png"), Path("out.glb"), {"seed": 9, "steps": 8, "gss": 10.0, "matte": False})
    assert command[:3] == [sys.executable, "-u", str(manifest_run.PIXAL3D_SCRIPT)]
    assert command[3:] == ["in.png", "out.glb", "--seed", "9", "--steps", "8",
                           "--gss", "10.0", "--no-matte"]
    assert "--matte" in manifest_run.pixal3d_command(Path("a"), Path("b"), {"matte": True})


def test_pixal3d_readiness_names_the_bootstrap_and_never_downloads(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(manifest_run, "_pixal3d_state", lambda: {
        "ready": False, "cli_built": True, "weights_present": 3})
    problems = manifest_run.pixal3d_readiness()
    assert problems and "bootstrap_pixal3d.py" in problems[0]
    assert calls == []


def test_run_pixal3d_collects_its_record_and_camera(tmp_path, monkeypatch):
    image = tiny_png(tmp_path / "cat.png")
    working = tmp_path / "working"

    def fake_run(command, **kwargs):
        output = Path(command[4])
        output.write_bytes(b"glTF")
        output.with_name(f"{output.stem}.json").write_text(json.dumps(
            {"backend": "pixal3d", "parameters": {"seed": 4, "steps": 12, "matted": True}}))
        (output.with_suffix(".svviews")).mkdir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(manifest_run, "run_worker", fake_run)
    generated = manifest_run.run_pixal3d([image], working, {"seed": 4}, print)
    assert generated.glb == working / "cat.glb"
    assert generated.parameters == {"seed": 4, "steps": 12, "matted": True}
    assert set(generated.companions) == {".json", ".svviews"}


def test_run_pixal3d_failure_carries_the_log_tail(tmp_path, monkeypatch):
    image = tiny_png(tmp_path / "cat.png")

    def fake_run(command, stdout, **kwargs):
        stdout.write("loading\nout of memory\n")
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(manifest_run, "run_worker", fake_run)
    with pytest.raises(RuntimeError, match="out of memory"):
        manifest_run.run_pixal3d([image], tmp_path / "w", {}, print)


def test_run_manifest_finalizes_and_moves_companions_beside_the_glb(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    glb, sidecar = manifest_run.run_manifest(_manifest(tmp_path))
    assert glb.parent == (tmp_path / "out" / "clear").resolve()
    assert glb.name.startswith("cat__fake__commercial-clear__")
    assert sidecar == glb.with_suffix(".provenance.json")
    assert glb.with_suffix(".json").is_file()        # the generator's own record
    record = json.loads(sidecar.read_text())
    assert record["model"] == {"backend": "fake", "parameters": {"seed": 1}}
    assert record["intent"]["use_case"] == "game"
    assert record["source_manifest"]["path"] == str((tmp_path / "run.json").resolve())


def test_run_manifest_refuses_before_the_generator_when_weights_are_missing(tmp_path,
                                                                             monkeypatch):
    generator = install_fake_generator(monkeypatch, ready=False)
    with pytest.raises(manifest_run.ManifestRunError, match="bootstrap_fake.py"):
        manifest_run.run_manifest(_manifest(tmp_path))
    assert generator.calls == []


@pytest.mark.parametrize("parameters, extra, message", [
    ({"sed": 1}, {}, "sed"),
    ({"seed": 1}, {"use_case": "arcade"}, "use_case"),
    ({"seed": 1}, {"input": {"views": ["cat.png", "cat.png"]}}, "one image"),
    ({"seed": 1}, {"input": {"views": []}}, "no input image"),
    ({"seed": 1}, {"license_policy": {"allow_conditional": "false"}}, "true or false"),
    ({"seed": 1}, {"input": {"path": 42}}, "file paths"),
    ({"seed": 1}, {"input": {"views": ["cat.png", 7]}}, "file paths"),
])
def test_run_manifest_rejects_bad_manifests_with_code_2(tmp_path, monkeypatch, parameters,
                                                        extra, message):
    install_fake_generator(monkeypatch)
    with pytest.raises(manifest_run.ManifestRunError, match=message) as raised:
        manifest_run.run_manifest(_manifest(tmp_path, parameters=parameters, **extra))
    assert raised.value.code == 2


def test_a_generator_failure_is_code_1(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch, FakeGenerator(fail_on="cat"))
    with pytest.raises(manifest_run.ManifestRunError) as raised:
        manifest_run.run_manifest(_manifest(tmp_path))
    assert raised.value.code == 1


def test_cli_runs_a_pixal3d_manifest_through_the_registry(tmp_path, monkeypatch, capsys):
    generator = FakeGenerator()
    monkeypatch.setitem(manifest_run.GENERATORS, "pixal3d", manifest_run.Generator(
        name="pixal3d", run=generator.run, readiness=list,
        parameters=manifest_run.PIXAL3D_PARAMETERS))
    assert cli.main(["--run-manifest", str(_manifest(tmp_path, backend="pixal3d"))]) == 0
    printed = capsys.readouterr().out.split()
    assert printed[0].endswith(".glb") and printed[1].endswith(".provenance.json")
    record = json.loads(Path(printed[1]).read_text())
    assert record["license"]["url"] == "https://huggingface.co/raven38/pixal3d-sv-q8_0-v1"
    assert generator.calls == ["cat"]


def test_cli_reports_a_manifest_error(tmp_path, monkeypatch, capsys):
    install_fake_generator(monkeypatch, ready=False)
    assert cli.main(["--run-manifest", str(_manifest(tmp_path))]) == 2
    assert "bootstrap_fake.py" in capsys.readouterr().err



def test_pixal3d_never_lets_rembg_download_a_remover(tmp_path, monkeypatch):
    from image_to_3dlab import matte

    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "no-models"))
    opaque = tiny_png(tmp_path / "opaque.png")
    problems = manifest_run.pixal3d_input_check([opaque], {})
    assert problems and "bootstrap_matte.py" in problems[0]
    assert manifest_run.pixal3d_input_check([opaque], {"matte": False}) == []
    (tmp_path / "no-models").mkdir()
    matte.model_file(matte.FALLBACK_MODEL).write_bytes(b"onnx")
    assert manifest_run.pixal3d_input_check([opaque], {}) == []


def test_a_cut_out_image_needs_no_remover(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "no-models"))
    cut = tmp_path / "cut.png"
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(cut)
    assert manifest_run.pixal3d_input_check([cut], {}) == []
    assert manifest_run.pixal3d_input_check([cut], {"matte": True})


def test_the_input_check_stops_a_run_before_the_generator(tmp_path, monkeypatch):
    generator = FakeGenerator()
    monkeypatch.setitem(manifest_run.GENERATORS, "fake", manifest_run.Generator(
        name="fake", run=generator.run, readiness=lambda: [], parameters=frozenset({"seed"}),
        input_check=lambda images, parameters: ["no remover"]))
    from batch_fakes import FAKE_PROFILE

    from image_to_3dlab import provenance

    monkeypatch.setitem(provenance.LICENSES, "fake", FAKE_PROFILE)
    monkeypatch.setattr(provenance, "_pipeline_revision", lambda: None)
    with pytest.raises(manifest_run.ManifestRunError, match="no remover"):
        manifest_run.run_manifest(_manifest(tmp_path))
    assert generator.calls == []


def test_the_generators_own_components_reach_the_provenance(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    original = manifest_run.GENERATORS["fake"].run

    def with_remover(images, working, parameters, log):
        generated = original(images, working, parameters, log)
        generated.components = [{"component": "rembg/u2net", "purpose": "background removal",
                                 "license": "MIT"}]
        return generated

    monkeypatch.setitem(manifest_run.GENERATORS, "fake", manifest_run.Generator(
        name="fake", run=with_remover, readiness=lambda: [], parameters=frozenset({"seed"})))
    glb, sidecar = manifest_run.run_manifest(_manifest(tmp_path))
    record = json.loads(sidecar.read_text())
    assert record["components"] == [{"component": "rembg/u2net",
                                     "purpose": "background removal", "license": "MIT"}]
    assert record["output"]["sha256"]


def test_run_pixal3d_carries_the_remover_it_used(tmp_path, monkeypatch):
    image = tiny_png(tmp_path / "cat.png")

    def fake_run(command, **kwargs):
        output = Path(command[4])
        output.write_bytes(b"glTF")
        output.with_name(f"{output.stem}.json").write_text(json.dumps({
            "parameters": {"seed": 1},
            "components": [{"component": "rembg/birefnet-general-lite", "license": "MIT"}]}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(manifest_run, "run_worker", fake_run)
    generated = manifest_run.run_pixal3d([image], tmp_path / "w", {}, print)
    assert generated.components == [{"component": "rembg/birefnet-general-lite",
                                     "license": "MIT"}]



@pytest.mark.parametrize("parameters, message", [
    ({"seed": "random"}, "seed"), ({"seed": True}, "seed"), ({"steps": 0}, "steps"),
    ({"gss": "strong"}, "gss"), ({"fov": 4}, "fov"), ({"matte": "false"}, "matte"),
])
def test_pixal3d_parameter_values_are_checked(parameters, message):
    (problem,) = manifest_run.pixal3d_check_parameters(parameters)
    assert message in problem
    assert manifest_run.pixal3d_check_parameters(
        {"seed": 7, "steps": 8, "gss": 10.0, "fov": 0.35, "matte": False}) == []



def test_a_broken_picture_is_caught_before_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "no-models"))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\nnot really")
    for parameters in ({}, {"matte": False}):
        (problem,) = manifest_run.pixal3d_input_check([broken], parameters)
        assert "not a readable picture" in problem



def test_a_direct_manifests_commercial_intent_must_be_a_boolean(tmp_path, monkeypatch):
    install_fake_generator(monkeypatch)
    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data["commercial_intent"] = "false"
    manifest.write_text(json.dumps(data))
    with pytest.raises(manifest_run.ManifestRunError, match="commercial_intent"):
        manifest_run.run_manifest(manifest)
