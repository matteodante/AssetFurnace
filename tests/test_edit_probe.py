"""Evidence is tied to actual input/output bytes and never overwrites an asset."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab.edit_probe import inspect_asset, run_probe
from image_to_3dlab.glb_turn import build_glb, parse_glb
from test_edit_glb import fixture


def test_probe_checks_reloaded_data_and_the_reweight_bind_invariant():
    data = fixture()
    report, rebuilt = inspect_asset(data)
    assert report["status"] == "inspection_passed"
    assert report["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert report["container_roundtrip"]["sha256"] == hashlib.sha256(rebuilt).hexdigest()
    assert report["materials"] == 2
    assert report["skins"][0]["reposition_then_reweight_probe"]["passed"]
    assert report["production_editing_ready"] is False
    assert "real_gap_and_detach" in report["pending_gates"]


def test_default_pose_outside_bind_is_reported_without_normalizing_or_rebinding():
    document, binary = parse_glb(fixture())
    document["nodes"][2]["translation"] = [0, 1.2, 0]
    report, rebuilt = inspect_asset(build_glb(document, binary))
    assert report["skins"][0]["bind_status"] == "needs_bind_adapter"
    assert report["production_editing_ready"] is False
    assert parse_glb(rebuilt)[0] == document


def test_invalid_input_has_a_report_and_no_fake_roundtrip():
    report, rebuilt = inspect_asset(b"broken")
    assert report["status"] == "unsupported"
    assert report["error"]["code"] == "invalid_glb"
    assert rebuilt is None


def test_output_is_exclusive_and_provenance_and_source_are_preserved(tmp_path):
    source = tmp_path / "asset.glb"
    data = fixture()
    source.write_bytes(data)
    provenance = b'{"source":"synthetic", "license":"test"}\n'
    source.with_suffix(".provenance.json").write_bytes(provenance)
    output = tmp_path / "evidence"
    report = run_probe(source, output)
    assert json.loads((output / "report.json").read_text()) == report
    assert (output / "roundtrip.provenance.json").read_bytes() == provenance
    assert source.read_bytes() == data
    with pytest.raises(FileExistsError):
        run_probe(source, output)
    assert (output / "roundtrip.glb").read_bytes() == data


def test_unknown_nested_extension_is_not_hidden_by_a_missing_declaration():
    document, binary = parse_glb(fixture())
    document["meshes"][0]["primitives"][0]["extensions"] = {"KHR_draco_mesh_compression": {}}
    report, _ = inspect_asset(build_glb(document, binary))
    assert report["error"]["code"] == "unsupported_extension"


def test_nonfinite_accessor_data_is_rejected():
    document, binary = parse_glb(fixture())
    damaged = bytearray(binary)
    damaged[:4] = np.array([np.inf], dtype="<f4").tobytes()
    report, rebuilt = inspect_asset(build_glb(document, bytes(damaged)))
    assert report["status"] == "unsupported"
    assert rebuilt is None


@pytest.mark.parametrize("valid,expected_code", [(True, 0), (False, 1)])
def test_cli_writes_inspectable_evidence_and_distinguishes_unsupported_input(tmp_path, valid, expected_code):
    source = tmp_path / "source.glb"
    data = fixture() if valid else b"invalid"
    source.write_bytes(data)
    output = tmp_path / "report"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / "scripts/inspect_edit_asset.py"),
                             str(source), "--output-dir", str(output)],
                            capture_output=True, text=True, cwd=tmp_path, timeout=20)
    assert result.returncode == expected_code, result.stderr
    assert json.loads(result.stdout)["production_editing_ready"] is False
    assert (output / "report.json").is_file()
    assert (output / "roundtrip.glb").exists() == valid
    assert source.read_bytes() == data
