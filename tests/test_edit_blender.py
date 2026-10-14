"""Opt-in real Blender tests; no generated assets or model downloads required."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from test_edit_glb import fixture

from image_to_3dlab.blender import find_blender
from image_to_3dlab.edit_document import EditDocument

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("I2L_TEST_EDIT_BLENDER") != "1", reason="set I2L_TEST_EDIT_BLENDER=1 for Blender integration")
def test_canonical_noop_through_actual_blender(tmp_path):
    blender = find_blender()
    assert blender is not None, "Blender is required for the requested integration suite"
    source = fixture(interleaved=True, normalized=True)
    document = EditDocument(source)
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    document.write_snapshot(input_dir)
    result = subprocess.run([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                             "--python", str(REPO / "scripts/blender_edit_bridge.py"), "--",
                             "--input", str(input_dir), "--output", str(output_dir)],
                            check=False, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output_dir / "bridge-report.json").read_text())
    assert report["passed"] and report["byte_exact_glb"]
    assert all(part["positions_equal"] and part["triangles_equal"] for part in report["parts"])
    assert (output_dir / "roundtrip.glb").read_bytes() == source
