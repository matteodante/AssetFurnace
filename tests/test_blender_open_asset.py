"""Opt-in real Blender import, manual-style mesh change, save and GLB export."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_edit_glb import fixture

from image_to_3dlab.blender import find_blender
from image_to_3dlab.blender_exchange import inspect_glb
from image_to_3dlab.glb_turn import build_glb, parse_glb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))
import assets_api
import blender_versions as bv


@pytest.mark.skipif(os.environ.get("I2L_TEST_EDIT_BLENDER") != "1", reason="set I2L_TEST_EDIT_BLENDER=1")
def test_open_edit_export_import_and_reopen_saved_workspace(tmp_path):
    blender = find_blender()
    assert blender, "Blender is required for the requested integration test"
    doc, binary = parse_glb(fixture())
    doc["scenes"] = [{"nodes": [0, 1]}]
    doc["scene"] = 0
    doc.pop("images")  # the byte-only fixture's unused fake image is not a real PNG
    source = tmp_path / "knight/knight.glb"
    source.parent.mkdir()
    source.write_bytes(build_glb(doc, binary))
    before = source.read_bytes()
    asset = assets_api.list_assets(tmp_path)[0]
    session = bv.create_session(tmp_path, asset["id"], "knight/knight.glb", "9" * 32)
    root = bv.session_path(tmp_path, session["id"])
    workspace, status = root / "work/model.blend", root / "open-status.json"
    args = [str(blender), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
            "--python", str(REPO / "scripts/blender_open_asset.py"), "--", "--source", str(root / "input/source.glb"),
            "--workspace", str(workspace), "--status", str(status)]
    opened = subprocess.run(args, capture_output=True, text=True, timeout=90, check=False)
    assert opened.returncode == 0, opened.stdout + opened.stderr
    assert json.loads(status.read_text())["state"] == "ready"
    assert workspace.is_file()
    exported = tmp_path / "edited.glb"
    change = tmp_path / "manual_change.py"
    change.write_text("\n".join([
        "import bpy", "import json", "from pathlib import Path",
        "mesh = next(o for o in bpy.context.scene.objects if o.type == 'MESH')",
        "mesh.data.vertices[0].co.x += 0.125", "mesh['manual_edit_test'] = True",
        "bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)",
        f"bpy.ops.export_scene.gltf(filepath={str(exported)!r}, export_format='GLB', export_animations=True)",
    ]), encoding="utf-8")
    edited = subprocess.run([str(blender), "--background", "--disable-autoexec", str(workspace),
                             "--python-exit-code", "1", "--python", str(change)],
                            capture_output=True, text=True, timeout=90, check=False)
    assert edited.returncode == 0, edited.stdout + edited.stderr
    facts = inspect_glb(exported.read_bytes())
    assert facts["skins"] == 1 and facts["joints"] == 2
    version = bv.import_version(tmp_path, session["id"], exported.read_bytes(), "edited.glb")
    assert (tmp_path / version["file"]).read_bytes() == exported.read_bytes()
    assert source.read_bytes() == before
    saved_bytes = workspace.read_bytes()
    reopened = subprocess.run(args, capture_output=True, text=True, timeout=90, check=False)
    assert reopened.returncode == 0, reopened.stdout + reopened.stderr
    assert workspace.read_bytes() == saved_bytes  # reopen must not replace saved manual changes
    assert len(assets_api.list_assets(tmp_path)[0]["blender_versions"]) == 1
