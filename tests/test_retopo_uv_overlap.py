"""The Finish's UVs do not paint a texel twice: Smart UV Project at the Finish's wide angle
folded islands onto themselves (the first real set's helmet and boots failed the uv.overlap
check at 10.1% and 11.1%); `blender_retopo_bake.unwrap_islands` unwraps them without that."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab.blender import find_blender
from image_to_3dlab.checks import mesh_checks

REPO = Path(__file__).resolve().parents[1]

SCRIPT = '''
import sys, math, json, bpy
sys.path.insert(0, {scripts!r})
import blender_retopo_bake as rb
bpy.ops.wm.read_factory_settings(use_empty=True)
for name, fixed in (("plain", False), ("fixed", True)):
    # a ramp turning twice round, like the overlapping plates of a helmet: one connected
    # surface whose faces all face up, so the wide angle keeps it one island stacked on itself
    verts, faces = [], []
    turns, steps, rings = 2, 64, 6
    for i in range(steps * turns + 1):
        a = 2 * math.pi * i / steps
        for j in range(rings + 1):
            r = 0.3 + 0.7 * j / rings
            verts.append((r * math.cos(a), r * math.sin(a), 0.02 * a))
    for i in range(steps * turns):
        for j in range(rings):
            k = i * (rings + 1) + j
            faces.append((k, k + 1, k + rings + 2, k + rings + 1))
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    angle = math.radians(89.0)
    if fixed:
        bpy.context.scene.tool_settings.mesh_select_mode = (True, False, False)   # as Blender starts
        print("LEFT", rb.unwrap_islands(bpy, obj.data, angle))
        print("MODE", tuple(bpy.context.scene.tool_settings.mesh_select_mode))
    else:
        bpy.ops.uv.smart_project(angle_limit=angle, island_margin=0.0002)
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data
    mesh.calc_loop_triangles()
    uv = mesh.uv_layers.active.data
    index, corners, faces = {{}}, [], []
    for tri in mesh.loop_triangles:
        face = []
        for loop in tri.loops:
            u, v = uv[loop].uv
            key = (mesh.loops[loop].vertex_index, round(u, 6), round(v, 6))
            if key not in index:
                index[key] = len(corners)
                corners.append([u, v])
            face.append(index[key])
        faces.append(face)
    json.dump({{"uv": corners, "faces": faces}}, open({out!r} + name + ".json", "w"))
    obj.hide_set(True)
'''


def _overlap(path: Path) -> float:
    data = json.loads(path.read_text())
    reused, used = mesh_checks.uv_overlap_pixels([(np.clip(np.asarray(data["uv"], float), 0, 1),
                                                   np.asarray(data["faces"]))], 1024)
    return 100 * reused / max(used, 1)


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_the_finish_unwraps_without_painting_a_texel_twice(tmp_path):
    script = tmp_path / "unwrap.py"
    script.write_text(SCRIPT.format(scripts=str(REPO / "scripts"), out=str(tmp_path / "uv_")))
    done = subprocess.run([str(find_blender()), "--background", "--factory-startup", "--python", str(script)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "MODE (True, False, False)" in done.stdout   # the person's select mode is given back
    plain, fixed = _overlap(tmp_path / "uv_plain.json"), _overlap(tmp_path / "uv_fixed.json")
    assert plain > 4.0, plain          # the wide-angle projection folds (the check warns from 4%)
    assert fixed < 1.0, fixed          # unwrapped, almost nothing is painted twice
