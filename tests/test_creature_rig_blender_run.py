"""The three Blender steps of the creature auto-rig, run for real on a creature of boxes (about
25 s; skipped without Blender). UniRig is stood in for by a skeleton file made the way it
makes them: the model scaled into a box of its own, with an armature in that frame."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import trimesh

from image_to_3dlab.blender import find_blender

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BLENDER = find_blender()
pytestmark = pytest.mark.skipif(BLENDER is None, reason="needs Blender")
TURN = 215.0     # the box creature is saved turned 35 degrees past facing +Y


def boxes():
    """Z up, facing -Y, centred, on the floor: body, raised head, tail, four legs with toes
    forward, and a small horn (a loose part with too few vertices for bone heat)."""
    parts = [((-0.15, -0.4, 0.35), (0.15, 0.4, 0.6)), ((-0.08, -0.58, 0.5), (0.08, -0.38, 0.8)),
             ((-0.03, 0.38, 0.3), (0.03, 0.58, 0.4)), ((-0.012, -0.5, 0.79), (0.012, -0.47, 0.86))]
    for x in (-0.12, 0.12):
        for y in (-0.3, 0.3):
            parts += [((x - 0.03, y - 0.03, 0.03), (x + 0.03, y + 0.03, 0.36)),
                      ((x - 0.035, y - 0.1, 0.0), (x + 0.035, y + 0.035, 0.035))]
    meshes = []
    for lo, hi in parts:
        lo, hi = np.array(lo), np.array(hi)
        m = trimesh.creation.box(extents=hi - lo)
        m.apply_translation((lo + hi) / 2)
        edge = 0.03 if (hi - lo).min() > 0.03 else 1.0       # the horn stays 8 corners
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, edge)
        meshes.append(trimesh.Trimesh(v, f, process=False))
    return trimesh.util.concatenate(meshes)


def write_box_creature(path: Path, turn_deg: float = TURN) -> None:
    mesh = boxes()
    a = math.radians(turn_deg)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    v = np.asarray(mesh.vertices) @ rot.T + [0.3, 0.2, 0.0]
    gltf = trimesh.Trimesh(v[:, [0, 2, 1]] * [1, 1, -1], mesh.faces, process=False)   # Z up -> glTF's Y up
    gltf.export(str(path))


def joints(height: float) -> dict:
    """A hand-made skeleton for the box creature, as the skeleton step writes joints.json."""
    def b(name, parent, kind, head, tail, side="", deform=True):
        return {"name": name, "parent": parent, "kind": kind, "head": head, "tail": tail, "side": side,
                "deform": deform}

    bones = [b("Root", None, "root", [0, 0, 0], [0, 0, 0.1], deform=False),
             b("Hips", "Root", "spine", [0, 0.3, 0.48], [0, 0, 0.48]),
             b("Spine1", "Hips", "spine", [0, 0, 0.48], [0, -0.35, 0.5]),
             b("Neck1", "Spine1", "neck", [0, -0.35, 0.5], [0, -0.45, 0.62]),
             b("Head", "Neck1", "head", [0, -0.45, 0.62], [0, -0.58, 0.7]),
             b("Tail1", "Hips", "tail", [0, 0.38, 0.35], [0, 0.58, 0.35])]
    for s, x in (("L", 0.12), ("R", -0.12)):
        bones += [b(f"Thigh.{s}", "Hips", "hind", [x, 0.3, 0.45], [x, 0.3, 0.24], s),
                  b(f"Shin.{s}", f"Thigh.{s}", "hind", [x, 0.3, 0.24], [x, 0.3, 0.04], s),
                  b(f"HindPaw.{s}", f"Shin.{s}", "hind", [x, 0.3, 0.04], [x, 0.22, 0.02], s),
                  b(f"UpperArm.{s}", "Spine1", "front", [x, -0.3, 0.45], [x, -0.3, 0.24], s),
                  b(f"Forearm.{s}", f"UpperArm.{s}", "front", [x, -0.3, 0.24], [x, -0.3, 0.04], s),
                  b(f"FrontPaw.{s}", f"Forearm.{s}", "front", [x, -0.3, 0.04], [x, -0.38, 0.02], s)]
    return {"body": "quadruped", "height": height, "facing": "-Y", "bones": bones, "draws": [], "chosen": None,
            "notes": []}


# A skeleton file the way UniRig writes one: its copy of the model centred and scaled so the
# largest side spans 2, an armature in the same frame. Run inside Blender.
FAKE_UNIRIG = r"""
import sys, bpy
from mathutils import Vector
glb, out = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
mesh = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
pts = [mesh.matrix_world @ v.co for v in mesh.data.vertices]
lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])
c, s = (lo + hi) / 2, 2.0 / max(hi - lo)
for v in mesh.data.vertices:
    v.co = (mesh.matrix_world @ v.co - c) * s
mesh.matrix_world.identity()
arm = bpy.data.objects.new('Armature', bpy.data.armatures.new('Armature'))
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
root = arm.data.edit_bones.new('bone_0')
root.head, root.tail = (Vector((0, 0.3, 0.48)) - c) * s, (Vector((0, 0, 0.48)) - c) * s
leg = arm.data.edit_bones.new('bone_1')
leg.head, leg.tail = (Vector((0.12, -0.3, 0.45)) - c) * s, (Vector((0.12, -0.3, 0.04)) - c) * s
leg.parent = root
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.export_scene.fbx(filepath=out, add_leaf_bones=False)
"""


def blender(script: Path, *args) -> str:
    done = subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python", str(script), "--",
                           *[str(a) for a in args]], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    return done.stdout


def report(out: str, tag: str) -> dict:
    return json.loads(next(line for line in out.splitlines() if line.startswith(tag + "::")).split("::", 1)[1])


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("box")
    model = root / "box_creature.glb"
    write_box_creature(model)
    work = root / "W"
    out = blender(SCRIPTS / "blender_creature_rig_prepare.py", model, work)
    return work, report(out, "I2L_CREATURE_PREPARE")


def test_prepare_turns_places_and_seals_the_creature(prepared):
    work, rec = prepared
    assert rec["front_cue"] == "toes"
    assert (rec["turn_deg"] + TURN) % 360 == pytest.approx(0, abs=1.5) or \
        (rec["turn_deg"] + TURN) % 360 == pytest.approx(360, abs=1.5)
    for name in ("model.blend", "unirig_input.glb", "shell.npz"):
        assert (work / name).is_file()
    shell = np.load(work / "shell.npz")
    v = shell["model_verts"]
    assert v[:, 2].min() == pytest.approx(0, abs=1e-4)
    assert (v[:, 1].min() + v[:, 1].max()) / 2 == pytest.approx(0, abs=1e-3)
    assert abs(v[:, 0].mean()) < 0.01 and float(shell["height"]) == pytest.approx(0.86, abs=0.01)
    assert np.allclose(np.abs(v).max(0), [0.155, 0.58, 0.86], atol=0.01)   # back where it was drawn
    assert len(shell["shell_faces"]) > 1000 and shell["shell_faces"].dtype == np.int32


def test_read_maps_a_unirig_skeleton_back_onto_the_model(prepared, tmp_path):
    work, _ = prepared
    script = tmp_path / "fake_unirig.py"
    script.write_text(FAKE_UNIRIG, encoding="utf-8")
    fbx = tmp_path / "draw0" / "skeleton.fbx"
    fbx.parent.mkdir()
    blender(script, work / "unirig_input.glb", fbx)
    out = blender(SCRIPTS / "blender_creature_rig_read.py", work, fbx, "--start", "3")
    assert report(out, "I2L_CREATURE_READ")[0]["draw"] == 3
    draw = json.loads((work / "draws" / "draw3.json").read_text())
    assert draw["names"] == ["bone_0", "bone_1"] and draw["parents"] == [-1, 0]
    heads = np.array([[0, 0.3, 0.48], [0.12, -0.3, 0.45]])
    assert np.allclose(draw["heads"], heads, atol=2e-3)
    stub = 0.03 * draw["scale"]                      # UniRig's short end, in the model's units
    along = (heads[1] - heads[0]) / np.linalg.norm(heads[1] - heads[0])
    assert draw["scale"] == pytest.approx(1.16 / 2, abs=0.01)
    assert np.allclose(draw["tails"], [heads[0] + [0, 0, stub], heads[1] + along * stub], atol=2e-3)
    assert "warning" not in draw


def test_build_rigs_and_skins_the_creature(prepared):
    work, _ = prepared
    height = float(np.load(work / "shell.npz")["height"])
    (work / "joints.json").write_text(json.dumps(joints(height)), encoding="utf-8")
    out = blender(SCRIPTS / "blender_creature_rig_build.py", work)
    rec = report(out, "I2L_CREATURE_BUILD")
    for name in ("rigged.blend", "rigged.glb", "weights.json", "poses-sheet.png"):
        assert (work / name).is_file(), name
    weights = json.loads((work / "weights.json").read_text())
    assert list(weights.values())[0]["empty_bones"] == []
    assert rec["loose"]["trunk"] >= 1                     # the horn rides on the head
    assert rec["deforming"] == 17 and rec["empty_bones"] == []
    glb = trimesh.load(str(work / "rigged.glb"), force="scene")
    assert glb.geometry, "the rigged GLB holds the mesh"
