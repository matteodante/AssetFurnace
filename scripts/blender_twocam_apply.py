"""Key a two-camera capture's motion onto its rig in Blender and export an animated GLB, headless.

blender -b --factory-startup RIGGED.blend --python-exit-code 1 --python scripts/blender_twocam_apply.py \
    -- RIG_INFO.json MOTION.json OUT_DIR [--armature NAME] [--name NAME] [--keep-paint] [--no-glb]

Ported from AnyMove's `blender/apply_motion.py` (Patch, Apache-2.0), without its renders.
Writes OUT_DIR/joints.json (every joint per frame, measured by Blender itself),
OUT_DIR/NAME.glb and OUT_DIR/apply.json, which says how far Blender's joints land from the
solver's (they share one set of kinematics, so anything over a millimetre is a bug).

The limb paint AnyMove's rig builder mixes over the texture (a capture aid, so the reader
can tell left legs from right) is taken out of the materials before export, unless
`--keep-paint`. Blender 5.2's exporter keeps the texture either way, but writes the paint
as two extra vertex-colour sets (one white, one all black) that nothing downstream needs;
without them the dragon's GLB is 0.3 MB smaller. The .blend on disk is never saved over.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAINT_LAYER = "legpaint"


def parse_args(argv: list[str]) -> dict:
    args = argv[argv.index("--") + 1:] if "--" in argv else list(argv)
    if len(args) < 3:
        raise SystemExit("usage: ... -- RIG_INFO.json MOTION.json OUT_DIR [--armature NAME] [--name NAME] "
                         "[--keep-paint] [--no-glb]")
    rest = args[3:]

    def value(flag, default=None):
        if flag not in rest:
            return default
        i = rest.index(flag)
        if i + 1 >= len(rest):
            raise SystemExit(f"{flag} needs a value")
        return rest[i + 1]

    return {"rig_info": Path(args[0]), "motion": Path(args[1]), "out": Path(args[2]),
            "armature": value("--armature"), "name": value("--name", "animation"),
            "keep_paint": "--keep-paint" in rest, "glb": "--no-glb" not in rest}


def iter_fcurves(action):
    """Every F-curve of an action: Blender 4.4+ keeps them per layer, strip and slot
    (`action.fcurves` is gone in 5.x), older Blender on the action itself."""
    if action is None:
        return
    layers = getattr(action, "layers", None)
    if layers:
        for layer in layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", ()):
                    yield from bag.fcurves
        return
    yield from getattr(action, "fcurves", ())


def is_identity(rows, tol: float = 1e-5) -> bool:
    return all(abs(rows[i][j] - (1.0 if i == j else 0.0)) <= tol for i in range(4) for j in range(4))


def max_joint_error_mm(solver, blender) -> float | None:
    """Largest distance between two (T, J, 3) joint tracks, in millimetres."""
    if not solver or not blender:
        return None
    worst = 0.0
    for a_frame, b_frame in zip(solver, blender):
        for a, b in zip(a_frame, b_frame):
            worst = max(worst, math.dist(a, b))
    return worst * 1000


def paint_mix_nodes(material):
    """(mix node, vertex-colour node, socket feeding the mix's A) for each limb-paint mix
    AnyMove's `common.paint_legs` put in front of a material's base colour."""
    found = []
    if material is None or not material.use_nodes:
        return found
    links = list(material.node_tree.links)
    for link in links:
        vc, mix = link.from_node, link.to_node
        if (vc.type != "VERTEX_COLOR" or getattr(vc, "layer_name", "") != PAINT_LAYER
                or mix.type != "MIX" or any(m == mix for m, _, _ in found)):
            continue
        # A mix's sockets share names across its data types, so go by the links: the
        # one input not fed by the paint is the texture (or nothing, a plain colour).
        # bpy hands out a new wrapper per access, so nodes compare with ==, never `is`.
        others = [x.from_socket for x in links if x.to_node == mix and x.from_node != vc]
        found.append((mix, vc, others[0] if others else None))
    return found


def strip_paint(meshes) -> int:
    """Take the limb paint out of every material: the texture goes straight back into the
    socket the mix fed, and the paint attribute is removed. Returns mixes removed."""
    removed = 0
    for mesh in meshes:
        for slot in mesh.material_slots:
            mat = slot.material
            for mix, vc, source in paint_mix_nodes(mat):
                tree = mat.node_tree
                targets = [link.to_socket for link in tree.links if link.from_node == mix]
                plain = next((s.default_value for s in mix.inputs
                              if s.enabled and s.identifier == "A_Color"), None)
                for target in targets:
                    if source is not None:
                        tree.links.new(source, target)
                    elif plain is not None:
                        target.default_value = plain
                tree.nodes.remove(mix)
                tree.nodes.remove(vc)
                removed += 1
        attr = mesh.data.color_attributes.get(PAINT_LAYER)
        if attr is not None:
            mesh.data.color_attributes.remove(attr)
    return removed


def main() -> None:
    import bpy
    from mathutils import Quaternion, Vector

    sys.path.insert(0, str(REPO))
    from image_to_3dlab.twocam.motion import load_motion, root_location, validate_motion

    opts = parse_args(sys.argv)
    out = opts["out"]
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads(opts["rig_info"].read_text(encoding="utf-8"))
    motion = load_motion(opts["motion"])

    scene = bpy.context.scene
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    if opts["armature"]:
        rig = bpy.data.objects.get(opts["armature"])
        if rig is None or rig.type != "ARMATURE":
            raise SystemExit(f"no armature named {opts['armature']!r} in this file")
    elif len(armatures) == 1:
        rig = armatures[0]
    else:
        raise SystemExit(f"{len(armatures)} armatures in this file: name one with --armature")
    if not is_identity([list(r) for r in rig.matrix_world]):
        raise SystemExit(f"armature {rig.name!r} is moved, turned or scaled: the capture is in "
                         "armature space, so apply its transform first")
    frames = validate_motion(motion, [b.name for b in rig.data.bones])
    meshes = [o for o in scene.objects if o.type == "MESH" and (
        o.parent == rig or any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers))]
    if not meshes:
        raise SystemExit(f"no mesh is skinned to armature {rig.name!r}")

    names = motion["bones"]
    fps = float(motion["fps"])
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start, scene.frame_end = 0, frames - 1

    root = names[0]
    rest = [list(r) for r in rig.data.bones[root].matrix_local]
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
    pr = rig.pose.bones[root]
    pr.rotation_mode = "QUATERNION"
    for t in range(frames):
        pr.rotation_quaternion = Quaternion(motion["root_quat"][t])
        pr.location = Vector(root_location(rest, motion["trans"][t]).tolist())
        pr.keyframe_insert("rotation_quaternion", frame=t)
        pr.keyframe_insert("location", frame=t)
        for i, n in enumerate(names[1:], 1):
            pb = rig.pose.bones[n]
            pb.rotation_euler = motion["euler"][t][i]
            pb.keyframe_insert("rotation_euler", frame=t)
    for fc in iter_fcurves(rig.animation_data.action if rig.animation_data else None):
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"

    joints = []
    for t in range(frames):
        scene.frame_set(t)
        mw = rig.matrix_world
        row = []
        for kp in info["keypoints"]:
            pb = rig.pose.bones[info["bones"][kp["bone"]]["name"]]
            row.append(list(mw @ (pb.head if kp["end"] == "head" else pb.tail)))
        joints.append(row)
    (out / "joints.json").write_text(json.dumps(
        {"names": [k["name"] for k in info["keypoints"]], "joints": joints, "fps": fps}), encoding="utf-8")

    report = {"armature": rig.name, "meshes": [m.name for m in meshes], "frames": frames, "fps": fps,
              "bones": len(names), "max_joint_error_mm": max_joint_error_mm(motion.get("joints"), joints),
              "paint_mixes_removed": 0, "glb": None}
    if not opts["keep_paint"]:
        report["paint_mixes_removed"] = strip_paint(meshes)
    if opts["glb"]:
        bpy.ops.object.select_all(action="DESELECT")
        for o in (rig, *meshes):
            o.select_set(True)
        glb = out / f"{opts['name']}.glb"
        bpy.ops.export_scene.gltf(filepath=str(glb), use_selection=True, export_animations=True,
                                  export_force_sampling=True, export_frame_range=True)
        report["glb"] = str(glb)
    (out / "apply.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("I2L_TWOCAM_APPLY::" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
