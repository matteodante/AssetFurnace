"""Make a rigged GLB ready for two-camera capture in Blender: turned, sized, sides named, limbs painted, headless.

blender -b --factory-startup --python scripts/blender_twocam_prepare.py -- \
    RIGGED.glb OUT_DIR --length-m 1.6 [--axis y] [--yaw-deg D] [--samples 64] [--no-render]

Takes any rigged model (a SkinTokens rig from the Animate tab, or one made by hand) and
writes what the capture needs, in AnyMove's layout so its own scripts read it too:

* OUT_DIR/rigged.blend: objects `rig` and `body`, facing -Y, `--length-m` long along
  `--axis`, standing on Z = 0, each limb painted its own colour (`legpaint`);
* OUT_DIR/rig_info.json: bones, rest matrices, joint limits, keypoints and the layout;
* OUT_DIR/plan.json: the turn, scale, new bone names and limb colours, for the record;
* OUT_DIR/ref_front.png, ref_side.png: the rest pose from both cameras (unless
  --no-render); `scripts/twocam_mocap.py prepare` lays them into ref.png.

The geometry decides everything (see image_to_3dlab/twocam/rigging.py); --yaw-deg overrides
the turn when the symmetry search picks the wrong one.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str]) -> dict:
    args = argv[argv.index("--") + 1:] if "--" in argv else list(argv)
    if len(args) < 2:
        raise SystemExit("usage: ... -- RIGGED.glb OUT_DIR --length-m L [--axis y] [--yaw-deg D] "
                         "[--samples N] [--no-render]")
    rest = args[2:]

    def value(flag, default=None, kind=str):
        if flag not in rest:
            return default
        i = rest.index(flag)
        if i + 1 >= len(rest):
            raise SystemExit(f"{flag} needs a value")
        return kind(rest[i + 1])

    length = value("--length-m", None, float)
    if length is None or length <= 0:
        raise SystemExit("--length-m (the creature's real size, in metres) is required")
    axis = value("--axis", "y")
    if axis not in ("x", "y", "z"):
        raise SystemExit("--axis must be x, y or z")
    yaw = value("--yaw-deg", None, float)
    return {"glb": Path(args[0]), "out": Path(args[1]), "length_m": length, "axis": axis,
            "yaw": None if yaw is None else math.radians(yaw), "samples": value("--samples", 64, int),
            "render": "--no-render" not in rest}


def main() -> None:
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import blender_twocam_studio as studio

    from image_to_3dlab.twocam import rigging

    opts = parse_args(sys.argv)
    out = opts["out"]
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(opts["glb"]))
    scene = bpy.context.scene

    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise SystemExit(f"the file holds {len(armatures)} armatures; one is needed")
    rig = armatures[0]
    bone_names = {b.name for b in rig.data.bones}
    meshes = [o for o in scene.objects if o.type == "MESH" and (
        any(m.type == "ARMATURE" and m.object == rig for m in o.modifiers)
        or (o.parent == rig and any(g.name in bone_names for g in o.vertex_groups)))]
    if not meshes:
        raise SystemExit("no mesh is skinned to the armature")
    keep = {rig, *meshes}
    for o in list(scene.objects):
        if o not in keep:
            bpy.data.objects.remove(o, do_unlink=True)

    # Everything into world space with identity object transforms.
    bpy.ops.object.select_all(action="DESELECT")
    for o in keep:
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    body = bpy.context.view_layer.objects.active if len(meshes) > 1 else meshes[0]
    for m in body.modifiers:
        if m.type == "ARMATURE":
            m.object = rig

    def skeleton():
        bones = list(rig.data.bones)
        index = {b.name: i for i, b in enumerate(bones)}
        return rigging.Skeleton([b.name for b in bones],
                                [index[b.parent.name] if b.parent else -1 for b in bones],
                                np.array([b.head_local[:] for b in bones]),
                                np.array([b.tail_local[:] for b in bones]))

    def vertices():
        co = np.empty(len(body.data.vertices) * 3)
        body.data.vertices.foreach_get("co", co)
        return co.reshape(-1, 3)

    matrix, yaw = rigging.normalising_matrix(vertices(), skeleton(), opts["length_m"], opts["yaw"], opts["axis"])
    bpy.ops.object.select_all(action="DESELECT")
    for o in (rig, body):
        o.matrix_world = Matrix(matrix.tolist()) @ o.matrix_world
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    verts = vertices()
    width = float(np.ptp(verts[:, 0]))
    sk = skeleton()
    plan = rigging.plan_bones(sk, tolerance=max(0.01, 0.05 * width))

    # Joined-up tails and AnyMove's roll, in edit mode; names after, so vertex groups follow.
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for i, name in enumerate(sk.names):
        eb = rig.data.edit_bones[name]
        eb.use_connect = False
        if np.linalg.norm(plan.tails[i] - np.array(eb.head[:])) > 1e-5:
            eb.tail = Vector(plan.tails[i].tolist())
        eb.align_roll(Vector(rigging.roll_vector(eb.head[:], eb.tail[:]).tolist()))
    bpy.ops.object.mode_set(mode="OBJECT")
    renames = {old: new for old, new in zip(sk.names, plan.names) if old != new}
    groups = {g.name: g for g in body.vertex_groups}
    for old, new in renames.items():
        rig.data.bones[old].name = new
    for old, new in renames.items():  # older Blender renames the groups itself; check, never assume
        if old in groups and body.vertex_groups.get(new) is None:
            groups[old].name = new
    rig.name = rig.data.name = "rig"
    body.name = body.data.name = "body"
    if body.parent != rig:
        body.parent = rig  # AnyMove's scripts look for the mesh under the rig

    painted = studio.paint_limbs(body, plan.paint)

    # rig_info.json, parents before children (the fit's kinematics need it)
    order = sk.order()
    position = {old: k for k, old in enumerate(order)}
    final = [plan.names[i] for i in order]
    bones = [rig.data.bones[n] for n in final]
    heads = np.array([b.head_local[:] for b in bones])
    tails = np.array([b.tail_local[:] for b in bones])
    parents = [position[sk.parents[i]] if sk.parents[i] >= 0 else -1 for i in order]
    children = [[k for k, p in enumerate(parents) if p == j] for j in range(len(final))]
    kps, ends = rigging.keypoints(final, parents, heads, tails, children)
    layout = rigging.layout_for(verts)
    info = {"bones": [], "keypoints": kps, "layout": {**layout, "Z_CENTER": studio.z_center(layout)}}
    for k, b in enumerate(bones):
        info["bones"].append({"name": b.name, "parent": parents[k],
                              "matrix_local": [list(r) for r in b.matrix_local], "length": b.length,
                              "head_joint": ends[k][0], "tail_joint": ends[k][1],
                              "limits_deg": plan.limits[order[k]]})
    (out / "rig_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    record = {"source": str(opts["glb"]), "length_m": opts["length_m"], "axis": opts["axis"],
              "yaw_deg": round(math.degrees(yaw), 2), "yaw_given": opts["yaw"] is not None,
              "renames": renames, "paint": plan.paint, "painted_vertices": painted,
              "limbs": len({k for k in plan.limbs if k}), "unpaired": plan.unpaired,
              "size_m": np.ptp(verts, 0).round(3).tolist(),
              "bones": len(final), "keypoints": len(kps), "layout": info["layout"]}
    (out / "plan.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "rigged.blend"))

    if opts["render"]:
        for pb in rig.pose.bones:
            pb.rotation_mode = "XYZ"
        record["render_device"] = studio.studio(scene, samples=opts["samples"])
        studio.render_views(out, "ref", info["layout"])
    print("I2L_TWOCAM_PREPARE::" + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
