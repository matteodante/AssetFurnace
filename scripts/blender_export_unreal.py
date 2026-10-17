"""Reshape a finished GLB for Unreal in headless Blender: size, ground, facing, bones, LODs.

    blender -b --factory-startup --python scripts/blender_export_unreal.py -- job.json

Only the Blender half of `scripts/export_unreal.py`, which writes the job and then names
the textures and packs the ORM map in plain Python. Everything that can be worked out
without Blender lives in `image_to_3dlab/unreal_profile.py`; this file only moves data.

What it does to the model:
- drops SkinTokens' `Icosphere` helper, as `kimodo_retarget.py` already does;
- scales it to the real size (height or longest side), turns a humanoid to face
  Blender's -Y (Unreal's +Y, like Manny) and stands it on the origin: between the feet
  for a character, the centre of the base for an object;
- scales animation translations with the model, which Blender does not do by itself;
- renames a humanoid's bones like the UE5 Mannequin and adds a `root` bone at the origin;
- puts each extra animation on its own NLA track, exported as one glTF animation each;
- adds baked LODs as `LODn_SM_<Name>` nodes and a `UBX_`/`UCX_` collision node;
- culls back faces (glTF `doubleSided` false) unless the job says two sided (cloth);
- equipment takes all of that from its body (`body` in the job), so it lands on the body.

Writes the GLB and a JSON report of what it measured and changed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from image_to_3dlab import unreal_profile as up  # noqa: E402

CONVEX_FACES = 60  # few enough for a cheap physics hull, enough to follow the shape


def _clear(bpy):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.materials):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _import(bpy, path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    removed = []
    rigged = any(o.type == "ARMATURE" for o in new)
    for obj in list(new):
        if _skintokens_helper(obj, rigged) or obj.type not in ("MESH", "ARMATURE", "EMPTY"):
            removed.append(obj.name)
            new.remove(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
    # A GLB may hang its model under empty nodes (a scene root, a scale node). Their
    # transforms are kept on the objects themselves and the empties dropped, so every
    # mesh and skeleton is a root the reshape below can move.
    for obj in [o for o in new if o.type != "EMPTY"]:
        parent = obj.parent
        if parent is not None and parent.type == "EMPTY":
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
    for obj in [o for o in new if o.type == "EMPTY"]:
        new.remove(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    return new, removed


def _skintokens_helper(obj, rigged):
    """SkinTokens' `glTF_not_exported` sphere, 2 m wide, never part of the model.

    Told apart by more than its name, since any prop made from Blender's primitive is
    called Icosphere too: it comes with a skeleton but is not skinned to it.
    """
    return (rigged and obj.type == "MESH" and obj.name.startswith("Icosphere")
            and not obj.vertex_groups and not any(m.type == "ARMATURE" for m in obj.modifiers)
            and (obj.parent is None or obj.parent.type != "ARMATURE"))


def _fcurves(action):
    """Every F-curve of an action, for Blender 4.4+ layered actions and older ones."""
    if getattr(action, "layers", None):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", []):
                    yield from bag.fcurves
    elif hasattr(action, "fcurves"):
        yield from action.fcurves


def _world_box(bpy, objects, turn=None):
    """Lowest and highest corner of the meshes in world space, after `turn` if given."""
    from mathutils import Vector

    depsgraph = bpy.context.evaluated_depsgraph_get()
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world if turn is None else turn @ evaluated.matrix_world
        for vertex in mesh.vertices:
            p = matrix @ Vector(vertex.co)
            for i in range(3):
                lo[i], hi[i] = min(lo[i], p[i]), max(hi[i], p[i])
        evaluated.to_mesh_clear()
    if lo[0] == math.inf:
        raise ValueError("the model has no mesh")
    return lo, hi


def _rest(bpy, armatures, rest=True):
    for arm in armatures:
        arm.data.pose_position = "REST" if rest else "POSE"
    bpy.context.view_layer.update()


def _apply(bpy, objects, matrix):
    """Pre-multiply every top-level object by `matrix`, then bake it into the data."""
    for obj in objects:
        if obj.parent is None:
            obj.matrix_world = matrix @ obj.matrix_world
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _humanoid(arm):
    """(SkinTokens -> SOMA mapping, character frame), or None for any other skeleton."""
    import numpy as np
    from kimodo_retarget import map_skintokens_to_soma

    bones = arm.data.bones
    parents = {b.name: (b.parent.name if b.parent else None) for b in bones}
    heads = {b.name: np.array(arm.matrix_world @ b.head_local) for b in bones}
    if sum(1 for p in parents.values() if p is None) != 1:
        return None
    try:
        return map_skintokens_to_soma(parents, heads)
    except (ValueError, StopIteration, KeyError, IndexError):
        return None


def _bone_order(arm):
    order, queue = [], [b for b in arm.data.bones if b.parent is None]
    while queue:
        bone = queue.pop(0)
        order.append(bone.name)
        queue += list(bone.children)
    return order


def _measure_skeleton(bpy, arm, meshes, job):
    """Scale, turn and ground offset for a skinned model, plus the bone renames."""
    from mathutils import Matrix

    _rest(bpy, [arm])
    found = _humanoid(arm)
    order = _bone_order(arm)
    parents = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    if found:
        mapping, frame = found
        plan = up.bone_rename_plan(order, parents, mapping)
        forward = [float(v) for v in frame[:, 2]]
    else:
        plan = up.clean_bone_names(order)
        forward = [float(v) for v in job.get("forward", (0.0, -1.0, 0.0))]
    yaw = up.yaw_to_face_front(forward)
    # Measured once turned: a creature facing half-way between two axes has a wider box
    # before the turn, and sizing it there made a 150 cm drake arrive 180 cm long.
    lo, hi = _world_box(bpy, meshes, Matrix.Rotation(yaw, 4, "Z"))
    dims = [h - low for low, h in zip(lo, hi)]
    scale = up.scale_factor(dims, job["size_cm"], job["measure"])
    turn = Matrix.Rotation(yaw, 4, "Z") @ Matrix.Diagonal((scale, scale, scale, 1.0))
    feet = [b for b, joint in (found[0].items() if found else []) if joint in ("LeftFoot", "RightFoot")]
    feet_xy = None
    if len(feet) == 2:
        points = [turn @ (arm.matrix_world @ arm.data.bones[b].head_local) for b in feet]
        feet_xy = ((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2)
    box_lo, box_hi = [v * scale for v in lo], [v * scale for v in hi]
    offset = up.ground_offset(box_lo, box_hi, centre=feet_xy)
    return {"scale": scale, "yaw": yaw, "offset": list(offset), "plan": plan, "humanoid": bool(found),
            "dims_before_m": dims, "forward_before": forward}


def _transform_matrix(t):
    from mathutils import Matrix

    return (Matrix.Translation(t["offset"]) @ Matrix.Rotation(t["yaw"], 4, "Z")
            @ Matrix.Diagonal((t["scale"], t["scale"], t["scale"], 1.0)))


def _rename_bones(bpy, arm, plan, actions):
    # Blender keeps bone names unique on every assignment: renaming `bone.0` to `bone_0`
    # while another bone still owns `bone_0` would give `bone_0.001`. Every bone moves to
    # a name nobody has first, then to its planned one.
    planned = [bone for bone in arm.data.bones if bone.name in plan]
    final = {}
    for index, bone in enumerate(planned):
        final[f"__af_rename_{index}"] = plan[bone.name]
        bone.name = f"__af_rename_{index}"
    for bone in planned:
        bone.name = final[bone.name]
    names = {b.name for b in arm.data.bones}
    for action in actions:
        for curve in _fcurves(action):
            renamed = up.renamed_path(curve.data_path, plan)
            # Blender already fixes the actions it can see; fix the rest, never twice.
            if renamed != curve.data_path and renamed.split('"')[1:2] and renamed.split('"')[1] in names:
                curve.data_path = renamed


def _add_root(bpy, arm):
    """A `root` bone on the floor under the model, parent of the old top bones.

    Added only once the model stands on the origin: added before, it would sit wherever
    the source's origin was (the middle of the body, for SkinTokens), and Unreal moves
    the whole character to the root's reference position whenever an animation locks
    its root motion there, which lifts it off the ground by that much.
    """
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    edit = arm.data.edit_bones
    tops = [b for b in edit if b.parent is None]
    root = edit.new(up.ROOT_BONE)
    root.head, root.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.25)
    root.use_deform = False
    for bone in tops:
        bone.parent = root
    bpy.ops.object.mode_set(mode="OBJECT")


def _scale_translations(actions, scale):
    for action in actions:
        for curve in _fcurves(action):
            if up.is_translation_path(curve.data_path):
                for key in curve.keyframe_points:
                    key.co[1] *= scale
                    key.handle_left[1] *= scale
                    key.handle_right[1] *= scale


def _nla(arm, named_actions):
    """One NLA track per animation, so the glTF exporter writes each under its own name."""
    data = arm.animation_data or arm.animation_data_create()
    data.action = None
    for name, action in named_actions:
        action.name = name
        action.use_fake_user = True
        track = data.nla_tracks.new()
        track.name = name
        strip = track.strips.new(name, int(action.frame_range[0]), action)
        if hasattr(strip, "action_slot") and getattr(action, "slots", None) and strip.action_slot is None:
            strip.action_slot = action.slots[0]
        track.mute = False


def _materials(objects, base, lod=0):
    seen = []
    for obj in objects:
        for slot in obj.material_slots:
            if slot.material is not None and slot.material not in seen:
                seen.append(slot.material)
    for i, material in enumerate(seen):
        name = up.lod_material(base, lod)
        material.name = name if i == 0 else f"{name}_{i:02d}"


def _join(bpy, meshes, name):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = obj.data.name = name
    return obj


def _hull(bmesh, mesh, points):
    """Replace `mesh` with the convex hull of `points`, and nothing else."""
    bm = bmesh.new()
    for point in points:
        bm.verts.new(point)
    result = bmesh.ops.convex_hull(bm, input=list(bm.verts))
    # a point can be both interior and unused (repeated points: a Pixal3D mesh's seams), and
    # delete refuses one named twice
    leftovers = list(dict.fromkeys(g for g in result["geom_interior"] + result["geom_unused"]
                                   if isinstance(g, bmesh.types.BMVert)))
    bmesh.ops.delete(bm, geom=leftovers, context="VERTS")
    bm.to_mesh(mesh)
    bm.free()


def _collision(bpy, target, kind, name):
    import bmesh

    lo, hi = _world_box(bpy, [target])
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=[(a + b) / 2 for a, b in zip(lo, hi)])
        obj = bpy.context.active_object
        obj.scale = [max(b - a, 1e-4) for a, b in zip(lo, hi)]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    else:
        mesh = bpy.data.meshes.new(name)
        _hull(bmesh, mesh, [target.matrix_world @ vertex.co for vertex in target.data.vertices])
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        if len(mesh.polygons) > CONVEX_FACES:
            decimate = obj.modifiers.new("decimate", "DECIMATE")
            decimate.ratio = CONVEX_FACES / len(mesh.polygons)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=decimate.name)
            # The decimated surface is only a set of points to hull again: a fresh hull from
            # those points alone, so no old face survives inside it.
            _hull(bmesh, mesh, [vertex.co.copy() for vertex in mesh.vertices])
    obj.name = obj.data.name = name
    obj.data.materials.clear()
    return obj


def run(job):
    import bpy
    from mathutils import Matrix

    names = up.asset_names(job["kind"], job["base"])
    report = {"names": names, "removed": [], "lods": 1, "collision": None, "animations": []}
    _clear(bpy)
    skinned = job["kind"] != "prop"

    body = None
    if job["kind"] == "equipment":
        objects, removed = _import(bpy, job["body"])
        arm = next((o for o in objects if o.type == "ARMATURE"), None)
        if arm is None:
            raise ValueError("the body GLB has no skeleton")
        body = _measure_skeleton(bpy, arm, [o for o in objects if o.type == "MESH"], job)
        _clear(bpy)

    objects, removed = _import(bpy, job["source"])
    report["removed"] += removed
    meshes = [o for o in objects if o.type == "MESH"]
    armatures = [o for o in objects if o.type == "ARMATURE"]
    if not meshes:
        raise ValueError("the source GLB has no mesh")

    if skinned:
        if len(armatures) != 1:
            raise ValueError(f"expected one skeleton, found {len(armatures)}")
        arm = armatures[0]
        actions = []
        if arm.animation_data and arm.animation_data.action:
            actions.append((job.get("source_animation") or names["animation_prefix"] + "Anim",
                            arm.animation_data.action))
        for extra in job.get("animations", []):
            extra_objects, _ = _import(bpy, extra["path"])
            extra_arm = next((o for o in extra_objects if o.type == "ARMATURE"), None)
            if not (extra_arm and extra_arm.animation_data and extra_arm.animation_data.action):
                raise ValueError(f"{Path(extra['path']).name} holds no animation")
            actions.append((extra["name"], extra_arm.animation_data.action))
            for obj in extra_objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        transform = body or _measure_skeleton(bpy, arm, meshes, job)
        _nla(arm, actions)
        _rename_bones(bpy, arm, transform["plan"], [a for _, a in actions])
        _rest(bpy, [arm])
        _apply(bpy, [arm, *meshes], _transform_matrix(transform))
        _add_root(bpy, arm)
        _scale_translations([a for _, a in actions], transform["scale"])
        _rest(bpy, [arm], rest=False)
        mesh = _join(bpy, meshes, names["mesh"] + "_Mesh") if len(meshes) > 1 else meshes[0]
        mesh.name = names["mesh"]
        arm.name = names["mesh"] + "_Rig"
        _materials([mesh], job["base"])
        report["animations"] = [name for name, _ in actions]
        report.update(humanoid=transform["humanoid"], scale=transform["scale"], yaw_deg=math.degrees(transform["yaw"]),
                      offset_m=transform["offset"], bones=transform["plan"], dims_before_m=transform["dims_before_m"])
        _rest(bpy, [arm])
        lo, hi = _world_box(bpy, [mesh])
        _rest(bpy, [arm], rest=False)
        report["root_head_m"] = list(arm.matrix_world @ arm.data.bones[up.ROOT_BONE].head_local)
    else:
        if job.get("grip"):   # a weapon or shield: the pivot on its grip, the blade or head up
            from mathutils import Vector

            x, y, z = job["grip"]["point"]
            ax, ay, az = job["grip"]["axis"]
            # glTF (x, y, z) is Blender (x, -z, y) once imported
            point, axis = Vector((x, -z, y)), Vector((ax, -az, ay)).normalized()
            turn = axis.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix().to_4x4() @ Matrix.Translation(-point)
            lo, hi = _world_box(bpy, meshes, turn)
            dims = [h - low for low, h in zip(lo, hi)]
            scale = up.scale_factor(dims, job["size_cm"], job["measure"])
            offset = [0.0, 0.0, 0.0]
            matrix = Matrix.Diagonal((scale, scale, scale, 1.0)) @ turn
        else:
            lo, hi = _world_box(bpy, meshes)
            dims = [h - low for low, h in zip(lo, hi)]
            scale = up.scale_factor(dims, job["size_cm"], job["measure"])
            turn = Matrix.Diagonal((scale, scale, scale, 1.0))
            offset = up.ground_offset([v * scale for v in lo], [v * scale for v in hi])
            matrix = Matrix.Translation(offset) @ turn
        _apply(bpy, meshes, matrix)
        lod_files = job.get("lods", [])
        node = up.lod_node(names["mesh"], 0) if lod_files else names["mesh"]
        mesh = _join(bpy, meshes, node)
        _materials([mesh], job["base"], 0)
        for index, path in enumerate(lod_files, start=1):
            lod_objects, lod_removed = _import(bpy, path)
            report["removed"] += lod_removed
            lod_meshes = [o for o in lod_objects if o.type == "MESH"]
            _apply(bpy, lod_meshes, matrix)
            lod = _join(bpy, lod_meshes, up.lod_node(names["mesh"], index))
            _materials([lod], job["base"], index)
        report["lods"] = 1 + len(lod_files)
        if job.get("collision") in ("box", "convex"):
            _collision(bpy, mesh, job["collision"], names["collision"][job["collision"]])
            report["collision"] = job["collision"]
        if job.get("grip"):   # the socket the game attaches it by, where the pivot is (Interchange reads SOCKET_)
            socket = bpy.data.objects.new("SOCKET_grip", None)
            bpy.context.scene.collection.objects.link(socket)
            report["grip_socket"] = "SOCKET_grip"
        report.update(humanoid=False, scale=scale, yaw_deg=0.0, offset_m=list(offset), dims_before_m=dims)
        lo, hi = _world_box(bpy, [mesh])

    report["box_after_m"] = [lo, hi]
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    exported_animations = bool(skinned and report["animations"])
    # A base body fitted with armour carries a vertex colour with alpha 0 under the pieces (the
    # fit step's `covered`; reimported, glTF calls it Color), read by the game's material as an
    # opacity mask. Blender exports colours only when a material uses them, so a colour whose
    # alpha drops below one half somewhere is asked for here, and nothing else.
    colours = {}
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        for attr in o.data.color_attributes:
            values = [0.0] * (len(attr.data) * 4)
            attr.data.foreach_get("color", values)
            if values and min(values[3::4]) < 0.5:
                o.data.color_attributes.active_color = attr
                colours = {"export_vertex_color": "ACTIVE"}
                report["vertex_colour"] = attr.name
                break
    # glTF's exporter writes doubleSided = not use_backface_culling
    two_sided = bool(job.get("two_sided", False))
    for o in bpy.context.scene.objects:
        for slot in getattr(o, "material_slots", []):
            if slot.material is not None:
                slot.material.use_backface_culling = not two_sided
    report["two_sided"] = two_sided
    bpy.ops.export_scene.gltf(
        filepath=str(out), export_format="GLB", export_yup=True,
        export_animations=exported_animations,
        **({"export_animation_mode": "NLA_TRACKS"} if exported_animations else {}),
        export_skins=skinned, export_cameras=False, export_lights=False, export_extras=False, **colours)
    report["blender"] = bpy.app.version_string
    return report


def main(argv=None):
    argv = sys.argv[sys.argv.index("--") + 1:] if argv is None and "--" in sys.argv else (argv or [])
    if len(argv) != 1:
        raise SystemExit("usage: blender -b --factory-startup --python blender_export_unreal.py -- job.json")
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        report = run(job)
        report["ok"] = True
    except Exception as exc:  # the caller reads the report, not Blender's exit code alone
        import traceback

        report = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
    Path(job["report"]).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("EXPORT_UNREAL::" + json.dumps({"ok": report["ok"], "out": job["out"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
