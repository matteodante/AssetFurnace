"""Fit armour pieces onto a rigged base body in Blender, skin them to its skeleton, mask the body.

    blender -b --factory-startup --python scripts/blender_fit_parts.py -- job.json

`job.json` (written by `scripts/slot_parts.py fit`):

    {"body": "output/bodies/knight/body.json", "concept_size": [1024, 1024], "out": "fit/",
     "pieces": [{"slot": "helmet", "glb": ".../helmet_finished.glb", "concept_box": [x0, y0, x1, y1]},
                {"slot": "torso", "glb": ".../torso.glb", "concept_box": [...], "frame": [x0, z0, x1, z1, yc]}],
     "offset": 0.002, "cage_voxel": 0.007, "cage_faces": 5000}

`concept_box` is where the piece sits on the concept (route C), or the whole subject's box
when the piece was cut from a whole-character model (route B0); then `frame` is that
model's extent (Blender x0, z0, x1, z1 and depth centre) and places it. Per piece: placement through the body's camera (`slot_parts.fit`); paired
pieces fitted per side; gauntlets turned onto the forearm; a low-poly cage shrinkwrapped
just outside the body drives the piece by Surface Deform (Mesh Deform when that will not
bind); cage spikes put back; any vertex still under the skin pushed out to it; skin weights copied from the body (at most 4 per vertex), or
one bone at 100% for the helmet. The body gets a `covered` vertex colour: white everywhere
(viewers multiply the base colour by it), with alpha 0 where a piece lies over the skin, which
the game's material uses as an opacity mask to hide the body there.

Writes `set.glb` (masked body, skeleton, every piece), `preview.glb` (the same with the body
trimmed: faces wholly under a piece removed), one `<piece>.glb` per piece with the skeleton,
`body_masked.glb`, `body_trimmed.glb` and `fit.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from image_to_3dlab.slot_parts import fit, slots
from image_to_3dlab.slot_parts.camera import Camera

LIMB_REFINE = {"gloves", "boots"}


def parse_args(argv: list[str]) -> Path:
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(rest) != 1:
        raise SystemExit("usage: blender -b --python blender_fit_parts.py -- job.json")
    return Path(rest[0])


def load_job(path: Path) -> dict:
    job = json.loads(Path(path).read_text())
    for key in ("body", "concept_size", "pieces", "out"):
        if key not in job:
            raise ValueError(f"job.json misses {key!r}")
    for piece in job["pieces"]:
        if piece.get("slot") not in slots.CATALOGUE:
            raise ValueError(f"unknown slot {piece.get('slot')!r}")
        if "concept_box" not in piece:
            raise ValueError(f"{piece['slot']}: give concept_box")
    return job


def slot_bones(roles: dict, slot: str, side: str | None) -> list[str]:
    """Bones a slot sits on; for a paired slot, only that side's (and the middle line's)."""
    wanted = slots.CATALOGUE[slot].roles
    return [b for b, (role, _, s) in roles.items() if role in wanted and (side is None or s in (side, ""))]


def rigid_bone(roles: dict, slot: str, side: str | None = None) -> str | None:
    """The one bone a rigid piece follows: the helmet the first head bone, the weapon the right
    hand, the shield the left forearm, unless `side` says otherwise."""
    role = slots.CATALOGUE[slot].rigid
    side = side or ("L" if slot == "shield" else "R")
    if role is None:
        return None
    found = [(i, b) for b, (r, i, s) in roles.items() if r == role and (role == "head" or s in (side, ""))]
    if not found and role == "head":
        found = [(i, b) for b, (r, i, _) in roles.items() if r == "neck"]
    return min(found)[1] if found else None


# ------------------------------------------------------------------ Blender from here


def main() -> None:  # pragma: no cover - runs inside Blender
    import bmesh
    import bpy
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree

    from image_to_3dlab.slot_parts import body as body_mod

    job = load_job(parse_args(sys.argv))
    record = body_mod.load(job["body"])
    cam = Camera.from_dict(record["camera"])
    roles = {b: tuple(r) for b, r in record["roles"].items()}
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)

    def ctx(obj):
        return bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                                         selected_editable_objects=[obj])

    def coords(obj):
        co = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("co", co)
        return co.reshape(-1, 3)

    def set_coords(obj, co):
        obj.data.vertices.foreach_set("co", np.asarray(co, float).ravel())
        obj.data.update()

    def apply(obj, m):
        obj.data.transform(Matrix([list(r) for r in m]))

    def import_glb(path):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(path))
        return [o for o in bpy.data.objects if o not in before]

    def join(objs, name):
        meshes = [o for o in objs if o.type == "MESH" and len(o.data.polygons)]
        for o in objs:
            if o not in meshes:
                bpy.data.objects.remove(o, do_unlink=True)
        for o in meshes:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = Matrix.Identity(4)
            o.data.transform(mw)
        if len(meshes) > 1:
            with bpy.context.temp_override(active_object=meshes[0], selected_editable_objects=meshes,
                                           object=meshes[0], selected_objects=meshes):
                bpy.ops.object.join()
        meshes[0].name = name
        # glTF splits vertices along UV seams; weld them (UVs live on the face corners and stay)
        # so a piece is one surface, not hundreds of islands a cage can fling apart one by one
        bm = bmesh.new()
        bm.from_mesh(meshes[0].data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bm.to_mesh(meshes[0].data)
        bm.free()
        return meshes[0]

    def mirror_copy(obj, name):
        dup = obj.copy()
        dup.data = obj.data.copy()
        bpy.context.collection.objects.link(dup)
        c = coords(dup).mean(0)
        apply(dup, np.array([[-1, 0, 0, 2 * c[0]], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float))
        bm = bmesh.new()
        bm.from_mesh(dup.data)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(dup.data)
        bm.free()
        dup.name = name
        return dup

    def make_cage(obj, voxel):
        cage = obj.copy()
        cage.data = obj.data.copy()
        bpy.context.collection.objects.link(cage)
        remesh = cage.modifiers.new("remesh", "REMESH")
        remesh.mode, remesh.voxel_size = "VOXEL", voxel
        with ctx(cage):
            bpy.ops.object.modifier_apply(modifier="remesh")
        dec = cage.modifiers.new("dec", "DECIMATE")
        dec.ratio = min(1.0, job.get("cage_faces", 5000) / max(len(cage.data.polygons), 1))
        with ctx(cage):
            bpy.ops.object.modifier_apply(modifier="dec")
        return cage

    def cage_fit(obj, body_obj):
        """Surface Deform onto a voxel cage, trying a few cage sizes: whether it binds hangs on
        small details of the cage's topology (one torso bound at 0.007 and not at 0.0065)."""
        base = job.get("cage_voxel", 0.007)
        deform = cage = None
        for voxel in fit.cage_sizes(base):
            cage = make_cage(obj, voxel)
            deform = obj.modifiers.new("sdef", "SURFACE_DEFORM")
            deform.target = cage
            with ctx(obj):
                bpy.ops.object.surfacedeform_bind(modifier="sdef")
            if deform.is_bound:
                method = f"surface_deform@{voxel:.4f}"
                break
            obj.modifiers.remove(deform)
            bpy.data.objects.remove(cage, do_unlink=True)
            deform = None
        if deform is None:
            cage = make_cage(obj, base)
            deform = obj.modifiers.new("mdef", "MESH_DEFORM")
            deform.object, deform.precision = cage, 4
            with ctx(obj):
                bpy.ops.object.meshdeform_bind(modifier="mdef")
            method = "mesh_deform"
        wrap = cage.modifiers.new("wrap", "SHRINKWRAP")
        wrap.target, wrap.wrap_method, wrap.wrap_mode = body_obj, "NEAREST_SURFACEPOINT", "OUTSIDE"
        wrap.offset = job.get("offset", 0.002)
        with ctx(cage):
            bpy.ops.object.modifier_apply(modifier="wrap")
        with ctx(obj):
            bpy.ops.object.modifier_apply(modifier=deform.name)
        bpy.data.objects.remove(cage, do_unlink=True)
        return method

    def bind(obj, arm):
        mod = obj.modifiers.new("skeleton", "ARMATURE")
        mod.object = arm
        obj.parent = arm
        obj.matrix_parent_inverse = arm.matrix_world.inverted()

    def copy_weights(obj, body_obj, arm):
        dt = obj.modifiers.new("dt", "DATA_TRANSFER")
        dt.object, dt.use_vert_data = body_obj, True
        dt.data_types_verts = {"VGROUP_WEIGHTS"}
        dt.vert_mapping = "POLYINTERP_NEAREST"
        dt.layers_vgroup_select_src, dt.layers_vgroup_select_dst = "ALL", "NAME"
        with ctx(obj):
            bpy.ops.object.datalayout_transfer(modifier="dt")
            bpy.ops.object.modifier_apply(modifier="dt")
            bpy.ops.object.vertex_group_normalize_all(lock_active=False)
            bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
            bpy.ops.object.vertex_group_normalize_all(lock_active=False)
        bind(obj, arm)

    def one_bone(obj, bone, arm):
        group = obj.vertex_groups.new(name=bone)
        group.add(range(len(obj.data.vertices)), 1.0, "REPLACE")
        bind(obj, arm)

    def bvh_of(objs):
        verts, polys, offset = [], [], 0
        for o in objs:
            co = coords(o) @ np.array(o.matrix_world)[:3, :3].T + np.array(o.matrix_world)[:3, 3]
            verts.append(co)
            polys += [tuple(i + offset for i in p.vertices) for p in o.data.polygons]
            offset += len(co)
        return BVHTree.FromPolygons([Vector(v) for v in np.concatenate(verts)], polys)

    def export(path, objs):
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        options = {"filepath": str(path), "export_format": "GLB", "use_selection": True, "export_skins": True,
                   "export_animations": False}
        try:
            bpy.ops.export_scene.gltf(**options, export_vertex_color="ACTIVE")
        except TypeError:
            bpy.ops.export_scene.gltf(**options)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    objs = import_glb(record["rigged"])
    arm = next(o for o in objs if o.type == "ARMATURE")
    body_obj = max((o for o in objs if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    for o in objs:
        if o.type == "MESH" and o is not body_obj:
            bpy.data.objects.remove(o, do_unlink=True)
    body_obj.name = "body"
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    missing = [b for b in roles if b not in arm.data.bones]
    if missing:
        raise SystemExit(f"the rig's bones do not match body.json: {missing[:5]}")
    heads = {b.name: np.array(arm.matrix_world @ b.head_local) for b in arm.data.bones}
    mw = np.array(body_obj.matrix_world)
    body_co = coords(body_obj) @ mw[:3, :3].T + mw[:3, 3]
    groups = {g.name: g.index for g in body_obj.vertex_groups}
    weights = np.zeros((len(body_co), max(len(groups), 1)))
    for v in body_obj.data.vertices:
        for g in v.groups:
            weights[v.index, g.group] = g.weight
    root = next(b for b, (r, _, _) in roles.items() if r == "hips")
    centre_px = fit.midline_px(cam, float(heads[root][0]))
    body_size = record["image_size"]

    def region(bones):
        idx = [groups[b] for b in bones if b in groups]
        sel = np.isin(weights.argmax(1), idx) & (weights.max(1) > 0)
        return body_co[sel] if sel.any() else body_co

    report = {"pieces": {}, "body": record["name"]}
    fitted = []
    for piece in job["pieces"]:
        slot = piece["slot"]
        obj = join(import_glb(piece["glb"]), slot)
        info = {"faces": len(obj.data.polygons)}
        paired = slots.CATALOGUE[slot].paired
        box = fit.concept_to_body(piece["concept_box"], job["concept_size"], body_size)
        if "frame" in piece:  # route B0: placed by the whole-character model's extent
            m, place = fit.placement(coords(obj), box, cam, float((body_co[:, 1].min() + body_co[:, 1].max()) / 2),
                                     frame=piece["frame"])
            apply(obj, m)
            group = [(obj, None)]
            info["placement"] = {obj.name: place}
        else:
            group = [(obj, "R" if paired else None)]
            if paired:
                group.append((mirror_copy(obj, slot + "_L"), "L"))
                obj.name = slot + "_R"
            for o, side in group:
                under = region(slot_bones(roles, slot, side))
                side_box = fit.mirror_box(box, centre_px) if side == "L" else box
                m, place = fit.placement(coords(o), side_box, cam, float((under[:, 1].min() + under[:, 1].max()) / 2),
                                         body_depth=float(np.ptp(under[:, 1])))
                apply(o, m)
                info.setdefault("placement", {})[o.name] = place
        for o, side in group:
            if slot in LIMB_REFINE and side is not None:
                bones = [b for b in slot_bones(roles, slot, side) if roles[b][0] != "upperarm" or slot == "shoulders"]
                ordered = sorted(bones, key=lambda b: roles[b][1])
                axis = heads[ordered[-1]] - heads[ordered[0]] if len(ordered) > 1 else None
                if slot == "gloves" and axis is not None and np.linalg.norm(axis) > 1e-6:
                    m, turned = fit.turn_onto(coords(o), axis)
                    apply(o, m)
                    info.setdefault("turn_degrees", {})[o.name] = round(turned, 1)
                shift = fit.centre_across(coords(o).mean(0), region(bones).mean(0),
                                          axis if slot != "shoulders" else None)
                apply(o, np.array([[1, 0, 0, shift[0]], [0, 1, 0, shift[1]], [0, 0, 1, shift[2]], [0, 0, 0, 1.0]]))
            before = coords(o)
            info.setdefault("deform", {})[o.name] = cage_fit(o, body_obj)
            after = coords(o)
            edges = np.array([e.vertices[:] for e in o.data.edges])
            bad = (fit.spikes(before, after, edges) | fit.stretched(before, after, edges)) if len(edges) \
                else np.zeros(len(after), bool)
            if bad.any():
                set_coords(o, fit.smooth_spikes(after, edges, bad))
            info.setdefault("spikes_fixed", {})[o.name] = int(bad.sum())
            # last pass on the piece itself: only vertices still under the skin come out to it
            push = o.modifiers.new("push", "SHRINKWRAP")
            push.target, push.wrap_method, push.wrap_mode = body_obj, "NEAREST_SURFACEPOINT", "OUTSIDE"
            push.offset = job.get("offset", 0.002)
            with ctx(o):
                bpy.ops.object.modifier_apply(modifier="push")
            bone = rigid_bone(roles, slot)
            if bone is not None:
                one_bone(o, bone, arm)
                info["rigid_bone"] = bone
            else:
                copy_weights(o, body_obj, arm)
            fitted.append(o)
        report["pieces"][slot] = info

    # the covered mask: a body vertex is covered when its outward ray meets a piece close by
    bvh = bvh_of(fitted)
    normals = np.empty(len(body_co) * 3)
    body_obj.data.vertices.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3) @ mw[:3, :3].T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    covered = np.zeros(len(body_co))
    for i, (p, n) in enumerate(zip(body_co, normals)):
        if bvh.ray_cast(Vector(p) + Vector(n) * 1e-4, Vector(n), job.get("cover_gap", 0.06))[0] is not None:
            covered[i] = 1.0
    colour = body_obj.data.color_attributes.new("covered", "BYTE_COLOR", "POINT")
    rgba = np.ones((len(covered), 4))
    rgba[:, 3] = 1.0 - covered  # white everywhere, so viewers that multiply by COLOR_0 see no change
    colour.data.foreach_set("color", rgba.ravel())
    body_obj.data.color_attributes.active_color = colour
    report["covered_share"] = float(covered.mean())

    # a trimmed body too: faces wholly under a piece removed, for viewers and engines that
    # do not read the mask (and the studio's preview, `preview.glb`)
    trimmed = body_obj.copy()
    trimmed.data = body_obj.data.copy()
    bpy.context.collection.objects.link(trimmed)
    trimmed.name = "body_trimmed"
    bm = bmesh.new()
    bm.from_mesh(trimmed.data)
    bm.verts.ensure_lookup_table()
    hidden = [f for f in bm.faces if all(covered[v.index] > 0.5 for v in f.verts)]
    bmesh.ops.delete(bm, geom=hidden, context="FACES")
    bm.to_mesh(trimmed.data)
    bm.free()
    report["trimmed_faces"] = len(hidden)

    export(out / "set.glb", [arm, body_obj, *fitted])
    export(out / "preview.glb", [arm, trimmed, *fitted])
    export(out / "body_masked.glb", [arm, body_obj])
    export(out / "body_trimmed.glb", [arm, trimmed])
    for o in fitted:
        export(out / f"{o.name}.glb", [arm, o])
    report["files"] = ["set.glb", "preview.glb", "body_masked.glb", "body_trimmed.glb",
                       *[f"{o.name}.glb" for o in fitted]]
    (out / "fit.json").write_text(json.dumps(report, indent=1) + "\n")
    print("FIT_DONE", json.dumps({"pieces": list(report["pieces"]), "covered": round(report["covered_share"], 3)}))


if __name__ == "__main__":
    main()
