"""Fit armour pieces onto a rigged base body in Blender, skin them to its skeleton, mask the body.

    blender -b --factory-startup --python scripts/blender_fit_parts.py -- job.json

`job.json` (written by `scripts/slot_parts.py fit`, or `refit` to move a fitted set onto
another body: then `source_body` names the first body's body.json and every piece has
`"transfer": true` and is that body's fitted piece GLB):

    {"body": "output/bodies/knight/body.json", "concept_size": [1024, 1024], "out": "fit/",
     "pieces": [{"slot": "helmet", "glb": ".../helmet_finished.glb", "concept_box": [x0, y0, x1, y1]},
                {"slot": "torso", "glb": ".../torso.glb", "concept_box": [...], "frame": [x0, z0, x1, z1, yc]}],
     "offset": 0.002, "cage_voxel": 0.007, "cage_faces": 5000}

Only worn pieces are fitted: a weapon or shield is an asset of its own, never fitted to a
body, and a job naming one is refused. `concept_box` is where a piece sits on the concept
(route C), or the whole subject's box when the piece was cut from a whole-character model
(route B0); then `frame` is that model's extent (Blender x0, z0, x1, z1 and depth centre) and places it. Per piece: placement through the body's camera (`slot_parts.fit`); paired
pieces fitted per side; gauntlets turned onto the forearm; a low-poly cage shrinkwrapped
just outside the body drives the piece by Surface Deform (Mesh Deform when that will not
bind); cage spikes put back; any vertex still under the skin pushed out to it; skin weights copied from the body (at most 4 per vertex), or
one bone at 100% for the helmet, which keeps its shape (no cage: it only follows the head).
The body gets a `covered` vertex colour: white everywhere
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
        if piece["slot"] in slots.STANDALONE:
            raise ValueError(f"{piece['slot']}: a weapon or shield is an asset of its own, never fitted to a body")
        if "concept_box" not in piece and not piece.get("transfer"):
            raise ValueError(f"{piece['slot']}: give concept_box")
        if piece.get("transfer") and "source_body" not in job:
            raise ValueError("a transferred piece needs the job's source_body")
    return job


def slot_bones(roles: dict, slot: str, side: str | None) -> list[str]:
    """Bones a slot sits on; for a paired slot, only that side's (and the middle line's)."""
    wanted = slots.CATALOGUE[slot].roles
    return [b for b, (role, _, s) in roles.items() if role in wanted and (side is None or s in (side, ""))]


def rigid_bone(roles: dict, slot: str, side: str | None = None, carried: dict | None = None) -> str | None:
    """The one bone a rigid piece follows: the helmet a head bone; None for a piece with no rigid
    bone. With `carried` (body vertices each bone carries most), the bone carrying most wins: a
    rig can split the head over two bones, and a helmet on the small one turned apart from the
    head (31% holes on a second body)."""
    role = slots.CATALOGUE[slot].rigid
    if role is None:
        return None
    found = [(i, b) for b, (r, i, s) in roles.items() if r == role and (role == "head" or s in (side, ""))]
    if not found and role == "head":
        found = [(i, b) for b, (r, i, _) in roles.items() if r == "neck"]
    if not found:
        return None
    if carried:
        return max(found, key=lambda f: (carried.get(f[1], 0), -f[0]))[1]
    return min(found)[1]


def head_bones(roles: dict) -> list[str]:
    """The bones whose skin the helmet must hold. A rig with one head-and-neck joint has no
    "head" bone: that joint carries the head. With neither, nothing, never the whole body."""
    return [b for b, (r, _, _) in roles.items() if r == "head"] or [b for b, (r, _, _) in roles.items() if r == "neck"]


def limb_axis(roles: dict, heads: dict, bones: list[str], slot: str):
    """The line a limb piece lies along, from the first bone of its chain to the last. For the
    gloves, the forearm's head to the hand's (the wrist): the fingers inherit the hand's role
    and the last of them can be the thumb, which turned the gauntlet toward it."""
    ordered = sorted(bones, key=lambda b: roles[b][1])
    if len(ordered) < 2:
        return None
    if slot == "gloves":
        forearm = [b for b in ordered if roles[b][0] == "forearm"]
        hand = [b for b in ordered if roles[b][0] == "hand"]
        if forearm and hand:
            return np.asarray(heads[hand[0]], float) - np.asarray(heads[forearm[0]], float)
    return np.asarray(heads[ordered[-1]], float) - np.asarray(heads[ordered[0]], float)


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
        for o in objs:  # a fitted piece arrives skinned to its first body: keep its rest shape only
            if o.type == "MESH":
                for mod in list(o.modifiers):
                    if mod.type == "ARMATURE":
                        o.modifiers.remove(mod)
                o.vertex_groups.clear()
        # importing a skinned GLB, Blender adds an icosphere for the bones' display, kept in the
        # importer's own glTF_not_exported collection: it is not part of the piece
        meshes = [o for o in objs if o.type == "MESH" and len(o.data.polygons)
                  and not any(c.name.startswith("glTF_not_exported") for c in o.users_collection)]
        bpy.context.view_layer.update()
        worlds = {o.name: o.matrix_world.copy() for o in meshes}  # before any parent goes away
        for o in meshes:
            o.parent = None
            o.matrix_world = Matrix.Identity(4)
            o.data.transform(worlds[o.name])
        for o in objs:
            if o not in meshes:
                bpy.data.objects.remove(o, do_unlink=True)
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
        for face in bm.faces:
            face.smooth = True
        bm.to_mesh(meshes[0].data)
        bm.free()
        # flat or custom normals make the exporter split every corner: three vertices a face
        if meshes[0].data.has_custom_normals:
            with ctx(meshes[0]):
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
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

    def split_sides(obj, slot, centre_x):
        """A paired piece cut from a whole model holds both sides: one object per side."""
        out = []
        for side, keep in (("R", lambda x: x < centre_x), ("L", lambda x: x >= centre_x)):
            dup = obj.copy()
            dup.data = obj.data.copy()
            bpy.context.collection.objects.link(dup)
            dup.name = f"{slot}_{side}"
            b = bmesh.new()
            b.from_mesh(dup.data)
            bmesh.ops.delete(b, geom=[f for f in b.faces if not keep(f.calc_center_median().x)], context="FACES")
            b.to_mesh(dup.data)
            b.free()
            if len(dup.data.polygons):
                out.append((dup, side))
            else:
                bpy.data.objects.remove(dup, do_unlink=True)
        bpy.data.objects.remove(obj, do_unlink=True)
        return out

    def push_off_skin(obj):
        """Every vertex under the skin (or closer than the offset) moved out along the skin's
        normal, its neighbours following smoothly (`fit.smooth_push`): the piece keeps its shape."""
        offset = job.get("offset", 0.002)
        mw = np.array(obj.matrix_world)
        co = coords(obj) @ mw[:3, :3].T + mw[:3, 3]
        obj.data.update()
        normals = np.empty(len(obj.data.vertices) * 3)
        obj.data.vertices.foreach_get("normal", normals)
        normals = normals.reshape(-1, 3) @ mw[:3, :3].T

        def needs(points):
            """Only the piece's outward-facing surface is pushed: a hollow piece's inner wall lies
            inside the body by design and stays hidden under the outer one; pushed, it went
            through the outer wall (the shards of a breastplate)."""
            need = np.zeros_like(points)
            for i, p in enumerate(points):
                loc, nrm, _, dist = body_bvh.find_nearest(Vector(p))
                if loc is None or np.dot(normals[i], np.array(nrm)) <= job.get("inner_wall", 0.0):
                    continue
                gap = dist if (Vector(p) - loc).dot(nrm) >= 0 else -dist
                if gap < offset:
                    need[i] = (offset - gap) * np.array(nrm)
            return need

        edges = np.array([e.vertices[:] for e in obj.data.edges], int)
        first = None
        for _ in range(job.get("push_passes", 3)):  # the spread moves some neighbours back under: again
            need = needs(co)
            first = int((np.linalg.norm(need, axis=1) > 0).sum()) if first is None else first
            if not need.any():
                break
            co = fit.smooth_push(co, edges, need, rounds=job.get("push_rounds", 80))
        inv = np.linalg.inv(mw)
        set_coords(obj, co @ inv[:3, :3].T + inv[:3, 3])
        return f"push@{first}"

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
    body_bvh = bvh_of([body_obj])  # the skin at rest, in world space, for pushing pieces off it
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

    source = None
    if job.get("source_body"):  # refit: the first body's regions, to carry its fitted pieces over
        first = body_mod.load(job["source_body"])
        src_objs = import_glb(first["rigged"])
        src_arm = next(o for o in src_objs if o.type == "ARMATURE")
        src_body = max((o for o in src_objs if o.type == "MESH"), key=lambda o: len(o.data.vertices))
        src_arm.data.pose_position = "REST"
        bpy.context.view_layer.update()
        src_mw = np.array(src_body.matrix_world)
        src_co = coords(src_body) @ src_mw[:3, :3].T + src_mw[:3, 3]
        src_groups = {g.name: g.index for g in src_body.vertex_groups}
        src_w = np.zeros((len(src_co), max(len(src_groups), 1)))
        for v in src_body.data.vertices:
            for g in v.groups:
                src_w[v.index, g.group] = g.weight
        source = {"roles": {b: tuple(r) for b, r in first["roles"].items()}, "co": src_co, "w": src_w,
                  "groups": src_groups}
        for o in src_objs:
            bpy.data.objects.remove(o, do_unlink=True)

    def source_region(bones):
        idx = [source["groups"][b] for b in bones if b in source["groups"]]
        sel = np.isin(source["w"].argmax(1), idx) & (source["w"].max(1) > 0)
        return source["co"][sel] if sel.any() else source["co"]

    report = {"pieces": {}, "body": record["name"]}
    fitted = []

    dominant = weights.argmax(1)
    names_by_index = {i: n for n, i in groups.items()}
    carried = {names_by_index[i]: int(c) for i, c in zip(*np.unique(dominant[weights.max(1) > 0], return_counts=True))
               if i in names_by_index}

    def shape_and_skin(o, slot, info):
        """Cage fit (not for a rigid piece: a helmet keeps its shape and only follows the head),
        spikes put back, sunken vertices pushed out, then skinned."""
        bone = rigid_bone(roles, slot, carried=carried)

        def stage(name):  # with "debug_dir" in the job: the piece after each step, to see where it goes wrong
            if job.get("debug_dir"):
                Path(job["debug_dir"]).mkdir(parents=True, exist_ok=True)
                export(Path(job["debug_dir"]) / f"{o.name}_{name}.glb", [o])
                export(Path(job["debug_dir"]) / f"{o.name}_{name}_on_body.glb", [o, body_obj])

        stage("1_placed")
        if slot == "helmet":  # room for the head inside, whatever the outline gave
            bones = head_bones(roles)
            if bones:  # region() of no bones is the whole body: the helmet would grow round it
                m, grown = fit.enclose(coords(o), region(bones))
                apply(o, m)
                info["enclose_scale"] = round(grown, 3)
        if bone is None and job.get("fit_method", "push") == "push":  # the default since 17/10
            info.setdefault("deform", {})[o.name] = push_off_skin(o)
            stage("2_pushed_off")
        elif bone is None:
            before = coords(o)
            info.setdefault("deform", {})[o.name] = cage_fit(o, body_obj)
            stage("2_caged")
            after = coords(o)
            edges = np.array([e.vertices[:] for e in o.data.edges])
            bad = (fit.spikes(before, after, edges) | fit.stretched(before, after, edges)) if len(edges) \
                else np.zeros(len(after), bool)
            if bad.any():
                set_coords(o, fit.smooth_spikes(after, edges, bad))
            info.setdefault("spikes_fixed", {})[o.name] = int(bad.sum())
            stage("3_spikes")
        else:
            info.setdefault("deform", {})[o.name] = "rigid"
        # last pass, for a rigid piece or the cage: vertices still under the skin come out to it
        # one by one (the smooth push has done this already, and keeps a hollow piece's inner wall)
        if bone is not None or job.get("fit_method", "push") != "push":
            push = o.modifiers.new("push", "SHRINKWRAP")
            push.target, push.wrap_method, push.wrap_mode = body_obj, "NEAREST_SURFACEPOINT", "OUTSIDE"
            push.offset = job.get("offset", 0.002)
            with ctx(o):
                bpy.ops.object.modifier_apply(modifier="push")
            stage("4_pushed")
        # a closed island the cage folded through itself faces inward: turned back out
        o.data.calc_loop_triangles()
        tris = np.array([lt.vertices[:] for lt in o.data.loop_triangles], int)
        owner = np.array([lt.polygon_index for lt in o.data.loop_triangles], int)
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", co)
        islands = fit.inward_islands(co.reshape(-1, 3), tris) if len(tris) else []
        if islands:
            flip = set(np.unique(owner[np.concatenate(islands)]).tolist())
            b = bmesh.new()
            b.from_mesh(o.data)
            b.faces.ensure_lookup_table()
            bmesh.ops.reverse_faces(b, faces=[b.faces[i] for i in flip])
            b.to_mesh(o.data)
            b.free()
            info.setdefault("islands_turned_out", {})[o.name] = len(islands)
        if bone is not None:
            one_bone(o, bone, arm)
            info["rigid_bone"] = bone
        else:
            copy_weights(o, body_obj, arm)
        fitted.append(o)
    for piece in job["pieces"]:
        slot = piece["slot"]
        obj = join(import_glb(piece["glb"]), slot)
        info = {"faces": len(obj.data.polygons)}
        paired = slots.CATALOGUE[slot].paired
        if piece.get("transfer"):  # refit: carried over region to region
            stem = Path(piece["glb"]).stem
            if paired and stem == slot:  # one file holding both shoulders: each side carried on its own
                group = split_sides(obj, slot, float(np.median(source["co"][:, 0])))
            else:
                side = ("L" if stem.endswith("_L") else "R") if paired else None
                if side:
                    obj.name = f"{slot}_{side}"
                group = [(obj, side)]
            for o, side in group:
                src = source_region(slot_bones(source["roles"], slot, side))
                dst = region(slot_bones(roles, slot, side))
                apply(o, fit.region_transfer((src.min(0), src.max(0)), (dst.min(0), dst.max(0))))
                info.setdefault("placement", {})[o.name] = "transferred"
                shape_and_skin(o, slot, info)
            merged = report["pieces"].setdefault(slot, {})
            for key, value in info.items():  # the two sides of a pair arrive one at a time
                if isinstance(value, dict):
                    merged.setdefault(key, {}).update(value)
                else:
                    merged[key] = value
            continue
        box = fit.concept_to_body(piece["concept_box"], job["concept_size"], body_size)
        if "frame" in piece:  # route B0: placed by the whole-character model's extent
            m, place = fit.placement(coords(obj), box, cam, float((body_co[:, 1].min() + body_co[:, 1].max()) / 2),
                                     frame=piece["frame"])
            apply(obj, m)
            info["placement"] = {obj.name: place}
            group = split_sides(obj, slot, float(heads[root][0])) if paired else [(obj, None)]
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
                axis = limb_axis(roles, heads, bones, slot)
                if slot == "gloves" and axis is not None and np.linalg.norm(axis) > 1e-6:
                    m, turned = fit.turn_onto(coords(o), axis)
                    apply(o, m)
                    info.setdefault("turn_degrees", {})[o.name] = round(turned, 1)
                shift = fit.centre_across(coords(o).mean(0), region(bones).mean(0),
                                          axis if slot != "shoulders" else None)
                apply(o, np.array([[1, 0, 0, shift[0]], [0, 1, 0, shift[1]], [0, 0, 1, shift[2]], [0, 0, 0, 1.0]]))
            shape_and_skin(o, slot, info)
        report["pieces"][slot] = info

    # the covered mask: a body vertex is hidden only where a worn piece lies over it at rest and in
    # every sampled moment of the body's clips. Hidden by its rest pose alone, the skin a shoulder
    # piece lifts off in a swing showed through (holes of 6-10% in the acceptance)
    worn = fitted
    gap = job.get("cover_gap", 0.06)

    def covered_now(depsgraph=None):
        """1 where the skin's outward ray meets a worn piece within the gap, in the pose shown."""
        objs = [body_obj, *worn]
        evaluated = {o.name: o.evaluated_get(depsgraph) if depsgraph else o for o in objs}
        meshes = {name: e.to_mesh() if depsgraph else e.data for name, e in evaluated.items()}
        verts, polys, offset = [], [], 0
        for o in worn:
            me, m = meshes[o.name], np.array(evaluated[o.name].matrix_world)
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            verts.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
            polys += [tuple(i + offset for i in poly.vertices) for poly in me.polygons]
            offset += len(me.vertices)
        tree = BVHTree.FromPolygons([Vector(v) for v in np.concatenate(verts)], polys)
        me, m = meshes[body_obj.name], np.array(evaluated[body_obj.name].matrix_world)
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        nrm = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("normal", nrm)
        nrm = nrm.reshape(-1, 3) @ m[:3, :3].T
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        out = np.array([tree.ray_cast(Vector(p) + Vector(n) * 1e-4, Vector(n), gap)[0] is not None
                        for p, n in zip(co, nrm)], float)
        if depsgraph:
            for e in evaluated.values():
                e.to_mesh_clear()
        return out

    covered = covered_now() if worn else np.zeros(len(body_obj.data.vertices))
    poses = 0
    for clip in job.get("clips", []) if worn else []:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(clip))
        imported = [o for o in bpy.data.objects if o not in before]
        clip_arm = next((o for o in imported if o.type == "ARMATURE" and o.animation_data and o.animation_data.action), None)
        action = clip_arm.animation_data.action if clip_arm else None
        if action is not None:
            action.use_fake_user = True
        for o in imported:
            bpy.data.objects.remove(o, do_unlink=True)
        if action is None:
            continue
        arm.animation_data_create()
        arm.animation_data.action = action
        if hasattr(arm.animation_data, "action_suitable_slots") and arm.animation_data.action_suitable_slots:
            arm.animation_data.action_slot = arm.animation_data.action_suitable_slots[0]
        arm.data.pose_position = "POSE"
        start, end = (int(v) for v in action.frame_range)
        for frame in fit.pose_frames(start, end, job.get("cover_frames", 8)):
            bpy.context.scene.frame_set(frame)
            covered *= covered_now(bpy.context.evaluated_depsgraph_get())
            poses += 1
        arm.animation_data.action = None
        arm.data.pose_position = "REST"
        bpy.context.scene.frame_set(start)
    bpy.context.view_layer.update()
    report["mask_poses"] = poses
    colour = body_obj.data.color_attributes.new("covered", "BYTE_COLOR", "POINT")
    rgba = np.ones((len(covered), 4))
    rgba[:, 3] = 1.0 - covered  # white everywhere, so viewers that multiply by COLOR_0 see no change
    colour.data.foreach_set("color", rgba.ravel())
    body_obj.data.color_attributes.active_color = colour
    report["covered_share"] = float(covered.mean())

    # the two shoulder pieces stay two files only when the person asked (`split`); fitted per
    # side all the same, then joined into one piece skinned to both shoulders
    for slot in {p["slot"] for p in job["pieces"] if p["slot"] == "shoulders" and not p.get("split", False)}:
        sides = [o for o in fitted if o.name in (f"{slot}_L", f"{slot}_R")]
        if len(sides) == 2:
            with bpy.context.temp_override(active_object=sides[0], object=sides[0], selected_objects=sides,
                                           selected_editable_objects=sides):
                bpy.ops.object.join()
            sides[0].name = slot
            fitted[:] = [o for o in fitted if o is not sides[1]]
            report["pieces"][slot]["joined"] = True

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
