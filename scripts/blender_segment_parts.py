"""Route B0: split a whole-character model into one piece per slot, cut edges closed.

    blender -b --factory-startup --python scripts/blender_segment_parts.py -- job.json

`job.json` (written by `scripts/slot_parts.py pieces` for slots on route B0):

    {"model": ".../concept_finished.glb", "camera": {...}, "slot_map": ".../slots.npy",
     "body": ".../body.json", "concept_size": [1024, 1024], "chosen": ["torso", "gloves"],
     "out": ".../b0"}

Faces are labelled by `slot_parts.segment.label_faces` (visible ones from the slot map, the
hidden ones from the body's bones), then each chosen slot's faces become one GLB with every
boundary loop filled, so a cut piece is closed like a piece made on its own. A paired slot
stays one file holding both sides; the fit splits it. `segment.json` records the faces per
slot and the model's extent, which places the pieces (Blender x0, z0, x1, z1 and depth centre).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from image_to_3dlab.slot_parts import segment, slots
from image_to_3dlab.slot_parts.camera import Camera


def parse_args(argv: list[str]) -> Path:
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(rest) != 1:
        raise SystemExit("usage: blender -b --python blender_segment_parts.py -- job.json")
    return Path(rest[0])


def load_job(path: Path) -> dict:
    job = json.loads(Path(path).read_text())
    for key in ("model", "camera", "slot_map", "body", "concept_size", "chosen", "out"):
        if key not in job:
            raise ValueError(f"job.json misses {key!r}")
    unknown = [s for s in job["chosen"] if s not in slots.DRAWN]
    if unknown:
        raise ValueError(f"route B0 cuts only drawn slots, not {', '.join(unknown)}")
    return job


def main() -> None:  # pragma: no cover - runs inside Blender
    import bmesh
    import bpy
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree

    from image_to_3dlab.slot_parts import body as body_mod

    job = load_job(parse_args(sys.argv))
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    cam = Camera.from_dict(job["camera"])
    record = body_mod.load(job["body"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(job["model"]))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    obj = max(meshes, key=lambda o: len(o.data.polygons))
    for o in list(bpy.data.objects):
        if o is not obj:
            bpy.data.objects.remove(o, do_unlink=True)
    mw = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = Matrix.Identity(4)
    obj.data.transform(mw)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)  # glTF splits along UV seams
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    eye = Vector((0.0, -cam.dist, 0.0))
    centres = np.array([f.calc_center_median()[:] for f in bm.faces])
    visible = np.zeros(len(centres), bool)
    for i, f in enumerate(bm.faces):
        c = Vector(centres[i])
        ray = c - eye
        if f.normal.dot(ray) >= 0:
            continue
        hit = bvh.ray_cast(eye, ray.normalized(), ray.length + 1e-3)
        visible[i] = hit[2] == i
    neighbours = [np.array([l.link_loop_radial_next.face.index for l in f.loops], int) for f in bm.faces]
    co = np.array([v.co[:] for v in bm.verts])
    y_centre = float((co[:, 1].min() + co[:, 1].max()) / 2)
    labels = segment.label_faces(centres, visible, cam, np.load(job["slot_map"]), record, job["concept_size"],
                                 y_centre, neighbours, chosen=set(job["chosen"]))
    bm.to_mesh(obj.data)
    bm.free()

    counts, files = {}, {}
    for name in job["chosen"]:
        keep = set(np.nonzero(labels == slots.label(name))[0].tolist())
        counts[name] = len(keep)
        if not keep:
            continue
        piece = obj.copy()
        piece.data = obj.data.copy()
        bpy.context.collection.objects.link(piece)
        piece.name = name
        b = bmesh.new()
        b.from_mesh(piece.data)
        b.faces.ensure_lookup_table()
        bmesh.ops.delete(b, geom=[f for f in b.faces if f.index not in keep], context="FACES")
        boundary = [e for e in b.edges if e.is_boundary]
        filled = bmesh.ops.holes_fill(b, edges=boundary, sides=0)["faces"] if boundary else []
        if filled:
            bmesh.ops.triangulate(b, faces=filled)
        bmesh.ops.recalc_face_normals(b, faces=b.faces)
        for f in b.faces:
            f.smooth = True
        b.to_mesh(piece.data)
        b.free()
        bpy.ops.object.select_all(action="DESELECT")
        piece.select_set(True)
        path = out / f"{name}.glb"
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True)
        files[name] = str(path)
        bpy.data.objects.remove(piece, do_unlink=True)
    frame = [float(co[:, 0].min()), float(co[:, 2].min()), float(co[:, 0].max()), float(co[:, 2].max()), y_centre]
    (out / "segment.json").write_text(json.dumps({"faces": counts, "files": files, "frame": frame,
                                                  "visible_share": float(visible.mean())}, indent=1) + "\n")
    print("SEGMENT_DONE", json.dumps(counts))


if __name__ == "__main__":
    main()
