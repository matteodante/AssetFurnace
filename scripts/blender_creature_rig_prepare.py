"""Get a generated creature ready for the auto-rig in Blender: turned to face -Y, placed, and a watertight shell saved.

blender -b --factory-startup --python scripts/blender_creature_rig_prepare.py -- MODEL WORK [--turn-deg D]

Step 1 of `scripts/creature_rig.py`. MODEL is a .glb, .gltf, .fbx, .obj or .blend; WORK is the
auto-rig's work folder, which gets:

* WORK/model.blend: the model's meshes (never joined), transforms applied, turned about Z so
  its mirror plane is X = 0 and it faces -Y, centred on X = Y = 0 and standing on Z = 0. An old
  armature, its modifiers and its vertex groups are dropped: the model is rigged afresh;
* WORK/unirig_input.glb: the same meshes as a GLB, without materials, for UniRig;
* WORK/shell.npz: a voxel remesh of all the meshes joined (1.2% of the height: closed, so
  inside and outside mean something), the model's own triangles beside it (a one-sided wing
  membrane has no inside, and the remesh keeps only shreds of it), the height and the turn.

Generated models stand at any turn. The turn that puts the mirror plane on X = 0 comes from
the surface (`image_to_3dlab.twocam.rigging.symmetry_yaw`); which end is the front from the
feet, whose toes reach forward of the leg above them, else from the head, the higher end.
`--turn-deg` overrides both.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# numpy only, so Blender's own Python runs it
from image_to_3dlab.twocam import rigging

SHELL_VOXEL = 0.012        # remesh voxel, a share of the height
SAMPLES = 12000            # surface points for the facing search
SYMMETRY_SAMPLES = 1500    # of those, for the mirror plane (the same as twocam.symmetry.mirror_gap)
PLANE_OFFSETS = np.linspace(-0.05, 0.05, 21)   # where the mirror plane may sit, a share of the size
IMPORTERS = {".glb": "gltf", ".gltf": "gltf", ".fbx": "fbx", ".obj": "obj", ".blend": "blend"}


def parse_args(argv: list[str]) -> dict:
    args = argv[argv.index("--") + 1:] if "--" in argv else list(argv[1:])
    if len(args) < 2:
        raise SystemExit("usage: ... -- MODEL WORK [--turn-deg D]")
    model, work, rest = Path(args[0]), Path(args[1]), args[2:]
    if model.suffix.lower() not in IMPORTERS:
        raise SystemExit(f"cannot read {model.suffix or 'a file without a suffix'}: use one of "
                         + ", ".join(sorted(IMPORTERS)))
    turn = None
    if "--turn-deg" in rest:
        i = rest.index("--turn-deg")
        if i + 1 >= len(rest):
            raise SystemExit("--turn-deg needs a value")
        turn = float(rest[i + 1])
    return {"model": model, "work": work, "turn_deg": turn}


# --- the facing --------------------------------------------------------------------------------


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def surface_samples(verts: np.ndarray, faces: np.ndarray, count: int = SAMPLES, seed: int = 0) -> np.ndarray:
    """Points spread evenly over the triangles' area: vertices crowd where the detail is (a
    face, the claws), and the mirror search would weigh those parts over the body."""
    v = np.asarray(verts, float)
    f = np.asarray(faces, int).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if len(f) == 0 or area.sum() <= 0:
        raise ValueError("the model has no surface")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(f), count, p=area / area.sum())
    u, w = rng.random(count), rng.random(count)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return a[pick] + u[:, None] * (b[pick] - a[pick]) + w[:, None] * (c[pick] - a[pick])


def _nearest_gap(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from each point of q to the nearest point of p (brute force: a few thousand)."""
    out = np.empty(len(q))
    for s in range(0, len(q), 500):
        d = ((q[s:s + 500, None, :] - p[None, :, :]) ** 2).sum(-1)
        out[s:s + 500] = np.sqrt(d.min(1))
    return out


def mirror_plane(points: np.ndarray, samples: int = SYMMETRY_SAMPLES, seed: int = 0) -> tuple[float, float]:
    """For points already turned so the mirror plane is parallel to X = 0: the X it sits at,
    and the median mirror gap there as a share of the size (as `twocam.symmetry` measures)."""
    p = np.asarray(points, float)
    if len(p) > samples:
        p = p[np.random.default_rng(seed).choice(len(p), samples, replace=False)]
    size = float(np.ptp(p, 0).max()) or 1.0
    centre = float(p[:, 0].mean())
    best = (math.inf, centre)
    for offset in PLANE_OFFSETS * size:
        x = centre + offset
        mirrored = p * [-1.0, 1.0, 1.0] + [2 * x, 0.0, 0.0]
        gap = float(np.median(_nearest_gap(p, mirrored)))
        if gap < best[0]:
            best = (gap, x)
    return best[1], best[0] / size


def link_groups(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Points closer than `radius` (chains of them) in one group; largest first."""
    p = np.asarray(points, float)
    n = len(p)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        near = np.nonzero(((p[i + 1:] - p[i]) ** 2).sum(1) < radius * radius)[0] + i + 1
        for j in near:
            a, b = find(i), find(int(j))
            if a != b:
                parent[a] = b
    roots = np.array([find(i) for i in range(n)], int)
    groups = [np.nonzero(roots == r)[0] for r in np.unique(roots)]
    return sorted(groups, key=len, reverse=True)


def facing_sign(points: np.ndarray, plane_x: float = 0.0) -> tuple[float, str]:
    """+1 when a creature whose mirror plane is X = plane_x faces -Y, -1 when it faces +Y,
    and the cue that said so.

    Toes first: on two legs or four, the part of a foot on the floor reaches forward of the
    leg standing on it (taken 8-20% of the height up). A tail tip on the floor has no leg
    above it and a robe hem no toes, so neither votes. Without feet, the head: the end of
    the body (a fifth of its length) that reaches higher. Neither clear: as it stands."""
    p = np.asarray(points, float)
    z0 = float(p[:, 2].min())
    h = float(np.ptp(p[:, 2])) or 1.0
    low = p[(p[:, 2] < z0 + 0.04 * h) & (np.abs(p[:, 0] - plane_x) > 0.02 * h)]
    if len(low) > 1500:
        low = low[np.random.default_rng(0).choice(len(low), 1500, replace=False)]
    band = p[(p[:, 2] > z0 + 0.08 * h) & (p[:, 2] < z0 + 0.2 * h)]
    shifts = []
    for idx in link_groups(low[:, :2], 0.06 * h):
        if len(idx) < 5:
            continue
        foot = low[idx]
        centre = foot[:, :2].mean(0)
        reach = max(float(np.linalg.norm(foot[:, :2] - centre, axis=1).max()), 0.04 * h)
        leg = band[np.linalg.norm(band[:, :2] - centre, axis=1) < reach]
        if len(leg) >= 5:
            shifts.append(float(foot[:, 1].mean() - leg[:, 1].mean()))
    if shifts and abs(float(np.mean(shifts))) > 0.015 * h:
        return (1.0 if np.mean(shifts) < 0 else -1.0), "toes"
    y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
    fifth = 0.2 * (y1 - y0)
    front_top = float(p[p[:, 1] < y0 + fifth, 2].max())
    back_top = float(p[p[:, 1] > y1 - fifth, 2].max())
    if abs(front_top - back_top) > 0.05 * h:
        return (1.0 if front_top > back_top else -1.0), "head"
    return 1.0, "none"


def facing(points: np.ndarray, turn_deg: float | None = None) -> dict:
    """The 4x4 that turns the creature about Z to face -Y with its mirror plane on X = 0, then
    stands it on Z = 0 centred on Y; with the turn used (degrees), the cue and the mirror gap."""
    p = np.asarray(points, float)
    if turn_deg is None:
        yaw = rigging.symmetry_yaw(p, samples=SYMMETRY_SAMPLES)
        turned = p @ rot_z(yaw).T
        plane, _ = mirror_plane(turned)
        sign, cue = facing_sign(turned, plane)
        if sign < 0:
            yaw += math.pi
    else:
        yaw, cue = math.radians(turn_deg), "given"
    turned = p @ rot_z(yaw).T
    plane, gap = mirror_plane(turned)
    lo, hi = turned.min(0), turned.max(0)
    m = np.eye(4)
    m[:3, :3] = rot_z(yaw)
    m[:3, 3] = [-plane, -(lo[1] + hi[1]) / 2, -lo[2]]
    return {"matrix": m, "turn_deg": math.degrees(yaw) % 360.0, "cue": cue, "mirror_gap": gap}


def shell_arrays(shell_verts, shell_faces, model_verts, model_faces, height: float, turn_deg: float,
                 mirror_gap: float | None = None) -> dict:
    """What `shell.npz` holds, with the types the skeleton step reads."""
    out = {"shell_verts": np.asarray(shell_verts, np.float32).reshape(-1, 3),
           "shell_faces": np.asarray(shell_faces, np.int32).reshape(-1, 3),
           "model_verts": np.asarray(model_verts, np.float32).reshape(-1, 3),
           "model_faces": np.asarray(model_faces, np.int32).reshape(-1, 3),
           "height": np.float64(height), "turn_deg": np.float64(turn_deg)}
    if mirror_gap is not None:
        out["mirror_gap"] = np.float64(mirror_gap)
    return out


# --- Blender ----------------------------------------------------------------------------------


def _load(bpy, path: Path) -> None:
    kind = IMPORTERS[path.suffix.lower()]
    if kind == "blend":
        bpy.ops.wm.open_mainfile(filepath=str(path))
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if kind == "gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif kind == "fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        bpy.ops.wm.obj_import(filepath=str(path))


def _mesh_arrays(objects) -> tuple[np.ndarray, np.ndarray]:
    """World-space vertices and triangles of the objects, as one array each."""
    verts, faces, base = [], [], 0
    for o in objects:
        me = o.data
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        m = np.array(o.matrix_world)
        verts.append(co @ m[:3, :3].T + m[:3, 3])
        tri = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        faces.append(tri.reshape(-1, 3) + base)
        base += len(co)
    return np.concatenate(verts), np.concatenate(faces)


def main() -> None:
    import bpy
    from mathutils import Matrix

    opts = parse_args(sys.argv)
    work = opts["work"].resolve()
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _load(bpy, opts["model"].resolve())
    scene = bpy.context.scene

    # The raw meshes in their own pose: old rigs, cameras, lights and empties go.
    for o in list(scene.objects):
        if o.type != "MESH":
            for child in [c for c in scene.objects if c.parent == o]:
                world = child.matrix_world.copy()
                child.parent = None
                child.matrix_world = world
    for o in list(scene.objects):
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    meshes = [o for o in scene.objects if o.type == "MESH" and len(o.data.vertices)]
    if not meshes:
        raise SystemExit(f"{opts['model'].name} has no mesh")
    for o in meshes:
        for mod in list(o.modifiers):
            if mod.type == "ARMATURE":
                o.modifiers.remove(mod)
        o.vertex_groups.clear()
        if o.data.users > 1:
            o.data = o.data.copy()
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    verts, faces = _mesh_arrays(meshes)
    place = facing(surface_samples(verts, faces), opts["turn_deg"])
    m = Matrix(place["matrix"].tolist())
    for o in meshes:
        o.matrix_world = m @ o.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    verts, faces = _mesh_arrays(meshes)
    height = float(np.ptp(verts[:, 2]))
    if height <= 0:
        raise SystemExit("the model is flat: it has no height to rig")
    print(f"creature-rig prepare: {len(meshes)} mesh(es), {len(verts)} vertices, height {height:.3f}, "
          f"turned {place['turn_deg']:.1f} deg (front from {place['cue']}), mirror gap "
          f"{100 * place['mirror_gap']:.1f}%", flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(work / "model.blend"))

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.export_scene.gltf(filepath=str(work / "unirig_input.glb"), export_format="GLB",
                              use_selection=True, export_materials="NONE", export_animations=False,
                              export_skins=False, export_morph=False)

    # The shell: a joined copy, voxel-remeshed (closes every gap and fur card at the voxel size).
    bpy.ops.object.duplicate()
    copies = list(bpy.context.selected_objects)
    bpy.context.view_layer.objects.active = copies[0]
    if len(copies) > 1:
        bpy.ops.object.join()
    shell = bpy.context.view_layer.objects.active
    remesh = shell.modifiers.new("shell", "REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = SHELL_VOXEL * height
    remesh.adaptivity = 0.0
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    shell_verts, shell_faces = _mesh_arrays([shell])
    np.savez_compressed(work / "shell.npz", **shell_arrays(shell_verts, shell_faces, verts, faces, height,
                                                           place["turn_deg"], place["mirror_gap"]))
    record = {"model": str(opts["model"]), "meshes": [o.name for o in meshes], "vertices": int(len(verts)),
              "height": round(height, 4), "turn_deg": round(place["turn_deg"], 2), "front_cue": place["cue"],
              "mirror_gap_pct": round(100 * place["mirror_gap"], 2), "shell_vertices": int(len(shell_verts)),
              "seconds": round(time.time() - started, 1)}
    print("I2L_CREATURE_PREPARE::" + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
