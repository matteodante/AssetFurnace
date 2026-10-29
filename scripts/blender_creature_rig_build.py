"""Rig a prepared creature in Blender: the armature from joints.json, skin weights by bone heat, a rigged GLB and a sheet of test poses.

blender -b --factory-startup --python scripts/blender_creature_rig_build.py -- WORK [--no-sheet]

Step 3 of `scripts/creature_rig.py`. Reads WORK/model.blend (step 1) and WORK/joints.json (the
skeleton step) and writes WORK/rigged.blend, WORK/rigged.glb (armature `rig` and the skinned
meshes, Y up as Blender's glTF exporter writes it), WORK/weights.json and WORK/poses-sheet.png.

* Each bone's roll puts its local X across the body (`twocam.rigging.roll_vector`), so a knee,
  an elbow or a neck bends about local X, as the two-camera fit expects.
* Bone heat runs on the model's big pieces only (2% of the vertices or more), with seams
  welded: a GLB splits every UV seam, and heat on thousands of scraps leaves bones empty.
  Where heat reaches under half the vertices (holes, inner walls), it runs again on a closed
  voxel copy and each vertex takes its nearest copy point's weights.
* A small loose piece (a claw, a horn, a plate, a sword) follows the limb its touching skin
  belongs to, or copies the trunk skin it sits on; a piece reaching well past a limb's last
  bone is held, and moves rigidly with one bone.
* At most four bones per vertex, and none from a limb other than the vertex's own.

The poses sheet shows rest, limbs bent, the spine turned, and both, each from the side, the
front and three-quarter: bad weights show as tears, twists or a limb dragging the body.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# numpy only, so Blender's own Python runs it
from image_to_3dlab.twocam import rigging

LIMB_KINDS = ("hind", "front", "biped", "arm", "wing", "chain", "part")
MAIN_SHARE = 0.02         # a piece with this share of the vertices gets bone heat of its own
HEAT_REACH = 0.5          # heat reaching fewer vertices than this share: the voxel copy instead
SOLID_VOXEL = 0.01        # the voxel copy's voxel, a share of the piece's largest side
WELD = 1e-5               # vertices this close (a share of the height) are one
TOUCH = 0.015             # a piece touches the body within this (x height) ...
LIMB_NEAR = 0.06          # ... a point this near a limb bone takes that limb's bones by distance
HELD = 0.10               # a piece reaching this far from its limb, past the last bone, is held
TRUNK_BLEND = (0.02, 0.06)   # a trunk piece blends from the skin to the trunk bones over this
MAX_INFLUENCES = 4
EMPTY = 1e-3              # a bone with less total weight on a mesh is reported empty there
HINGES = ("Shin", "Forearm", "LowerArm")
LIMB_TOPS = ("Thigh", "UpperArm", "WingUpper")
POSES = ("rest", "bend", "turn", "all")
VIEWS = {"side": (1.0, 0.0, 0.0), "front": (0.0, -1.0, 0.0), "three-quarter": (0.7, -0.7, 0.35)}
TILE = 360


def parse_args(argv: list[str]) -> dict:
    args = argv[argv.index("--") + 1:] if "--" in argv else list(argv[1:])
    if not args:
        raise SystemExit("usage: ... -- WORK [--no-sheet]")
    return {"work": Path(args[0]), "sheet": "--no-sheet" not in args[1:]}


# --- the skeleton -----------------------------------------------------------------------------


def check_bones(spec: dict) -> list[dict]:
    """The bones of joints.json, each parent listed before its children; refuses a broken file
    before Blender builds half an armature from it."""
    bones = spec.get("bones") or []
    if not bones:
        raise ValueError("joints.json has no bones")
    seen: set[str] = set()
    for b in bones:
        for key in ("name", "head", "tail", "kind"):
            if key not in b:
                raise ValueError(f"a bone without `{key}`: {b}")
        if b["name"] in seen:
            raise ValueError(f"two bones are called {b['name']}")
        if b.get("parent") and b["parent"] not in seen:
            raise ValueError(f"{b['name']} comes before its parent {b['parent']}")
        seen.add(b["name"])
    return bones


def limb_chains(bones: list[dict]) -> dict[str, int]:
    """Bone name -> limb number for every limb bone; trunk bones are left out. A limb starts at
    a limb bone whose parent is not one and takes the limb bones below it."""
    kind = {b["name"]: b["kind"] for b in bones}
    kids: dict[str | None, list[str]] = {}
    for b in bones:
        kids.setdefault(b.get("parent"), []).append(b["name"])
    chains: dict[str, int] = {}
    for b in bones:
        if b["kind"] in LIMB_KINDS and kind.get(b.get("parent")) not in LIMB_KINDS:
            limb, stack = len(set(chains.values())), [b["name"]]
            while stack:
                name = stack.pop()
                if kind[name] in LIMB_KINDS:
                    chains[name] = limb
                    stack += kids.get(name, [])
    return chains


def chain_ends(bones: list[dict], chains: dict[str, int]) -> set[str]:
    """Limb bones with no child in their own limb: the paws, hands, wing tips."""
    inner = {b["parent"] for b in bones if b["name"] in chains and b.get("parent") in chains
             and chains[b["parent"]] == chains[b["name"]]}
    return {n for n in chains if n not in inner}


# --- loose parts and nearest points -----------------------------------------------------------


def loose_parts(co: np.ndarray, edges: np.ndarray, weld: float) -> np.ndarray:
    """A part number per vertex: vertices joined by edges, or lying within `weld` of each other
    (a GLB splits every UV seam: a man came in as four thousand pieces). Biggest part first."""
    co = np.asarray(co, float)
    n = len(co)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in np.asarray(edges, int).reshape(-1, 2):
        union(int(a), int(b))
    cells = np.floor(co / max(weld, 1e-12)).astype(np.int64)
    _, same = np.unique(cells, axis=0, return_inverse=True)
    same = np.asarray(same).reshape(-1)
    first: dict[int, int] = {}
    for i, k in enumerate(same.tolist()):
        if k in first:
            union(i, first[k])
        else:
            first[k] = i
    roots = np.array([find(i) for i in range(n)], int)
    _, part, sizes = np.unique(roots, return_inverse=True, return_counts=True)
    part = np.asarray(part).reshape(-1)
    rank = np.empty(len(sizes), int)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(len(sizes))
    return rank[part]


def main_parts(part: np.ndarray, share: float = MAIN_SHARE) -> set[int]:
    """The parts bone heat runs on: every part with `share` of the vertices, and the biggest."""
    sizes = np.bincount(part)
    return {int(i) for i in np.nonzero(sizes >= share * len(part))[0]} | {int(np.argmax(sizes))}


def nearest_brute(points: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of and distance to the nearest of `points` for each query (Blender uses a k-d
    tree with the same signature)."""
    p, q = np.asarray(points, float), np.asarray(queries, float)
    idx = np.empty(len(q), int)
    dist = np.empty(len(q))
    for s in range(0, len(q), 1024):
        d = ((q[s:s + 1024, None, :] - p[None, :, :]) ** 2).sum(-1)
        idx[s:s + 1024] = d.argmin(1)
        dist[s:s + 1024] = np.sqrt(d[np.arange(len(d)), idx[s:s + 1024]])
    return idx, dist


def segment_distances(points: np.ndarray, heads: np.ndarray, tails: np.ndarray) -> np.ndarray:
    """(points, bones) distance from each point to each bone's segment."""
    x = np.asarray(points, float)[:, None, :]
    h, t = np.asarray(heads, float)[None], np.asarray(tails, float)[None]
    d = t - h
    u = np.clip(((x - h) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(x - (h + u[..., None] * d), axis=-1)


def by_distance(dist: np.ndarray, cols: list[int], width: int) -> np.ndarray:
    """Weights on the two nearest of `cols`, by the inverse fourth power of their distance."""
    out = np.zeros((len(dist), width))
    if not cols:
        return out
    two = np.argsort(dist, 1)[:, :2]
    d = np.take_along_axis(dist, two, 1)
    w = 1.0 / np.maximum(d, 1e-4) ** 4
    w /= w.sum(1, keepdims=True)
    rows = np.arange(len(dist))[:, None]
    np.add.at(out, (rows, np.asarray(cols)[two]), w)
    return out


# --- loose parts follow their limb --------------------------------------------------------------


def follow_limbs(co: np.ndarray, part: np.ndarray, mains: set[int], weights: np.ndarray, near: np.ndarray,
                 dist: np.ndarray, owner: np.ndarray, heads: np.ndarray, tails: np.ndarray,
                 chain_of: np.ndarray, ends: np.ndarray, height: float,
                 nearest: Callable = nearest_brute) -> tuple[np.ndarray, dict]:
    """Weights for every vertex, the loose parts settled.

    `weights` holds the heat-weighted skin vertices' weights; `near`/`dist` give each vertex's
    nearest such vertex and how far it is; `owner` the limb of each skin vertex (its
    strongest bone's, -1 for the trunk); `chain_of` each bone's limb; `ends` which bones end
    a limb. Returns the weights and what was done (counts of parts by kind)."""
    co = np.asarray(co, float)
    width = weights.shape[1]
    out = weights[near].copy()
    done = {"limb": 0, "trunk": 0, "held": 0}
    trunk_cols = [c for c in range(width) if chain_of[c] < 0]
    for q in np.unique(part):
        if int(q) in mains:
            continue
        idx = np.nonzero(part == q)[0]
        reach = max(TOUCH * height, float(np.quantile(dist[idx], 0.05)))
        touching = idx[dist[idx] <= reach]
        limbs = owner[near[touching]]
        votes = np.bincount(limbs + 1)
        mine = int(np.argmax(votes)) - 1
        if mine >= 0:
            cols = [c for c in range(width) if chain_of[c] == mine]
            d = segment_distances(co[idx], heads[cols], tails[cols])
            gap = d.min(1)
            close = gap <= LIMB_NEAR * height
            if close.any() and gap.max() > HELD * height and \
                    np.mean(ends[np.asarray(cols)[np.argmin(d[~close], 1)]]) > 0.5:
                handle = cols[int(np.bincount(np.argmin(d[close], 1), minlength=len(cols)).argmax())]
                out[idx] = 0.0
                out[idx, handle] = 1.0
                done["held"] += 1
                continue
            if close.any():
                out[idx[close]] = by_distance(d[close], cols, width)
                if (~close).any():
                    k, _ = nearest(co[idx[close]], co[idx[~close]])
                    out[idx[~close]] = out[idx[close][k]]
                done["limb"] += 1
                continue
        # on the trunk (or touching a limb nowhere near its bones): the skin it touches
        anchors = touching[limbs == mine]
        k, _ = nearest(co[anchors], co[idx])
        out[idx] = weights[near[anchors[k]]]
        if mine < 0 and trunk_cols:
            lo, hi = TRUNK_BLEND
            blend = np.clip((dist[idx] - lo * height) / ((hi - lo) * height), 0.0, 1.0)
            if (blend > 0).any():
                bones = by_distance(segment_distances(co[idx], heads[trunk_cols], tails[trunk_cols]),
                                    trunk_cols, width)
                out[idx] = (1 - blend[:, None]) * out[idx] + blend[:, None] * bones
        done["trunk"] += 1
    return out, done


def clean_weights(weights: np.ndarray, chain_of: np.ndarray, keep: int = MAX_INFLUENCES) -> np.ndarray:
    """At most `keep` bones per vertex, no weight from a limb other than the one its strongest
    bone is on (a toe following the other foot), and each vertex's weights summing to one."""
    w = np.array(weights, float)
    if w.size == 0:
        return w
    own = np.asarray(chain_of)[np.argmax(w, 1)]
    other = (own[:, None] >= 0) & (chain_of[None, :] >= 0) & (chain_of[None, :] != own[:, None])
    w[other] = 0.0
    if w.shape[1] > keep:
        drop = np.argsort(-w, 1, kind="stable")[:, keep:]
        np.put_along_axis(w, drop, 0.0, 1)
    total = w.sum(1, keepdims=True)
    return np.divide(w, total, out=np.zeros_like(w), where=total > 0)


# --- the closed voxel copy for bone heat ---------------------------------------------------------


def _grow(m: np.ndarray) -> np.ndarray:
    g = m.copy()
    g[1:] |= m[:-1]
    g[:-1] |= m[1:]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    g[:, :, 1:] |= m[:, :, :-1]
    g[:, :, :-1] |= m[:, :, 1:]
    return g


def sealed_solid(co: np.ndarray, tris: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The volume a surface encloses, as voxels: the skin marked (triangles sampled every half
    voxel), grown by one so holes up to a voxel wide close, the outside flooded in from the
    border. What the flood does not reach is solid, inner walls and all. Returns the grid
    and its corner."""
    co = np.asarray(co, float)
    tris = np.asarray(tris, int).reshape(-1, 3)
    lo = co.min(0) - 2 * voxel
    dims = np.ceil((co.max(0) + 2 * voxel - lo) / voxel).astype(int) + 1
    skin = np.zeros(dims, bool)
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], axis=0)
    steps = np.clip(np.ceil(longest / (0.5 * voxel)).astype(int), 1, 200)
    for n in np.unique(steps):
        sel = steps == n
        u, v = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        inside = (u + v) <= n
        u, v = u[inside] / n, v[inside] / n
        pts = (a[sel][:, None] * (1 - u - v)[None, :, None] + b[sel][:, None] * u[None, :, None]
               + c[sel][:, None] * v[None, :, None]).reshape(-1, 3)
        cell = np.clip(np.floor((pts - lo) / voxel).astype(int), 0, dims - 1)
        skin[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    wall = _grow(skin)
    outside = np.zeros(dims, bool)
    outside[[0, -1]] = True
    outside[:, [0, -1]] = True
    outside[:, :, [0, -1]] = True
    outside &= ~wall
    while True:
        grown = _grow(outside) & ~wall
        if grown.sum() == outside.sum():
            break
        outside = grown
    return ~outside, lo


def voxel_skin(solid: np.ndarray, lo: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The square faces between solid and empty voxels, wound to face out: vertices, quads."""
    pad = np.pad(solid, 1)
    corners = []
    for axis in range(3):
        o1, o2 = (axis + 1) % 3, (axis + 2) % 3     # o1 x o2 = +axis
        for sign in (1, -1):
            face = pad & ~np.roll(pad, -sign, axis)
            cells = np.argwhere(face)
            if sign == 1:
                cells[:, axis] += 1
            quad = []
            for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = cells.copy()
                p[:, o1] += d1
                p[:, o2] += d2
                quad.append(p)
            quad = np.stack(quad, 1)
            corners.append(quad if sign == 1 else quad[:, ::-1])
    allq = np.concatenate(corners) if corners else np.zeros((0, 4, 3), int)
    keys, inverse = np.unique(allq.reshape(-1, 3), axis=0, return_inverse=True)
    verts = np.asarray(lo, float) + (keys - 1) * voxel
    return verts, np.asarray(inverse).reshape(-1, 4)


# --- the poses sheet --------------------------------------------------------------------------


def base_name(name: str) -> str:
    """`Thigh2.L` -> `Thigh`, `Tail3` -> `Tail`."""
    return name.split(".")[0].rstrip("0123456789")


def sheet_pose(bones: list[dict], pose: str) -> dict[str, tuple[float, float, float]]:
    """Each posed bone's turn (degrees about its local X, Y, Z) in one of the sheet's poses.
    Bends: knees and elbows 45 degrees, one side forward and the other back; thighs, upper
    arms and wings 30; every bone of a many-legged chain below its first 30. Turns: every
    spine, neck and tail bone but the hips 12 up and 10 sideways."""
    if pose not in POSES:
        raise ValueError(f"no pose called {pose}")
    kind = {b["name"]: b["kind"] for b in bones}
    out = {}
    for b in bones:
        name, base = b["name"], base_name(b["name"])
        side = b.get("side") or (name[-1] if name[-2:] in (".L", ".R") else "")
        s = -1.0 if side == "R" else 1.0
        x = z = 0.0
        if pose in ("bend", "all"):
            if base in HINGES:
                x = 45.0 * s
            elif base in LIMB_TOPS:
                x = -30.0
            elif b["kind"] == "chain" and kind.get(b.get("parent")) == "chain":
                x = 30.0 * s
        if pose in ("turn", "all") and b["kind"] in ("spine", "neck", "tail") and name != "Hips":
            x, z = x + 12.0, z + 10.0
        if x or z:
            out[name] = (x, 0.0, z)
    return out


def tile_sheet(rows: list[list[np.ndarray]]) -> np.ndarray:
    """Tiles (top row first, each H x W x C, top line first) laid out in one image."""
    h, w, c = rows[0][0].shape
    sheet = np.zeros((h * len(rows), w * max(len(r) for r in rows), c), np.float32)
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            sheet[i * h:(i + 1) * h, j * w:(j + 1) * w] = tile
    return sheet


def weights_report(names: list[str], meshes: list[tuple[str, int, int]], weights: np.ndarray) -> dict:
    """Per mesh (name, vertex count, loose parts): its vertices, loose parts and the bones that
    move nothing on it."""
    report, start = {}, 0
    for mesh, count, parts in meshes:
        mass = weights[start:start + count].sum(0)
        report[mesh] = {"vertices": int(count), "loose_parts": int(parts),
                        "empty_bones": [n for n, m in zip(names, mass) if m < EMPTY]}
        start += count
    return report


# --- Blender ----------------------------------------------------------------------------------


def _kd_nearest(points: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from mathutils.kdtree import KDTree

    tree = KDTree(len(points))
    for i, p in enumerate(np.asarray(points, float).tolist()):
        tree.insert(p, i)
    tree.balance()
    idx = np.empty(len(queries), int)
    dist = np.empty(len(queries))
    for k, q in enumerate(np.asarray(queries, float).tolist()):
        _, idx[k], dist[k] = tree.find(q)
    return idx, dist


def _build_armature(bpy, scene, bones: list[dict], height: float):
    from mathutils import Vector

    data = bpy.data.armatures.new("rig")
    rig = bpy.data.objects.new("rig", data)
    scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for b in bones:
        eb = data.edit_bones.new(b["name"])
        eb.head = Vector(b["head"])
        eb.tail = Vector(b["tail"])
        if (eb.tail - eb.head).length < 1e-3 * height:
            eb.tail = eb.head + Vector((0.0, 0.0, 0.01 * height))
        eb.align_roll(Vector(rigging.roll_vector(eb.head[:], eb.tail[:]).tolist()))
        if b.get("parent"):
            eb.parent = data.edit_bones[b["parent"]]
        eb.use_connect = False
        eb.use_deform = bool(b.get("deform", True))
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def _world_arrays(meshes):
    co, edges, tris, base = [], [], [], 0
    for o in meshes:
        me = o.data
        me.calc_loop_triangles()
        v = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", v)
        m = np.array(o.matrix_world)
        co.append(v.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
        e = np.empty(len(me.edges) * 2, np.int64)
        me.edges.foreach_get("vertices", e)
        edges.append(e.reshape(-1, 2) + base)
        t = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", t)
        tris.append(t.reshape(-1, 3) + base)
        base += len(me.vertices)
    return np.concatenate(co), np.concatenate(edges), np.concatenate(tris)


def _heat(bpy, rig, verts: np.ndarray, faces: np.ndarray, names: list[str], weld: float | None):
    """Bone heat on a temporary mesh; its vertices and their (vertices, bones) weights."""
    import bmesh

    me = bpy.data.meshes.new("heat")
    me.from_pydata(np.asarray(verts, float).tolist(), [], np.asarray(faces, int).tolist())
    me.update()
    if weld:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=weld)
        bm.to_mesh(me)
        bm.free()
    tmp = bpy.data.objects.new("heat", me)
    bpy.context.scene.collection.objects.link(tmp)
    bpy.ops.object.select_all(action="DESELECT")
    tmp.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    col = {n: k for k, n in enumerate(names)}
    group = {g.index: col.get(g.name) for g in tmp.vertex_groups}
    w = np.zeros((len(me.vertices), len(names)))
    for i, v in enumerate(me.vertices):
        for g in v.groups:
            c = group.get(g.group)
            if c is not None:
                w[i, c] = g.weight
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.meshes.remove(me)
    return co.reshape(-1, 3), w


def _remeshed(bpy, verts: np.ndarray, quads: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The voxel faces smoothed by Blender's voxel remesh: vertices and triangles."""
    me = bpy.data.meshes.new("solid")
    me.from_pydata(verts.tolist(), [], quads.tolist())
    me.update()
    obj = bpy.data.objects.new("solid", me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("smooth", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    mod.adaptivity = 0.0
    bpy.ops.object.modifier_apply(modifier=mod.name)
    me = obj.data
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    tri = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(me)
    return co.reshape(-1, 3), tri.reshape(-1, 3)


def _solid_heat(bpy, rig, verts, faces, names):
    """Bone heat on a closed voxel copy. Its square faces as they are first: smoothed by
    Blender's voxel remesh at the same size, heat found no solution at all on the dragon
    (0% against 100%, 2026-10-29), so the smoothed copy is only the second try."""
    voxel = SOLID_VOXEL * float(np.ptp(verts, 0).max())
    solid, lo = sealed_solid(verts, faces, voxel)
    v, quads = voxel_skin(solid, lo, voxel)
    print(f"creature-rig build: voxel copy of {int(solid.sum())} voxels at {1000 * voxel:.1f} mm", flush=True)
    co, w = _heat(bpy, rig, v, np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]), names, None)
    if (w.sum(1) > 1e-6).mean() >= HEAT_REACH:
        return co, w
    co2, w2 = _heat(bpy, rig, *_remeshed(bpy, v, quads, voxel), names, None)
    return (co2, w2) if (w2.sum(1) > 1e-6).mean() > (w.sum(1) > 1e-6).mean() else (co, w)


def skin(bpy, rig, meshes, bones: list[dict], height: float) -> tuple[np.ndarray, list[str], dict]:
    """Weights for every vertex of every mesh (in mesh order), the deforming bones' names, and
    what was done."""
    names = [b["name"] for b in bones if b.get("deform", True)]
    chains = limb_chains(bones)
    ends_set = chain_ends(bones, chains)
    chain_of = np.array([chains.get(n, -1) for n in names])
    ends = np.array([n in ends_set for n in names])
    by_name = {b["name"]: b for b in bones}
    heads = np.array([by_name[n]["head"] for n in names], float)
    tails = np.array([by_name[n]["tail"] for n in names], float)

    co, edges, tris = _world_arrays(meshes)
    part = loose_parts(co, edges, WELD * height)
    mains = main_parts(part)
    on_main = np.isin(part, sorted(mains))
    keep = np.nonzero(on_main)[0]
    remap = -np.ones(len(co), int)
    remap[keep] = np.arange(len(keep))
    main_tris = remap[tris[on_main[tris].all(1)]]
    heat_co, heat_w = _heat(bpy, rig, co[keep], main_tris, names, WELD * height)
    reached = float((heat_w.sum(1) > 1e-6).mean()) if len(heat_w) else 0.0
    info = {"parts": int(part.max() + 1), "heat_parts": len(mains), "heat_reached": round(reached, 3),
            "voxel_copy": False}
    print(f"creature-rig build: bone heat on {len(mains)} piece(s) of {info['parts']} reached "
          f"{100 * reached:.0f}% of their vertices", flush=True)
    if reached < HEAT_REACH:
        heat_co, heat_w = _solid_heat(bpy, rig, co[keep], main_tris, names)
        reached = float((heat_w.sum(1) > 1e-6).mean()) if len(heat_w) else 0.0
        info.update(voxel_copy=True, voxel_copy_reached=round(reached, 3))
        print(f"creature-rig build: heat on the voxel copy reached {100 * reached:.0f}%", flush=True)
    k, _ = _kd_nearest(heat_co, co[keep])
    main_w = heat_w[k]
    weighted = main_w.sum(1) > 1e-6
    if not weighted.any():
        print("creature-rig build: bone heat found no solution: weights by distance to the bones", flush=True)
        w = by_distance(segment_distances(co, heads, tails), list(range(len(names))), len(names))
        info["by_distance"] = True
        return clean_weights(w, chain_of), names, info
    rows = keep[weighted]
    near, dist = _kd_nearest(co[rows], co)
    owner = chain_of[np.argmax(main_w[weighted], 1)]
    w, done = follow_limbs(co, part, mains, main_w[weighted], near, dist, owner, heads, tails, chain_of,
                           ends, height, _kd_nearest)
    info["loose"] = done
    return clean_weights(w, chain_of), names, info


def _apply_weights(rig, meshes, names, weights):
    start = 0
    for o in meshes:
        n = len(o.data.vertices)
        w = weights[start:start + n]
        start += n
        o.vertex_groups.clear()
        for c, name in enumerate(names):
            idx = np.nonzero(w[:, c] > 1e-4)[0]
            if not len(idx):
                continue
            g = o.vertex_groups.new(name=name)
            for i in idx.tolist():
                g.add([i], float(w[i, c]), "REPLACE")
        for mod in list(o.modifiers):
            if mod.type == "ARMATURE":
                o.modifiers.remove(mod)
        mod = o.modifiers.new("Armature", "ARMATURE")
        mod.object = rig
        world = o.matrix_world.copy()
        o.parent = rig
        o.matrix_parent_inverse = rig.matrix_world.inverted()
        o.matrix_world = world


def _pose(rig, turns: dict) -> None:
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
        x, y, z = turns.get(pb.name, (0.0, 0.0, 0.0))
        pb.rotation_euler = (math.radians(x), math.radians(y), math.radians(z))


def poses_sheet(bpy, scene, rig, meshes, bones: list[dict], path: Path) -> None:
    from mathutils import Vector

    co = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in meshes])
    centre = Vector(((co.min(0) + co.max(0)) / 2).tolist())
    size = float(np.ptp(co, 0).max())
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_cavity = True          # creases and outlines: a flat side view still reads
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.show_object_outline = True
    scene.render.resolution_x = scene.render.resolution_y = TILE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    world = bpy.data.worlds.new("sheet")
    world.color = (0.42, 0.41, 0.40)
    scene.world = world
    cam_data = bpy.data.cameras.new("sheet")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 1.35 * size
    cam_data.clip_end = 20 * size
    cam = bpy.data.objects.new("sheet", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    tmp = path.parent / "sheet-tiles"
    tmp.mkdir(exist_ok=True)
    rows = []
    for pose in POSES:
        _pose(rig, sheet_pose(bones, pose))
        bpy.context.view_layer.update()
        row = []
        for view, look in VIEWS.items():
            d = Vector(look).normalized()
            cam.location = centre + d * size * 4
            cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
            tile = tmp / f"{pose}-{view}.png"
            scene.render.filepath = str(tile)
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(str(tile))
            px = np.empty(img.size[0] * img.size[1] * 4, np.float32)
            img.pixels.foreach_get(px)
            row.append(np.flipud(px.reshape(img.size[1], img.size[0], 4)))
            bpy.data.images.remove(img)
            tile.unlink()
        rows.append(row)
    _pose(rig, {})
    sheet = np.flipud(tile_sheet(rows))
    out = bpy.data.images.new("poses-sheet", sheet.shape[1], sheet.shape[0], alpha=True)
    out.pixels.foreach_set(sheet.ravel())
    out.filepath_raw = str(path)
    out.file_format = "PNG"
    out.save()
    tmp.rmdir()


def main() -> None:
    import bpy

    opts = parse_args(sys.argv)
    work = opts["work"].resolve()
    started = time.time()
    spec = json.loads((work / "joints.json").read_text(encoding="utf-8"))
    bones = check_bones(spec)
    bpy.ops.wm.open_mainfile(filepath=str(work / "model.blend"))
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH" and len(o.data.vertices)]
    if not meshes:
        raise SystemExit("model.blend has no mesh")
    height = float(spec.get("height") or 0.0)
    if height <= 0:
        height = float(max(o.dimensions.z for o in meshes))
    rig = _build_armature(bpy, scene, bones, height)
    weights, names, info = skin(bpy, rig, meshes, bones, height)
    _apply_weights(rig, meshes, names, weights)
    parts = loose_parts_per_mesh(meshes, height)
    report = weights_report(names, [(o.name, len(o.data.vertices), p) for o, p in zip(meshes, parts)], weights)
    (work / "weights.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(work / "rigged.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    for o in (rig, *meshes):
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=str(work / "rigged.glb"), export_format="GLB", use_selection=True,
                              export_skins=True, export_animations=False, export_yup=True)
    built = time.time() - started
    if opts["sheet"]:
        try:
            poses_sheet(bpy, scene, rig, meshes, bones, work / "poses-sheet.png")
        except RuntimeError as exc:   # no GPU to draw with: the rig is still good
            info["sheet_error"] = str(exc)
            print(f"creature-rig build: no poses sheet ({exc})", flush=True)
    info.update(bones=len(bones), deforming=len(names), seconds=round(built, 1),
                sheet_seconds=round(time.time() - started - built, 1),
                empty_bones=sorted(set.intersection(*[set(r["empty_bones"]) for r in report.values()])))
    print("I2L_CREATURE_BUILD::" + json.dumps(info), flush=True)


def loose_parts_per_mesh(meshes, height: float) -> list[int]:
    out = []
    for o in meshes:
        co, edges, _ = _world_arrays([o])
        out.append(int(loose_parts(co, edges, WELD * height).max() + 1))
    return out


if __name__ == "__main__":
    main()
