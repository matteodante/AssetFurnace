"""Read the skeletons UniRig drew into the auto-rig's draw files, in the model's own frame.

blender -b --factory-startup --python scripts/blender_creature_rig_read.py -- WORK SKELETON.fbx [...] [--start K]
python scripts/blender_creature_rig_read.py WORK SKELETON.fbx [...] [--start K]   (any Python with `bpy`)

Writes WORK/draws/draw{K+i}.json for the i-th file: `names`, `parents` (index, -1 for a root,
parents listed before their children), `heads` and `tails` in the frame of WORK/model.blend.

UniRig draws on its own copy of the model, centred and scaled into a box of its own, and
writes that copy into the skeleton file beside the bones. The two boxes give the way back:
the ratio of their largest sides and their centres. Each axis is checked as well; a box off
by more than 5% on any axis means the copy is not the model (a wrong file, or turned), and
the draw file says so in `warning`.

A skeleton file keeps only UniRig's joints: FBX has no bone lengths, so an importer makes
up the tails (Blender gave a box creature's leg 50% too long). The tails are rebuilt the
way UniRig made them: a bone with one child ends at that child, any other bone 0.03 (of its
box) on along the line from its parent, the root 0.03 straight up.

Blender's own FBX importer has been known to die on a file UniRig's `bpy` wrote and reads
fine, so `scripts/creature_rig.py` runs this script again with UniRig's Python for any file
Blender could not read: hence the second form.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AXIS_TOLERANCE = 0.05      # a share of the model's largest side
UNIRIG_STUB = 0.03         # UniRig's short bone end, in its box (largest side 2)


def parse_args(argv: list[str]) -> dict:
    args = argv[argv.index("--") + 1:] if "--" in argv else list(argv[1:])
    start = 0
    if "--start" in args:
        i = args.index("--start")
        if i + 1 >= len(args):
            raise SystemExit("--start needs a value")
        start = int(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        raise SystemExit("usage: ... -- WORK SKELETON.fbx [SKELETON.fbx ...] [--start K]")
    return {"work": Path(args[0]), "files": [Path(a) for a in args[1:]], "start": start}


def box_map(model_lo, model_hi, drawn_lo, drawn_hi) -> dict:
    """How to carry a point from UniRig's box to the model's: scale by the ratio of the
    largest sides, keep the centres together. `axis_gap` is the worst disagreement of one
    side, after scaling, as a share of the model's largest side."""
    mlo, mhi = np.asarray(model_lo, float), np.asarray(model_hi, float)
    dlo, dhi = np.asarray(drawn_lo, float), np.asarray(drawn_hi, float)
    model_ext, drawn_ext = mhi - mlo, dhi - dlo
    if drawn_ext.max() <= 0 or model_ext.max() <= 0:
        raise ValueError("a box with no size")
    scale = float(model_ext.max() / drawn_ext.max())
    gap = float(np.abs(drawn_ext * scale - model_ext).max() / model_ext.max())
    return {"scale": scale, "model_centre": (mlo + mhi) / 2, "drawn_centre": (dlo + dhi) / 2,
            "axis_gap": gap}


def to_model(points, mapping: dict) -> np.ndarray:
    p = np.asarray(points, float)
    return (p - mapping["drawn_centre"]) * mapping["scale"] + mapping["model_centre"]


def parents_first(names: list[str], parent_names: list[str | None]) -> list[int]:
    """An order of the bones with every parent before its children (roots in file order)."""
    index = {n: i for i, n in enumerate(names)}
    children: dict[int, list[int]] = {i: [] for i in range(len(names))}
    roots = []
    for i, p in enumerate(parent_names):
        if p is None or p not in index:
            roots.append(i)
        else:
            children[index[p]].append(i)
    order, stack = [], list(reversed(roots))
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(reversed(children[i]))
    if len(order) != len(names):
        raise ValueError("the bones' parents loop")
    return order


def unirig_tails(heads: np.ndarray, parents: list[int], stub: float) -> np.ndarray:
    """Tails as UniRig sets them from its joints: a bone with one child ends at it; the root
    `stub` straight up; any other bone `stub` on along the line from its parent's joint."""
    heads = np.asarray(heads, float)
    kids = [[j for j, p in enumerate(parents) if p == i] for i in range(len(heads))]
    tails = heads.copy()
    for i, p in enumerate(parents):
        if p < 0:
            tails[i] = heads[i] + [0.0, 0.0, stub]
        elif len(kids[i]) == 1:
            tails[i] = heads[kids[i][0]]
        else:
            d = heads[i] - heads[p]
            n = np.linalg.norm(d)
            tails[i] = heads[i] + (d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])) * stub
    return tails


def draw_record(names, parent_names, heads, mapping: dict, source: str = "") -> dict:
    """One draw file: the bones parents first, in the model's frame, tails as UniRig made them."""
    order = parents_first(list(names), list(parent_names))
    position = {names[i]: k for k, i in enumerate(order)}
    heads = to_model(np.asarray(heads, float)[order], mapping)
    parents = [position.get(parent_names[i], -1) if parent_names[i] else -1 for i in order]
    tails = unirig_tails(heads, parents, UNIRIG_STUB * mapping["scale"])
    record = {"names": [names[i] for i in order], "parents": parents,
              "heads": np.round(heads, 6).tolist(), "tails": np.round(tails, 6).tolist(),
              "file": source, "scale": round(mapping["scale"], 6), "axis_gap": round(mapping["axis_gap"], 4)}
    if mapping["axis_gap"] > AXIS_TOLERANCE:
        record["warning"] = (f"UniRig's copy of the model disagrees with it by {100 * mapping['axis_gap']:.0f}% "
                             "of its size on one axis: the skeleton may not sit in the model")
    return record


def model_box(work: Path) -> tuple[np.ndarray, np.ndarray]:
    verts = np.load(Path(work) / "shell.npz")["model_verts"]
    return verts.min(0).astype(float), verts.max(0).astype(float)


# --- Blender ----------------------------------------------------------------------------------


def _import_fbx(bpy, path: Path) -> None:
    try:
        bpy.ops.import_scene.fbx(filepath=str(path))
    except (RuntimeError, AttributeError):
        bpy.ops.wm.fbx_import(filepath=str(path))   # the newer importer, where the old one is gone


def read_skeleton(bpy, path: Path) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Bone names, parents and joints (world space), and the box of the mesh copy, in one
    skeleton file."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _import_fbx(bpy, path)
    rigs = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(rigs) != 1:
        raise ValueError(f"{path.name} holds {len(rigs)} armatures, one expected")
    rig = rigs[0]
    corners = []
    for o in bpy.data.objects:
        if o.type == "MESH" and len(o.data.vertices):
            co = np.empty(len(o.data.vertices) * 3)
            o.data.vertices.foreach_get("co", co)
            m = np.array(o.matrix_world)
            co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            corners += [co.min(0), co.max(0)]
    if not corners:
        raise ValueError(f"{path.name} holds no mesh to place its skeleton by")
    m = np.array(rig.matrix_world)
    bones = list(rig.data.bones)
    heads = np.array([b.head_local[:] for b in bones]) @ m[:3, :3].T + m[:3, 3]
    corners = np.array(corners)
    return ([b.name for b in bones], [b.parent.name if b.parent else None for b in bones], heads,
            corners.min(0), corners.max(0))


def main() -> None:
    import bpy

    opts = parse_args(sys.argv)
    work = opts["work"].resolve()
    out = work / "draws"
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = model_box(work)
    done = []
    for i, path in enumerate(opts["files"]):
        k = opts["start"] + i
        names, parents, heads, dlo, dhi = read_skeleton(bpy, path.resolve())
        record = draw_record(names, parents, heads, box_map(lo, hi, dlo, dhi), str(path))
        target = out / f"draw{k}.json"
        target.write_text(json.dumps(record, indent=1), encoding="utf-8")
        done.append({"draw": k, "bones": len(names), "axis_gap": record["axis_gap"],
                     "warning": record.get("warning")})
        print(f"creature-rig read: draw {k}: {len(names)} bones from {path.name}"
              + (f" ({record['warning']})" if "warning" in record else ""), flush=True)
    print("I2L_CREATURE_READ::" + json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
