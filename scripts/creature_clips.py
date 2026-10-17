#!/usr/bin/env python3
"""Recognise a rigged creature's family and write its test clips, one animated GLB each.

Runs after the auto-rig, in the same Python that runs SkinTokens (it has the `bpy` wheel):

    python scripts/creature_clips.py --rig out/wolf_rigged.glb [--family auto|digitigrade|...]
        [--no-clips] [--no-library] [--no-variations] [--variation-factor 2] [--variation-seed 0]

Writes `wolf_rigged.family.json` (the family and the role of every chain, in glTF space)
and, for a family that has them, `wolf_<family>-<motion>.glb` beside the rig: the names
the studio already lists as clips of `wolf_rigged.glb`. With the CC0 clip library
installed (`scripts/bootstrap_clip_library.py`) a family first gets its library clips,
carried onto the rig by `chain_retarget.py`; the procedural clips of `creature_motion.py`
fill in the motions the library does not have. `--no-clips` writes only the family file
(the Unreal export reads a creature's facing from it). `--no-library` makes only the procedural
clips; `--no-variations` leaves out the longer takes of the looping library clips (GenMM,
`--variation-factor` times as long, varied with `--variation-seed`). This file only reads
the armature and keys it.

A test clip is a bonus on top of a rig, so a family or clip that cannot be made does not
undo the rig: the reason goes into the family file and the exit code stays 0. Each clip
gets a provenance sidecar carrying the rig's licences (the studio rewrites it with the
rig job's own record once the job ends).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_to_3dlab import clip_library as cl  # noqa: E402
from image_to_3dlab import creature_family as cf  # noqa: E402

AUTO = "auto"
RIGGED = "_rigged"


def clip_path(rigged: Path, name: str) -> Path:
    """`wolf_rigged.glb` + `digitigrade-walk` -> `wolf_digitigrade-walk.glb`."""
    stem = rigged.stem.removesuffix(RIGGED)
    return rigged.with_name(f"{stem}_{name}.glb")


def recognise(rigged: Path, chosen: str | None) -> dict:
    """The family file's content: the family and roles, or why there are none."""
    try:
        found = cf.classify_glb(rigged, family=chosen)
    except Exception as exc:  # noqa: BLE001 - an unreadable rig still is the rig the user asked for
        traceback.print_exc()
        return {"schema_version": cf.SCHEMA_VERSION, "family": None, "clips": [],
                "clips_error": f"could not recognise the family: {type(exc).__name__}: {exc}"}
    found["clips"], found["clips_error"] = [], None
    return found


def write_provenance(rigged: Path, clip: Path, name: str, family: str | None, made: dict | None = None) -> Path:
    """`<clip>.provenance.json`, from the rig's own sidecar when it has one."""
    import json

    try:
        rig_record = json.loads(rigged.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rig_record = None
    record = cf.clip_record(rig_record, name, family, tool="scripts/creature_clips.py", output=clip, made=made)
    path = clip.with_suffix(".provenance.json")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def wanted_motions(family: str, chosen: bool) -> tuple[str, ...]:
    """The clips to make: a family's own, but for humanoids (who have the preset moves)
    only when a person asked for that family."""
    import creature_motion

    if family not in creature_motion.DEFAULT_CLIPS and not chosen:
        return ()
    return creature_motion.clips_for(family)


def plan(family: str, chosen: bool, legs: int, installed: bool | None = None, library: bool = True,
         variations: bool = True, clips: bool = True) -> list[tuple[str, str | None]]:
    """(motion, library creature or None for a procedural clip), library clips first. A
    library motion replaces the procedural one of the same name. Without `library` only
    the procedural clips; without `variations` no longer takes of the library's loops;
    without `clips` none (the family alone)."""
    import creature_motion

    if not clips or (family not in creature_motion.DEFAULT_CLIPS and not chosen):
        return []
    installed = (cl.installed() if installed is None else installed) and library
    source = cl.source_for(family, legs) if installed else None
    motions = [*cl.SOURCES[source]["clips"], *(cl.variations(source) if variations else [])] if source else []
    library = [(motion, source) for motion in motions]
    have = {motion for motion, _ in library}
    return library + [(motion, None) for motion in wanted_motions(family, chosen) if motion not in have]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rig", required=True, type=Path, help="rigged GLB (SkinTokens output)")
    ap.add_argument("--family", default=AUTO, choices=(AUTO, *cf.FAMILIES))
    ap.add_argument("--no-clips", dest="clips", action="store_false",
                    help="only the family file, no test clips")
    ap.add_argument("--no-library", dest="library", action="store_false",
                    help="only the procedural test clips, not the CC0 library's")
    ap.add_argument("--no-variations", dest="variations", action="store_false",
                    help="no longer takes of the library's looping clips")
    ap.add_argument("--variation-factor", type=float, default=cl.LONGER_FACTOR,
                    help="how much longer a varied take is (default %(default)s)")
    ap.add_argument("--variation-seed", type=int, default=cl.LONGER_SEED, help="the variation's seed")
    ap.add_argument("--blend", type=Path, help="also save the last clip as a .blend for inspection")
    return ap


def main() -> None:  # pragma: no cover - needs bpy
    args = build_parser().parse_args()

    chosen = None if args.family == AUTO else args.family
    found = recognise(args.rig, chosen)
    if found["family"] is None:
        cf.family_path(args.rig).write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
        print(found["clips_error"])
        return
    errors = []
    found["clip_sources"] = {}
    for motion, source in plan(found["family"], chosen is not None, len(found.get("legs") or []),
                               library=args.library, variations=args.variations, clips=args.clips):
        try:
            name = _write_clip(args.rig, found["family"], motion, args.blend, source,
                               factor=args.variation_factor, seed=args.variation_seed)
        except Exception as exc:  # noqa: BLE001 - a missing clip must not cost the rig, or the other clips
            traceback.print_exc()
            errors.append(f"{motion}: {type(exc).__name__}: {exc}")
            continue
        made = cl.motion_record(source, motion, args.variation_factor, args.variation_seed) if source else None
        write_provenance(args.rig, clip_path(args.rig, name), name, found["family"], made)
        found["clips"].append(name)
        if made:
            found["clip_sources"][name] = made
        print(f"wrote {'library' if source else 'test'} clip {name}")
    found["clips_error"] = "; ".join(errors) or None
    cf.write_family(args.rig, found)
    print(f"family: {found['family']} ({found['confidence']} confidence; {found['detail']})")


def _write_clip(rigged: Path, family: str, motion: str, blend: Path | None,  # pragma: no cover
                source: str | None = None, factor: float = cl.LONGER_FACTOR, seed: int = cl.LONGER_SEED) -> str:
    import bpy
    from mathutils import Matrix

    import creature_motion as cm
    from kimodo_retarget import pose_basis

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(rigged))
    for o in list(bpy.data.objects):  # SkinTokens' `glTF_not_exported` helper
        if o.type == "MESH" and o.name.startswith("Icosphere"):
            bpy.data.objects.remove(o, do_unlink=True)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bones = arm.data.bones
    parents = {b.name: (b.parent.name if b.parent else None) for b in bones}
    order = [b.name for b in bones if b.parent is None]
    i = 0
    while i < len(order):
        order += [c.name for c in bones[order[i]].children]
        i += 1
    rest = {b.name: np.array(b.matrix_local) for b in bones}
    heads = {n: m[:3, 3].copy() for n, m in rest.items()}
    to_arm = np.array(arm.matrix_world.inverted())
    points = [(to_arm @ np.array(o.matrix_world) @ np.append(np.array(v.co), 1.0))[:3]
              for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == arm for v in o.data.vertices]
    arm_rot = np.array(arm.matrix_world.to_3x3().normalized())
    up, front = arm_rot.T @ np.array([0.0, 0.0, 1.0]), arm_rot.T @ np.array([0.0, -1.0, 0.0])  # glTF +Z is Blender -Y
    roles = cf.classify(parents, heads, up=up, front=front, family=family, points=np.array(points))
    rig = cm.Rig(parents, order, heads)
    if source:
        import chain_retarget as cr

        clip, loop = cl.SOURCES[source]["clips"][cl.base_motion(motion)]
        clip_source = cr.load_source(cl.pack_dir() / cl.SOURCES[source]["file"], cl.source_roles(source), clip,
                                     loop=loop)
        if motion != cl.base_motion(motion):
            clip_source = cr.longer(clip_source, factor, seed, loop)
        frames = cr.retarget(rig, roles, clip_source)
    else:
        frames = cm.make(motion, rig, roles)

    scene = bpy.context.scene
    scene.render.fps = cm.FPS
    scene.frame_start, scene.frame_end = 1, len(frames)
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    rest_rot = {b: m[:3, :3] for b, m in rest.items()}
    for t, (turn, root) in enumerate(frames, start=1):
        world_rot, root_pos = cm.frame_rotations(rig, rest_rot, turn, root)
        for name, mtx in pose_basis(rest, parents, order, world_rot, root_pos).items():
            pb = arm.pose.bones[name]
            pb.matrix_basis = Matrix(mtx.tolist())
            pb.keyframe_insert("rotation_quaternion", frame=t)
            if parents[name] is None:
                pb.keyframe_insert("location", frame=t)
    name = cm.clip_name(family, motion)
    arm.animation_data.action.name = name
    out = clip_path(rigged, name)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_animations=True)
    if blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    return name


if __name__ == "__main__":
    main()
