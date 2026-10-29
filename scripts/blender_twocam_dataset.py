"""Render a prepared creature in random poses from both layout cameras, with every joint's true pixel position, headless.

blender -b RIGGED.blend --python scripts/blender_twocam_dataset.py -- RIG_INFO.json OUT_DIR START COUNT SEED [SCALE]

The joint reader learns one creature from these. Each sample i writes OUT_DIR/{i:05d}_front.png
and _side.png (RGBA) and _meta.json (joints in metres and in layout pixels for both views,
plus the pose). Ported from AnyMove's `blender/render_dataset.py` (Patch, Apache-2.0),
changed to read the layout from rig_info.json and to treat any rig's limbs (`_L`/`_R` bones)
as the fast-moving parts, not the wolf's named ones. About half the samples carry motion
blur from a random velocity, the way a video model smears fast limbs; light, turn, place
and airtime are random too. A sample already on disk is skipped, so a stopped run resumes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str]) -> dict:
    args = argv[argv.index("--") + 1:] if "--" in argv else list(argv)
    if len(args) < 5:
        raise SystemExit("usage: ... -- RIG_INFO.json OUT_DIR START COUNT SEED [SCALE]")
    return {"rig_info": Path(args[0]), "out": Path(args[1]), "start": int(args[2]), "count": int(args[3]),
            "seed": int(args[4]), "scale": float(args[5]) if len(args) > 5 else 0.6}


def fast_bone(name: str, leaf: bool) -> bool:
    """Limbs and loose ends swing fast in a video; the trunk does not."""
    return name.endswith(("_L", "_R")) or leaf


def sample_pose(rng, bones, limits, root):
    """Euler angles per bone: half uniform inside the limits, half a narrower spread around rest."""
    import numpy as np

    pose = {}
    wide = rng.random() < 0.5
    for name in bones:
        lo, hi = limits[name][:, 0].copy(), limits[name][:, 1].copy()
        if name == root and rng.random() < 0.15:  # sometimes rear up or dive
            lo[0] = math.radians(-60)
        pose[name] = rng.uniform(lo, hi) if wide else np.clip(rng.normal(0, 0.3 * (hi - lo) / 2), lo, hi)
    return pose


def main() -> None:
    import bpy
    import numpy as np
    from mathutils import Euler, Vector

    sys.path.insert(0, str(REPO / "scripts"))
    import blender_twocam_studio as studio

    opts = parse_args(sys.argv)
    out = opts["out"]
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    info = json.loads(opts["rig_info"].read_text(encoding="utf-8"))
    layout = info["layout"]
    rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
    bones = [b["name"] for b in info["bones"]]
    parents = [b["parent"] for b in info["bones"]]
    leaves = {bones[i] for i in range(len(bones)) if i not in parents}
    limits = {b["name"]: np.radians(np.array(b["limits_deg"], float)) for b in info["bones"]}
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
    device = studio.studio(scene, samples=16)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.denoising_use_gpu = device != "CPU"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    cams = studio.cameras(scene, layout)
    key, fill, rim = (bpy.data.objects[n] for n in ("key", "fill", "rim"))
    bg = scene.world.node_tree.nodes["Background"]
    m, h, zc = layout["M_PER_PX"], layout["H"], studio.z_center(layout)

    def apply(pose, loc, yaw, frame):
        for name, a in pose.items():
            pb = rig.pose.bones[name]
            pb.rotation_euler = Euler(a.tolist(), "XYZ")
            if frame is not None:
                pb.keyframe_insert("rotation_euler", frame=frame)
        rig.location = loc
        rig.rotation_euler = (0, 0, yaw)
        if frame is not None:
            rig.keyframe_insert("location", frame=frame)
            rig.keyframe_insert("rotation_euler", frame=frame)

    def lowest_z():
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        mw = np.array(body.matrix_world)
        return (co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])[:, 2].min()

    def joints_world():
        mw = rig.matrix_world
        pts = []
        for kp in info["keypoints"]:
            pb = rig.pose.bones[bones[kp["bone"]]]
            pts.append(list(mw @ (pb.head if kp["end"] == "head" else pb.tail)))
        return np.array(pts)

    rng = np.random.default_rng(opts["seed"] + opts["start"])
    for i in range(opts["start"], opts["start"] + opts["count"]):
        stem = out / f"{i:05d}"
        pose = sample_pose(rng, bones, limits, bones[0])
        r = rng.random()
        yaw = math.radians(rng.uniform(-30, 30) if r < 0.75 else rng.uniform(-70, 70) if r < 0.85 else 0)
        loc = Vector((rng.uniform(-0.08, 0.08), rng.uniform(-0.25, 0.25), 0))
        blur = rng.random() < 0.5
        vel = {n: rng.normal(0, math.radians(6 if fast_bone(n, n in leaves) else 2.5), 3) for n in bones}
        vloc = Vector(rng.normal(0, 0.015, 3).tolist())
        vyaw = rng.normal(0, math.radians(3))
        lights = (rng.uniform(20, 70), rng.uniform(-math.pi, math.pi), rng.uniform(1.5, 4.5),
                  rng.uniform(30, 80), rng.uniform(-math.pi, math.pi), rng.uniform(0.2, 1.5),
                  rng.uniform(0.0, 2.5), rng.uniform(0.3, 0.9), rng.uniform(0.4, 0.6),
                  rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1))
        airtime = rng.uniform(0.05, 0.45) if rng.random() < 0.2 else 0
        shutter = rng.uniform(0.5, 1.0)
        if (out / f"{i:05d}_meta.json").exists():  # drawn above all the same, so a resumed run matches
            continue
        rig.animation_data_clear()
        for a in list(bpy.data.actions):
            bpy.data.actions.remove(a)
        for pb in rig.pose.bones:
            pb.location = (0, 0, 0)
        apply(pose, loc, yaw, None)
        bpy.context.view_layer.update()
        loc.z = -lowest_z() + airtime
        if blur:  # the pose keyed at frames 0, 1, 2 along a random velocity; frame 1 rendered with blur
            for f, s in ((0, -1), (1, 0), (2, 1)):
                apply({n: pose[n] + s * vel[n] for n in bones}, loc + s * vloc, yaw + s * vyaw, f)
            scene.frame_set(1)
            scene.render.use_motion_blur = True
            scene.render.motion_blur_shutter = shutter
        else:
            apply(pose, loc, yaw, None)
            scene.render.use_motion_blur = False
            bpy.context.view_layer.update()
        key.rotation_euler = (math.radians(lights[0]), 0, lights[1])
        key.data.energy = lights[2]
        fill.rotation_euler = (math.radians(lights[3]), 0, lights[4])
        fill.data.energy = lights[5]
        rim.data.energy = lights[6]
        bg.inputs["Strength"].default_value = lights[7]
        g = lights[8]
        bg.inputs["Color"].default_value = (g * lights[9], g, g * lights[10], 1)

        jw = joints_world()
        front = np.stack([layout["W_FRONT"] / 2 + jw[:, 0] / m, h / 2 - (jw[:, 2] - zc) / m], 1)
        side = np.stack([layout["W_SIDE"] / 2 - jw[:, 1] / m, h / 2 - (jw[:, 2] - zc) / m], 1)
        for name, (cam, w) in cams.items():
            scene.camera = cam
            scene.render.resolution_x, scene.render.resolution_y = round(w * opts["scale"]), round(h * opts["scale"])
            scene.render.filepath = f"{stem}_{name.split('_')[1]}.png"
            bpy.ops.render.render(write_still=True)
        meta = {"joints_world": jw.round(5).tolist(), "front_uv": front.round(3).tolist(),
                "side_uv": side.round(3).tolist(), "scale": opts["scale"], "blur": bool(blur), "yaw": yaw,
                "root": list(loc), "pose": {n: np.round(a, 5).tolist() for n, a in pose.items()}}
        (out / f"{i:05d}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"I2L_TWOCAM_SAMPLE::{i}", flush=True)


if __name__ == "__main__":
    main()
