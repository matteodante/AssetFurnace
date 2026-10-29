"""Blender helpers shared by the two-camera scripts: the layout's cameras, the studio light, and limb paint.

Imported by `blender_twocam_prepare.py` and `blender_twocam_dataset.py`, inside Blender.
Ported from AnyMove's `blender/common.py` (Patch, Apache-2.0), changed to take the layout
as an argument (AnyMove read it from an environment variable) and to paint any limbs
`image_to_3dlab.twocam.rigging` names, not only the wolf's legs.

The layout is the contract between every stage. One frame holds two orthographic views at
the same scale: the front camera (left) looks along +Y at the creature's face, the side
camera (right) looks along +X at its right flank, head to the right. Both share the
vertical axis, so a joint's height is the same row in both halves.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np

PAINT_LAYER = "legpaint"  # AnyMove's name, so its own scripts read our rigs


def z_center(layout: dict) -> float:
    return (layout["GROUND_V"] - layout["H"] / 2) * layout["M_PER_PX"]


def srgb_to_linear(hexcol: str) -> np.ndarray:
    c = np.array([int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def paint_limbs(body, groups: dict, threshold: float = 0.55) -> dict[str, int]:
    """Colour each limb from its first bone down, as a colour attribute mixed over the
    texture: so the video model and the reader can tell a left leg from a right one."""
    index = {g.name: g.index for g in body.vertex_groups}
    n = len(body.data.vertices)
    weights = {}
    for key, group in groups.items():
        w = np.zeros(n)
        wanted = {index[b] for b in group["bones"] if b in index}
        for v in body.data.vertices:
            for g in v.groups:
                if g.group in wanted:
                    w[v.index] += g.weight
        weights[key] = w
    rgba = np.zeros((n, 4))
    for key, group in groups.items():
        m = weights[key] > threshold
        rgba[m, :3] = srgb_to_linear(group["color"])
        rgba[m, 3] = 1.0
    attr = body.data.color_attributes.get(PAINT_LAYER) or body.data.color_attributes.new(
        PAINT_LAYER, "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", rgba.ravel())
    for slot in body.material_slots:
        if slot.material is None or not slot.material.use_nodes:
            continue
        nt = slot.material.node_tree
        bsdf = next((nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        base = bsdf.inputs["Base Color"]
        src = base.links[0].from_socket if base.links else None
        ca = nt.nodes.new("ShaderNodeVertexColor")
        ca.layer_name = PAINT_LAYER
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(ca.outputs["Alpha"], mix.inputs["Factor"])
        if src is not None:
            nt.links.new(src, mix.inputs["A"])
        else:
            mix.inputs["A"].default_value = base.default_value
        nt.links.new(ca.outputs["Color"], mix.inputs["B"])
        nt.links.new(mix.outputs["Result"], base)
    return {key: int((w > threshold).sum()) for key, w in weights.items()}


def cameras(scene, layout: dict) -> dict:
    """The front and side orthographic cameras of the layout (created once, reused)."""
    m, h = layout["M_PER_PX"], layout["H"]
    zc = z_center(layout)
    cams = {}
    for name, loc, rot, w in (("cam_front", (0, -20, zc), (math.pi / 2, 0, 0), layout["W_FRONT"]),
                              ("cam_side", (-20, 0, zc), (math.pi / 2, 0, -math.pi / 2), layout["W_SIDE"])):
        cam = bpy.data.objects.get(name)
        if cam is None:
            cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            scene.collection.objects.link(cam)
        cam.location, cam.rotation_euler = loc, rot
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = m * max(w, h)
        cam.data.clip_end = 100
        cams[name] = (cam, w)
    return cams


def studio(scene, samples: int = 32, device: str = "auto") -> str:
    """Cycles with a transparent film, Standard view transform (keeps the limb paint
    saturated) and three suns. Renders on the GPU when Blender finds one (CUDA or OptiX on
    NVIDIA, HIP on AMD, Metal on a Mac), else on the CPU. Returns the device used."""
    scene.render.engine = "CYCLES"
    used = "CPU"
    if device != "cpu":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                used = kind
                break
    scene.cycles.device = "CPU" if used == "CPU" else "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    bg.inputs["Strength"].default_value = 0.55
    for name, rot, energy in (("key", (math.radians(50), 0, math.radians(-35)), 3.2),
                              ("fill", (math.radians(60), 0, math.radians(120)), 1.0),
                              ("rim", (math.radians(35), 0, math.radians(200)), 1.6)):
        ob = bpy.data.objects.get(name)
        if ob is None:
            ob = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
            scene.collection.objects.link(ob)
        ob.rotation_euler = rot
        ob.data.energy = energy
        ob.data.angle = math.radians(8)
    return used


def render_views(out: Path, prefix: str, layout: dict, scale: float = 1.0) -> dict[str, str]:
    """Render the front and side views as RGBA PNGs at the layout's pixel scale x `scale`."""
    scene = bpy.context.scene
    paths = {}
    for name, (cam, w) in cameras(scene, layout).items():
        scene.camera = cam
        scene.render.resolution_x = round(w * scale)
        scene.render.resolution_y = round(layout["H"] * scale)
        scene.render.resolution_percentage = 100
        view = name.split("_")[1]
        scene.render.filepath = str(Path(out) / f"{prefix}_{view}.png")
        bpy.ops.render.render(write_still=True)
        paths[view] = scene.render.filepath
    return paths
