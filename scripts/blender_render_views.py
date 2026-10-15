"""Render a GLB's albedo unlit from an orbit of views, with the UV and normal of every pixel.

Headless, on the CPU (the GPU stays with generation work):

    blender -b --factory-startup -P scripts/blender_render_views.py -- <folder>/plan.json

``plan.json`` comes from ``image_to_3dlab.multiview.render_plan``: the tagged copy of the
GLB (materials named ``afmat_<glTF index>``), the image size and the directions the
cameras look along. For each view this writes, in the plan's folder:

- ``view_NN_color.png``: the base colour as shown, no light at all (emission);
- ``view_NN_uv.npz``: R, G = the pixel's UV (Blender's, v upward), B = glTF material index + 1
  (0 is background), A = coverage; float32, top row first;
- ``view_NN_normal.npz``: the world normal, the same way;

and ``views.json`` with every view's direction. The data passes are rendered to EXR and
saved as NumPy arrays by Blender itself, so reading them needs no EXR support elsewhere. Cameras are orthographic and frame the
whole model, so every view sees it at one scale. One sample and no pixel filter: each pixel
is one point of the surface, never a blend of two.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def frame(corners, direction):
    """(camera location, orthographic scale) for a camera looking along `direction` at the box."""
    low = [min(c[i] for c in corners) for i in range(3)]
    high = [max(c[i] for c in corners) for i in range(3)]
    centre = [(a + b) / 2 for a, b in zip(low, high)]
    diagonal = math.dist(low, high) or 1.0
    location = [c - d * diagonal * 2.0 for c, d in zip(centre, direction)]
    return location, diagonal * 1.05


def save_array(bpy, exr: Path, size: int) -> None:
    """Keep an EXR pass as ``.npz`` float32, top row first, and drop the EXR."""
    import numpy as np

    image = bpy.data.images.load(str(exr))
    pixels = np.empty(size * size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    np.savez_compressed(exr.with_suffix(".npz"), rgba=pixels.reshape(size, size, 4)[::-1])
    exr.unlink()


def main(plan_path: Path) -> int:
    import bpy
    from mathutils import Vector

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    folder = plan_path.parent
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(folder / plan["glb"]))
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]

    passes = {"color": [], "uv": [], "normal": []}
    for material in {s.material for o in meshes for s in o.material_slots if s.material}:
        material.use_nodes = True
        tree = material.node_tree
        nodes, links = tree.nodes, tree.links
        output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
        principled = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        colour = nodes.new("ShaderNodeEmission")
        base = principled.inputs["Base Color"] if principled else None
        if base is not None and base.links:
            links.new(base.links[0].from_socket, colour.inputs["Color"])
        elif base is not None:
            colour.inputs["Color"].default_value = base.default_value
        index = int(material.name.split("_")[1].split(".")[0]) if material.name.startswith("afmat_") else -1
        uv_map = nodes.new("ShaderNodeUVMap")
        source = base.links[0].from_node if base is not None and base.links else None
        vector = source.inputs.get("Vector") if source is not None else None
        if vector is not None and vector.links and vector.links[0].from_node.type == "UVMAP":
            uv_map.uv_map = vector.links[0].from_node.uv_map   # the base colour's own UV set
        split = nodes.new("ShaderNodeSeparateXYZ")
        links.new(uv_map.outputs["UV"], split.inputs["Vector"])
        join = nodes.new("ShaderNodeCombineXYZ")
        links.new(split.outputs["X"], join.inputs["X"])
        links.new(split.outputs["Y"], join.inputs["Y"])
        join.inputs["Z"].default_value = index + 1
        uv = nodes.new("ShaderNodeEmission")
        links.new(join.outputs["Vector"], uv.inputs["Color"])
        geometry = nodes.new("ShaderNodeNewGeometry")
        normal = nodes.new("ShaderNodeEmission")
        links.new(geometry.outputs["Normal"], normal.inputs["Color"])
        for name, node in (("color", colour), ("uv", uv), ("normal", normal)):
            passes[name].append((links, node, output))
        material.blend_method = "OPAQUE" if hasattr(material, "blend_method") else None

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.filter_width = 0.01
    scene.render.use_persistent_data = True
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = plan["size"]
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("black")

    camera = bpy.data.objects.new("view", bpy.data.cameras.new("view"))
    camera.data.type = "ORTHO"
    scene.collection.objects.link(camera)
    scene.camera = camera
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]

    written = []
    for index, direction in enumerate(plan["directions"]):
        location, scale = frame(corners, direction)
        camera.location = location
        camera.data.ortho_scale = scale
        camera.data.clip_end = scale * 10
        camera.rotation_euler = Vector(direction).to_track_quat("-Z", "Z" if abs(direction[2]) < 0.99 else "Y").to_euler()
        for name, extension in (("color", "PNG"), ("uv", "OPEN_EXR"), ("normal", "OPEN_EXR")):
            for links, node, output in passes[name]:
                links.new(node.outputs["Emission"], output.inputs["Surface"])
            settings = scene.render.image_settings
            settings.file_format = extension
            settings.color_mode = "RGBA"
            if extension == "OPEN_EXR":
                settings.color_depth = "32"
                settings.exr_codec = "ZIP"
                scene.view_settings.view_transform = "Raw"
            else:
                settings.color_depth = "8"
                scene.view_settings.view_transform = "Standard"
            suffix = "png" if extension == "PNG" else "exr"
            path = folder / f"view_{index:02d}_{name}.{suffix}"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            if extension == "OPEN_EXR":
                save_array(bpy, path, plan["size"])
        written.append({"index": index, "direction": direction})
    (folder / "views.json").write_text(json.dumps({"size": plan["size"], "views": written}, indent=2),
                                       encoding="utf-8")
    print(f"rendered {len(written)} views into {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[sys.argv.index("--") + 1])))
