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
- ``view_NN_lit.png``, only when the plan asks for ``"lit"``: the base colour (and its normal
  map) on a neutral dielectric, roughness 0.5, under a key light from the camera's upper left
  and a dim grey sky. Models that read materials from shading (Marigold-IID Appearance) need
  light to read; it is neutral so that no reflection in the picture is one we invented.
  With ``"light"`` (the direction the light travels, world axes) the key light stays fixed in
  the world instead, as light painted into a texture does;

and ``views.json`` with every view's direction. The data passes are rendered to EXR and
saved as NumPy arrays by Blender itself, so reading them needs no EXR support elsewhere. Cameras are orthographic, keep the world's up
up the picture, and fit the model's box as each one sees it. One sample and no pixel filter: each pixel
is one point of the surface, never a blend of two.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def basis(direction):
    """(right, up, forward) of a camera looking along `direction` with the world's Z kept up.

    Built here rather than with ``to_track_quat``, so the roll is never left to chance: a
    2D model shown a figure lying on its side is shown something it was not trained on.
    Straight down or up, the picture's up is the world's +Y.
    """
    length = math.sqrt(sum(d * d for d in direction)) or 1.0
    forward = [d / length for d in direction]
    reference = (0.0, 0.0, 1.0) if abs(forward[2]) < 0.99 else (0.0, 1.0, 0.0)
    right = _cross(forward, reference)
    norm = math.sqrt(sum(r * r for r in right))
    right = [r / norm for r in right]
    return right, _cross(right, forward), forward


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def frame(corners, direction):
    """(camera location, orthographic scale, far clip) that fit the box's outline as this camera sees it.

    The box's corners are projected on the picture plane, so a tall figure seen from the
    front fills the frame instead of sitting inside its own diagonal.
    """
    right, up, forward = basis(direction)
    across = [sum(c[i] * right[i] for i in range(3)) for c in corners]
    upward = [sum(c[i] * up[i] for i in range(3)) for c in corners]
    depth = [sum(c[i] * forward[i] for i in range(3)) for c in corners]
    low = [min(c[i] for c in corners) for i in range(3)]
    high = [max(c[i] for c in corners) for i in range(3)]
    diagonal = math.dist(low, high) or 1.0
    mid_across, mid_up = (min(across) + max(across)) / 2, (min(upward) + max(upward)) / 2
    centre = [mid_across * right[i] + mid_up * up[i] + min(depth) * forward[i] for i in range(3)]
    location = [c - f * diagonal for c, f in zip(centre, forward)]
    span = max(max(across) - min(across), max(upward) - min(upward)) or diagonal
    return location, span * 1.05, diagonal * 3.0


def uv_set(node, depth: int = 0) -> str | None:
    """The UV map a texture reads, found upstream through Mapping and other vector nodes."""
    if node is None or depth > 8:
        return None
    if node.type == "UVMAP":
        return node.uv_map or None
    for socket in node.inputs:
        if socket.links:
            found = uv_set(socket.links[0].from_node, depth + 1)
            if found:
                return found
    return None


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
    from mathutils import Matrix, Vector

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    folder = plan_path.parent
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(folder / plan["glb"]))
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH"]

    lit = bool(plan.get("lit"))
    passes = {"color": [], "uv": [], "normal": [], "lit": []}
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
        found = uv_set(base.links[0].from_node) if base is not None and base.links else None
        if found:
            uv_map.uv_map = found          # the base colour's own UV set, through any Mapping node
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
        studio = nodes.new("ShaderNodeBsdfPrincipled")
        studio.inputs["Roughness"].default_value = 0.5
        studio.inputs["Metallic"].default_value = 0.0
        if base is not None and base.links:
            links.new(base.links[0].from_socket, studio.inputs["Base Color"])
        elif base is not None:
            studio.inputs["Base Color"].default_value = base.default_value
        bump = principled.inputs.get("Normal") if principled else None
        if bump is not None and bump.links:
            links.new(bump.links[0].from_socket, studio.inputs["Normal"])
        alpha = principled.inputs.get("Alpha") if principled else None
        for name, node in (("color", colour), ("uv", uv), ("normal", normal), ("lit", studio)):
            shader = node.outputs["BSDF"] if name == "lit" else node.outputs["Emission"]
            if alpha is not None and (alpha.links or alpha.default_value < 1.0):
                # Keep cut-outs and blends see-through in every pass: a hair card or a leaf
                # must not hide the surface behind it, nor claim its pixels.
                mix = nodes.new("ShaderNodeMixShader")
                if alpha.links:
                    links.new(alpha.links[0].from_socket, mix.inputs["Fac"])
                else:
                    mix.inputs["Fac"].default_value = alpha.default_value
                links.new(nodes.new("ShaderNodeBsdfTransparent").outputs["BSDF"], mix.inputs[1])
                links.new(shader, mix.inputs[2])
                shader = mix.outputs["Shader"]
            passes[name].append((links, shader, output))

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
    if lit:
        scene.world.use_nodes = True
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)
        sun = bpy.data.objects.new("key", bpy.data.lights.new("key", "SUN"))
        sun.data.energy = 3.0
        sun.data.angle = 0.2
        scene.collection.objects.link(sun)

    camera = bpy.data.objects.new("view", bpy.data.cameras.new("view"))
    camera.data.type = "ORTHO"
    scene.collection.objects.link(camera)
    scene.camera = camera
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]

    written = []
    for index, direction in enumerate(plan["directions"]):
        location, scale, far = frame(corners, direction)
        camera.location = location
        camera.data.ortho_scale = scale
        camera.data.clip_end = far
        camera.data.clip_start = far / 3000.0       # the default 0.1 m cuts away models a few cm wide
        right, up, forward = basis(direction)
        # Blender cameras look down their local -Z, with local +Y up the picture.
        camera.rotation_euler = Matrix((right, up, [-f for f in forward])).transposed().to_euler()
        if lit:
            # The light travels from the camera's upper left towards the model, or as the plan says.
            travel = plan.get("light") or [forward[i] - 0.6 * up[i] + 0.5 * right[i] for i in range(3)]
            sun.rotation_euler = Vector(travel).to_track_quat("-Z", "Y").to_euler()
        names = (("color", "PNG"), ("uv", "OPEN_EXR"), ("normal", "OPEN_EXR")) + ((("lit", "PNG"),) if lit else ())
        for name, extension in names:
            for links, shader, output in passes[name]:
                links.new(shader, output.inputs["Surface"])
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
            # Data passes: one sample, no filter, so a pixel is one point. The lit picture is a picture.
            scene.cycles.samples = 16 if name == "lit" else 1
            scene.cycles.use_denoising = name == "lit"
            scene.cycles.filter_width = 1.5 if name == "lit" else 0.01
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
