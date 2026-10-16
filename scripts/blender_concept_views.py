"""Render a GLB for a concept: a lit colour picture and a depth map per view, plus the bones.

Headless, on the CPU (the GPU stays with generation work):

    blender -b --factory-startup -P scripts/blender_concept_views.py -- model.glb outdir front,side,back [size]

For each view this writes, in ``outdir``:

- ``{view}_colour.png``: the model lit by three soft lights on a transparent background,
  the picture a concept is drawn over (the base body, the front of a creature);
- ``{view}_depth.png``: nearest white, farthest dark, transparent background
  (``image_to_3dlab.concept.maps.depth_on_black`` stretches it onto black);

and ``bones.json`` with each bone's head and tail in pixels of every view, top-left origin.
Cameras are orthographic and fit the model with a margin. Bone display shapes and small meshes
(under 2% of the largest one's faces, such as a helper left in an export) are hidden, or
they would wreck the framing.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

AZIMUTH = {"front": 0.0, "side": 90.0, "back": 180.0}
MARGIN = 1.12
STRAY_FACES = 0.02
LIGHTS = (("key", 110, (-1, -1.5, 1.2)), ("fill", 45, (1.5, -1, 0.4)), ("rim", 55, (0.5, 1.5, 1.5)))


def main(glb: Path, out: Path, views: list[str], size: int) -> None:
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    out.mkdir(parents=True, exist_ok=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))

    # Not the bones' display shapes: the glTF importer adds an icosphere for them.
    shapes = {pb.custom_shape for o in bpy.context.scene.objects if o.type == "ARMATURE"
              for pb in o.pose.bones if pb.custom_shape}
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o not in shapes]
    for shape in shapes:
        shape.hide_render = True
    if not meshes:
        raise SystemExit(f"{glb.name} has no mesh to render")
    most = max(len(o.data.polygons) for o in meshes)
    for obj in meshes:
        if len(obj.data.polygons) < STRAY_FACES * most:
            obj.hide_render = True
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in meshes:
        if obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        points += [evaluated.matrix_world @ v.co for v in mesh.vertices]
        evaluated.to_mesh_clear()
    low = Vector([min(p[i] for p in points) for i in range(3)])
    high = Vector([max(p[i] for p in points) for i in range(3)])
    centre, radius = (low + high) / 2, max((high - low).length, 1e-6)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("concept")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.25
    scene.world = world

    camera_data = bpy.data.cameras.new("concept")
    camera_data.type = "ORTHO"
    camera_data.clip_end = radius * 10
    camera = bpy.data.objects.new("concept", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for name, energy, offset in LIGHTS:
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
        light.data.energy = energy * radius * radius
        light.data.size = radius
        light.location = centre + Vector(offset) * radius
        light.rotation_euler = (centre - light.location).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)

    # Depth: the camera's distance remapped to white (near) .. 0.15 (far), as emission.
    depth = bpy.data.materials.new("concept_depth")
    depth.use_nodes = True
    nodes, links = depth.node_tree.nodes, depth.node_tree.links
    nodes.clear()
    camera_node = nodes.new("ShaderNodeCameraData")
    remap = nodes.new("ShaderNodeMapRange")
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(camera_node.outputs["View Z Depth"], remap.inputs["Value"])
    remap.inputs["To Min"].default_value = 1.0
    remap.inputs["To Max"].default_value = 0.15
    links.new(remap.outputs["Result"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    armature = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    bones = {}
    for view in views:
        angle = math.radians(AZIMUTH[view])
        camera.location = centre + Vector((math.sin(angle), -math.cos(angle), 0)) * radius * 3
        camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        local = [camera.matrix_world.inverted() @ p for p in points]
        width = max(p.x for p in local) - min(p.x for p in local)
        height = max(p.y for p in local) - min(p.y for p in local)
        camera_data.ortho_scale = MARGIN * max(width, height)
        distances = [-p.z for p in local]
        remap.inputs["From Min"].default_value = min(distances)
        remap.inputs["From Max"].default_value = max(distances)

        bpy.context.view_layer.material_override = None
        scene.cycles.samples, scene.cycles.use_denoising = 32, True
        scene.render.filepath = str(out / f"{view}_colour.png")
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.material_override = depth
        scene.cycles.samples, scene.cycles.use_denoising = 1, False
        scene.render.filepath = str(out / f"{view}_depth.png")
        bpy.ops.render.render(write_still=True)

        if armature is not None:
            placed = {}
            for bone in armature.data.bones:
                ends = []
                for co in (bone.head_local, bone.tail_local):
                    p = world_to_camera_view(scene, camera, armature.matrix_world @ co)
                    ends.append([p.x * size, (1 - p.y) * size])
                placed[bone.name] = {"head": ends[0], "tail": ends[1],
                                     "parent": bone.parent.name if bone.parent else None}
            bones[view] = placed
    if bones:
        (out / "bones.json").write_text(json.dumps(bones, indent=1), encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    main(Path(args[0]), Path(args[1]), args[2].split(","), int(args[3]) if len(args) > 3 else 1024)
