#!/usr/bin/env python3
"""Render a game-materials GLB before and after a tint change, under two different lights.

Headless proof for ``game_materials.py``: the tint formula from the asset's
``.materials.json`` is built in shader nodes exactly as Unreal will rebuild it, then the
asset is rendered with its original colours and with new ones, under a front key light
(the concept's) and under a low light from the side and behind. Painted shadows show up
as dark patches that do not move with the new light.

    blender -b -P scripts/blender_tint_preview.py -- asset.glb out/ --tint R=#1f3a7a G=#2f6b2a
    blender -b -P scripts/blender_tint_preview.py -- asset.glb out/ --plain    # no sidecar

Writes ``original_key.png``, ``original_side.png``, ``tinted_key.png``, ``tinted_side.png``
(only the ``original_*`` pair with ``--plain`` or no ``--tint``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

CHANNELS = ("R", "G", "B")
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
# Where each light shines toward, in Blender axes (Z up, the asset's front facing -Y).
LIGHTS = {
    "key": [(-0.35, 0.8, -0.5)],                       # front, above, slightly right
    "side": [(0.85, -0.35, 0.25), (-0.5, 0.6, 0.0)],   # low from the left-behind, weak fill
}
LIGHT_STRENGTH = {"key": [3.0], "side": [4.0, 0.4]}


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def parse_tints(items: list[str]) -> dict[str, list[float]]:
    """``["R=#1f3a7a"]`` -> ``{"R": [linear r, g, b]}``. Tints are given as sRGB hex."""
    tints = {}
    for item in items or []:
        channel, _, colour = item.partition("=")
        channel = channel.strip().upper()
        colour = colour.strip().lstrip("#")
        if channel not in CHANNELS:
            raise ValueError(f"unknown zone {channel!r}: use R, G or B")
        if len(colour) != 6 or any(c not in "0123456789abcdefABCDEF" for c in colour):
            raise ValueError(f"expected a #rrggbb colour, got {item!r}")
        tints[channel] = [_srgb_to_linear(int(colour[i:i + 2], 16) / 255) for i in (0, 2, 4)]
    return tints


def tint_terms(zones: list[dict], tints: dict[str, list[float]]) -> list[dict]:
    """The per-zone constants of ``out = albedo + m * g * (t - c)``, ready for nodes.

    ``g = Y(albedo) / Y(c)``, so each term is ``m * Y(albedo) * scale * delta`` with
    ``scale = 1 / Y(c)`` and ``delta = t - c``. Zones without a new tint add nothing.
    """
    terms = []
    for zone in zones:
        channel = zone["channel"]
        if channel not in tints:
            continue
        mean = zone["mean_linear"]
        y = sum(w * c for w, c in zip(LUMINANCE_WEIGHTS, mean))
        terms.append({"channel": channel,
                      "scale": 1.0 / y if y > 1e-6 else 0.0,
                      "delta": [t - c for t, c in zip(tints[channel], mean)]})
    return terms


def mask_for_material(sidecar: dict, name: str) -> str | None:
    """The tint mask file for one Blender material, matched by its glTF name.

    Blender suffixes repeated names (``arm.001``); a name the sidecar holds twice cannot
    be told apart here, so that material is left untinted rather than given the wrong mask.
    """
    materials = sidecar.get("materials") or []
    if len(materials) <= 1:
        return (sidecar.get("textures") or {}).get("tint_mask")
    for wanted in (name or "", re.sub(r"\.\d{3}$", "", name or "")):   # exact name first
        matches = [m for m in materials if m.get("name") == wanted]
        if matches:
            return matches[0].get("textures", {}).get("tint_mask") if len(matches) == 1 else None
    return None


def light_rotation(toward) -> tuple[float, float, float]:
    """XYZ Euler angles that turn a light's -Z axis onto the direction `toward`."""
    x, y, z = toward
    length = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / length, y / length, z / length
    # -Z rotated by Ry(b) then Rz(c) is (-sin b cos c, -sin b sin c, -cos b).
    b = math.acos(max(-1.0, min(1.0, -z)))
    c = math.atan2(-y, -x) if abs(math.sin(b)) > 1e-9 else 0.0
    return 0.0, b, c


def _build_tint_nodes(bpy, material, mask_path: Path, terms: list[dict]) -> None:
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    base_link = principled.inputs["Base Color"].links[0]
    albedo = base_link.from_socket

    mask = nodes.new("ShaderNodeTexImage")
    mask.image = bpy.data.images.load(str(mask_path))
    mask.image.colorspace_settings.name = "Non-Color"
    uv_source = base_link.from_node.inputs.get("Vector")
    if uv_source is not None and uv_source.links:
        links.new(uv_source.links[0].from_socket, mask.inputs["Vector"])
    split = nodes.new("ShaderNodeSeparateColor")
    links.new(mask.outputs["Color"], split.inputs["Color"])

    lum = nodes.new("ShaderNodeVectorMath")
    lum.operation = "DOT_PRODUCT"
    links.new(albedo, lum.inputs[0])
    lum.inputs[1].default_value = LUMINANCE_WEIGHTS

    total = albedo
    for term in terms:
        weight = nodes.new("ShaderNodeMath")
        weight.operation = "MULTIPLY"
        links.new(split.outputs["RGB".index(term["channel"])], weight.inputs[0])
        links.new(lum.outputs["Value"], weight.inputs[1])
        scaled = nodes.new("ShaderNodeMath")
        scaled.operation = "MULTIPLY"
        links.new(weight.outputs[0], scaled.inputs[0])
        scaled.inputs[1].default_value = term["scale"]
        shift = nodes.new("ShaderNodeVectorMath")
        shift.operation = "SCALE"
        shift.inputs[0].default_value = term["delta"]
        links.new(scaled.outputs[0], shift.inputs["Scale"])
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        links.new(total, add.inputs[0])
        links.new(shift.outputs["Vector"], add.inputs[1])
        total = add.outputs["Vector"]
    low = nodes.new("ShaderNodeVectorMath")
    low.operation = "MAXIMUM"
    links.new(total, low.inputs[0])
    low.inputs[1].default_value = (0.0, 0.0, 0.0)
    high = nodes.new("ShaderNodeVectorMath")
    high.operation = "MINIMUM"
    links.new(low.outputs["Vector"], high.inputs[0])
    high.inputs[1].default_value = (1.0, 1.0, 1.0)
    links.new(high.outputs["Vector"], principled.inputs["Base Color"])


def _stage(bpy, objects, size: int, samples: int):
    from mathutils import Vector

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"          # the GPU is shared with generation work
    scene.cycles.samples = samples
    scene.render.resolution_x, scene.render.resolution_y = size, int(size * 1.25)
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("preview") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.03, 0.035, 1)

    corners = [o.matrix_world @ Vector(c) for o in objects if o.type == "MESH" for c in o.bound_box]
    low = Vector([min(c[i] for c in corners) for i in range(3)])
    high = Vector([max(c[i] for c in corners) for i in range(3)])
    centre, extent = (low + high) / 2, max(high - low)
    camera = bpy.data.objects.new("preview_camera", bpy.data.cameras.new("preview_camera"))
    scene.collection.objects.link(camera)
    camera.data.lens = 85
    camera.location = centre + Vector((0, -extent * 3.4, extent * 0.15))
    camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    return scene


def _set_lights(bpy, scene, name: str) -> None:
    for obj in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    for i, (toward, strength) in enumerate(zip(LIGHTS[name], LIGHT_STRENGTH[name])):
        light = bpy.data.lights.new(f"{name}_{i}", type="SUN")
        light.energy = strength
        light.angle = math.radians(3)
        obj = bpy.data.objects.new(f"{name}_{i}", light)
        obj.rotation_euler = light_rotation(toward)
        scene.collection.objects.link(obj)


def main(argv: list[str]) -> int:
    import bpy

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--tint", nargs="*", default=[], help="zone=#rrggbb, e.g. R=#1f3a7a")
    p.add_argument("--plain", action="store_true", help="no sidecar: render the asset as it is")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--samples", type=int, default=64)
    args = p.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.asset))
    objects = list(bpy.context.scene.objects)
    scene = _stage(bpy, objects, args.size, args.samples)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = [("original", [])]
    if not args.plain and args.tint:
        sidecar = json.loads(args.asset.with_name(f"{args.asset.stem}.materials.json").read_text())
        terms = tint_terms(sidecar["zones"], parse_tints(args.tint))
        variants.append(("tinted", terms))
    else:
        sidecar = None

    for label, terms in variants:
        if terms:
            for material in {s.material for o in objects if o.type == "MESH" for s in o.material_slots}:
                mask = mask_for_material(sidecar, material.name)
                if mask is None:
                    print(f"no tint mask for material {material.name!r}: left untinted")
                    continue
                _build_tint_nodes(bpy, material, args.asset.with_name(mask), terms)
        for light in LIGHTS:
            _set_lights(bpy, scene, light)
            scene.render.filepath = str(args.out_dir / f"{label}_{light}.png")
            bpy.ops.render.render(write_still=True)
            print(f"wrote {scene.render.filepath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
