"""Structural checks for self-contained Blender exports, without rewriting their bytes.

This is not a full glTF conformance or Unreal compatibility validator. Unlike the
experimental guided editor, manual exchange permits morphs and multiple skins.
"""

from __future__ import annotations

import hashlib
import json
import struct

MAX_BYTES = 512 * 1024 * 1024


def inspect_glb(data: bytes) -> dict:
    # Keep the HTTP server cheap to import; NumPy is only needed for a submitted file.
    import numpy as np

    from image_to_3dlab.edit_glb import EditGLB

    try:
        if len(data) >= 20 and struct.unpack_from("<I", data, 12)[0] > 16 * 1024 * 1024:
            raise ValueError("GLB JSON metadata exceeds 16 MiB")
        glb = EditGLB(data)
        doc = glb.document

        def ref(items, index):
            if type(index) is not int or not 0 <= index < len(items):
                raise ValueError("GLB contains an invalid reference")
            return items[index]

        def objects(name):
            values = doc.get(name, [])
            if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
                raise ValueError(f"Invalid GLB {name}")
            return values

        # These require decoders which Studio does not configure. Never fetch a URI.
        forbidden = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_texture_basisu"}
        pending = [doc]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if forbidden.intersection(value.get("extensions", {})):
                    raise ValueError("Export GLB without Draco/meshopt compression or KTX textures")
                pending.extend(v for key, v in value.items() if key != "extras")
            elif isinstance(value, list):
                pending.extend(value)
        if forbidden.intersection(doc.get("extensionsRequired", []) + doc.get("extensionsUsed", [])):
            raise ValueError("Export GLB without Draco/meshopt compression or KTX textures")
        supported = {"KHR_lights_punctual", "KHR_texture_transform", "KHR_mesh_quantization", "EXT_mesh_gpu_instancing",
                     "EXT_texture_webp", "EXT_texture_avif", "EXT_materials_bump"}
        supported.update("KHR_materials_" + suffix for suffix in (
            "clearcoat", "dispersion", "ior", "sheen", "specular", "transmission", "iridescence", "anisotropy",
            "unlit", "volume", "emissive_strength"))
        unknown = set(doc.get("extensionsRequired", [])) - supported
        if unknown:
            raise ValueError("Studio cannot preview required GLB extensions: " + ", ".join(sorted(unknown)))
        for image in objects("images"):
            if "uri" in image:
                raise ValueError("Textures must be embedded in the GLB; export glTF Binary (.glb)")
            ref(glb.views, image.get("bufferView"))
            if image.get("mimeType") not in {"image/png", "image/jpeg", "image/webp", "image/avif"}:
                raise ValueError("Use PNG, JPEG, WebP or AVIF textures")
        total, logical_total = 0, 0
        for i, accessor in enumerate(glb.accessors):
            count = accessor.get("count")
            width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}.get(accessor.get("type"))
            if type(count) is not int or count < 1 or width is None:
                raise ValueError("Invalid accessor count or type")
            logical_total += count * width * 8
            if count * width * 8 > 128 * 1024 * 1024 or logical_total > MAX_BYTES:
                raise ValueError("Decoded GLB accessors exceed the memory limit")
            if "sparse" not in accessor:
                values = glb.accessor(i)
                total += values.nbytes
                if total > MAX_BYTES or not np.isfinite(values).all():
                    raise ValueError("GLB accessor data exceeds the limit or contains nonfinite values")
            else:
                # Blender may export sparse morph deltas. Check both stores using
                # the same bounded accessor reader, then check the sparse indices.
                sparse = accessor["sparse"]
                count = sparse["count"]
                if type(count) is not int or not 0 < count <= accessor["count"]:
                    raise ValueError("Invalid sparse accessor count")
                base = dict(accessor)
                base.pop("sparse")
                original = glb.accessors[i]
                try:
                    if "bufferView" in base:
                        glb.accessors[i] = base
                        total += glb.accessor(i).nbytes
                    for name in ("indices", "values"):
                        entry = sparse[name]
                        kind = entry["componentType"] if name == "indices" else accessor["componentType"]
                        if name == "indices" and kind not in (5121, 5123, 5125):
                            raise ValueError("Invalid sparse index type")
                        glb.accessors[i] = {"bufferView": entry["bufferView"], "byteOffset": entry.get("byteOffset", 0),
                                            "count": count, "componentType": kind,
                                            "type": "SCALAR" if name == "indices" else accessor["type"]}
                        values = glb.accessor(i)
                        total += values.nbytes
                        if total > MAX_BYTES or not np.isfinite(values).all():
                            raise ValueError("Sparse accessor exceeds limits")
                        if name == "indices" and (values.max() >= accessor["count"] or (np.diff(values.astype(np.int64)) <= 0).any()):
                            raise ValueError("Invalid sparse indices")
                finally:
                    glb.accessors[i] = original
        nodes, meshes, skins = objects("nodes"), objects("meshes"), objects("skins")
        glb.node_globals()  # includes cycle, parent and transform checks
        mesh_nodes = 0
        for node in nodes:
            if "mesh" in node:
                ref(meshes, node["mesh"])
                mesh_nodes += 1
            if "skin" in node:
                ref(skins, node["skin"])
        if not mesh_nodes:
            raise ValueError("The GLB has no mesh. Export the model, not an empty selection")
        for scene in objects("scenes"):
            for index in scene.get("nodes", []):
                ref(nodes, index)
        if "scene" in doc:
            ref(doc.get("scenes", []), doc["scene"])
        for mesh in meshes:
            if not mesh.get("primitives"):
                raise ValueError("Empty mesh")
            for primitive in mesh["primitives"]:
                attrs = primitive["attributes"]
                position = ref(glb.accessors, attrs["POSITION"])
                if position["type"] != "VEC3" or position["count"] < 1:
                    raise ValueError("Invalid mesh positions")
                for index in attrs.values():
                    if ref(glb.accessors, index)["count"] != position["count"]:
                        raise ValueError("Mismatched vertex attributes")
                for target in primitive.get("targets", []):
                    for index in target.values():
                        if ref(glb.accessors, index)["count"] != position["count"]:
                            raise ValueError("Mismatched morph target")
                if "indices" in primitive:
                    index = primitive["indices"]
                    meta = ref(glb.accessors, index)
                    if meta["type"] != "SCALAR" or meta["componentType"] not in (5121, 5123, 5125) or "sparse" in meta:
                        raise ValueError("Unsupported mesh indices")
                    if glb.accessor(index).max() >= position["count"]:
                        raise ValueError("Triangle index exceeds vertex count")
                if "material" in primitive:
                    ref(objects("materials"), primitive["material"])
        for skin in skins:
            if not skin.get("joints"):
                raise ValueError("Empty skeleton")
            for index in skin["joints"]:
                ref(nodes, index)
            if "inverseBindMatrices" in skin:
                accessor = ref(glb.accessors, skin["inverseBindMatrices"])
                if accessor["type"] != "MAT4" or accessor["count"] < len(skin["joints"]):
                    raise ValueError("Invalid inverse bind matrices")
        animations = objects("animations")
        for animation in animations:
            for sampler in animation["samplers"]:
                ref(glb.accessors, sampler["input"])
                ref(glb.accessors, sampler["output"])
            for channel in animation["channels"]:
                ref(animation["samplers"], channel["sampler"])
                ref(nodes, channel["target"]["node"])
        rig = {"skins": skins, "nodes": [{k: n[k] for k in ("name", "children", "matrix", "translation", "rotation", "scale") if k in n}
                                          for n in nodes]}
        return {"meshes": mesh_nodes, "skins": len(skins), "joints": sum(len(s["joints"]) for s in skins),
                "animations": [str(a.get("name") or f"Animation {i + 1}")[:160] for i, a in enumerate(animations)],
                "materials": len(objects("materials")), "images": len(objects("images")),
                "rig_signature": hashlib.sha256(json.dumps(rig, sort_keys=True).encode()).hexdigest(),
                "generator": str(doc.get("asset", {}).get("generator", "unknown"))[:200]}
    except (KeyError, IndexError, TypeError, AttributeError, RecursionError, struct.error) as exc:
        raise ValueError("Invalid or unsupported GLB structure; export glTF Binary (.glb) from Blender") from exc
