"""Game-ready materials for a generated GLB: the GLB plumbing every material step shares.

What is here is the part of the first design that stood up (see
``docs/plans/2026-10-15-game-materials-v2-design.md``): read the albedo the way the renderer
does, pack occlusion, roughness and metallic into one ORM image, edit the GLB in its JSON
so skins and animations survive, and carry the source's licence record to the result.

Nothing is invented: roughness and metallic come from the asset's own map (Pixal3D writes
one) or its factors, the normal map stays whatever the finish step baked from the dense
mesh, and occlusion is the asset's own or a bake. Tint zones by material, light taken out
at the source and learned checks arrive in their own steps, on ``multiview.py``.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from image_to_3dlab.edit_glb import EditGLB
from image_to_3dlab.glb_turn import build_glb
from image_to_3dlab.provenance import sha256_file

SIDECAR_SCHEMA = "assetfurnace.materials/2"
CHANNELS = ("R", "G", "B")
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
GUTTER_PX = 8
REPEAT, CLAMP, MIRROR = 10497, 33071, 33648
IMAGE_FORMAT_EXTENSIONS = ("EXT_texture_webp", "EXT_texture_avif", "KHR_texture_basisu")
COMPRESSED_GEOMETRY = ("EXT_meshopt_compression", "KHR_draco_mesh_compression")
OPERATION = "Game materials: ORM packed from the asset's own maps (no models)"


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Relative luminance Y of linear RGB (last axis)."""
    return np.asarray(linear_rgb, dtype=np.float64) @ np.array(LUMINANCE_WEIGHTS)


def apply_tint(albedo_linear: np.ndarray, mask: np.ndarray, means: np.ndarray,
               tints: np.ndarray) -> np.ndarray:
    """The tint formula, in linear light: ``out = albedo + m * g * (t - c)``.

    ``g = luminance(albedo) / luminance(c)`` is the zone's brightness detail, 1 on
    average, so a zone's mean lands exactly on its tint and ``t = c`` changes nothing.
    Unreal reproduces this same formula from the sidecar.
    """
    albedo = np.asarray(albedo_linear, dtype=np.float64)
    out = albedo.copy()
    y = luminance(albedo)
    for i in range(min(3, len(means))):
        yc = float(luminance(np.asarray(means[i])))
        g = y / yc if yc > 1e-6 else np.ones_like(y)
        delta = np.asarray(tints[i], dtype=np.float64) - np.asarray(means[i], dtype=np.float64)
        out += mask[..., i:i + 1] * g[..., None] * delta
    return np.clip(out, 0.0, 1.0)


def check_record(check: str, status: str, value, threshold, message: str, details=None) -> dict:
    record = {"check": check, "status": status, "value": value,
              "threshold": threshold, "message": message}
    if details is not None:
        record["details"] = details
    return json.loads(json.dumps(record, default=_plain))


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def worst(statuses) -> str:
    order = {"skipped": 0, "pass": 1, "warn": 2, "fail": 3}
    present = [s for s in statuses if s in order]
    picked = max(present, key=order.__getitem__) if present else "pass"
    return "pass" if picked == "skipped" else picked


def wrap_uv(values: np.ndarray, mode: int) -> np.ndarray:
    """One UV axis wrapped as the sampler would: repeat, clamp to edge or mirrored repeat."""
    if mode == CLAMP:
        return np.clip(values, 0.0, 1.0)
    if mode == MIRROR:
        period = np.mod(values, 2.0)
        return np.where(period > 1.0, 2.0 - period, period)
    return np.mod(values, 1.0)


def uv_coverage(uv: np.ndarray, faces: np.ndarray, size: tuple[int, int],
                wrap: tuple = (REPEAT, REPEAT), grow: bool = True) -> np.ndarray:
    """Texels covered by at least one UV triangle (glTF UVs: v = 0 is the top row).

    Under REPEAT a triangle outside 0..1 is moved into it, and drawn once more one tile
    back where it crosses the edge; under CLAMP_TO_EDGE the part outside is dropped;
    MIRRORED_REPEAT folds each corner, which is exact unless a triangle crosses a fold.
    """
    width, height = size
    tri = uv[faces].astype(np.float64)
    shifts = [(0, 0)]
    for axis in (0, 1):
        if wrap[axis] == REPEAT:
            tri[..., axis] -= np.floor(tri[..., axis].min(axis=1, keepdims=True))
        elif wrap[axis] == MIRROR:
            tri[..., axis] = wrap_uv(tri[..., axis], MIRROR)
    if REPEAT in wrap:
        shifts = [(-du, -dv) for du in (0, 1) if du == 0 or wrap[0] == REPEAT
                  for dv in (0, 1) if dv == 0 or wrap[1] == REPEAT]
    coverage = np.zeros((height, width), np.uint8)
    for du, dv in shifts:
        pixels = (tri + np.array([du, dv])) * np.array([width, height])
        try:
            import cv2
            cv2.fillPoly(coverage, np.round(pixels * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        except ImportError:  # pragma: no cover - opencv ships with rembg in this repo
            from PIL import ImageDraw
            canvas = Image.fromarray(coverage * 255)
            draw = ImageDraw.Draw(canvas)
            for t in pixels:
                draw.polygon([tuple(p) for p in t], fill=255)
            coverage = (np.asarray(canvas) > 0).astype(np.uint8)
    # Sub-texel triangles fill nothing; one texel of growth catches them.
    return ndimage.binary_dilation(coverage.astype(bool), iterations=1) if grow else coverage.astype(bool)


def fill_gutter(image: np.ndarray, coverage: np.ndarray, max_px: float | None = GUTTER_PX):
    """Copy each covered texel outward into the gutter, so mipmaps do not bleed black."""
    if coverage.all() or not coverage.any():
        return image
    distance, (rows, cols) = ndimage.distance_transform_edt(~coverage, return_indices=True)
    filled = image[rows, cols]
    if max_px is None:
        return filled
    reach = (distance <= max_px)
    if image.ndim == 3:
        reach = reach[..., None]
    return np.where(reach, filled, image)


def island_labels(coverage: np.ndarray, core: np.ndarray | None = None) -> np.ndarray:
    """Number the UV islands of a coverage that was grown by one texel.

    Growing joins islands whose gutter is one or two texels wide, so they are numbered on
    `core`, the coverage before growth (shrunk back when not given), and each grown texel
    then joins its nearest island.
    """
    core = ndimage.binary_erosion(coverage) if core is None else core & coverage
    labels, count = ndimage.label(core)
    if not count:
        return ndimage.label(coverage)[0]
    _, (rows, cols) = ndimage.distance_transform_edt(~core, return_indices=True)
    return np.where(coverage, labels[rows, cols], 0)


def pack_orm(occlusion: np.ndarray, roughness: np.ndarray, metallic: np.ndarray) -> np.ndarray:
    """glTF and Unreal ORM layout: R occlusion, G roughness, B metallic, linear 0..255."""
    stacked = np.dstack([occlusion, roughness, metallic]).astype(np.float64)
    return np.round(np.clip(stacked, 0.0, 1.0) * 255.0).astype(np.uint8)


def set_view_bytes(document: dict, binary: bytes, view_index: int, data: bytes) -> bytes:
    """Replace one bufferView's bytes and lay every view out again, 4-byte aligned."""
    out = bytearray()
    for index, view in enumerate(document["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunk = data if index == view_index else binary[start:start + view["byteLength"]]
        out += b"\0" * (-len(out) % 4)
        view["byteOffset"] = len(out)
        view["byteLength"] = len(chunk)
        out += chunk
    document["buffers"][0]["byteLength"] = len(out)
    return bytes(out)


def add_view(document: dict, binary: bytes, data: bytes) -> tuple[int, bytes]:
    binary = binary[:document["buffers"][0]["byteLength"]]
    binary += b"\0" * (-len(binary) % 4)
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
    binary += data
    document["buffers"][0]["byteLength"] = len(binary)
    return len(document["bufferViews"]) - 1, binary


def add_texture(document: dict, binary: bytes, data: bytes, name: str, sampler) -> tuple[int, bytes]:
    view, binary = add_view(document, binary, data)
    document.setdefault("images", []).append({"bufferView": view, "mimeType": "image/png", "name": name})
    texture = {"source": len(document["images"]) - 1}
    if sampler is not None:
        texture["sampler"] = sampler
    document.setdefault("textures", []).append(texture)
    return len(document["textures"]) - 1, binary


def _png(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG", optimize=False, compress_level=6)
    return out.getvalue()


def _texture_users(document: dict) -> dict[int, list[tuple[int, str]]]:
    """Which (material, slot) pairs reference each texture index."""
    users: dict[int, list[tuple[int, str]]] = {}

    def walk(node, material, key):
        if isinstance(node, dict):
            if "index" in node and isinstance(node["index"], int) and (
                    key.endswith("Texture") or key == "tintMask"):
                users.setdefault(node["index"], []).append((material, key))
            for k, v in node.items():
                walk(v, material, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, material, key)

    for index, material in enumerate(document.get("materials", [])):
        walk(material, index, "")
    return users


def view_shared(document: dict, image_index: int) -> bool:
    """Whether another image or an accessor reads the same bufferView as this image."""
    view = document["images"][image_index].get("bufferView")
    return (any(i != image_index and image.get("bufferView") == view
                for i, image in enumerate(document.get("images", [])))
            or any(accessor.get("bufferView") == view for accessor in document.get("accessors", [])))


def referenced_outside_materials(document: dict, texture_index: int) -> bool:
    """Whether anything but a material (a metadata extension, say) points at this texture.

    Such a texture is never rewritten in place: its other user would read ORM data.
    """
    def walk(node):
        if isinstance(node, dict):
            if node.get("index") == texture_index and isinstance(node.get("index"), int):
                return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    skip = {"materials", "textures", "images", "samplers", "accessors", "bufferViews", "buffers",
            "nodes", "meshes", "skins", "animations", "scenes", "cameras", "asset"}
    return any(walk(value) for key, value in document.items() if key not in skip)


def _decode_image(glb: EditGLB, image_index: int, mode: str = "RGB") -> np.ndarray:
    image = glb.document["images"][image_index]
    if "bufferView" not in image:
        raise ValueError("only images embedded in the GLB are supported")
    view = glb.document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return np.asarray(Image.open(io.BytesIO(glb.binary[start:start + view["byteLength"]])).convert(mode))


def texture_source(texture: dict) -> int:
    """The image a texture reads: core ``source``, or one from an image-format extension."""
    if "source" in texture:
        return texture["source"]
    for extension in (texture.get("extensions") or {}).values():
        if isinstance(extension, dict) and isinstance(extension.get("source"), int):
            return extension["source"]
    raise ValueError("texture has no image source")


def _texture_image(glb: EditGLB, info: dict | None, mode: str = "RGB") -> np.ndarray | None:
    if not info:
        return None
    texture = glb.document["textures"][info["index"]]
    return _decode_image(glb, texture_source(texture), mode)


def transform_uv(uv: np.ndarray, transform: dict | None) -> np.ndarray:
    """Apply KHR_texture_transform as three.js does (Matrix3.setUvTransform): rotate, scale, offset."""
    if not transform:
        return uv
    sx, sy = transform.get("scale", [1.0, 1.0])
    ox, oy = transform.get("offset", [0.0, 0.0])
    c, s = np.cos(transform.get("rotation", 0.0)), np.sin(transform.get("rotation", 0.0))
    u, v = uv[:, 0], uv[:, 1]
    return np.c_[sx * (c * u + s * v) + ox, sy * (-s * u + c * v) + oy]


def primitive_triangles(indices: np.ndarray, mode: int) -> np.ndarray | None:
    """Triangles of a primitive: plain lists, strips and fans; None for points and lines."""
    indices = np.asarray(indices, dtype=np.int64)
    if mode == 4:
        return indices[: len(indices) // 3 * 3].reshape(-1, 3)
    if mode == 5:
        count = max(len(indices) - 2, 0)
        tri = np.stack([indices[:count], indices[1:count + 1], indices[2:count + 2]], axis=1)
        tri[1::2, [0, 1]] = tri[1::2, [1, 0]]   # every other strip triangle flips to keep winding
        return tri
    if mode == 6:
        count = max(len(indices) - 2, 0)
        return np.stack([np.full(count, indices[0]), indices[1:count + 1], indices[2:count + 2]], axis=1)
    return None


def scene_nodes(document: dict) -> list[int]:
    """The nodes the displayed scene draws, parents before children; every node if no scene."""
    nodes = document.get("nodes", [])
    scenes = document.get("scenes") or []
    if not scenes:
        return list(range(len(nodes)))
    stack = list(reversed(scenes[document.get("scene", 0)].get("nodes", [])))
    order, seen = [], set()
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        order.append(index)
        stack.extend(reversed(nodes[index].get("children", [])))
    return order


def _sampler_wrap(document: dict, info: dict) -> tuple:
    texture = document["textures"][info["index"]]
    sampler = document.get("samplers", [])[texture["sampler"]] if "sampler" in texture else {}
    return sampler.get("wrapS", REPEAT), sampler.get("wrapT", REPEAT)


def _drawn_primitives(document: dict):
    """(node index, primitive) for every triangle primitive the displayed scene draws."""
    for node_index in scene_nodes(document):
        node = document["nodes"][node_index]
        if "mesh" not in node:
            continue
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            if primitive.get("mode", 4) in (4, 5, 6):   # points and lines have no surface to paint
                yield node_index, primitive


def untextured_materials(document: dict) -> list[str]:
    """Names of drawn materials with no base colour texture to mask, pack or derive from."""
    names = []
    for _, primitive in _drawn_primitives(document):
        if "material" not in primitive:
            label = "(default material)"
        else:
            material = document["materials"][primitive["material"]]
            if material.get("pbrMetallicRoughness", {}).get("baseColorTexture"):
                continue
            label = material.get("name") or f"material {primitive['material']}"
        if label not in names:
            names.append(label)
    return names


def _lookup(image: np.ndarray, uv: np.ndarray, wrap: tuple = (REPEAT, REPEAT)) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip((wrap_uv(uv[:, 0], wrap[0]) * width).astype(int), 0, width - 1)
    y = np.clip((wrap_uv(uv[:, 1], wrap[1]) * height).astype(int), 0, height - 1)
    return image[y, x]


def _uv_layout(document: dict, info: dict) -> tuple:
    return (info.get("texCoord", 0), (info.get("extensions") or {}).get("KHR_texture_transform"),
            _sampler_wrap(document, info))


def _same_uv_layout(document: dict, base: dict, other: dict | None) -> bool:
    return not other or _uv_layout(document, other) == _uv_layout(document, base)


def _load_ao(path: Path | None, surfaces) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    ao = Image.open(path).convert("L")
    return {index: np.asarray(ao.resize(s.image.shape[1::-1], Image.BILINEAR)) / 255.0
            for index, s in surfaces.items()}


def drop_unused_extensions(document: dict) -> None:
    """Forget an image-format extension once no texture uses it, so importers that lack it
    are not told it is required."""
    textures = document.get("textures", [])
    for name in IMAGE_FORMAT_EXTENSIONS:
        if any(name in (t.get("extensions") or {}) for t in textures):
            continue
        for key in ("extensionsUsed", "extensionsRequired"):
            if name in document.get(key, []):
                document[key] = [e for e in document[key] if e != name]
                if not document[key]:
                    del document[key]


def source_provenance(source: Path) -> dict:
    """The source's licence record, or one that says its licence is unknown, never a guess."""
    sidecar = source.with_suffix(".provenance.json")
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return {"schema_version": 1, "license": None, "classification": "unknown", "models": [],
            "note": "the source carried no provenance record"}


def write_provenance(source: Path, source_sha256: str, record: dict, output: Path) -> Path:
    """Carry the source's licence record to the result, adding no models.

    The step only does image maths, so the result is licensed exactly as its source.
    `source_sha256` and `record` are taken when the source is read, so a file replaced
    while the step runs cannot lend the result another hash or licence.
    """
    record = copy.deepcopy(record)
    record["operation"] = OPERATION
    record["input"] = {"path": str(source), "sha256": source_sha256}
    # finalize_output keeps the licence classification under "output": update, never replace
    record["output"] = {**(record.get("output") or {}), "path": str(output), "sha256": sha256_file(output)}
    path = output.with_suffix(".provenance.json")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def safe_name(name: str) -> str:
    """A material name made fit for a file name on every platform."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", name or "").strip("-")[:60]
    return cleaned or "material"


# --------------------------------------------------------------------------- surfaces

@dataclass
class Surface:
    """One textured material as drawn: its base colour image and its triangles in UV."""
    material: int
    image: np.ndarray            # base colour, sRGB uint8 HxWx3
    factor: np.ndarray           # baseColorFactor rgb
    texcoord: int
    transform: dict | None = None
    wrap: tuple = (REPEAT, REPEAT)
    uv: list = field(default_factory=list)       # per primitive, transformed into image space
    faces: list = field(default_factory=list)

    def coverage(self, grow: bool = True) -> np.ndarray:
        """Texels this material's triangles cover in its own atlas."""
        offsets = np.cumsum([0] + [len(u) for u in self.uv[:-1]])
        faces = np.concatenate([f + o for f, o in zip(self.faces, offsets)])
        height, width = self.image.shape[:2]
        return uv_coverage(np.concatenate(self.uv), faces, (width, height), self.wrap, grow=grow)


def surfaces(glb: EditGLB) -> dict[int, Surface]:
    """Every textured material the displayed scene draws, keyed by glTF material index."""
    document = glb.document
    packed = [e for e in COMPRESSED_GEOMETRY if e in document.get("extensionsRequired", [])]
    if packed:
        raise ValueError(f"the geometry is compressed ({', '.join(packed)}): decompress the GLB first, "
                         "or its bytes would be read as plain positions and UVs")
    found: dict[int, Surface] = {}
    for _, primitive in _drawn_primitives(document):
        if "material" not in primitive:
            continue
        index = primitive["material"]
        material = document["materials"][index]
        if "KHR_materials_unlit" in (material.get("extensions") or {}):
            raise ValueError(f"material {material.get('name') or index!r} is unlit (KHR_materials_unlit): "
                             "an ORM would be ignored; make it a lit PBR material first")
        pbr = material.get("pbrMetallicRoughness", {})
        base = pbr.get("baseColorTexture")
        if base is None:
            continue
        transform = (base.get("extensions") or {}).get("KHR_texture_transform")
        texcoord = (transform or {}).get("texCoord", base.get("texCoord", 0))
        if f"TEXCOORD_{texcoord}" not in primitive["attributes"]:
            continue
        if index not in found:
            found[index] = Surface(index, _texture_image(glb, base),
                                   np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float64),
                                   texcoord, transform, _sampler_wrap(document, base))
        positions = glb.accessor(primitive["attributes"]["POSITION"])
        order = glb.accessor(primitive["indices"]) if "indices" in primitive else np.arange(len(positions))
        found[index].faces.append(primitive_triangles(order, primitive.get("mode", 4)))
        found[index].uv.append(transform_uv(glb.accessor(primitive["attributes"][f"TEXCOORD_{texcoord}"])
                                            .astype(np.float64), transform))
    if not found:
        raise ValueError("the GLB has no textured triangle primitives")
    return found


# --------------------------------------------------------------------------- ORM

def orm_image(glb: EditGLB, material: dict, size: tuple[int, int], ao: np.ndarray | None):
    """The material's own occlusion, roughness and metallic packed as one ORM image.

    Nothing is derived: roughness and metallic are the asset's map times its factors, or
    the factors alone. Occlusion is a bake when one is given, else the asset's own.
    """
    height, width = size
    pbr = material.get("pbrMetallicRoughness", {})
    for slot, info in (("metallic-roughness", pbr.get("metallicRoughnessTexture")),
                       ("occlusion", material.get("occlusionTexture"))):
        if not _same_uv_layout(glb.document, pbr["baseColorTexture"], info):
            raise ValueError(f"the {slot} texture uses another UV set, transform or wrap than the base "
                             "colour; repack it onto the base colour's UVs before making game materials")
    base_layout = _uv_layout(glb.document, pbr["baseColorTexture"])
    if ao is not None and (base_layout[0] != 0 or base_layout[1]):
        raise ValueError("the AO bake is on the raw first UV set, but the base colour uses another UV set "
                         "or a texture transform; bake AO onto the base colour's UVs instead")

    def fit(image):
        if image is not None and image.shape[:2] != (height, width):
            image = np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))
        return image

    mr = fit(_texture_image(glb, pbr.get("metallicRoughnessTexture")))
    rough_factor = float(pbr.get("roughnessFactor", 1.0))
    # glTF's default metallicFactor is 1, but an asset that never mentions metal is not metal.
    metal_factor = float(pbr.get("metallicFactor", 1.0 if mr is not None else 0.0))
    maps = {}
    if mr is not None:
        rough, metal = mr[..., 1] / 255.0 * rough_factor, mr[..., 2] / 255.0 * metal_factor
        maps["roughness"] = maps["metallic"] = "kept"
    else:
        rough, metal = np.full((height, width), rough_factor), np.full((height, width), metal_factor)
        maps["roughness"] = maps["metallic"] = "factor"
    existing = fit(_texture_image(glb, material.get("occlusionTexture")))
    if ao is not None:
        occlusion, maps["occlusion"] = ao, "baked"
    elif existing is not None:
        occlusion, maps["occlusion"] = existing[..., 0] / 255.0, "kept"
    else:
        occlusion, maps["occlusion"] = np.ones((height, width)), "none"
    return pack_orm(occlusion, rough, metal), maps


# --------------------------------------------------------------------------- entry point

@dataclass
class MaterialsResult:
    status: str
    glb_path: Path
    sidecar_path: Path
    textures: dict[str, Path]
    maps: dict[str, str]


def _material_names(document: dict, surfaces_: dict[int, Surface], prefix: str) -> dict[int, str]:
    """Texture file stems per material: the prefix alone for one, else made safe and unique."""
    if len(surfaces_) == 1:
        return {next(iter(surfaces_)): prefix}
    names, used = {}, set()
    for index in surfaces_:
        base = f"{prefix}_{safe_name(document['materials'][index].get('name') or str(index))}"
        name, extra = base, 0
        while name.casefold() in used:   # Skin and skin are one file on Windows and macOS
            extra += 1
            name = f"{base}-{index}" + (f"-{extra}" if extra > 1 else "")
        used.add(name.casefold())
        names[index] = name
    return names


def make_game_materials(asset: Path, out_dir: Path, *, ao: Path | None = None,
                        texture_prefix: str | None = None) -> MaterialsResult:
    """Write the asset back with its own maps packed as ORM, plus its sidecar and licence.

    Writes ``<out_dir>/<stem>.glb``, ``<prefix>_ORM.png`` per material beside it,
    ``<stem>.materials.json`` and ``<stem>.provenance.json``. Raises with a reason for what
    cannot be packed without damage (compressed geometry, unlit materials, maps on another
    UV layout, one AO bake for several atlases); writes nothing then.
    """
    asset, out_dir = Path(asset), Path(out_dir)
    if texture_prefix is not None and safe_name(texture_prefix) != texture_prefix:
        raise ValueError(f"texture prefix {texture_prefix!r} must be a plain file name "
                         "(letters, digits, '_' and '-')")
    stem = asset.stem
    target = out_dir / f"{stem}.glb"
    if target.exists() and target.samefile(asset):
        raise ValueError("out_dir is the asset's own folder: the result would overwrite the source")
    data = asset.read_bytes()
    source_sha256, licence = hashlib.sha256(data).hexdigest(), source_provenance(asset)
    glb = EditGLB(data)
    found = surfaces(glb)
    if ao is not None and len(found) > 1:
        raise ValueError(f"one AO image cannot serve {len(found)} material atlases that each fill "
                         "the UV square; run without --ao, or bake one AO per material")
    ao_maps = _load_ao(ao, found)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = texture_prefix or f"T_{stem}"
    names = _material_names(glb.document, found, prefix)

    document = copy.deepcopy(glb.document)
    binary = glb.binary[:glb.buffer_size]
    users = _texture_users(document)
    per_material = []
    for surface in found.values():
        material = document["materials"][surface.material]
        pbr = material["pbrMetallicRoughness"]
        base_info = pbr["baseColorTexture"]
        sampler = document["textures"][base_info["index"]].get("sampler")
        coverage = surface.coverage()
        orm, maps = orm_image(glb, material, surface.image.shape[:2], ao_maps.get(surface.material))
        orm = fill_gutter(orm, coverage)
        written = {"orm": out_dir / f"{names[surface.material]}_ORM.png"}
        Image.fromarray(orm).save(written["orm"])

        old = pbr.get("metallicRoughnessTexture")
        old_texture = document["textures"][old["index"]] if old is not None else None
        exclusive = (old is not None and not referenced_outside_materials(document, old["index"])
                     and all(m == surface.material and slot in ("metallicRoughnessTexture", "occlusionTexture")
                             for m, slot in users.get(old["index"], []))
                     and sum(texture_source(t) == texture_source(old_texture) for t in document["textures"]) == 1
                     and not view_shared(document, texture_source(old_texture)))
        if exclusive:
            # Reuse the old metallic-roughness slot rather than leave a dead 4K image behind.
            source = texture_source(old_texture)
            image = document["images"][source]
            binary = set_view_bytes(document, binary, image["bufferView"], _png(orm))
            image["mimeType"] = "image/png"
            old_texture["source"] = source
            old_texture.pop("extensions", None)     # a PNG is core glTF, not WebP
            orm_tex = old["index"]
        else:
            orm_tex, binary = add_texture(document, binary, _png(orm), f"{names[surface.material]}_ORM", sampler)
        info = {"texCoord": base_info["texCoord"]} if base_info.get("texCoord") else {}
        if (base_info.get("extensions") or {}).get("KHR_texture_transform"):
            info["extensions"] = {"KHR_texture_transform": copy.deepcopy(
                base_info["extensions"]["KHR_texture_transform"])}
        pbr["metallicRoughnessTexture"] = {"index": orm_tex, **info}
        pbr["metallicFactor"] = 1.0
        pbr["roughnessFactor"] = 1.0
        if maps["occlusion"] in ("baked", "kept"):
            strength = (material.get("occlusionTexture") or {}).get("strength")
            material["occlusionTexture"] = {"index": orm_tex, **info}
            if strength is not None:
                material["occlusionTexture"]["strength"] = strength
        else:
            material.pop("occlusionTexture", None)
        # The normal map is the finish step's bake from the dense mesh, or none: never invented.
        maps["normal"] = "kept" if "normalTexture" in material else "none"
        per_material.append({"material": surface.material, "name": material.get("name"),
                             "textures": {k: p.name for k, p in written.items()}, "maps": maps})

    drop_unused_extensions(document)
    glb_path = out_dir / f"{stem}.glb"
    sidecar_path = out_dir / f"{stem}.materials.json"
    try:
        # The GLB and its licence record first; the sidecar, which Studio and batches read
        # as "done", only once both exist.
        glb_path.write_bytes(build_glb(document, binary))
        write_provenance(asset, source_sha256, licence, glb_path)
    except BaseException:
        for leftover in (glb_path, glb_path.with_suffix(".provenance.json"), sidecar_path):
            leftover.unlink(missing_ok=True)
        raise
    first = per_material[0]
    sidecar = {"schema": SIDECAR_SCHEMA, "asset": asset.name, "source_sha256": source_sha256,
               "status": "pass", "glb": glb_path.name, "zones": [], "checks": [],
               "textures": {**first["textures"],
                            "orm_channels": {"R": "occlusion", "G": "roughness", "B": "metallic"},
                            "srgb": {"orm": False}},
               "maps": first["maps"], "materials": per_material,
               "untextured": untextured_materials(glb.document)}
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return MaterialsResult("pass", glb_path, sidecar_path,
                           {"orm": out_dir / first["textures"]["orm"]}, first["maps"])
