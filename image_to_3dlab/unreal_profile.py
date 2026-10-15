"""Unreal export profile: the pure parts, testable without Blender or Unreal.

A finished AssetFurnace GLB becomes a GLB that Unreal Engine 5.8 imports through
Interchange with no touch-ups: Unreal names, real size in centimetres, origin on the
ground, LODs, simple collision, one packed ORM texture per material and, for humanoids,
bones named like the UE5 Mannequin (Manny) so Unreal's IK Rig recognises the skeleton.

Measured on UE 5.8.3 (docs/plans/2026-10-15-unreal-profile-design.md):
- Interchange converts glTF metres to centimetres and Y-up to Z-up by itself, and
  Blender's -Y (glTF's front) becomes Unreal's +Y, the way Manny faces.
- Nodes named `LOD0_SM_X`, `LOD1_SM_X` become one Static Mesh with LODs; a node named
  `UBX_SM_X_00` becomes its box collision.
- It refuses `EXT_meshopt_compression` and `EXT_texture_webp`, so nothing here compresses.
- glTF's metallicRoughness texture already uses Unreal's ORM channel order (G roughness,
  B metal); occlusion in R makes it a true ORM, used as both textures.

Blender does the geometry (`scripts/blender_export_unreal.py`), this module the names,
the numbers, the texture rewrite and the checks, and `scripts/export_unreal.py` ties
them together.
"""

from __future__ import annotations

import io
import math
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from image_to_3dlab.glb_turn import build_glb, parse_glb

KINDS = ("prop", "creature", "equipment")
# What the batch rows and the checks call the same kinds.
KIND_ALIASES = {"object": "prop", "equipment_part": "equipment"}
MEASURES = ("height", "longest")
COLLISIONS = ("box", "convex", "none")
ROOT_BONE = "root"
MAX_NAME = 40  # keeps output paths well under Windows' 260 characters
SIDECAR_SCHEMA = 1
EXTRAS_KEY = "image_to_3dlab_unreal"
UNREAL_ENV = "I2L_UNREAL"
# Extensions UE 5.8 Interchange was seen to refuse, or that this profile never needs.
REFUSED_EXTENSIONS = {"EXT_meshopt_compression", "KHR_meshopt_compression", "EXT_texture_webp",
                      "KHR_draco_mesh_compression", "KHR_texture_basisu", "EXT_texture_avif"}
_PREFIXES = ("SK_", "SM_", "T_", "MI_", "M_", "AS_", "A_")
SIZE_TOLERANCE = 0.01   # measured size may differ from the request by 1%
BASE_TOLERANCE_CM = 0.5

# SOMA joint (what kimodo_retarget.map_skintokens_to_soma returns) -> UE5 Mannequin bone.
SOMA_TO_UE: dict[str, str] = {
    "Hips": "pelvis", "Spine1": "spine_01", "Spine2": "spine_02", "Chest": "spine_03",
    "Neck1": "neck_01", "Neck2": "neck_02", "Head": "head",
}
for _soma, _ue in (("Left", "l"), ("Right", "r")):
    SOMA_TO_UE.update({
        f"{_soma}Shoulder": f"clavicle_{_ue}", f"{_soma}Arm": f"upperarm_{_ue}",
        f"{_soma}ForeArm": f"lowerarm_{_ue}", f"{_soma}Hand": f"hand_{_ue}",
        f"{_soma}Leg": f"thigh_{_ue}", f"{_soma}Shin": f"calf_{_ue}",
        f"{_soma}Foot": f"foot_{_ue}", f"{_soma}ToeBase": f"ball_{_ue}",
    })
    for _i in (1, 2, 3):
        SOMA_TO_UE[f"{_soma}HandThumb{_i}"] = f"thumb_0{_i}_{_ue}"
    # SOMA fingers count from the metacarpal (1); SkinTokens' start at the knuckle (2),
    # which is Manny's finger_01.
    for _finger in ("Index", "Middle", "Ring", "Pinky"):
        for _i in (2, 3, 4):
            SOMA_TO_UE[f"{_soma}Hand{_finger}{_i}"] = f"{_finger.lower()}_0{_i - 1}_{_ue}"


# --- names -----------------------------------------------------------------------------

def base_name(raw: str) -> str:
    """`SK_BrennholdWarrior_9k` -> `BrennholdWarrior9k`: one CamelCase word Unreal accepts.

    Known prefixes, a `_LODn` suffix and the extension are dropped, so a name already
    in Unreal's style does not get prefixed twice.
    """
    stem = Path(raw).name
    stem = re.sub(r"\.(glb|gltf)$", "", stem, flags=re.IGNORECASE)
    for prefix in _PREFIXES:
        if stem.upper().startswith(prefix):
            stem = stem[len(prefix):]
            break
    stem = re.sub(r"_LOD\d+$", "", stem, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stem) if w]
    if not words:
        raise ValueError(f"no usable characters in the name {raw!r}")
    name = "".join(w[0].upper() + w[1:] for w in words)
    if not name[0].isalpha():
        name = "Asset" + name
    return name[:MAX_NAME]


def asset_names(kind: str, base: str) -> dict[str, Any]:
    """Every Unreal name one exported asset uses."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    mesh = ("SM_" if kind == "prop" else "SK_") + base
    return {
        "base": base,
        "kind": kind,
        "mesh": mesh,
        "material": f"MI_{base}",
        "base_color": f"T_{base}_BaseColor",
        "normal": f"T_{base}_N",
        "orm": f"T_{base}_ORM",
        "tint": f"T_{base}_Tint",
        "skeleton": f"{mesh}_Skeleton",  # what Interchange calls the Skeleton it creates
        "animation_prefix": f"AS_{base}_",
        "collision": {"box": f"UBX_{mesh}_00", "convex": f"UCX_{mesh}_00"},
    }


def lod_node(mesh: str, index: int) -> str:
    """The node name Interchange folds into LOD `index` of `mesh`."""
    return f"LOD{index}_{mesh}"


def lod_material(base: str, index: int) -> str:
    """LODs baked by finish_props carry their own atlas, hence their own material."""
    return f"MI_{base}" if index == 0 else f"MI_{base}_LOD{index}"


_ROLE_SUFFIX = {"base_color": "BaseColor", "normal": "N", "orm": "ORM", "tint": "Tint", "emissive": "E"}


def texture_name(material: str, role: str) -> str:
    """`MI_Barrel_LOD1` + `orm` -> `T_Barrel_LOD1_ORM`."""
    stem = material.removeprefix("MI_")
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stem) if w]
    stem = "_".join(words) if material.startswith("MI_") else "".join(w[0].upper() + w[1:] for w in words)
    return f"T_{stem or 'Texture'}_{_ROLE_SUFFIX[role]}"


def sibling_lods(path: Path) -> list[Path]:
    """`chest_LOD0.glb` -> the `chest_LOD1.glb`, `chest_LOD2.glb`... beside it, in order.

    These are the LODs `scripts/finish_props.py` baked from the original, which beat any
    simplification; the compressed `.web.glb` copies are for browsers, not Unreal.
    """
    match = re.fullmatch(r"(.*)_LOD0\.glb", path.name, flags=re.IGNORECASE)
    if not match:
        return []
    found = []
    for index in range(1, 8):
        candidate = path.with_name(f"{match[1]}_LOD{index}.glb")
        if not candidate.is_file():
            break
        found.append(candidate)
    return found


def model_prefix(stems: Iterable[str]) -> str:
    """What a model's files share: `X_rigged`, `X_idle`, `X_hit_react` -> `X`.

    The auto-rig writes `<model>_rigged.glb` and one `<model>_<move>.glb` per move.
    """
    stems = list(stems)
    for stem in stems:
        if stem.lower().endswith("_rigged"):
            return stem[:-len("_rigged")]
    if len(stems) == 1:
        return stems[0]
    common = os.path.commonprefix(stems)
    return common[:common.rfind("_")] if "_" in common else ""


def clip_name(stem: str, prefix: str, base: str) -> str:
    """`SK_Warrior_9k_hit_react` under prefix `SK_Warrior_9k` -> `AS_<base>_HitReact`."""
    rest = stem[len(prefix):] if prefix and stem.startswith(prefix) else stem
    words = [w for w in re.split(r"[^A-Za-z0-9]+", rest) if w]
    return f"AS_{base}_" + ("".join(w[0].upper() + w[1:] for w in words) or "Anim")


def find_provenance(path: Path, levels: int = 3) -> Path | None:
    """The `.provenance.json` that describes `path`: its own, else the nearest run's.

    A model's own record sits beside it (`X.glb` -> `X.provenance.json`); a prop cut
    from a sheet shares its run's `props.provenance.json` a couple of folders up.
    """
    path = Path(path)
    own = path.with_name(path.name[:-len(path.suffix)] + ".provenance.json")
    if own.is_file():
        return own
    folder = path.parent
    for _ in range(levels):
        found = sorted(folder.glob("*.provenance.json"))
        if found:
            return found[0]
        if folder.parent == folder:
            break
        folder = folder.parent
    return None


# --- size, ground, facing --------------------------------------------------------------

def scale_factor(dims: Iterable[float], size_cm: float, measure: str) -> float:
    """How much to scale a model whose box is `dims` metres (Blender, Z up) so that its
    height, or longest side, is `size_cm`."""
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {', '.join(MEASURES)}, not {measure!r}")
    if not size_cm or size_cm <= 0:
        raise ValueError("the real size must be a positive number of centimetres")
    x, y, z = (float(d) for d in dims)
    measured = z if measure == "height" else max(x, y, z)
    if measured <= 1e-9:
        raise ValueError(f"the model has no {measure} to measure")
    return size_cm / 100.0 / measured


def ground_offset(box_min: Iterable[float], box_max: Iterable[float],
                  centre: Iterable[float] | None = None) -> tuple[float, float, float]:
    """The move that puts the model on the ground (z = 0) and centres it on the origin.

    `centre` (x, y) overrides the box centre: between the feet for a character, so a
    cape or an outstretched arm does not pull the origin off the body.
    """
    lo, hi = list(box_min), list(box_max)
    cx, cy = centre if centre is not None else ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
    return (-float(cx), -float(cy), -float(lo[2]))


def yaw_to_face_front(forward: Iterable[float]) -> float:
    """Radians about Z that turn `forward` to Blender's -Y, which Unreal imports as +Y."""
    x, y = list(forward)[:2]
    if math.hypot(x, y) < 1e-6:
        raise ValueError("the forward direction is vertical")
    return math.atan2(-1.0, 0.0) - math.atan2(y, x)


# --- bones -----------------------------------------------------------------------------

def bone_rename_plan(order: list[str], parents: Mapping[str, str | None],
                     soma: Mapping[str, str]) -> dict[str, str]:
    """{old bone: Mannequin name} from a SkinTokens -> SOMA mapping.

    The first bone of each SOMA joint gets the Mannequin name; a second chain mapped to
    the same joint (a mitten hand's other finger) or a bone that merely follows its
    parent (hair under the head) becomes `<name>_extraN`, so every name stays unique and
    the recognised chain stays clean. `order` must list parents before children.
    """
    plan: dict[str, str] = {}
    used: set[str] = {ROOT_BONE}
    extras: dict[str, int] = {}
    counted = {"spine": 0, "neck": 0}
    for bone in order:
        joint = soma.get(bone)
        target = SOMA_TO_UE.get(joint or "", clean_name(joint or bone))
        parent = parents.get(bone)
        follows_parent = parent is not None and soma.get(parent) == joint
        # Manny numbers its spine 01..05 and neck 01..02 from the hips up; SkinTokens
        # has as many as it likes, so they are numbered in order rather than by SOMA.
        family = target.split("_")[0]
        if family in counted and counted[family] < (5 if family == "spine" else 2):
            counted[family] += 1
            target, follows_parent = f"{family}_{counted[family]:02d}", False
        if target in used or follows_parent:
            count = extras.get(target, 0)
            extras[target] = count + 1
            name = f"{target}_extra{count}"
        else:
            name = target
        used.add(name)
        plan[bone] = name
    return plan


def clean_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "bone"


def clean_bone_names(order: list[str]) -> dict[str, str]:
    """For skeletons that are not humanoid: same bones, names Unreal and FBX/glTF agree on.

    `root` is reserved for the bone this profile adds at the origin.
    """
    plan, used = {}, {ROOT_BONE}
    for bone in order:
        name = clean_name(bone)
        candidate, n = name, 1
        while candidate in used:
            candidate = f"{name}_{n}"
            n += 1
        used.add(candidate)
        plan[bone] = candidate
    return plan


_BONE_PATH = re.compile(r'pose\.bones\["((?:[^"\\]|\\.)*)"\]')


def renamed_path(data_path: str, plan: Mapping[str, str]) -> str:
    """An animation curve's data path, pointing at the renamed bone."""
    return _BONE_PATH.sub(lambda m: f'pose.bones["{plan.get(m[1], m[1])}"]', data_path)


def is_translation_path(data_path: str) -> bool:
    """Curves whose values are distances, which must grow with the model when it is scaled.

    Rotations and scales are ratios and stay as they are.
    """
    return data_path == "location" or data_path.endswith(".location")


# --- ORM -------------------------------------------------------------------------------

def pack_orm(metal_rough: np.ndarray, occlusion: np.ndarray | None) -> np.ndarray:
    """R occlusion (white when there is none), G roughness, B metal: Unreal's ORM.

    `metal_rough` is glTF's metallicRoughness image (G roughness, B metal). `occlusion`
    is an occlusion image (R, or grey) of the same size.
    """
    mr = np.asarray(metal_rough)
    if mr.ndim != 3 or mr.shape[2] < 3:
        raise ValueError("the metallicRoughness image must have at least three channels")
    out = np.empty(mr.shape[:2] + (3,), np.uint8)
    out[..., 1] = mr[..., 1]
    out[..., 2] = mr[..., 2]
    if occlusion is None:
        out[..., 0] = 255
    else:
        ao = np.asarray(occlusion)
        if ao.shape[:2] != mr.shape[:2]:
            raise ValueError(f"occlusion is {ao.shape[1]}x{ao.shape[0]}, metallicRoughness "
                             f"{mr.shape[1]}x{mr.shape[0]}")
        out[..., 0] = ao if ao.ndim == 2 else ao[..., 0]
    return out


# --- GLB rewrite -----------------------------------------------------------------------

def _image_bytes(document: dict, binary: bytes, index: int) -> bytes:
    image = document["images"][index]
    if "bufferView" not in image:
        raise ValueError(f"image {image.get('name', index)!r} is not embedded")
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def _decode(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB"))


def _encode_png(pixels: np.ndarray) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG", optimize=False)
    return out.getvalue()


def _resized(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if pixels.shape[:2] == shape:
        return pixels
    return np.array(Image.fromarray(pixels).resize((shape[1], shape[0]), Image.BILINEAR))


def _repack(document: dict, binary: bytes, new_images: dict[int, bytes]) -> bytes:
    """Drop unused textures, images and buffer views; rebuild one aligned buffer.

    `new_images` maps an image index to replacement bytes.
    """
    materials = document.get("materials", [])
    used_textures: set[int] = set()

    def visit(value):
        if isinstance(value, dict):
            if "index" in value and isinstance(value.get("index"), int) and set(value) <= {
                    "index", "texCoord", "scale", "strength", "extensions", "extras"}:
                used_textures.add(value["index"])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(materials)
    textures = document.get("textures", [])
    keep_textures = [i for i in range(len(textures)) if i in used_textures]
    texture_map = {old: new for new, old in enumerate(keep_textures)}
    used_images = {textures[i]["source"] for i in keep_textures if "source" in textures[i]}
    images = document.get("images", [])
    keep_images = [i for i in range(len(images)) if i in used_images]
    image_map = {old: new for new, old in enumerate(keep_images)}

    def remap(value):
        if isinstance(value, dict):
            if "index" in value and isinstance(value.get("index"), int) and value["index"] in texture_map and set(value) <= {
                    "index", "texCoord", "scale", "strength", "extensions", "extras"}:
                value["index"] = texture_map[value["index"]]
            for item in value.values():
                remap(item)
        elif isinstance(value, list):
            for item in value:
                remap(item)

    remap(materials)
    document["textures"] = [dict(textures[i], source=image_map[textures[i]["source"]]) for i in keep_textures]
    old_images = images
    document["images"] = [old_images[i] for i in keep_images]
    if not document["textures"]:
        document.pop("textures", None)
    if not document["images"]:
        document.pop("images", None)

    # Every buffer view still referenced, in its original order, then the new images.
    views = document.get("bufferViews", [])
    referenced = {a["bufferView"] for a in document.get("accessors", []) if "bufferView" in a}
    for accessor in document.get("accessors", []):
        sparse = accessor.get("sparse")
        if sparse:
            referenced.add(sparse["indices"]["bufferView"])
            referenced.add(sparse["values"]["bufferView"])
    image_payload = {}
    for new, old in enumerate(keep_images):
        if old in new_images:
            image_payload[new] = new_images[old]
            document["images"][new] = {k: v for k, v in document["images"][new].items() if k != "uri"}
        else:
            referenced.add(old_images[old]["bufferView"])
    out = bytearray()
    new_views, view_map = [], {}
    for old, view in enumerate(views):
        if old not in referenced:
            continue
        out += b"\0" * (-len(out) % 4)
        start = view.get("byteOffset", 0)
        chunk = binary[start:start + view["byteLength"]]
        view_map[old] = len(new_views)
        new_views.append(dict(view, byteOffset=len(out), buffer=0))
        out += chunk
    for new, payload in image_payload.items():
        out += b"\0" * (-len(out) % 4)
        new_views.append({"buffer": 0, "byteOffset": len(out), "byteLength": len(payload)})
        document["images"][new]["bufferView"] = len(new_views) - 1
        document["images"][new]["mimeType"] = "image/png"
        out += payload
    for accessor in document.get("accessors", []):
        if "bufferView" in accessor:
            accessor["bufferView"] = view_map[accessor["bufferView"]]
        sparse = accessor.get("sparse")
        if sparse:
            sparse["indices"]["bufferView"] = view_map[sparse["indices"]["bufferView"]]
            sparse["values"]["bufferView"] = view_map[sparse["values"]["bufferView"]]
    for index, image in enumerate(document.get("images", [])):
        if "bufferView" in image and index not in image_payload:
            image["bufferView"] = view_map[image["bufferView"]]
    out += b"\0" * (-len(out) % 4)
    document["bufferViews"] = new_views
    document["buffers"] = [{"byteLength": len(out)}]
    return bytes(out)


def finalize_glb(data: bytes, stamp: Mapping[str, Any] | None = None) -> tuple[bytes, dict]:
    """Pack each material's ORM texture and give every texture its Unreal name.

    Material names (`MI_<Name>`, set by the Blender step) decide the texture names. The
    metallicRoughness image is rebuilt as an ORM (occlusion from the material's own
    occlusion texture when it has one, else white) and used for both slots, so Unreal
    gets one linear `T_<Name>_ORM` instead of two half-used maps.
    """
    document, binary = parse_glb(data)
    textures = document.get("textures", [])
    images = document.get("images", [])
    new_images: dict[int, bytes] = {}
    added_textures: list[dict] = []
    info: dict[str, Any] = {"materials": {}}
    pixels_cache: dict[int, np.ndarray] = {}

    def pixels(texture_index: int) -> np.ndarray:
        source = textures[texture_index]["source"]
        if source not in pixels_cache:
            pixels_cache[source] = _decode(_image_bytes(document, binary, source))
        return pixels_cache[source]

    names: dict[int, str] = {}
    for material in document.get("materials", []):
        name = material.get("name") or "MI_Material"
        pbr = material.setdefault("pbrMetallicRoughness", {})
        record: dict[str, Any] = {"orm": None, "occlusion": None}
        mr_ref = pbr.get("metallicRoughnessTexture")
        if mr_ref is not None:
            mr = pixels(mr_ref["index"])
            occlusion_ref = material.get("occlusionTexture")
            ao = None
            if occlusion_ref is not None:
                ao = _resized(pixels(occlusion_ref["index"]), mr.shape[:2])
            orm = pack_orm(mr, ao)
            images.append({"name": texture_name(name, "orm")})
            new_index = len(images) - 1
            new_images[new_index] = _encode_png(orm)
            added_textures.append({"source": new_index})
            texture_index = len(textures) + len(added_textures) - 1
            texcoord = mr_ref.get("texCoord", 0)
            pbr["metallicRoughnessTexture"] = {"index": texture_index, "texCoord": texcoord}
            material["occlusionTexture"] = {"index": texture_index, "texCoord": texcoord}
            record.update(orm=texture_name(name, "orm"), occlusion="baked" if ao is not None else "none")
        for role, ref in (("base_color", pbr.get("baseColorTexture")), ("normal", material.get("normalTexture")),
                          ("emissive", material.get("emissiveTexture"))):
            if ref is not None:
                names[textures[ref["index"]]["source"]] = texture_name(name, role)
                record[role] = texture_name(name, role)
        info["materials"][name] = record
    textures.extend(added_textures)
    if textures:
        document["textures"] = textures
    if images:
        document["images"] = images
    for index, name in names.items():
        images[index]["name"] = name
    if stamp is not None:
        document.setdefault("asset", {}).setdefault("extras", {})[EXTRAS_KEY] = dict(stamp)
    binary = _repack(document, binary, new_images)
    return build_glb(document, binary), info


# --- checks ----------------------------------------------------------------------------

def check_document(document: Mapping[str, Any], names: Mapping[str, Any]) -> list[str]:
    """What would make Unreal import this badly, in words. Empty when it is clean."""
    problems = []
    for extension in document.get("extensionsRequired", []) or []:
        problems.append(f"requires {extension}, which Unreal's glTF import cannot read")
    for extension in document.get("extensionsUsed", []) or []:
        if extension in REFUSED_EXTENSIONS and extension not in (document.get("extensionsRequired") or []):
            problems.append(f"uses {extension}, which Unreal's glTF import cannot read")
    mesh = names["mesh"]
    allowed = {mesh, *names["collision"].values()}
    lod = re.compile(rf"LOD\d+_{re.escape(mesh)}")
    for node in document.get("nodes", []):
        if "mesh" not in node:
            continue
        name = node.get("name", "")
        if name not in allowed and not lod.fullmatch(name) and not (
                name.startswith(("UBX_", "UCX_")) and mesh in name):
            problems.append(f"mesh node {name!r} is not {mesh}, a LODn_ node or a collision node")
    for material in document.get("materials", []):
        if not str(material.get("name", "")).startswith("MI_"):
            problems.append(f"material {material.get('name')!r} is not named MI_...")
    for image in document.get("images", []):
        if not str(image.get("name", "")).startswith("T_"):
            problems.append(f"texture {image.get('name')!r} is not named T_...")
    nodes = document.get("nodes", [])
    for skin in document.get("skins", []):
        joint_names = {nodes[j].get("name") for j in skin.get("joints", [])}
        if ROOT_BONE not in joint_names:
            problems.append("the skeleton has no 'root' bone at the origin")
    return problems


# --- sidecar and Unreal report ---------------------------------------------------------

def sidecar(*, kind: str, names: Mapping[str, Any], glb: str, size_cm: float | None, measure: str | None,
            lods: int, collision: str | None, humanoid: bool, skeleton_from: str | None,
            animations: list[str], tint: str | None, problems: list[str]) -> dict[str, Any]:
    """`<mesh>.unreal.json`: what Unreal must do with the GLB, and what to expect."""
    skinned = kind != "prop"
    return {
        "schema": SIDECAR_SCHEMA,
        "kind": kind,
        "glb": glb,
        "mesh": names["mesh"],
        "skinned": skinned,
        "material": names["material"],
        "textures": {"base_color": names["base_color"], "normal": names["normal"], "orm": names["orm"]},
        "tint": tint,
        "lods": lods,
        "collision": collision,
        "skeleton": None if not skinned else (f"{skeleton_from}_Skeleton" if skeleton_from else names["skeleton"]),
        "skeleton_from": skeleton_from,
        "humanoid": humanoid,
        "animations": animations,
        "problems": problems,
        "import": {
            "pipeline": "Interchange glTF",
            # Skeletal LODs are reduced by Unreal (gltfpack stalls on UV seams); static
            # LODs travel inside the GLB as LODn_ nodes.
            "generate_skeletal_lods": lods if skinned and kind != "equipment" else 0,
            "reuse_skeleton": skeleton_from is not None,
        },
        "expect": {
            "size_cm": size_cm,
            "measure": measure,
            "base_z_cm": 0.0,
            "facing": "+Y" if skinned and kind != "equipment" else None,
            "manny_compatible": humanoid and kind != "equipment",
        },
    }


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": detail}


def evaluate_unreal(measured: Mapping[str, Any], expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn what Unreal measured after import into pass/fail lines, one per criterion."""
    results = []
    want = expected["expect"]
    equipment = expected["kind"] == "equipment"
    bounds = measured.get("bounds_cm") or [0, 0, 0]
    if want.get("size_cm") and not equipment:
        got = bounds[2] if want["measure"] == "height" else max(bounds)
        ok = abs(got - want["size_cm"]) <= SIZE_TOLERANCE * want["size_cm"]
        results.append(_check("size", ok, f"{want['measure']} {got:.1f} cm, wanted {want['size_cm']} cm"))
    if not equipment:
        min_z = measured.get("min_z", 0.0)
        results.append(_check("base", abs(min_z) <= BASE_TOLERANCE_CM, f"lowest point at z = {min_z:.2f} cm"))
    names = measured.get("names", [])
    results.append(_check("names", expected["mesh"] in names, f"assets: {', '.join(names)}"))
    want_lods = expected["lods"]
    results.append(_check("lods", measured.get("lods", 0) >= want_lods, f"{measured.get('lods')} LODs, wanted {want_lods}"))
    if expected.get("collision") in ("box", "convex"):
        key = "box_elems" if expected["collision"] == "box" else "convex_elems"
        count = (measured.get("collision") or {}).get(key, 0)
        results.append(_check("collision", count >= 1, f"{count} {expected['collision']} collision shape(s)"))
    srgb = measured.get("orm_srgb", [])
    results.append(_check("orm", bool(srgb) and not any(srgb), f"ORM textures sRGB: {srgb}"))
    if expected.get("tint"):
        srgb = measured.get("tint_srgb")
        results.append(_check("tint", srgb is False, "tint mask imported as linear data" if srgb is False else
                              "tint mask missing" if srgb is None else "tint mask imported as sRGB colour"))
    if expected.get("skinned") and "root_cm" in measured:
        root = measured["root_cm"]
        results.append(_check("root", all(abs(v) <= BASE_TOLERANCE_CM for v in root),
                              f"root bone at {[round(v, 1) for v in root]} cm in the reference pose, wanted the origin"))
    if want.get("facing"):
        forward = measured.get("forward") or [0, 0, 0]
        results.append(_check("facing", forward[1] > 0.7, f"faces {forward}, wanted +Y like Manny"))
    if want.get("manny_compatible"):
        results.append(_check("skeleton", measured.get("ik_recognised") is True,
                              "IK Rig recognised the skeleton" if measured.get("ik_recognised") else
                              "IK Rig did not recognise the skeleton"))
        results.append(_check("retarget", measured.get("retarget") == "ok", f"retarget of MM_Idle: {measured.get('retarget')}"))
        if "walk" in measured:
            results.append(_check("walk", measured["walk"] == "ok", f"retarget of a walk: {measured['walk']}"))
    if equipment and expected.get("skeleton_from"):
        reused = str(measured.get("skeleton", "")).endswith(expected["skeleton"]) and measured.get("new_skeletons", 1) == 0
        results.append(_check("skeleton", reused, f"skeleton {measured.get('skeleton')}, new skeletons {measured.get('new_skeletons')}"))
    if expected.get("animations"):
        got = set(measured.get("animations", []))
        missing = [a for a in expected["animations"] if a not in got]
        results.append(_check("animations", not missing, f"missing: {missing}" if missing else f"{len(got)} imported"))
    return results


def stride(widths: Iterable[float]) -> float:
    """The widest step in a walk cycle, from the feet's fore-aft spread sampled over it."""
    return max(widths, default=0.0)


def stage_layout(widths: list[float], height: float, sun: Iterable[float], fov: float = 50.0,
                 aspect: float = 16 / 9, side_angle: float = 25.0) -> dict[str, Any]:
    """Where to stand a row of assets and the camera so the sun lights their faces.

    The camera looks roughly the way the sunlight travels (turned `side_angle` degrees
    for some shading), the row runs across its view, every actor is turned so its front
    (local +Y, Manny's) faces the camera, and the distance fits the whole row and the
    tallest figure in a `fov`-degree lens. Unreal axes: X forward, Y right, Z up; yaw in
    degrees.
    """
    sx, sy = list(sun)[:2]
    length = math.hypot(sx, sy)
    view = (sx / length, sy / length) if length > 1e-3 else (0.0, -1.0)
    turn = math.radians(side_angle)
    view = (view[0] * math.cos(turn) - view[1] * math.sin(turn), view[0] * math.sin(turn) + view[1] * math.cos(turn))
    camera_yaw = math.degrees(math.atan2(view[1], view[0]))
    right = (-view[1], view[0])
    total = sum(widths)
    offsets, edge = [], -total / 2
    for width in widths:
        offsets.append(edge + width / 2)
        edge += width
    half_h = math.tan(math.radians(fov) / 2)
    half_v = half_h / aspect
    distance = 1.12 * max(total / 2 / half_h, height / 2 / half_v) + max(widths, default=0) / 2
    return {
        "positions": [(right[0] * o, right[1] * o) for o in offsets],
        "actor_yaw": camera_yaw + 90.0,
        "camera": (-view[0] * distance, -view[1] * distance, height / 2),
        "camera_yaw": camera_yaw,
        "fov": fov,
    }


def closeup_camera(position: Iterable[float], camera_yaw: float, width: float, height: float,
                   fov: float = 35.0, aspect: float = 16 / 9) -> tuple[float, float, float]:
    """A camera that fills the frame with one figure, looking the same way as the wide
    shot (so the same side is lit), level with its middle so head and feet both fit."""
    px, py = list(position)[:2]
    vx, vy = math.cos(math.radians(camera_yaw)), math.sin(math.radians(camera_yaw))
    half_h = math.tan(math.radians(fov) / 2)
    distance = 1.3 * max(width / 2 / half_h, height / 2 / (half_h / aspect)) + width / 2
    return (px - vx * distance, py - vy * distance, height * 0.5)


# --- finding Unreal --------------------------------------------------------------------

def _default_roots() -> list[Path]:
    roots = []
    for key in ("ProgramFiles", "ProgramW6432"):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]) / "Epic Games")
    roots += [Path("C:/Program Files/Epic Games"), Path("/Users/Shared/Epic Games"), Path.home() / "UnrealEngine"]
    return roots


def _version(path: Path) -> tuple[int, ...]:
    match = re.search(r"UE_(\d+)\.(\d+)", str(path))
    return (int(match[1]), int(match[2])) if match else (0, 0)


def find_unreal_editor(env: Mapping[str, str] | None = None, roots: list[Path] | None = None) -> Path | None:
    """UnrealEditor.exe: `I2L_UNREAL` when set (a wrong path is not silently skipped),
    else the newest `UE_5.x` install in the usual places."""
    env = os.environ if env is None else env
    configured = env.get(UNREAL_ENV)
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    found = []
    for root in _default_roots() if roots is None else roots:
        found += [p for p in Path(root).glob("UE_5.*/Engine/Binaries/Win64/UnrealEditor.exe") if p.is_file()]
    return max(found, key=_version) if found else None
