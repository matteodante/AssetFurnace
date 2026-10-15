"""Small GLB assets for the checks tests: one clean asset, and copies broken on purpose.

Each broken copy carries exactly one defect, so a test can say not only "this asset is
stopped" but "it is stopped by the check that should stop it, and only by that one".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

HEIGHT = 1.8
RADIUS = 0.3


def capsule_arrays(rows: int = 24, cols: int = 32, radius: float = RADIUS, height: float = HEIGHT):
    """A closed capsule with a latitude-longitude UV layout, Y up, standing on y=0.

    The seam column is duplicated (different UVs, same position), the way glTF stores it,
    so the checks have to weld by position to see one closed surface.
    """
    lat = np.linspace(0.0, np.pi, rows + 1)
    lon = np.linspace(0.0, 2 * np.pi, cols + 1)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    ring = np.sin(lat_grid)
    x = radius * ring * np.cos(lon_grid)
    z = radius * ring * np.sin(lon_grid)
    # A sphere stretched into a capsule: the middle band carries the height.
    y = np.cos(lat_grid)
    y = np.where(y >= 0, y * radius + (height - radius), y * radius + radius)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    uv = np.stack([lon_grid / (2 * np.pi), 1.0 - lat_grid / np.pi], axis=-1).reshape(-1, 2)
    faces = []
    width = cols + 1
    for r in range(rows):
        for c in range(cols):
            a, b = r * width + c, r * width + c + 1
            d, e = (r + 1) * width + c, (r + 1) * width + c + 1
            if r > 0:
                faces.append((a, b, d))
            if r < rows - 1:
                faces.append((b, e, d))
    return vertices, np.array(faces, dtype=np.int64), uv


def textured(vertices, faces, uv, texture_size=(1024, 1024), colour=(180, 120, 90)) -> trimesh.Trimesh:
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", texture_size, colour), metallicFactor=0.0, roughnessFactor=0.8)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))


def write_glb(meshes, path: Path) -> Path:
    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes if isinstance(meshes, list) else [meshes]):
        scene.add_geometry(mesh, geom_name=f"part{index}")
    path.write_bytes(scene.export(file_type="glb"))
    return path


def clean(path: Path) -> Path:
    return write_glb(textured(*capsule_arrays()), path)


def flipped(path: Path) -> Path:
    """Half the capsule (x > 0) wound inward."""
    vertices, faces, uv = capsule_arrays()
    centres = vertices[faces].mean(axis=1)
    faces = faces.copy()
    faces[centres[:, 0] > 0] = faces[centres[:, 0] > 0][:, ::-1]
    return write_glb(textured(vertices, faces, uv), path)


def inside_out(path: Path) -> Path:
    vertices, faces, uv = capsule_arrays()
    return write_glb(textured(vertices, faces[:, ::-1], uv), path)


def holed(path: Path) -> Path:
    """Every twelfth face missing: a torn surface, still one piece."""
    vertices, faces, uv = capsule_arrays()
    keep = np.ones(len(faces), dtype=bool)
    keep[::12] = False
    return write_glb(textured(vertices, faces[keep], uv), path)


def with_specks(path: Path, count: int = 1) -> Path:
    """Tiny closed four-face specks floating inside the body: extraction debris."""
    vertices, faces, uv = capsule_arrays()
    all_vertices, all_faces, all_uv = [vertices], [faces], [uv]
    offset = len(vertices)
    corners = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]])
    tetra = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    for i in range(count):
        all_vertices.append(corners + (0.0, 0.5 + 0.1 * i, 0.0))
        all_faces.append(tetra + offset)
        all_uv.append(np.full((4, 2), 0.999))
        offset += 4
    return write_glb(textured(np.vstack(all_vertices), np.vstack(all_faces), np.vstack(all_uv)), path)


def overlapping_uv(path: Path) -> Path:
    """Both halves of the capsule painted from the same half of the texture."""
    vertices, faces, uv = capsule_arrays()
    uv = uv.copy()
    uv[:, 0] = np.where(uv[:, 0] > 0.5, uv[:, 0] - 0.5, uv[:, 0]) * 2 * 0.5
    return write_glb(textured(vertices, faces, uv), path)


def many_materials(path: Path, count: int = 8) -> Path:
    """The same capsule cut into `count` slices, each with its own material."""
    vertices, faces, uv = capsule_arrays()
    centres = vertices[faces].mean(axis=1)
    band = np.minimum((centres[:, 1] / HEIGHT * count).astype(int), count - 1)
    parts = [textured(vertices, faces[band == i], uv, colour=(30 * i, 90, 200 - 20 * i))
             for i in range(count)]
    return write_glb(parts, path)


def big_texture(path: Path) -> Path:
    return write_glb(textured(*capsule_arrays(), texture_size=(8192, 16)), path)


def scaled(path: Path, factor: float) -> Path:
    vertices, faces, uv = capsule_arrays()
    return write_glb(textured(vertices * factor, faces, uv), path)


def stretched(path: Path, x: float = 1.0, z: float = 1.0) -> Path:
    vertices, faces, uv = capsule_arrays()
    return write_glb(textured(vertices * (x, 1.0, z), faces, uv), path)


def cube(path: Path) -> Path:
    """A different shape at the same height: the silhouette should not recognise it."""
    box = trimesh.creation.box(extents=(HEIGHT, HEIGHT, HEIGHT))
    box.apply_translation((0, HEIGHT / 2, 0))
    uv = (box.vertices[:, [0, 2]] / HEIGHT) + 0.5
    return write_glb(textured(box.vertices, box.faces, uv), path)


def broken_accessor(source: Path, path: Path) -> Path:
    """A GLB whose index accessor claims more indices than its buffer holds."""
    from image_to_3dlab.glb_turn import build_glb, parse_glb

    document, binary = parse_glb(source.read_bytes())
    for primitive in document["meshes"][0]["primitives"]:
        document["accessors"][primitive["indices"]]["count"] *= 50
    path.write_bytes(build_glb(document, binary))
    return path


def sidecar(glb: Path, record: dict, suffix: str = ".provenance.json") -> Path:
    path = glb.with_name(glb.stem + suffix)
    path.write_text(json.dumps(record))
    return path


PIXAL3D_RECORD = {
    "schema_version": 1,
    "backend": "pixal3d",
    "components": [{"component": "rembg/birefnet-general-lite", "license": "MIT (BiRefNet)",
                    "purpose": "background removal"}],
    "license": {"name": "MIT (code + flow weights); DINOv3 License (bundled encoder)",
                "url": "https://huggingface.co/raven38/pixal3d-sv-q8_0-v1"},
    "output": {"classification": "commercial-conditional"},
}

HUNYUAN_RECORD = {
    "schema_version": 1,
    "license": {"name": "Tencent Hunyuan 3D 2.1 Community License", "conditions": [
        "Use and outputs are not licensed in the EU, UK, or South Korea."]},
    "output": {"classification": "territory-restricted"},
    "components": [],
}


def one_material_many_draws(path: Path, count: int = 10) -> Path:
    """The capsule repeated `count` times, every copy sharing one material: many draw calls."""
    base = textured(*capsule_arrays())
    parts = []
    for i in range(count):
        copy = base.copy()
        copy.apply_translation((i * 0.7, 0, 0))
        copy.visual.material = base.visual.material
        parts.append(copy)
    return write_glb(parts, path)


def with_inward_piece(path: Path) -> Path:
    """The clean capsule plus a smaller, separate capsule wound inside out."""
    vertices, faces, uv = capsule_arrays()
    small = vertices * 0.5 + (1.0, 0.0, 0.0)
    all_vertices = np.vstack([vertices, small])
    all_faces = np.vstack([faces, faces[:, ::-1] + len(vertices)])
    return write_glb(textured(all_vertices, all_faces, np.vstack([uv, uv])), path)


def with_cavity(path: Path) -> Path:
    """A solid with a hollow inside: an inner shell facing inward, into the cavity. Correct."""
    vertices, faces, uv = capsule_arrays()
    inner = vertices * 0.5 + (0.0, 0.45, 0.0)
    all_vertices = np.vstack([vertices, inner])
    all_faces = np.vstack([faces, faces[:, ::-1] + len(vertices)])
    return write_glb(textured(all_vertices, all_faces, np.vstack([uv, uv])), path)


def points_only(path: Path) -> Path:
    """A valid GLB that draws no triangles at all: only points."""
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.points.PointCloud(np.random.default_rng(0).random((50, 3))), geom_name="points")
    path.write_bytes(scene.export(file_type="glb"))
    return path


def external_texture(path: Path, name: str = "large texture.png", size=(8192, 16)) -> Path:
    """The clean capsule with its texture beside it, under a URI-encoded file name."""
    from urllib.parse import quote

    from image_to_3dlab.glb_turn import build_glb, parse_glb

    document, binary = parse_glb(clean(path).read_bytes())
    Image.new("RGB", size, (90, 90, 90)).save(path.parent / name)
    document["images"] = [{"uri": quote(name), "mimeType": "image/png"}]
    path.write_bytes(build_glb(document, binary))
    return path


def normal_map_only(path: Path, overlapping: bool = True) -> Path:
    """A capsule painted by a flat colour, with only a normal map, whose UVs overlap."""
    vertices, faces, uv = capsule_arrays()
    if overlapping:
        uv = uv.copy()
        uv[:, 0] = np.where(uv[:, 0] > 0.5, uv[:, 0] - 0.5, uv[:, 0])
    material = trimesh.visual.material.PBRMaterial(
        normalTexture=Image.new("RGB", (256, 256), (128, 128, 255)), baseColorFactor=[200, 100, 50, 255])
    return write_glb(trimesh.Trimesh(vertices, faces, process=False,
                                     visual=trimesh.visual.TextureVisuals(uv=uv, material=material)), path)
