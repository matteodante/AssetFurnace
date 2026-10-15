"""Mesh cleanliness: flipped faces, holes, floating parts, non-manifold edges, UV overlap."""

from __future__ import annotations

import numpy as np

from image_to_3dlab.checks import geometry
from image_to_3dlab.checks.asset import base_colour
from image_to_3dlab.checks.registry import grade_max, register, result


def _limits(config: dict, warn: float, fail: float) -> dict:
    return {"warn": config.get("warn", warn), "fail": config.get("fail", fail)}


@register("mesh.flipped_faces")
def flipped_faces(asset, config):
    welded = asset.welded
    flipped, volume = geometry.orientation(welded.vertices, welded.faces)
    share = 100.0 * float(flipped.mean()) if len(flipped) else 0.0
    limits = _limits(config, 0.1, 1.0)
    status = grade_max(share, **limits)
    _, open_faces = geometry.boundary_shares(welded.faces)
    # The volume says something only when the surface is mostly closed.
    inside_out = volume < 0 and open_faces < 5.0
    details = {"flipped_faces": int(flipped.sum()), "volume": round(volume, 6)}
    if inside_out:
        return result("mesh.flipped_faces", "fail", round(share, 3), limits,
                      f"the whole surface faces inward (volume {volume:.4g}); "
                      f"{share:.2f}% of faces also disagree with their neighbours", details)
    return result("mesh.flipped_faces", status, round(share, 3), limits,
                  f"{share:.2f}% of faces are wound against their neighbours", details)


@register("mesh.holes")
def holes(asset, config):
    edge_share, face_share = geometry.boundary_shares(asset.welded.faces)
    limits = _limits(config, 1.0, 3.0)
    return result("mesh.holes", grade_max(face_share, **limits), round(face_share, 3), limits,
                  f"{face_share:.2f}% of faces touch an open edge",
                  {"open_edges_percent": round(edge_share, 3)})


@register("mesh.loose_parts")
def loose_parts(asset, config):
    """Pieces far smaller than the main body: specks left over from extraction.

    A piece counts as a speck below `speck_share` percent of the largest piece's faces, so
    a sword with a separate guard is two real parts, not one part and debris.
    """
    labels = geometry.components(asset.welded.faces)
    sizes = np.bincount(labels) if len(labels) else np.zeros(0, dtype=int)
    speck_share = config.get("speck_share", 0.5)
    largest = int(sizes.max()) if len(sizes) else 0
    specks = sizes[sizes < largest * speck_share / 100.0]
    speck_faces = 100.0 * float(specks.sum()) / max(len(labels), 1)
    limits = {"warn": config.get("warn", 0), "fail": config.get("fail", 1.0)}
    status = "fail" if speck_faces > limits["fail"] else "warn" if len(specks) > limits["warn"] else "pass"
    return result("mesh.loose_parts", status, len(specks),
                  {"warn_above_specks": limits["warn"], "fail_above_percent_faces": limits["fail"]},
                  f"{len(specks)} floating speck(s) holding {speck_faces:.2f}% of the faces, "
                  f"in {len(sizes)} piece(s)",
                  {"pieces": len(sizes), "speck_faces_percent": round(speck_faces, 3)})


@register("mesh.nonmanifold")
def nonmanifold(asset, config):
    count, share = geometry.nonmanifold_share(asset.welded.faces)
    limits = _limits(config, 0.0, 0.5)
    status = "fail" if share > limits["fail"] else "warn" if count and share > limits["warn"] else "pass"
    return result("mesh.nonmanifold", status, round(share, 4), limits,
                  f"{count} edge(s) shared by three or more faces ({share:.3f}% of edges)",
                  {"edges": count})


def _uv_charts(faces: np.ndarray) -> np.ndarray:
    """Chart label per face. glTF splits vertices at UV seams, so charts are just the
    components of the mesh as stored, before any welding."""
    return geometry.components(faces) if len(faces) else np.zeros(0, dtype=np.int64)


def uv_overlap_pixels(groups: list[tuple[np.ndarray, np.ndarray]], resolution: int = 1024) -> tuple[float, int]:
    """(texture pixels the surface uses in all, distinct pixels) for UVs sharing one texture.

    The two are equal when no texture pixel is used twice; the gap is the reuse.

    Each group is (uv, faces) for one primitive. Every chart is drawn on its own and added
    to a count, so two charts on the same pixels show as overlap. A chart folded onto
    itself does not add up that way, so it is caught by area instead: its triangles' areas
    sum to clearly more than the pixels it covers.
    """
    count = np.zeros((resolution, resolution), dtype=np.uint16)
    folded = 0.0
    for uv, faces in groups:
        labels = _uv_charts(faces)
        if not len(labels):
            continue
        pixels = uv * resolution
        pixels[:, 1] = resolution - pixels[:, 1]
        areas = geometry.triangle_areas(pixels, faces)
        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(labels.max() + 2))
        for chart in range(len(bounds) - 1):
            members = faces[order[bounds[chart]:bounds[chart + 1]]]
            corners = pixels[members].reshape(-1, 2)
            lo = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, resolution - 1)
            hi = np.clip(np.ceil(corners.max(axis=0)).astype(int) + 1, lo + 1, resolution)
            drawn = geometry.rasterize(pixels - lo, members, int(hi[0] - lo[0]), int(hi[1] - lo[1]))
            count[lo[1]:hi[1], lo[0]:hi[0]] += drawn
            area, covered = float(areas[order[bounds[chart]:bounds[chart + 1]]].sum()), float(drawn.sum())
            if area > covered * 1.05:
                folded += area - covered
    return float(count.sum()) + folded, int((count > 0).sum())


@register("uv.overlap")
def uv_overlap(asset, config):
    """Share of the surface painted from texture pixels that another piece also uses.

    Mirrored or stacked UVs on purpose show up here too; a profile can make it warn only.
    """
    resolution = int(config.get("resolution", 1024))
    groups: dict[int, list] = {}
    missing, outside, total = 0, 0, 0
    for mesh in asset.primitives:
        image = base_colour(mesh)
        uv = getattr(mesh.visual, "uv", None)
        if image is None:
            continue
        if uv is None or len(uv) != len(mesh.vertices):
            missing += 1
            continue
        uv = np.asarray(uv, dtype=np.float64)
        outside += int(((uv < -1e-3) | (uv > 1 + 1e-3)).any(axis=1).sum())
        total += len(uv)
        groups.setdefault(id(image), []).append((np.clip(uv, 0.0, 1.0), np.asarray(mesh.faces)))
    limits = _limits(config, 2.0, 10.0)
    if missing:
        return result("uv.overlap", "fail", None, limits,
                      f"{missing} textured part(s) have no UV coordinates")
    if not groups:
        return result("uv.overlap", "pass", 0.0, limits, "no texture, so no UV layout to overlap")
    used = distinct = 0.0
    for members in groups.values():
        u, d = uv_overlap_pixels(members, resolution)
        used += u
        distinct += d
    share = 100.0 * (used - distinct) / max(used, 1.0)
    status = grade_max(share, **limits)
    outside_share = 100.0 * outside / max(total, 1)
    message = f"{share:.2f}% of the surface is painted from texture pixels used elsewhere too"
    if outside_share > 0 and status == "pass":
        status = "warn"
        message += f"; {outside_share:.2f}% of UVs fall outside the texture"
    return result("uv.overlap", status, round(share, 3), limits, message,
                  {"outside_percent": round(outside_share, 3), "textures": len(groups)})
