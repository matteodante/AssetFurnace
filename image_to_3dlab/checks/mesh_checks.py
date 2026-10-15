"""Mesh cleanliness: flipped faces, holes, floating parts, non-manifold edges, UV overlap."""

from __future__ import annotations

import numpy as np

from image_to_3dlab.checks import geometry
from image_to_3dlab.checks.asset import texture_of
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
    # And piece by piece: a big outward piece must not hide a smaller, closed inward one.
    inward = geometry.inward_pieces(welded.vertices, welded.faces, flipped, config.get("speck_share", 0.5))
    details = {"flipped_faces": int(flipped.sum()), "volume": round(volume, 6), "inward_pieces": len(inward)}
    if inward and not inside_out:
        faces = sum(n for n, _ in inward)
        return result("mesh.flipped_faces", "fail", round(share, 3), limits,
                      f"{len(inward)} closed piece(s) ({faces:,} faces) face inward; "
                      f"{share:.2f}% of faces also disagree with their neighbours", details)
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


def uv_overlap_pixels(groups: list[tuple[np.ndarray, np.ndarray]], resolution: int = 1024) -> tuple[float, float]:
    """(surface painted from a pixel another piece also uses, surface painted in all), in pixels.

    Each chart counts each of its pixels once; a pixel two charts share counts for both of
    them, so two equal charts overlapping by 15% reuse 15% of the surface. A chart folded
    onto itself does not add up that way, so it is caught by area instead: its triangles'
    areas sum to clearly more than the pixels it covers, and both layers of the fold count.
    All charts are drawn in one pass.
    """
    pairs, folded, offset = [], 0.0, 0
    for uv, faces in groups:
        labels = _uv_charts(faces)
        if not len(labels):
            continue
        pixels = uv * resolution
        pixels[:, 1] = resolution - pixels[:, 1]
        hit, owner = geometry.raster_hits(pixels, faces, resolution, resolution)
        chart = labels[owner] + offset
        unique = np.unique(np.stack([chart, hit], axis=1), axis=0)
        pairs.append(unique)
        areas = np.bincount(labels, weights=geometry.triangle_areas(pixels, faces))
        covered = np.bincount(unique[:, 0] - offset, minlength=len(areas)).astype(float)
        excess = areas - covered
        folded += float(excess[areas > covered * 1.05].sum())
        offset += int(labels.max()) + 1
    if not pairs:
        return 0.0, 0.0
    every = np.vstack(pairs)
    _, slot, charts_per_pixel = np.unique(every[:, 1], return_inverse=True, return_counts=True)
    shared = float((charts_per_pixel[slot.reshape(-1)] > 1).sum())
    return shared + 2.0 * folded, float(len(every)) + folded


@register("uv.overlap")
def uv_overlap(asset, config):
    """Share of the surface painted from texture pixels that another piece also uses.

    Mirrored or stacked UVs on purpose show up here too; a profile can make it warn only.
    """
    resolution = int(config.get("resolution", 1024))
    groups: dict[int, list] = {}
    missing, outside, total = 0, 0, 0
    for mesh in asset.primitives:
        image = texture_of(mesh)
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
    limits = _limits(config, 4.0, 10.0)
    if missing:
        return result("uv.overlap", "fail", None, limits,
                      f"{missing} textured part(s) have no UV coordinates")
    if not groups:
        return result("uv.overlap", "pass", 0.0, limits, "no texture, so no UV layout to overlap")
    reused = used = 0.0
    for members in groups.values():
        r, u = uv_overlap_pixels(members, resolution)
        reused += r
        used += u
    share = min(100.0, 100.0 * reused / max(used, 1.0))
    status = grade_max(share, **limits)
    outside_share = 100.0 * outside / max(total, 1)
    message = f"{share:.2f}% of the surface is painted from texture pixels used elsewhere too"
    if outside_share > 0 and status == "pass":
        status = "warn"
        message += f"; {outside_share:.2f}% of UVs fall outside the texture"
    return result("uv.overlap", status, round(share, 3), limits, message,
                  {"outside_percent": round(outside_share, 3), "textures": len(groups)})
