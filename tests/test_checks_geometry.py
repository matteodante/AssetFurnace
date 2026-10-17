"""The package's mesh measures agree with the scripts they were copied from, and rasterise right."""

from __future__ import annotations

import _check_fixtures as fx
import mesh_health
import numpy as np
import pytest
import ribbon_metric
import trimesh

from image_to_3dlab.checks import geometry


def torn_capsule():
    vertices, faces, _ = fx.capsule_arrays()
    keep = np.ones(len(faces), dtype=bool)
    keep[::9] = False
    return vertices, faces[keep]


def test_welding_matches_mesh_health():
    vertices, faces = torn_capsule()
    ours = geometry.weld(vertices, faces)
    theirs = mesh_health.weld_by_position(trimesh.Trimesh(vertices, faces, process=False))
    assert len(ours.vertices) == len(theirs.vertices)
    assert len(ours.faces) == len(theirs.faces)


def test_boundary_shares_match_ribbon_metric():
    vertices, faces = torn_capsule()
    welded = geometry.weld(vertices, faces)
    assert geometry.boundary_shares(welded.faces) == pytest.approx(ribbon_metric.boundary_stats(welded.faces))


def test_nonmanifold_count_matches_mesh_health():
    vertices, faces, _ = fx.capsule_arrays()
    extra = np.vstack([faces, faces[:3][:, [0, 1, 2]] + 0])  # duplicate faces: edges used 4 times
    welded = geometry.weld(vertices, extra)
    stats = mesh_health.measure(trimesh.Trimesh(vertices, extra, process=False))
    assert geometry.nonmanifold_share(welded.faces)[0] == stats["non_manifold_edges"]


def test_orientation_counts_a_large_flipped_patch_by_its_area():
    vertices, faces, _ = fx.capsule_arrays()
    welded = geometry.weld(vertices, faces)
    flipped = welded.faces.copy()
    patch = welded.vertices[flipped].mean(axis=1)[:, 0] > 0.15
    flipped[patch] = flipped[patch][:, ::-1]
    mask, volume = geometry.orientation(welded.vertices, flipped)
    assert mask.sum() == patch.sum()
    assert volume > 0


def test_orientation_spots_an_inside_out_surface():
    vertices, faces, _ = fx.capsule_arrays()
    welded = geometry.weld(vertices, faces)
    mask, volume = geometry.orientation(welded.vertices, welded.faces[:, ::-1])
    assert not mask.any()
    assert volume < 0


def test_components_split_on_shared_edges_only():
    a = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
    labels = geometry.components(a)
    assert labels[0] == labels[1] != labels[2]


def test_rasterize_fills_the_union_where_fill_poly_would_cancel():
    """Overlapping triangles must add up, not cancel (cv2.fillPoly's even-odd rule)."""
    square = np.array([[2, 2], [12, 2], [12, 12], [2, 12]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 2], [2, 1, 0]])  # front and back, both windings
    assert geometry.rasterize(square, faces, 20, 20).sum() == 100


def test_rasterize_keeps_triangles_smaller_than_a_pixel():
    tiny = np.array([[5.2, 5.2], [5.4, 5.2], [5.2, 5.4]])
    assert geometry.rasterize(tiny, np.array([[0, 1, 2]]), 10, 10).sum() == 1


def test_rasterize_draws_large_triangles_and_slivers():
    big = np.array([[0, 0], [100, 0], [0, 100], [50, 50.2], [50.2, 50]], dtype=float)
    image = geometry.rasterize(big, np.array([[0, 1, 2], [3, 4, 1]]), 100, 100)
    assert 4900 < image.sum() < 5200


def test_winding_number_is_one_inside_a_closed_surface_and_zero_outside():
    vertices, faces, _ = fx.capsule_arrays()
    welded = geometry.weld(vertices, faces)
    points = np.array([[0.0, 0.9, 0.0], [2.0, 0.9, 0.0]])
    assert geometry.winding_number(points, welded.vertices, welded.faces) == pytest.approx([1.0, 0.0], abs=1e-6)


def test_uv_overlap_counts_every_surface_use_of_a_shared_pixel():
    """Codex review 3: two equal charts overlapping by 15% of their area reuse 15% of the surface."""
    from image_to_3dlab.checks.mesh_checks import uv_overlap_pixels

    square = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]])
    shifted = square + (0.4 * 0.85, 0.0)          # 15% of the width overlaps
    uv = np.vstack([square, shifted])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    reused, used = uv_overlap_pixels([(uv, faces)], resolution=512)
    assert reused / used == pytest.approx(0.15, abs=0.01)


def test_a_copy_sharing_the_texture_counts_once_when_copies_are_allowed():
    """A pair of pauldrons in one file: the mirrored one repeats the other's UVs exactly."""
    from image_to_3dlab.checks.mesh_checks import uv_overlap_pixels, without_copied_uvs

    square = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]])
    shifted = square + (0.4 * 0.85, 0.0)
    pair = [(np.vstack([square, square]), np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]))]
    reused, used = uv_overlap_pixels(pair, resolution=512)
    assert reused / used == pytest.approx(1.0)
    assert uv_overlap_pixels(without_copied_uvs(pair), resolution=512)[0] == 0.0
    near = [(np.vstack([square, shifted]), np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]))]
    reused, used = uv_overlap_pixels(without_copied_uvs(near), resolution=512)
    assert reused / used == pytest.approx(0.15, abs=0.01)  # a real overlap still counts
