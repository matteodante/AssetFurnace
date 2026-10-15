"""The shared multi-view piece: cameras, the tagged copy Blender renders, and the way back
from rendered pixels to texels of the atlas. Pure parts only; the real render is in
test_multiview_blender.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_materials import banded_albedo, make_glb, two_material_glb

from image_to_3dlab import multiview as mv
from image_to_3dlab.glb_turn import parse_glb


def test_orbit_sees_all_round_and_from_above_and_below():
    directions = mv.orbit_directions(around=16, elevation=20.0)
    assert directions.shape == (18, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert directions[:, 2].min() == pytest.approx(-1.0) and directions[:, 2].max() == pytest.approx(1.0)
    ring = directions[:16]
    # Alternate rings look slightly down and slightly up, all the way round.
    assert (ring[::2, 2] < 0).all() and (ring[1::2, 2] > 0).all()
    azimuths = np.degrees(np.arctan2(ring[:, 1], ring[:, 0])) % 360
    assert np.allclose(np.sort(azimuths), np.arange(16) * 22.5, atol=1e-6)


def test_the_tagged_copy_names_every_material_by_its_index(tmp_path):
    source = two_material_glb(tmp_path / "two.glb", ["arm", "arm"])
    document, _ = parse_glb(mv.tag_materials(source.read_bytes()))
    assert [mat["name"] for mat in document["materials"]] == ["afmat_0", "afmat_1"]
    assert mv.material_id("afmat_1") == 1
    assert mv.material_id("afmat_1.001") == 1        # Blender's suffix on a clash
    assert mv.material_id("arm") is None


def test_pixels_find_their_texels_with_blenders_v_turned_up():
    """Blender stores v upward; the atlas row is counted from the top as in glTF."""
    uv = np.array([[[0.0, 1.0], [0.5, 0.5]], [[0.999, 0.001], [0.25, 0.75]]])
    ids = np.array([[3, 3], [3, -1]])
    view = mv.View(index=0, direction=np.array([0, 0, -1.0]), uv=uv, material=ids,
                   normal=np.zeros((2, 2, 3)), color=None)
    pixels, texels = mv.view_texels(view, material=3, size=(4, 8))
    rows, cols = np.divmod(texels, 8)
    assert pixels.tolist() == [0, 1, 2]
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, 0), (2, 4), (3, 7)]


def test_texture_transform_and_wrap_are_applied_like_the_renderer():
    uv = np.array([[[1.25, 1.0 - 0.5]]])           # Blender v of glTF v = 0.5
    view = mv.View(0, np.array([0, 0, -1.0]), uv, np.array([[0]]), np.zeros((1, 1, 3)), None)
    _, repeat = mv.view_texels(view, material=0, size=(10, 10))
    assert np.divmod(repeat, 10)[1].tolist() == [2]                     # 1.25 wraps to 0.25
    _, scaled = mv.view_texels(view, material=0, size=(10, 10), transform={"scale": [0.5, 1.0]})
    assert np.divmod(scaled, 10)[1].tolist() == [6]                     # 0.625


def test_a_surface_seen_head_on_counts_fully_and_one_seen_edge_on_not_at_all():
    direction = np.array([0.0, 0.0, -1.0])           # the camera looks down
    normals = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0, -1.0], [0, 0.6, 0.8]])
    np.testing.assert_allclose(mv.facing_weight(normals, direction), [1.0, 0.0, 0.0, 0.8])


def test_texels_average_what_the_views_saw_weighted_by_how_squarely():
    acc = mv.Accumulator((2, 2), channels=1)
    acc.add(np.array([0, 0, 3]), np.array([[1.0], [3.0], [5.0]]), np.array([1.0, 3.0, 0.5]))
    mean, seen = acc.mean()
    assert mean[0, 0, 0] == pytest.approx((1 * 1 + 3 * 3) / 4)
    assert mean[1, 1, 0] == pytest.approx(5.0)
    assert seen.tolist() == [[True, False], [False, True]]


def test_a_vote_picks_the_most_supported_class_per_texel():
    acc = mv.Accumulator((1, 3), channels=3)
    acc.add(np.array([0, 0, 1, 2]), np.eye(3)[[0, 1, 2, 2]], np.array([0.9, 0.4, 1.0, 1.0]))
    labels, share = mv.vote(acc)
    assert labels.tolist() == [[0, 2, 2]]
    assert share[0, 0] == pytest.approx(0.9 / 1.3)


def test_covered_texels_no_view_saw_take_the_nearest_seen_value():
    values = np.zeros((1, 6, 1))
    values[0, 0, 0], values[0, 5, 0] = 1.0, 9.0
    seen = np.array([[True, False, False, False, False, True]])
    coverage = np.array([[True, True, True, True, True, False]])
    filled = mv.fill_unseen(values, seen, coverage)
    # Texel 5 is seen but outside the island, so it lends nothing to the island's holes.
    assert filled[0, :, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 9.0]


def test_views_are_read_back_as_written(tmp_path):
    from PIL import Image

    size = 4
    uv = np.random.default_rng(0).random((size, size, 2)).astype(np.float32)
    ids = np.full((size, size), 2, np.float32)
    ids[0, 0] = -1
    normal = np.zeros((size, size, 3), np.float32)
    normal[..., 2] = 1
    rgba_uv = np.dstack([uv[..., 0], uv[..., 1], ids + 1, np.ones((size, size), np.float32)])
    np.savez_compressed(tmp_path / "view_00_uv.npz", rgba=rgba_uv)
    np.savez_compressed(tmp_path / "view_00_normal.npz", rgba=np.dstack([normal, np.ones((size, size))]))
    Image.fromarray(np.full((size, size, 4), 200, np.uint8)).save(tmp_path / "view_00_color.png")
    (tmp_path / "views.json").write_text(json.dumps({"views": [{"index": 0, "direction": [0, -1, 0]}]}))
    [view] = mv.load_views(tmp_path)
    np.testing.assert_allclose(view.uv, uv, atol=1e-6)
    assert view.material[0, 0] == -1 and (view.material.ravel()[1:] == 2).all()
    np.testing.assert_allclose(view.normal[..., 2], 1.0)
    assert view.color.shape == (size, size, 3) and view.color[0, 0, 0] == 200


def test_the_render_plan_says_what_blender_must_draw(tmp_path):
    asset = make_glb(tmp_path / "ball.glb", banded_albedo())
    plan = mv.render_plan(asset, tmp_path / "views", size=64, around=8)
    assert plan["size"] == 64 and len(plan["directions"]) == 10
    tagged, _ = parse_glb((tmp_path / "views" / "tagged.glb").read_bytes())
    assert tagged["materials"][0]["name"] == "afmat_0"


def test_unseen_texels_are_filled_from_their_own_island():
    """Atlas neighbours are not surface neighbours: a fill never crosses into another island."""
    values = np.zeros((1, 12, 1))
    seen = np.zeros((1, 12), bool)
    coverage = np.zeros((1, 12), bool)
    islands = np.zeros((1, 12), np.int32)
    coverage[0, 0:5] = True; islands[0, 0:5] = 1            # island 1: seen only at its left end
    values[0, 0, 0], seen[0, 0] = 1.0, True
    coverage[0, 6:9] = True; islands[0, 6:9] = 2            # island 2: seen at its left end, close by
    values[0, 6, 0], seen[0, 6] = 9.0, True
    coverage[0, 10:12] = True; islands[0, 10:12] = 3        # island 3: never seen
    filled = mv.fill_unseen(values, seen, coverage, islands=islands)
    assert filled[0, 0:5, 0].tolist() == [1.0] * 5           # not 9 from the nearer island
    assert filled[0, 6:9, 0].tolist() == [9.0] * 3
    assert filled[0, 10:12, 0].tolist() == [9.0, 9.0]        # no sample of its own: nearest anywhere


def test_blender_gets_an_absolute_plan_path(tmp_path, monkeypatch):
    asset = make_glb(tmp_path / "ball.glb", banded_albedo())
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_run(args, **kwargs):
        plan = Path(args[-1])
        seen["absolute"], seen["exists"] = plan.is_absolute(), plan.is_file()
        (plan.parent / "views.json").write_text(json.dumps({"views": []}))

    assert mv.render_views(asset, Path("views"), size=16, around=4, blender=Path("blender"), run=fake_run) == []
    assert seen == {"absolute": True, "exists": True}
