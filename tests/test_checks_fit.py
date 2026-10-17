"""fit.clipping and fit.holes: the geometry measures on toy meshes, the bone matching by name,
and the registry skipping pieces that have no body."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from image_to_3dlab.checks import Asset, fit_checks, run_checks
from tests._skinned_glb import write as write_skinned_glb


def _box(size, centre=(0.0, 0.0, 0.0)):
    return trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(centre))


def test_a_shell_outside_the_body_does_not_clip():
    body = _box((1.0, 1.0, 1.0))
    shell = _box((1.2, 1.2, 1.2))
    masked = np.zeros(len(body.faces), bool)
    assert fit_checks.outer_skin_inside(shell, body, masked, np.arange(len(shell.vertices)), reach=0.5) == 0.0


def test_a_shell_sunk_into_visible_skin_clips_unless_the_mask_hides_that_skin():
    body = _box((1.0, 1.0, 1.0))
    shell = _box((0.9, 0.9, 0.9))  # all of it under the body's skin
    sample = np.arange(len(shell.vertices))
    assert fit_checks.outer_skin_inside(shell, body, np.zeros(len(body.faces), bool), sample, reach=0.5) == 1.0
    assert fit_checks.outer_skin_inside(shell, body, np.ones(len(body.faces), bool), sample, reach=0.5) == 0.0


def test_rays_only_count_hits_within_reach():
    target = _box((1.0, 1.0, 0.1), centre=(0, 0, 1.0))
    origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    directions = np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, 1.0]])
    assert fit_checks.rays_hit(origins, directions, target, reach=2.0).tolist() == [True, False, False]
    assert fit_checks.rays_hit(origins, directions, target, reach=0.5).tolist() == [False, False, False]


class _Rig:
    """Enough of SkinnedGLB for world_by_name: two joints under a root node."""
    joint_nodes = (1, 2)

    def name(self, node):
        return ["Armature", "hips", "spine"][node]

    def rest_locals(self):
        return [np.eye(4)] * 3

    def world(self, locals_):
        return list(locals_)


def test_bones_are_matched_by_name_and_the_rest_stays():
    moved = np.eye(4)
    moved[:3, 3] = [0, 1, 0]
    world = fit_checks.world_by_name(_Rig(), {"spine": moved, "elsewhere": np.zeros((4, 4))})
    assert np.allclose(world[2], moved) and np.allclose(world[1], np.eye(4)) and np.allclose(world[0], np.eye(4))


def test_a_piece_without_a_body_is_skipped_not_passed(tmp_path):
    glb = write_skinned_glb(tmp_path / "piece.glb")
    report = run_checks(Asset(glb=glb), profile="equipment_part", only=["fit.clipping", "fit.holes"])
    assert not report["results"]
    assert {s["check"] for s in report["skipped"]} == {"fit.clipping", "fit.holes"}
    assert "base body" in report["skipped"][0]["reason"]


def test_the_body_must_exist(tmp_path):
    glb = write_skinned_glb(tmp_path / "piece.glb")
    asset = Asset(glb=glb, body=tmp_path / "missing.glb", body_clips=[str(tmp_path / "walk.glb")])
    assert asset.body_clips == (tmp_path / "walk.glb",)
    assert "does not exist" in asset.missing(("body",))


def test_a_strip_on_itself_neither_clips_nor_leaves_holes(tmp_path):
    """The toy rig as both piece and body: rest plus its own clip, measured end to end."""
    body = write_skinned_glb(tmp_path / "body.glb", clips={"bend": [(0.0, [0, 0, 0, 1]), (1.0, [0, 0, 0.38, 0.92])]})
    fit_checks._CACHE.clear()
    measured = fit_checks.measure(Asset(glb=body, body=body, body_clips=[body]), {"frames": 3, "sample": 6})
    assert [p["pose"] for p in measured["poses"]][:2] == ["rest", "bend@0.00"]
    assert len(measured["poses"]) == 4
    assert all(p["clipping"] == pytest.approx(0.0) for p in measured["poses"])


def test_a_held_weapon_has_no_holes_to_measure(tmp_path):
    glb = write_skinned_glb(tmp_path / "weapon.glb")
    body = write_skinned_glb(tmp_path / "body.glb")
    report = run_checks(Asset(glb=glb, body=body), profile="equipment_part", only=["fit.holes"])
    assert not report["results"] and "held in the hand" in report["skipped"][0]["reason"]


def test_the_worn_set_is_the_pieces_beside_the_body(tmp_path):
    for name in ("body_masked", "set", "preview", "body_trimmed", "weapon", "helmet", "torso", "gloves_L"):
        (tmp_path / f"{name}.glb").write_bytes(b"glb")
    found = fit_checks.worn_set(Asset(glb=tmp_path / "helmet.glb", body=tmp_path / "body_masked.glb"))
    assert [p.name for p in found] == ["helmet.glb", "gloves_L.glb", "torso.glb"]
