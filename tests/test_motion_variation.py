"""Motion variation by example (a numpy port of GenMM's patch synthesis)."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_3dlab import motion_variation as mv


def _wave(t=60, channels=4, period=20):
    k = np.arange(t)
    return np.stack([np.sin(2 * np.pi * (k / period + c / channels)) for c in range(channels)])


@pytest.mark.parametrize("loop", [False, True])
def test_patches_put_back_together_give_the_motion_back(loop):
    x = np.random.default_rng(0).normal(size=(3, 40))
    patches = mv.extract_patches(x, 7, loop=loop)
    assert np.allclose(mv.combine_patches(patches, x.shape, 7, loop=loop), x)


def test_the_exemplar_itself_is_found_again_without_noise():
    x = _wave()
    out = mv.synthesize(x, x.shape[1], np.random.default_rng(0), noise=0.0)
    assert out.shape == x.shape
    assert np.abs(out - x).max() < 0.05


def test_a_longer_variation_is_made_of_the_exemplars_own_pieces():
    x = _wave(t=80)
    out = mv.synthesize(x, 160, np.random.default_rng(1))
    assert out.shape == (4, 160)
    ours = mv.extract_patches(out, 11)
    theirs = mv.extract_patches(x, 11, loop=True)
    gap = ((ours[:, None, :] - theirs[None, :, :]) ** 2).mean(-1).min(1)
    power = float((x ** 2).mean())
    assert gap.mean() < 0.02 * power  # almost everywhere a piece of the example
    assert gap.max() < 0.1 * power  # and where two pieces meet, an average of them


def test_a_seed_gives_the_same_variation_and_another_seed_a_different_one():
    x = np.random.default_rng(5).normal(size=(4, 60)).cumsum(1) / 5
    a = mv.synthesize(x, 120, np.random.default_rng(3))
    b = mv.synthesize(x, 120, np.random.default_rng(3))
    c = mv.synthesize(x, 120, np.random.default_rng(4))
    assert np.array_equal(a, b) and not np.allclose(a, c)


def _chain_poses(frames=48):
    """A root walking forward along +Z with a two-bone arm swinging: world matrices."""
    parents = {"root": None, "a": "root", "b": "a"}
    offsets = {"a": np.array([0, 0.5, 0]), "b": np.array([0, 0.5, 0])}

    def world(angle, z):
        c, s = np.cos(angle), np.sin(angle)
        r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        out = {"root": np.eye(4)}
        out["root"][:3, 3] = [0, 1.0, z]
        for b in ("a", "b"):
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = r, offsets[b]
            out[b] = out[parents[b]] @ m
        return out

    rest = world(0.0, 0.0)
    poses = [world(0.6 * np.sin(2 * np.pi * k / 16), 0.02 * k) for k in range(frames)]
    return parents, rest, poses


def test_a_varied_clip_is_a_valid_pose_sequence_of_the_same_skeleton():
    parents, rest, poses = _chain_poses()
    out = mv.vary(parents, rest, poses, up=np.array([0, 1.0, 0]), length=96, seed=0)
    assert len(out) == 96 and set(out[0]) == set(rest)
    for pose in out:
        r = pose["a"][:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(r), 1.0)
        bone = pose["b"][:3, 3] - pose["a"][:3, 3]
        assert np.isclose(np.linalg.norm(bone), 0.5)  # bones keep their length
        assert abs(pose["root"][1, 3] - 1.0) < 1e-6  # the height is kept
    forward = out[-1]["root"][2, 3] - out[0]["root"][2, 3]
    assert 0.02 * 95 * 0.7 < forward < 0.02 * 95 * 1.3  # and so is the pace


def test_a_bone_that_moves_under_a_still_helper_root_keeps_moving():
    """Mesh2Motion keeps its root still and moves the bone below it: the fox's hips bounce
    as it runs, the snake's head sweeps from side to side. Rebuilt from rest offsets, the
    longer take lost all of it."""
    parents = {"root": None, "hips": "root", "tail": "hips"}

    def world(k):
        out = {"root": np.eye(4)}
        hips = np.eye(4)
        hips[:3, 3] = [0.3 * np.sin(2 * np.pi * k / 12), 1.0 + 0.1 * np.cos(2 * np.pi * k / 12), 0.0]
        out["hips"] = out["root"] @ hips
        tail = np.eye(4)
        tail[:3, 3] = [0, 0, -0.5]
        out["tail"] = out["hips"] @ tail
        return out

    rest, poses = world(0), [world(k) for k in range(36)]
    out = mv.vary(parents, rest, poses, up=np.array([0, 1.0, 0]), length=72, seed=0, loop=True)
    side = np.array([p["hips"][0, 3] for p in out])
    height = np.array([p["hips"][1, 3] for p in out])
    assert np.ptp(side) > 0.5 and np.ptp(height) > 0.15  # the source sweeps 0.6 and bobs 0.2
    tail = [np.linalg.norm(p["tail"][:3, 3] - p["hips"][:3, 3]) for p in out]
    assert np.allclose(tail, 0.5)  # a bone that does not move keeps its length
