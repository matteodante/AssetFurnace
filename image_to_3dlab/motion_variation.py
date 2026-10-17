"""New motion from one example clip, by patches: a numpy port of GenMM's synthesis.

GenMM (Li et al., SIGGRAPH 2023, https://github.com/wyysf-98/GenMM, Apache-2.0) makes a
longer or different take of a clip without any trained model: it builds the clip at a
few time resolutions, starts from a noisy stretched copy of the coarsest, and at each
resolution replaces every short window ("patch") of the new motion by its nearest window
in the example, averaging where windows overlap. The result is made of the example's own
pieces in a new order, so it moves like the example and loops where asked.

Ported rather than vendored: the upstream `main` no longer runs (`GenMM.KEYFRAME_INDICES`
is used but never defined since commit b2b1539), the code needs torch and unfoldNd, and
its folder also holds a GPL-3 Blender add-on. This port follows `GenMM.py`,
`nearest_neighbor/` (itself from Efficient-GPNN, Apache-2.0) and `configs/default.yaml`
at commit fae65cd, with these changes: numpy instead of torch, one example clip, and the
two pyramids aligned so the result has exactly the length asked for.

`vary` applies it to a skeleton's poses: each bone's turn relative to its parent (6D), and
for each root its height and its step along the ground.
"""

from __future__ import annotations

import numpy as np

# GenMM's defaults (configs/default.yaml)
PATCH = 11
ALPHA = 0.01  # completeness against diversity: smaller uses more of the example
STEPS = 3
NOISE = 10.0
PYRAMID = 0.75
COARSE = 5.0  # the coarsest level is this many patches long
# A bone whose offset from its parent changes by more than this (metres) over a clip moves.
MOVES = 1e-4


def pyramid_lengths(final: int, coarse_ratio: float, factor: float = PYRAMID) -> list[int]:
    lengths = [int(np.round(final * coarse_ratio))]
    while lengths[-1] < final:
        lengths.append(int(np.round(lengths[-1] / factor)))
        if lengths[-1] == lengths[-2]:
            lengths[-1] += 1
    lengths[-1] = final
    return lengths


def resample(x: np.ndarray, length: int) -> np.ndarray:
    """(C, T) -> (C, length), linear, as torch's `interpolate(align_corners=False)`."""
    t = x.shape[1]
    pos = np.clip((np.arange(length) + 0.5) * t / length - 0.5, 0, t - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    u = pos - lo
    return x[:, lo] * (1 - u) + x[:, hi] * u


def extract_patches(x: np.ndarray, size: int, loop: bool = False) -> np.ndarray:
    """(C, T) -> (windows, C * size). A looping clip also gets the windows across its seam."""
    if loop:
        half = size // 2
        x = np.concatenate([x[:, -half:], x, x[:, :half]], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(x, size, axis=1)  # (C, W, size)
    return np.ascontiguousarray(windows.transpose(1, 0, 2)).reshape(windows.shape[1], -1)


def combine_patches(patches: np.ndarray, shape: tuple[int, int], size: int, loop: bool = False) -> np.ndarray:
    """Put windows back in place, averaging where they overlap."""
    c, t = shape
    half = size // 2
    full = t + 2 * half if loop else t
    out = np.zeros((c, full))
    count = np.zeros(full)
    for start, patch in enumerate(patches.reshape(len(patches), c, size)):
        out[:, start:start + size] += patch
        count[start:start + size] += 1
    out /= count
    if loop:  # the padding on each side is the other end of the clip: fold it back
        front, tail = out[:, :half].copy(), out[:, -half:].copy()
        out[:, half:2 * half] = (out[:, half:2 * half] + tail) / 2
        out[:, -2 * half:-half] = (front + out[:, -2 * half:-half]) / 2
        out = out[:, half:-half]
    return out


def nearest(x: np.ndarray, y: np.ndarray, alpha: float | None = ALPHA) -> np.ndarray:
    """For each window of x, the index of its nearest window of y. With `alpha`, distances
    are divided by how close y's window is to anything in x, so every part of the example
    tends to be used (GenMM's completeness term)."""
    dist = ((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * x @ y.T) / x.shape[1]
    if alpha is not None:
        dist = dist / (alpha + dist.min(0)[None, :])
    return dist.argmin(1)


def synthesize(example: np.ndarray, length: int, rng: np.random.Generator, patch: int = PATCH,
               alpha: float | None = ALPHA, steps: int = STEPS, noise: float = NOISE,
               factor: float = PYRAMID, coarse: float = COARSE, loop: bool = False) -> np.ndarray:
    """A new (C, length) motion made of `example`'s (C, T) windows. A window is at most half
    the example (odd, at least 3): on a 14-frame run, windows of 11 frames averaged out of
    phase halved its swing."""
    t = example.shape[1]
    patch = max(3, min(patch, (t // 2) - (1 - (t // 2) % 2)))
    ratio = patch * coarse / t
    mine, theirs = pyramid_lengths(length, ratio, factor), pyramid_lengths(t, ratio, factor)
    levels = min(len(mine), len(theirs))
    mine = [max(n, patch) for n in mine[-levels:]]
    theirs = [max(n, patch) for n in theirs[-levels:]]
    out = resample(resample(example, theirs[0]), mine[0])
    if noise > 0:
        out = np.fmod(out + rng.normal(size=out.shape) * noise, 1.0)
    for level, (n, m) in enumerate(zip(mine, theirs)):
        if level:
            out = resample(out, n)
        target = extract_patches(resample(example, m), patch)
        for _ in range(steps):
            idx = nearest(extract_patches(out, patch, loop), target, alpha)
            out = combine_patches(target[idx], out.shape, patch, loop)
    return out


# -- poses ----------------------------------------------------------------------------------

def _order(parents: dict[str, str | None], bones) -> list[str]:
    bones = set(bones)
    out = [b for b in bones if parents.get(b) not in bones]
    i = 0
    while i < len(out):
        out += sorted(c for c in bones if parents.get(c) == out[i])
        i += 1
    return out


def _six(r: np.ndarray) -> np.ndarray:
    return r[:, :2].T.reshape(6)


def _from_six(v: np.ndarray) -> np.ndarray:
    a, b = v[:3], v[3:]
    e1 = a / max(np.linalg.norm(a), 1e-12)
    b = b - (e1 @ b) * e1
    e2 = b / max(np.linalg.norm(b), 1e-12)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def vary(parents: dict[str, str | None], rest: dict[str, np.ndarray], poses: list[dict[str, np.ndarray]],
         up: np.ndarray, length: int, seed: int = 0, loop: bool = False) -> list[dict[str, np.ndarray]]:
    """`length` new frames of world matrices for the bones of `rest`, varied from `poses`.

    Roots keep the example's heights and pace along the ground. A bone that moves against
    its parent in the example (Mesh2Motion keeps its root still and moves the bone below:
    a fox's hips bounce, a snake's head sweeps) keeps that motion too; every other bone
    keeps its rest offset, so its length."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    order = _order(parents, rest)
    roots = [b for b in order if parents.get(b) not in rest]

    def local(pose, b):
        p = parents.get(b)
        return pose[b] if p not in rest else np.linalg.inv(pose[p]) @ pose[b]

    offsets = {b: local(rest, b)[:3, 3] for b in order}
    tracks = {b: np.array([local(pose, b)[:3, 3] for pose in poses]) for b in order if b not in roots}
    moving = [b for b, t in tracks.items() if np.ptp(t, axis=0).max() > MOVES]
    rows = []
    for pose in poses:
        row = [_six(local(pose, b)[:3, :3]) for b in order]
        for b in roots:
            p = pose[b][:3, 3]
            row.append(np.append(p - (p @ up) * up, p @ up))
        rows.append(np.concatenate(row))
    features = np.array(rows).T  # (C, T)
    n_rot = 6 * len(order)
    n_root = 4 * len(roots)
    ground = features[n_rot:n_rot + n_root].reshape(len(roots), 4, -1)
    step = np.diff(ground[:, :3], axis=2)
    step = np.concatenate([step[:, :, :1] if step.shape[2] else np.zeros_like(ground[:, :3, :1]), step], axis=2)
    features[n_rot:n_rot + n_root] = np.concatenate([step, ground[:, 3:]], axis=1).reshape(-1, features.shape[1])
    if moving:
        features = np.concatenate([features, np.concatenate([tracks[b].T for b in moving])])

    made = synthesize(features, length, np.random.default_rng(seed), loop=loop)

    start = ground[:, :3, 0]
    moved = made[n_rot:n_rot + n_root].reshape(len(roots), 4, -1)
    where = start[:, :, None] + np.cumsum(moved[:, :3], axis=2) - moved[:, :3, :1]
    shifted = {b: made[n_rot + n_root + 3 * i:n_rot + n_root + 3 * i + 3] for i, b in enumerate(moving)}
    out = []
    for k in range(length):
        pose = {}
        for i, b in enumerate(order):
            m = np.eye(4)
            m[:3, :3] = _from_six(made[6 * i:6 * i + 6, k])
            if b in roots:
                j = roots.index(b)
                m[:3, 3] = where[j, :, k] + moved[j, 3, k] * up
                pose[b] = m
            else:
                m[:3, 3] = shifted[b][:, k] if b in shifted else offsets[b]
                pose[b] = pose[parents[b]] @ m
        out.append(pose)
    return out
