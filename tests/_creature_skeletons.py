"""Small synthetic skeletons for the creature-family tests, Y up and facing +Z.

Bones are named `bone_N` in shuffled order so nothing can lean on numbering. `werewolf()`
is the real leg and spine layout SkinTokens produced for one (fingers dropped).
"""

import numpy as np


def build(rows, shuffle_seed=0):
    """rows: (role, parent_role, (x, y, z)). Returns (parents, heads, role -> bone name)."""
    order = np.random.default_rng(shuffle_seed).permutation(len(rows))
    names = {role: f"bone_{order[i]}" for i, (role, _, _) in enumerate(rows)}
    parents = {names[r]: (names[p] if p else None) for r, p, _ in rows}
    heads = {names[r]: np.array(pos, float) for r, _, pos in rows}
    return parents, heads, names


def chain(prefix, parent, points):
    rows, prev = [], parent
    for i, point in enumerate(points):
        rows.append((f"{prefix}{i}", prev, point))
        prev = f"{prefix}{i}"
    return rows


def humanoid():
    rows = [("hips", None, (0, 1.0, 0))]
    rows += chain("spine", "hips", [(0, 1.15, 0), (0, 1.35, 0), (0, 1.5, 0), (0, 1.65, 0)])  # spine3 = neck
    for side, x in (("L", 1), ("R", -1)):
        rows += chain(f"arm{side}", "spine1", [(0.15 * x, 1.42, 0), (0.4 * x, 1.42, 0), (0.65 * x, 1.42, 0)])
        rows += chain(f"leg{side}", "hips", [(0.1 * x, 0.95, 0), (0.1 * x, 0.5, 0.02), (0.1 * x, 0.08, 0),
                                              (0.1 * x, 0.0, 0.15)])
    return build(rows)


def werewolf():
    """Real joint heads of a SkinTokens werewolf rig (glTF units), fingers dropped."""
    rows = [("root", None, (0.001, 0.049, 0.082))]
    rows += chain("spine", "root", [(0.001, 0.1, 0.1), (0.001, 0.166, 0.118), (0.001, 0.239, 0.137),
                                    (0.001, 0.322, 0.158), (0.001, 0.344, 0.228)])
    rows += chain("armL", "spine2", [(0.041, 0.308, 0.147), (0.122, 0.282, 0.122), (0.194, 0.25, 0.067),
                                     (0.304, 0.209, 0.071)])
    rows += chain("armR", "spine2", [(-0.042, 0.308, 0.151), (-0.123, 0.282, 0.14), (-0.192, 0.242, 0.067),
                                     (-0.305, 0.202, 0.045)])
    rows += chain("legL", "root", [(0.06, 0.02, 0.075), (0.107, -0.173, 0.056), (0.133, -0.337, -0.064),
                                   (0.158, -0.414, 0.02)])
    rows += chain("legR", "root", [(-0.057, 0.02, 0.075), (-0.093, -0.173, 0.042), (-0.079, -0.337, -0.079),
                                   (-0.152, -0.414, -0.006)])
    return build(rows, shuffle_seed=3)


def quadruped(wings=False, tail=True):
    """A dog-ish animal facing +Z: hips at the back, chest at the front."""
    rows = [("hips", None, (0, 0.6, -0.4))]
    rows += chain("spine", "hips", [(0, 0.62, -0.1), (0, 0.63, 0.25), (0, 0.8, 0.45), (0, 0.85, 0.65)])
    for side, x in (("L", 1), ("R", -1)):
        rows += chain(f"front{side}", "spine1", [(0.12 * x, 0.55, 0.25), (0.12 * x, 0.3, 0.22),
                                                   (0.12 * x, 0.06, 0.24), (0.12 * x, 0.0, 0.32)])
        rows += chain(f"back{side}", "hips", [(0.12 * x, 0.55, -0.42), (0.12 * x, 0.32, -0.32),
                                                (0.12 * x, 0.1, -0.48), (0.12 * x, 0.0, -0.4)])
        if wings:
            rows += chain(f"wing{side}", "spine1", [(0.1 * x, 0.75, 0.2), (0.45 * x, 0.9, 0.1),
                                                      (0.8 * x, 1.0, 0.0), (1.1 * x, 0.95, -0.15)])
    if tail:
        rows += chain("tail", "hips", [(0, 0.58, -0.6), (0, 0.5, -0.8), (0, 0.42, -1.0)])
    return build(rows)


def spider(legs=8):
    rows = [("body", None, (0, 0.35, 0)), ("abdomen", "body", (0, 0.4, -0.35)), ("head", "body", (0, 0.36, 0.2))]
    for k in range(legs // 2):
        z = 0.15 - 0.1 * k
        for side, x in (("L", 1), ("R", -1)):
            rows += chain(f"leg{side}{k}", "body", [(0.08 * x, 0.36, z), (0.35 * x, 0.6, z * 1.5),
                                                      (0.6 * x, 0.25, z * 2), (0.7 * x, 0.0, z * 2.2)])
    return build(rows)


def snake():
    rows = chain("seg", None, [(0, 0.05, 0.9 - 0.15 * i + 0.0) for i in range(12)])
    rows[0] = ("seg0", None, (0, 0.25, 0.95))  # the head is raised a little
    return build(rows)


def plant():
    rows = chain("stem", None, [(0, 0.0, 0), (0, 0.3, 0.02), (0, 0.6, 0.05), (0, 0.9, 0.1), (0, 1.1, 0.2)])
    rows += chain("leafL", "stem1", [(0.1, 0.32, 0), (0.3, 0.25, 0)])
    rows += chain("leafR", "stem1", [(-0.1, 0.32, 0), (-0.3, 0.25, 0)])
    rows += chain("root", "stem0", [(0.15, -0.02, 0.05), (0.3, -0.03, 0.1)])
    return build(rows)
