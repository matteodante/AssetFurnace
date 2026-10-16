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


def humanoid(a_pose=False):
    """A T-pose, or with `a_pose` arms hanging 45 degrees down."""
    rows = [("hips", None, (0, 1.0, 0))]
    rows += chain("spine", "hips", [(0, 1.15, 0), (0, 1.35, 0), (0, 1.5, 0), (0, 1.65, 0)])  # spine3 = neck
    drop = 1.0 if a_pose else 0.0
    for side, x in (("L", 1), ("R", -1)):
        rows += chain(f"arm{side}", "spine1", [(0.15 * x, 1.42, 0), (0.33 * x, 1.42 - 0.18 * drop, 0),
                                                (0.5 * x, 1.42 - 0.35 * drop, 0)])
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


def quadruped(wings=False, tail=True, hub=False):
    """A dog-ish animal facing +Z: hips at the back, chest at the front. With `hub`, all
    four legs hang from the hips joint, as an auto-rig sometimes builds them."""
    rows = [("hips", None, (0, 0.6, -0.4))]
    rows += chain("spine", "hips", [(0, 0.62, -0.1), (0, 0.63, 0.25), (0, 0.8, 0.45), (0, 0.85, 0.65)])
    for side, x in (("L", 1), ("R", -1)):
        rows += chain(f"front{side}", "hips" if hub else "spine1", [(0.12 * x, 0.55, 0.25), (0.12 * x, 0.3, 0.22),
                                                   (0.12 * x, 0.06, 0.24), (0.12 * x, 0.0, 0.32)])
        rows += chain(f"back{side}", "hips", [(0.12 * x, 0.55, -0.42), (0.12 * x, 0.32, -0.32),
                                                (0.12 * x, 0.1, -0.48), (0.12 * x, 0.0, -0.4)])
        if wings:
            rows += chain(f"wing{side}", "spine1", [(0.1 * x, 0.75, 0.2), (0.45 * x, 0.9, 0.1),
                                                      (0.8 * x, 1.0, 0.0), (1.1 * x, 0.95, -0.15)])
    if tail:
        rows += chain("tail", "hips", [(0, 0.58, -0.6), (0, 0.5, -0.8), (0, 0.42, -1.0)])
    return build(rows)


def spider(legs=8, lifted=(), lift=0.15, splay=0.0, palps=False, twin=None):
    """`lifted`: legs (like "R1") whose tips stop `lift` above the ground, as a generated
    spider's do. `splay` moves every foot back (+) or forward (-). `palps` adds two short
    feelers reaching forward and down from the head. `twin` (a leg number) doubles that
    left leg's chain, a centimetre off."""
    rows = [("body", None, (0, 0.35, 0)), ("abdomen", "body", (0, 0.4, -0.35)), ("head", "body", (0, 0.36, 0.2))]
    for k in range(legs // 2):
        z = 0.15 - 0.1 * k
        for side, x in (("L", 1), ("R", -1)):
            tip = lift if f"{side}{k}" in lifted else 0.0
            rows += chain(f"leg{side}{k}", "body", [(0.08 * x, 0.36, z), (0.35 * x, 0.6, z * 1.5),
                                                      (0.6 * x, 0.25 + tip, z * 2 - splay / 2),
                                                      (0.7 * x, tip, z * 2.2 - splay)])
    if twin:  # the auto-rig sometimes puts two chains inside one leg, side by side
        x, z = 1, 0.15 - 0.1 * twin
        rows += chain("twin", "body", [(0.08 * x, 0.37, z + 0.01), (0.35 * x, 0.61, z * 1.5 + 0.01),
                                       (0.6 * x, 0.26, z * 2 + 0.01), (0.7 * x, 0.0, z * 2.2 + 0.01)])
    if palps:
        for side, x in (("L", 1), ("R", -1)):
            rows += chain(f"palp{side}", "head", [(0.05 * x, 0.34, 0.26), (0.12 * x, 0.25, 0.34),
                                                    (0.16 * x, 0.12, 0.4)])
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


def reroot(skeleton, new_root):
    """The same skeleton with `new_root` (a role) as its root: SkinTokens may root a snake at
    its head, its tail or its middle."""
    parents, heads, names = skeleton
    root = names[new_root]
    links = {b: set() for b in parents}
    for b, p in parents.items():
        if p is not None:
            links[b].add(p)
            links[p].add(b)
    out, todo = {root: None}, [root]
    while todo:
        b = todo.pop()
        for n in links[b]:
            if n not in out:
                out[n] = b
                todo.append(n)
    return out, heads, names


def flat_snake():
    """The real chain SkinTokens gave a generated serpent lying stretched out along X: 13
    joints, a few centimetres of height over almost a metre, head (bone 0) at -X."""
    points = [(-0.396, -0.074, 0.228), (-0.371, -0.041, 0.174), (-0.331, -0.03, 0.101), (-0.295, -0.059, 0.029),
              (-0.255, -0.092, -0.036), (-0.19, -0.102, -0.109), (-0.092, -0.066, -0.177), (0.002, -0.063, -0.199),
              (0.096, -0.077, -0.196), (0.187, -0.095, -0.188), (0.281, -0.084, -0.174), (0.357, -0.081, -0.156),
              (0.419, -0.088, -0.138)]
    return build(chain("seg", None, points), shuffle_seed=5)
