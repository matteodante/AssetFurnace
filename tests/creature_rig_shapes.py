"""Small creatures of boxes and rods for the creature auto-rig tests, in the rig's frame:
Z up, facing -Y, left on +X, standing on Z = 0, about 1 tall.

Each shape is a list of (verts, faces) parts; `work_folder` writes them as the prepare
step's `shell.npz`. Draws are joint trees written the way the Blender reader writes them
(`draws/draw{k}.json`): names, parent indices, heads and tails in the model frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

Part = tuple[np.ndarray, np.ndarray]


def box(lo, hi, edge: float = 0.03) -> Part:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, edge)
    return np.asarray(v), np.asarray(f)


def rod(a, b, radius: float, edge: float = 0.02) -> Part:
    m = trimesh.creation.cylinder(radius=radius, segment=[a, b], sections=10)
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, edge)
    return np.asarray(v), np.asarray(f)


def sheet_cone(z_top: float, r_top: float, r_hem: float, rings: int = 30, around: int = 72) -> Part:
    """An open cone of one-sided cloth from a waist down to a hem on the floor: a robe."""
    t = np.linspace(0, 1, rings)
    a = np.linspace(0, 2 * np.pi, around, endpoint=False)
    r = r_top + (r_hem - r_top) * t
    z = z_top * (1 - t)
    v = np.stack([np.outer(r, np.cos(a)), np.outer(r, np.sin(a)), np.repeat(z[:, None], around, 1)], -1).reshape(-1, 3)
    faces = []
    for i in range(rings - 1):
        for j in range(around):
            p, q = i * around + j, i * around + (j + 1) % around
            faces += [[p, q, p + around], [q, q + around, p + around]]
    return v, np.array(faces)


def mirrored(parts: list[Part]) -> list[Part]:
    return parts + [(v * [-1.0, 1.0, 1.0], f[:, ::-1]) for v, f in parts]


# --- bodies ------------------------------------------------------------------------------


def quadruped(mouth: bool = False) -> list[Part]:
    """Body, four legs, a neck and head in front (-Y), a tail behind. Height 1.05."""
    parts = [box([-0.15, -0.5, 0.45], [0.15, 0.5, 0.75]),
             box([-0.075, -0.65, 0.7], [0.075, -0.45, 1.0]),
             box([-0.03, 0.5, 0.57], [0.03, 1.0, 0.63])]
    if mouth:   # a skull and a lower jaw joined at the back, open to the front
        parts += [box([-0.075, -0.95, 0.94], [0.075, -0.6, 1.05]),
                  box([-0.06, -0.92, 0.82], [0.06, -0.6, 0.87]),
                  box([-0.075, -0.72, 0.82], [0.075, -0.6, 1.05])]
    else:
        parts.append(box([-0.075, -0.95, 0.85], [0.075, -0.6, 1.05]))
    for y in (-0.38, 0.38):
        parts.append(box([0.05, y - 0.05, 0.0], [0.15, y + 0.05, 0.5]))
        parts.append(box([-0.15, y - 0.05, 0.0], [-0.05, y + 0.05, 0.5]))
    return parts


def biped(wings: bool = False) -> list[Part]:
    """Two legs with feet pointing forward, a torso, arms held out (or flat wings), a head.
    Height 1.2."""
    parts = [box([-0.2, -0.1, 0.5], [0.2, 0.1, 0.95]), box([-0.09, -0.1, 1.0], [0.09, 0.1, 1.2]),
             box([-0.05, -0.05, 0.94], [0.05, 0.05, 1.01])]
    side = [box([0.06, -0.06, 0.05], [0.18, 0.06, 0.52]), box([0.06, -0.2, 0.0], [0.18, 0.06, 0.06])]
    if wings:
        side.append(box([0.19, -0.15, 0.86], [1.0, 0.3, 0.89]))
    else:
        side.append(box([0.19, -0.04, 0.83], [0.65, 0.04, 0.91]))
    return parts + mirrored(side)


def spider() -> list[Part]:
    """A body on eight thin legs that bend up and down to the floor. Height about 0.6."""
    parts = [box([-0.15, -0.25, 0.25], [0.15, 0.25, 0.45])]
    side = []
    for y in (-0.2, -0.07, 0.07, 0.2):
        root = np.array([0.13, y, 0.35])
        knee = np.array([0.42, y * 1.8, 0.58])
        foot = np.array([0.72, y * 2.6, 0.02])
        side += [rod(root, knee, 0.025), rod(knee, foot, 0.025)]
    return parts + mirrored(side)


def robed(legs: bool = False) -> list[Part]:
    """An upper body over a cloth robe whose hem lies on the floor, optionally with legs
    standing inside it."""
    parts = [box([-0.15, -0.1, 0.6], [0.15, 0.1, 1.0]), box([-0.08, -0.08, 1.02], [0.08, 0.08, 1.2]),
             box([-0.04, -0.04, 0.98], [0.04, 0.04, 1.03]), sheet_cone(0.62, 0.14, 0.4)]
    if legs:
        parts += mirrored([box([0.07, -0.05, 0.0], [0.13, 0.05, 0.62])])
    return parts


def work_folder(work: Path, parts: list[Part]) -> Path:
    """`shell.npz` as the prepare step writes it, the parts serving as shell and model."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    verts, faces, n = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + n)
        n += len(v)
    v, f = np.concatenate(verts).astype(np.float32), np.concatenate(faces).astype(np.int32)
    height = float(v[:, 2].max() - v[:, 2].min())
    np.savez(work / "shell.npz", shell_verts=v, shell_faces=f, model_verts=v, model_faces=f,
             height=height, turn_deg=0.0)
    return work


# --- draws -------------------------------------------------------------------------------


def draw(joints: dict[str, tuple], links: list[tuple[str, str | None]], ends: dict[str, tuple]) -> dict:
    """A draw from joints: `links` (joint, parent joint) make one bone per joint, its tail
    at its only child or at `ends[joint]` for a joint without children."""
    names = [j for j, _ in links]
    index = {n: i for i, n in enumerate(names)}
    kids: dict[str, list[str]] = {}
    for j, p in links:
        if p is not None:
            kids.setdefault(p, []).append(j)
    heads = [list(map(float, joints[n])) for n in names]
    tails = []
    for n in names:
        if n in ends:
            tails.append(list(map(float, ends[n])))
        else:
            tails.append(list(map(float, np.mean([joints[c] for c in kids[n]], axis=0))))
    return {"names": names, "parents": [index[p] if p is not None else -1 for _, p in links],
            "heads": heads, "tails": tails}


def side_joints(base: dict[str, tuple]) -> dict[str, tuple]:
    """`name.L` joints given on the left (+X), with their `.R` mirror images."""
    out = dict(base)
    for n, p in base.items():
        if n.endswith(".L"):
            out[n[:-2] + ".R"] = (-p[0], p[1], p[2])
    return out


def side_links(links: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    out = list(links)
    for j, p in links:
        if j.endswith(".L"):
            out.append((j[:-2] + ".R", p[:-2] + ".R" if p and p.endswith(".L") else p))
    return out


QUAD_JOINTS = {
    "hips": (0, 0.35, 0.62), "back": (0, 0.0, 0.62), "chest": (0, -0.38, 0.62), "neck": (0, -0.55, 0.8),
    "head": (0, -0.65, 0.95), "tail1": (0, 0.5, 0.6), "tail2": (0, 0.75, 0.6),
    "hip.L": (0.1, 0.38, 0.55), "knee.L": (0.1, 0.38, 0.36), "hock.L": (0.1, 0.38, 0.16), "hpaw.L": (0.1, 0.38, 0.05),
    "shoulder.L": (0.1, -0.38, 0.55), "elbow.L": (0.1, -0.38, 0.36), "wrist.L": (0.1, -0.38, 0.16),
    "fpaw.L": (0.1, -0.38, 0.05),
}
QUAD_LINKS = [("hips", None), ("back", "hips"), ("chest", "back"), ("neck", "chest"), ("head", "neck"),
              ("tail1", "hips"), ("tail2", "tail1"),
              ("hip.L", "hips"), ("knee.L", "hip.L"), ("hock.L", "knee.L"), ("hpaw.L", "hock.L"),
              ("shoulder.L", "chest"), ("elbow.L", "shoulder.L"), ("wrist.L", "elbow.L"), ("fpaw.L", "wrist.L")]
QUAD_ENDS = {"head": (0, -0.92, 0.95), "tail2": (0, 0.97, 0.6),
             "hpaw.L": (0.1, 0.4, 0.01), "fpaw.L": (0.1, -0.4, 0.01)}


def quadruped_draw(drop: tuple[str, ...] = (), move: dict[str, tuple] | None = None,
                   extra: dict[str, tuple[tuple, str]] | None = None, ends: dict[str, tuple] | None = None) -> dict:
    """A clean draw of `quadruped()`; `drop` takes joints (and everything below them) out,
    `move` puts joints elsewhere, `extra` adds {joint: (position, parent)}."""
    joints = side_joints(QUAD_JOINTS)
    links = side_links(QUAD_LINKS)
    tips = {**QUAD_ENDS, **{n[:-2] + ".R": (-p[0], p[1], p[2]) for n, p in QUAD_ENDS.items() if n.endswith(".L")}}
    for n, p in (extra or {}).items():
        joints[n] = p[0]
        links.append((n, p[1]))
    joints.update(move or {})
    tips.update(ends or {})
    gone = set(drop)
    changed = True
    while changed:
        changed = False
        for j, p in links:
            if p in gone and j not in gone:
                gone.add(j)
                changed = True
    links = [(j, p) for j, p in links if j not in gone]
    kids = {p for _, p in links}
    for j, _ in links:
        if j not in kids and j not in tips:
            tips[j] = joints[j]
    return draw(joints, links, {j: tips[j] for j, _ in links if j not in kids or j in tips})


BIPED_JOINTS = {
    "hips": (0, 0.0, 0.55), "spine": (0, 0.0, 0.75), "chest": (0, 0.0, 0.9), "neck": (0, 0.0, 1.0),
    "head": (0, 0.0, 1.08),
    "hip.L": (0.12, 0.0, 0.5), "knee.L": (0.12, 0.0, 0.28), "ankle.L": (0.12, 0.0, 0.05),
    "toe.L": (0.12, -0.12, 0.03),
    "shoulder.L": (0.22, 0.0, 0.87), "elbow.L": (0.42, 0.0, 0.87), "wrist.L": (0.6, 0.0, 0.87),
}
BIPED_LINKS = [("hips", None), ("spine", "hips"), ("chest", "spine"), ("neck", "chest"), ("head", "neck"),
               ("hip.L", "hips"), ("knee.L", "hip.L"), ("ankle.L", "knee.L"), ("toe.L", "ankle.L"),
               ("shoulder.L", "chest"), ("elbow.L", "shoulder.L"), ("wrist.L", "elbow.L")]
BIPED_ENDS = {"head": (0, -0.02, 1.19), "toe.L": (0.12, -0.19, 0.03), "wrist.L": (0.64, 0.0, 0.87)}


def biped_draw(fingers: bool = False, wings: bool = False) -> dict:
    """A clean draw of `biped()`; `fingers` gives each hand four two-bone fingers;
    `wings` stretches the arm chain along the wing to its tip."""
    joints = side_joints(BIPED_JOINTS)
    links = side_links(BIPED_LINKS)
    ends = dict(BIPED_ENDS)
    if wings:
        joints.update(side_joints({"elbow.L": (0.5, 0.05, 0.875), "wrist.L": (0.8, 0.05, 0.875)}))
        ends["wrist.L"] = (0.98, 0.05, 0.875)
    if fingers:
        del ends["wrist.L"]
        for k, y in enumerate((-0.03, -0.01, 0.01, 0.03)):
            a, b = f"f{k}a.L", f"f{k}b.L"
            joints[a] = (0.63, y, 0.87)
            joints[b] = (0.66, y, 0.87)
            links += [(a, "wrist.L"), (b, a)]
            ends[b] = (0.69, y, 0.87)
        joints = side_joints(joints)
        links = side_links([lk for lk in links if not lk[0].endswith(".R")])
    tips = {**ends, **{n[:-2] + ".R": (-p[0], p[1], p[2]) for n, p in ends.items() if n.endswith(".L")}}
    return draw(joints, links, tips)


def write_draws(work: Path, draws: list[dict]) -> list[Path]:
    folder = Path(work) / "draws"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, d in enumerate(draws):
        p = folder / f"draw{k}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        paths.append(p)
    return paths
