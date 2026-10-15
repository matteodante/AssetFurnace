"""How a rig's clips deform its mesh: `rig.deformation`.

Every clip of a rig is played in numpy (`image_to_3dlab.glb_skin`) at evenly spaced
times, and three things are measured against the rest pose:

* **stretched edges**: an edge more than `stretch` times its rest length, or shorter than
  `squash` times it, is skin tearing or collapsing;
* **flipped triangles**: a triangle whose normal points against where its own skinning
  carries the rest normal has folded through itself (a candy-wrapper twist, a knee bent
  through the thigh);
* **crossed feet**: with a family file beside the rig (`creature_family`), the left and
  right feet must stay on their own sides. A walk whose feet pass across each other looks
  broken however clean the skin is.

The thresholds are provisional and warn only until a pilot calibrates them: the preset
moves already accepted on generated humanoids measure 0.25-1.2% stretched and 0.3-1.7%
flipped (mostly armpits and hips folding), the werewolf's test walk 0.05% and 0.09%.

The clips are the GLB's own animations, plus, for a `<base>_rigged.glb`, every
`<base>_<clip>.glb` beside it: the files the studio lists as that rig's clips.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from image_to_3dlab import creature_family
from image_to_3dlab.checks.registry import grade_max, register, result, worst

RIGGED = "_rigged"
TAKE_SUFFIX = re.compile(r"_[b-z]$")  # a second take of a preset move (the studio's rule)


def clip_files(glb: Path) -> list[Path]:
    """The GLB itself, then, for a rig, its sibling clip files."""
    glb = Path(glb)
    if not glb.stem.endswith(RIGGED):
        return [glb]
    base = glb.stem.removesuffix(RIGGED)
    siblings = [p for p in sorted(glb.parent.glob(f"{base}_*.glb"))
                if p != glb and not TAKE_SUFFIX.search(p.stem.removeprefix(f"{base}_"))]
    return [glb, *siblings]


def family_for(glb: Path) -> dict | None:
    """The family file of a rig, or of the rig a clip file belongs to."""
    glb = Path(glb)
    found = creature_family.read_family(glb)
    if found is not None:
        return found
    for rigged in glb.parent.glob(f"*{RIGGED}.glb"):
        if glb.stem.startswith(rigged.stem.removesuffix(RIGGED) + "_"):
            return creature_family.read_family(rigged)
    return None


def has_clips(glb: Path) -> bool:
    from image_to_3dlab.glb_turn import parse_glb

    for path in clip_files(glb):
        try:
            if parse_glb(path.read_bytes())[0].get("animations"):
                return True
        except (OSError, ValueError):
            continue
    return False


def feet_swap(left: np.ndarray, right: np.ndarray, axis: np.ndarray) -> bool:
    """True when the left foot ends up on the right of the right foot, or the other way.

    `left` and `right` are (frames, 3) tracks, rest first; only a change of side counts.
    """
    side = (np.asarray(left, float) - np.asarray(right, float)) @ np.asarray(axis, float)
    if side[0] == 0:
        return False
    return bool(np.any(np.sign(side[1:]) * np.sign(side[0]) < 0))


def _edges(faces: np.ndarray) -> np.ndarray:
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(pairs, axis=1), axis=0)


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = vertices[faces]
    return np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])


def measure_clip(rig, animation: int, duration: float, config: dict,
                 feet: list[tuple[str, str]] | None = None, left_axis: np.ndarray | None = None) -> dict:
    """Worst stretched / flipped share over the sampled frames, and which feet crossed."""
    mesh = rig.mesh
    edges = _edges(mesh.faces)
    rest_len = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    keep = rest_len > 1e-9
    rest_normal = _normals(mesh.vertices, mesh.faces)
    keep_faces = np.linalg.norm(rest_normal, axis=1) > 1e-12
    names = {rig.name(j): j for j in rig.joint_nodes}
    rest_world = rig.world(rig.rest_locals())
    tracks = {b: [rest_world[names[b]][:3, 3]] for pair in feet or [] for b in pair if b in names}
    stretched = flipped = 0.0
    for t in np.linspace(0.0, duration, max(2, int(config.get("frames", 24)))):
        world = rig.pose(animation, float(t))
        blend = rig.blend(world)
        posed = rig.deform(world, blend=blend)
        ratio = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1)[keep] / rest_len[keep]
        bad = (ratio > config.get("stretch", 2.0)) | (ratio < config.get("squash", 0.4))
        stretched = max(stretched, 100.0 * float(bad.mean()) if len(bad) else 0.0)
        # where each triangle's own skinning carries its rest normal (its first corner's matrix)
        carried = np.einsum("fij,fj->fi", blend[mesh.faces[:, 0], :3, :3], rest_normal)
        turned = np.einsum("fi,fi->f", _normals(posed, mesh.faces), carried)[keep_faces]
        flipped = max(flipped, 100.0 * float((turned < 0).mean()) if len(turned) else 0.0)
        for bone, track in tracks.items():
            track.append(world[names[bone]][:3, 3])
    crossed = [f"{left}/{right}" for left, right in feet or []
               if left in tracks and right in tracks and feet_swap(tracks[left], tracks[right], left_axis)]
    return {"stretched_percent": round(stretched, 3), "flipped_percent": round(flipped, 3), "crossed": crossed}


@register("rig.deformation", needs=("rig", "animation"))
def deformation(asset, config):
    from image_to_3dlab.glb_skin import SkinnedGLB

    limits = {key: config.get(key, default) for key, default in
              (("stretch_warn", 0.5), ("stretch_fail", 1.5), ("flip_warn", 0.5), ("flip_fail", 1.0))}
    family = family_for(asset.glb)
    feet, left_axis = None, None
    if family and len(family.get("legs", [])) >= 2:
        sides = {s: [leg["bones"][-1] for leg in family["legs"] if leg.get("side") == s] for s in ("L", "R")}
        feet = list(zip(sides["L"], sides["R"]))
        left_axis = np.asarray(family["frame"]["left"], float)
    clips, statuses = {}, []
    for path in clip_files(asset.glb):
        rig = SkinnedGLB(path)
        for index, (name, duration) in enumerate(rig.animations()):
            label = name if Path(path) == Path(asset.glb) else f"{Path(path).name}: {name}"
            found = measure_clip(rig, index, duration, config, feet, left_axis)
            clips[label] = found
            statuses += [grade_max(found["stretched_percent"], limits["stretch_warn"], limits["stretch_fail"]),
                         grade_max(found["flipped_percent"], limits["flip_warn"], limits["flip_fail"]),
                         "fail" if found["crossed"] else "pass"]
    value = {"stretched_percent": max(c["stretched_percent"] for c in clips.values()),
             "flipped_percent": max(c["flipped_percent"] for c in clips.values())}
    crossed = [label for label, c in clips.items() if c["crossed"]]
    message = (f"{len(clips)} clip(s): at worst {value['stretched_percent']:.2f}% of edges stretched, "
               f"{value['flipped_percent']:.2f}% of triangles flipped")
    if crossed:
        message += f"; the feet cross in {', '.join(crossed)}"
    return result("rig.deformation", worst(statuses), value, limits, message, {"clips": clips})
