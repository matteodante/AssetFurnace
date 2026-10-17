"""How a fitted armour piece sits on its base body in motion: `fit.clipping` and `fit.holes`.

The piece and the masked body (`body_masked.glb` from the fit step, its vertex alpha 0
where a piece covers the skin) share one skeleton. Both are posed in numpy with the body's
own clips (`Asset.body_clips`, Kimodo clips retargeted onto that body), matched bone by
bone by name, at rest and at evenly spaced times of each clip.

* **fit.clipping**: the share of the piece's outer skin (vertices facing the same way as
  the body under them) that sinks into skin the mask leaves visible. Buried inner faces of
  a solid piece and skin hidden under armour do not count. The worst pose is the value.
* **fit.holes**: of the body this piece covers at rest and the mask hides (the outward ray
  from hidden skin meets the piece within `gap`), the share that a pose leaves bare: the ray
  misses every worn piece of the set within `reach`. Skin a shoulder piece lifts off but the
  torso still covers does not show, so it is not a hole. The set is the other piece GLBs beside
  the masked body (the fit step's folder), weapons and shields left out.

Geometry only, measured by rays and nearest points: no pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_to_3dlab.checks.registry import grade_max, register, result

_CACHE: dict = {}


def _limits(config: dict, warn: float, fail: float) -> dict:
    return {"warn": config.get("warn", warn), "fail": config.get("fail", fail)}


def world_by_name(rig, named: dict[str, np.ndarray]) -> list[np.ndarray]:
    """`rig`'s node world matrices at rest, with every joint found in `named` replaced."""
    world = list(rig.world(rig.rest_locals()))
    for j in rig.joint_nodes:
        m = named.get(rig.name(j))
        if m is not None:
            world[j] = m
    return world


def poses(clips: list[Path], frames: int) -> list[tuple[str, dict[str, np.ndarray]]]:
    """("clip@time", {joint name: world matrix}) for evenly spaced times of every clip."""
    from image_to_3dlab.glb_skin import SkinnedGLB

    out = []
    for path in clips:
        rig = SkinnedGLB(path)
        for index, (name, duration) in enumerate(rig.animations()):
            for t in np.linspace(0.0, duration, max(frames, 1), endpoint=False) if duration > 0 else [0.0]:
                world = rig.pose(index, float(t))
                out.append((f"{name}@{t:.2f}", {rig.name(j): world[j] for j in rig.joint_nodes}))
    return out


def vertex_alpha(rig) -> np.ndarray:
    """COLOR_0 alpha of every skinned vertex, in `rig.mesh` order (1 where there is none)."""
    alphas = []
    for node in rig.nodes:
        if node.get("skin") is None or node.get("mesh") is None:
            continue
        for prim in rig.doc["meshes"][node["mesh"]]["primitives"]:
            attr = prim.get("attributes", {})
            if "JOINTS_0" not in attr or "WEIGHTS_0" not in attr or prim.get("mode", 4) != 4:
                continue
            count = rig.doc["accessors"][attr["POSITION"]]["count"]
            if "COLOR_0" in attr:
                colour = rig.glb.accessor(attr["COLOR_0"]).astype(float)
                if colour.max() > 1.0:  # unsigned shorts or bytes, not normalised by the reader
                    colour = colour / (65535.0 if colour.max() > 255 else 255.0)
                alphas.append(colour[:, 3] if colour.shape[1] == 4 else np.ones(count))
            else:
                alphas.append(np.ones(count))
    return np.concatenate(alphas) if alphas else np.zeros(0)


def _mesh(vertices, faces):
    import trimesh

    return trimesh.Trimesh(vertices, faces, process=False)


def outer_skin_inside(piece, body, masked_faces: np.ndarray, sample: np.ndarray, reach: float = 0.05) -> float:
    """Share of sampled piece vertices facing out that lie under visible body skin."""
    points = piece.vertices[sample]
    normals = piece.vertex_normals[sample]
    closest, distance, face = body.nearest.on_surface(points)
    near = distance < reach
    body_normals = body.face_normals[face]
    outer = np.einsum("ij,ij->i", normals, body_normals) > 0
    inside = np.einsum("ij,ij->i", points - closest, body_normals) < 0
    counted = near & outer & ~masked_faces[face]
    return float((counted & inside & (distance > 1e-5)).sum() / max(len(sample), 1))


def rays_hit(origins: np.ndarray, directions: np.ndarray, target, reach: float) -> np.ndarray:
    """Which rays meet `target` within `reach`."""
    if not len(origins):
        return np.zeros(0, bool)
    locations, rays, _ = target.ray.intersects_location(origins, directions, multiple_hits=False)
    hit = np.zeros(len(origins), bool)
    if len(rays):
        hit[rays] = np.linalg.norm(locations - origins[rays], axis=1) <= reach
    return hit


# The fit step writes these beside the pieces; none is a worn piece.
NOT_WORN = ("set", "preview", "body_masked", "body_trimmed", "weapon", "shield")


def worn_set(asset) -> list[Path]:
    """This piece and every other worn piece fitted to the same body (the GLBs beside it)."""
    folder = Path(asset.body).parent
    others = [p for p in sorted(folder.glob("*.glb")) if p.stem not in NOT_WORN and p.resolve() != Path(asset.glb).resolve()]
    return [Path(asset.glb), *others]


def measure(asset, config: dict) -> dict:
    """Clipping and holes of one piece over every pose; computed once per asset."""
    from image_to_3dlab.glb_skin import SkinnedGLB

    key = (str(asset.glb), str(asset.body), tuple(map(str, asset.body_clips)), config.get("frames", 8))
    if key in _CACHE:
        return _CACHE[key]
    piece_rig, body_rig = SkinnedGLB(asset.glb), SkinnedGLB(asset.body)
    worn_rigs = []
    for path in worn_set(asset)[1:]:
        try:
            worn_rigs.append(SkinnedGLB(path))
        except (LookupError, ValueError, OSError):
            continue
    alpha = vertex_alpha(body_rig)
    masked_faces = (alpha[body_rig.mesh.faces] < 0.5).all(axis=1)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(piece_rig.mesh.bind), min(config.get("sample", 1500), len(piece_rig.mesh.bind)),
                        replace=False)
    gap, reach = config.get("gap", 0.06), config.get("reach", 0.2)

    rest_body = _mesh(body_rig.mesh.vertices, body_rig.mesh.faces)
    rest_piece = _mesh(piece_rig.mesh.vertices, piece_rig.mesh.faces)
    origins = rest_body.vertices + rest_body.vertex_normals * 1e-4
    # only skin the mask hides can show through: a held weapon hides none, so it has no holes
    region = np.nonzero(rays_hit(origins, rest_body.vertex_normals, rest_piece, gap) & (alpha < 0.5))[0]
    if len(region) > config.get("region_sample", 3000):
        region = rng.choice(region, config.get("region_sample", 3000), replace=False)

    found = []
    for label, named in [("rest", {})] + poses(list(asset.body_clips), config.get("frames", 8)):
        body = _mesh(body_rig.deform(world_by_name(body_rig, named)) if named else body_rig.mesh.vertices,
                     body_rig.mesh.faces)
        piece = _mesh(piece_rig.deform(world_by_name(piece_rig, named)) if named else piece_rig.mesh.vertices,
                      piece_rig.mesh.faces)
        clipping = outer_skin_inside(piece, body, masked_faces, sample)
        n = body.vertex_normals[region]
        bare = np.zeros(0, bool)
        if len(region):
            meshes = [piece] + [_mesh(r.deform(world_by_name(r, named)) if named else r.mesh.vertices, r.mesh.faces)
                                for r in worn_rigs]
            import trimesh

            every = trimesh.util.concatenate(meshes) if len(meshes) > 1 else piece
            bare = ~rays_hit(body.vertices[region] + n * 1e-4, n, every, reach)
        found.append({"pose": label, "clipping": clipping, "holes": float(bare.mean()) if len(bare) else 0.0})
    _CACHE[key] = {"poses": found, "region": len(region)}
    return _CACHE[key]


@register("fit.clipping", needs=("body",))
def clipping(asset, config):
    measured = measure(asset, config)
    worst = max(measured["poses"], key=lambda p: p["clipping"])
    value = round(100 * worst["clipping"], 2)
    limits = _limits(config, 3.0, 10.0)
    return result("fit.clipping", grade_max(value, **limits), value, limits,
                  f"{value:.2f}% of the piece's outer skin sinks into visible skin at worst "
                  f"({worst['pose']}, {len(measured['poses'])} poses)",
                  {"poses": [{"pose": p["pose"], "percent": round(100 * p["clipping"], 2)} for p in measured["poses"]]})


@register("fit.holes", needs=("body", "worn"))
def holes(asset, config):
    measured = measure(asset, config)
    worst = max(measured["poses"], key=lambda p: p["holes"])
    value = round(100 * worst["holes"], 2)
    limits = _limits(config, 2.0, 5.0)
    return result("fit.holes", grade_max(value, **limits), value, limits,
                  f"{value:.2f}% of the body this piece covers shows through at worst "
                  f"({worst['pose']}, {len(measured['poses'])} poses)",
                  {"region_vertices": measured["region"],
                   "poses": [{"pose": p["pose"], "percent": round(100 * p["holes"], 2)} for p in measured["poses"]]})
