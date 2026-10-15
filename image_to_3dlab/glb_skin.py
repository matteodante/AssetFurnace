"""A rigged GLB's skeleton, rest mesh and animations, posed and skinned in numpy.

Enough of glTF skinning to look at a clip without Blender: sample an animation at a time,
build every node's world matrix, and move the vertices with linear blend skinning, the way
an engine does. The checks use it to measure how a clip deforms the mesh; the family
detector uses the rest skeleton and mesh.

Accessors are read through `edit_glb.EditGLB`, which refuses what it cannot read safely
(external buffers, sparse accessors) instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from image_to_3dlab.edit_glb import EditGLB, node_matrix


@dataclass
class Skinned:
    bind: np.ndarray       # (N, 3) positions as stored, before skinning
    vertices: np.ndarray   # (N, 3) rest positions, world space
    faces: np.ndarray      # (F, 3) indices into vertices
    joints: np.ndarray     # (N, 4) indices into the skin's joint list
    weights: np.ndarray    # (N, 4) normalised


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _trs(t, r, s) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(np.asarray(r, float)) @ np.diag(s)
    m[:3, 3] = t
    return m


class SkinnedGLB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.glb = EditGLB(self.path.read_bytes())
        self.doc = self.glb.document
        self.nodes = self.doc.get("nodes", [])
        skins = self.doc.get("skins") or []
        if not skins:
            raise LookupError(f"{self.path.name} has no skin")
        self.skin = skins[0]
        self.joint_nodes: list[int] = list(self.skin["joints"])
        self.parent_of = {c: i for i, n in enumerate(self.nodes) for c in n.get("children", [])}

    # -- the skeleton ------------------------------------------------------------------

    def name(self, node: int) -> str:
        return self.nodes[node].get("name") or f"node_{node}"

    @cached_property
    def inverse_binds(self) -> np.ndarray:
        if "inverseBindMatrices" in self.skin:
            return self.glb.accessor(self.skin["inverseBindMatrices"])
        return np.repeat(np.eye(4)[None], len(self.joint_nodes), axis=0)

    def rest_locals(self) -> list[np.ndarray]:
        return [node_matrix(n) for n in self.nodes]

    def world(self, locals_: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray | None] = [None] * len(self.nodes)

        def resolve(i: int) -> np.ndarray:
            if out[i] is None:
                p = self.parent_of.get(i)
                out[i] = locals_[i] if p is None else resolve(p) @ locals_[i]
            return out[i]

        for i in range(len(self.nodes)):
            resolve(i)
        return out  # type: ignore[return-value]

    def skeleton(self) -> tuple[dict[str, str | None], dict[str, np.ndarray]]:
        """(parent of each joint, rest head of each joint), by joint name, world space."""
        world = self.world(self.rest_locals())
        joints = set(self.joint_nodes)
        parents = {}
        for j in self.joint_nodes:
            p = self.parent_of.get(j)
            while p is not None and p not in joints:
                p = self.parent_of.get(p)
            parents[self.name(j)] = None if p is None else self.name(p)
        heads = {self.name(j): world[j][:3, 3].copy() for j in self.joint_nodes}
        return parents, heads

    # -- the mesh ----------------------------------------------------------------------

    @cached_property
    def mesh(self) -> Skinned:
        """Every primitive skinned by this skin, as one mesh, at rest."""
        verts, faces, joints, weights, offset = [], [], [], [], 0
        for node in self.nodes:
            if node.get("skin") is None or node.get("mesh") is None:
                continue
            for prim in self.doc["meshes"][node["mesh"]]["primitives"]:
                attr = prim.get("attributes", {})
                if "JOINTS_0" not in attr or "WEIGHTS_0" not in attr or prim.get("mode", 4) != 4:
                    continue
                v = self.glb.accessor(attr["POSITION"]).astype(np.float64)
                f = (self.glb.accessor(prim["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in prim
                     else np.arange(len(v)).reshape(-1, 3))
                w = self.glb.accessor(attr["WEIGHTS_0"]).astype(np.float64)
                verts.append(v)
                faces.append(f + offset)
                joints.append(self.glb.accessor(attr["JOINTS_0"], normalized=False).astype(np.int64))
                weights.append(w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12))
                offset += len(v)
        if not verts:
            raise LookupError(f"{self.path.name} has no skinned triangles")
        bind = np.vstack(verts)
        rest = Skinned(bind, bind, np.vstack(faces), np.vstack(joints), np.vstack(weights))
        rest.vertices = self.deform(self.world(self.rest_locals()), rest)
        return rest

    def deform(self, world: list[np.ndarray], mesh: Skinned | None = None) -> np.ndarray:
        """Linear blend skinning with these node world matrices."""
        mesh = mesh or self.mesh
        palette = np.stack([world[j] @ ib for j, ib in zip(self.joint_nodes, self.inverse_binds)])
        blend = np.einsum("nk,nkij->nij", mesh.weights, palette[mesh.joints])
        return np.einsum("nij,nj->ni", blend[:, :3, :3], mesh.bind) + blend[:, :3, 3]

    # -- animations --------------------------------------------------------------------

    def animations(self) -> list[tuple[str, float]]:
        """(name, duration in seconds) of every animation."""
        out = []
        for k, anim in enumerate(self.doc.get("animations") or []):
            end = max((float(self.glb.accessor(s["input"]).max()) for s in anim["samplers"]), default=0.0)
            out.append((anim.get("name") or f"animation_{k}", end))
        return out

    def pose(self, animation: int, time: float) -> list[np.ndarray]:
        """Node world matrices with `animation` sampled at `time` seconds."""
        locals_ = self.rest_locals()
        anim = self.doc["animations"][animation]
        current = {}
        for channel in anim["channels"]:
            target = channel["target"]
            if target.get("path") not in ("translation", "rotation", "scale") or "node" not in target:
                continue
            node = target["node"]
            if "matrix" in self.nodes[node]:
                raise ValueError("an animated node must use translation/rotation/scale")
            sampler = anim["samplers"][channel["sampler"]]
            value = self._sample(sampler, time, target["path"] == "rotation")
            current.setdefault(node, {})[target["path"]] = value
        for node, values in current.items():
            n = self.nodes[node]
            locals_[node] = _trs(values.get("translation", n.get("translation", [0, 0, 0])),
                                 values.get("rotation", n.get("rotation", [0, 0, 0, 1])),
                                 values.get("scale", n.get("scale", [1, 1, 1])))
        return self.world(locals_)

    def _sample(self, sampler: dict, time: float, rotation: bool) -> np.ndarray:
        times = self.glb.accessor(sampler["input"]).astype(np.float64)
        values = self.glb.accessor(sampler["output"]).astype(np.float64)
        mode = sampler.get("interpolation", "LINEAR")
        if mode == "CUBICSPLINE":  # (in-tangent, value, out-tangent) per key: keep the values
            values = values.reshape(len(times), 3, -1)[:, 1]
        if time <= times[0]:
            return values[0]
        if time >= times[-1]:
            return values[-1]
        i = int(np.searchsorted(times, time, side="right")) - 1
        if mode == "STEP":
            return values[i]
        a, b = values[i], values[i + 1]
        u = (time - times[i]) / max(times[i + 1] - times[i], 1e-12)
        if rotation and float(a @ b) < 0:
            b = -b
        out = a + (b - a) * u
        return out / np.linalg.norm(out) if rotation else out
