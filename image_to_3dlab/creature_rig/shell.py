"""The creature's watertight shell as voxels: inside or out, how deep, where it stands.

Everything the auto-rig measures on the body goes through one grid whose cell is 1% of
the height, so a creature is about 100 cells tall whatever its units or face count: once
its triangles are in the grid a 1M-face dragon costs what a 20k one does. Inside is a cell
lookup, never a ray test.

The grid holds both the voxel-remeshed shell and the model's own triangles: a robe or a
wing membrane modelled as one sheet has no inside for the remesh to keep, but it is body.

Frame (set by the prepare step): Z up, facing -Y, the creature's left on +X, the floor at
the lowest vertex.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

VOXEL = 0.01                              # grid cell, a share of the height
FORWARD = np.array([0.0, -1.0, 0.0])      # the creature faces -Y
LEFT = np.array([1.0, 0.0, 0.0])          # its left is +X
UP = np.array([0.0, 0.0, 1.0])

FLOOR_BAND = 0.02        # vertices this close above the floor stand on it
FLOOR_POINTS = 1500      # floor vertices kept to group (every k-th)
CONTACT_LINK = 0.08      # floor points closer than this are one contact
CONTACT_MIN = 4          # points a contact needs
SHEET_BAND = 0.08        # the cloth test looks at cells this low ...
SHEET_NEAR = 0.03        # ... and this close beside the contact
SHEET_DEPTH = 2.0        # a foot has cells at least this deep (in cells); a robe hem never does
REACH_HIGH = 0.15        # a bone point this low ...
REACH_NEAR = 0.06        # ... and this close beside a contact reaches it
POINTS_PER_BATCH = 2_000_000

Mesh = tuple[np.ndarray, np.ndarray]


def _mark(grid: np.ndarray, points: np.ndarray, lo: np.ndarray, size: float) -> None:
    ijk = np.floor((points - lo) / size).astype(np.int64)
    ijk = np.clip(ijk, 0, np.array(grid.shape) - 1)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True


def surface_cells(meshes: Sequence[Mesh], size: float, lo: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """The cells the triangles pass through: each triangle sampled every half cell.

    A triangle smaller than half a cell is its three corners, which the vertices already
    mark; only the larger ones are sampled on a lattice, in batches so a big mesh never
    holds more than a couple of million points at once."""
    grid = np.zeros(shape, bool)
    for verts, faces in meshes:
        v = np.asarray(verts, np.float64)
        f = np.asarray(faces, np.int64).reshape(-1, 3)
        if not len(f):
            continue
        _mark(grid, v[np.unique(f)], lo, size)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        longest = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1),
                          np.linalg.norm(c - b, axis=1)], axis=0)
        steps = np.clip(np.ceil(longest / (0.5 * size)).astype(np.int64), 1, 1024)
        for k in np.unique(steps[steps > 1]):
            u, w = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
            keep = (u + w) <= k
            u, w = u[keep] / k, w[keep] / k
            sel = np.nonzero(steps == k)[0]
            batch = max(1, POINTS_PER_BATCH // len(u))
            for start in range(0, len(sel), batch):
                s = sel[start:start + batch]
                pts = (a[s, None] * (1 - u - w)[None, :, None] + b[s, None] * u[None, :, None]
                       + c[s, None] * w[None, :, None])
                _mark(grid, pts.reshape(-1, 3), lo, size)
    return grid


def voxelize(meshes: Sequence[Mesh], size: float) -> tuple[np.ndarray, np.ndarray]:
    """(solid cells, the grid's low corner): the surfaces in cells, closed parts filled.

    Two empty cells of margin round the body so the outside is one piece."""
    allv = np.concatenate([np.asarray(v, np.float64).reshape(-1, 3) for v, _ in meshes])
    lo = allv.min(0) - 2 * size
    shape = tuple(int(n) for n in np.ceil((allv.max(0) + 2 * size - lo) / size).astype(int) + 1)
    surface = surface_cells(meshes, size, lo, shape)
    return ndimage.binary_fill_holes(surface), lo


class Shell:
    """The body on its grid, and what the rig steps ask of it.

    `meshes` are (verts, faces) pairs, all in the model frame; `height` is the model's."""

    def __init__(self, meshes: Sequence[Mesh], height: float, voxel: float = VOXEL):
        self.meshes = [(np.asarray(v, np.float64).reshape(-1, 3), np.asarray(f, np.int64).reshape(-1, 3))
                       for v, f in meshes]
        self.height = float(height)
        if self.height <= 0:
            raise ValueError("the model has no height")
        self.size = voxel * self.height
        self.solid, self.lo = voxelize(self.meshes, self.size)
        self.verts = np.concatenate([v for v, _ in self.meshes])
        if not self.solid.any():
            raise ValueError("the shell has no cells")

    @classmethod
    def load(cls, work: Path) -> Shell:
        """The shell the prepare step wrote: `W/shell.npz` (remeshed shell and the model's
        own triangles)."""
        d = np.load(Path(work) / "shell.npz")
        meshes = [(d["shell_verts"], d["shell_faces"])]
        if "model_verts" in d.files and "model_faces" in d.files and len(d["model_faces"]):
            meshes.append((d["model_verts"], d["model_faces"]))
        return cls(meshes, float(d["height"]))

    # --- cells -----------------------------------------------------------------------------

    @functools.cached_property
    def depth(self) -> np.ndarray:
        """How deep every cell lies, in cells: 1 on the skin, 0 outside."""
        return ndimage.distance_transform_edt(self.solid).astype(np.float32)

    @functools.cached_property
    def ijk(self) -> np.ndarray:
        return np.argwhere(self.solid)

    @functools.cached_property
    def cells(self) -> np.ndarray:
        """Centres of the solid cells."""
        return self.lo + (self.ijk + 0.5) * self.size

    @functools.cached_property
    def cell_depth(self) -> np.ndarray:
        return self.depth[tuple(self.ijk.T)]

    @functools.cached_property
    def kd(self) -> cKDTree:
        return cKDTree(self.cells)

    @functools.cached_property
    def mid(self) -> float:
        """The middle plane's X: the median of the solid cells (a sword held out to one side
        moves a mean, not this)."""
        return float(np.median(self.cells[:, 0]))

    @functools.cached_property
    def floor(self) -> float:
        return float(self.verts[:, 2].min())

    @functools.cached_property
    def longest(self) -> float:
        """The model's longest side (a lizard is three times as long as it is high)."""
        return float(np.ptp(self.verts, axis=0).max())

    def index(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(cell indices, which of them lie in the grid)."""
        ijk = np.floor((np.atleast_2d(points) - self.lo) / self.size).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.solid.shape)), axis=1)
        return ijk, ok

    def inside(self, points: np.ndarray) -> np.ndarray:
        ijk, ok = self.index(points)
        out = np.zeros(len(ijk), bool)
        out[ok] = self.solid[tuple(ijk[ok].T)]
        return out

    def off_skin(self, points: np.ndarray) -> np.ndarray:
        """How far each point lies off the body: 0 inside, else the distance to the nearest
        solid cell's centre."""
        p = np.atleast_2d(np.asarray(points, float))
        out = np.zeros(len(p))
        outside = ~self.inside(p)
        if outside.any():
            out[outside] = self.kd.query(p[outside])[0]
        return out

    def mirror(self, points: np.ndarray) -> np.ndarray:
        """Points mirrored across the middle plane."""
        p = np.array(points, float, copy=True)
        p[..., 0] = 2 * self.mid - p[..., 0]
        return p

    def lateral(self, points: np.ndarray) -> np.ndarray:
        """How far points lie to the left (+) or right (-) of the middle plane."""
        return np.asarray(points, float)[..., 0] - self.mid

    def symmetric_near(self, points: np.ndarray, radius: float = 0.05) -> float:
        """Share of the solid cells within `radius` (x height) of the points whose mirror
        cell is solid too: 1 where the body round them mirrors itself."""
        p = np.atleast_2d(np.asarray(points, float))
        if not len(p):
            return 0.0
        near = self.kd.query_ball_point(p, radius * self.height)
        idx = np.unique(np.concatenate([np.asarray(n, np.int64) for n in near]))
        if not len(idx):
            return 0.0
        return float(self.inside(self.mirror(self.cells[idx])).mean())

    # --- floor contacts --------------------------------------------------------------------

    @functools.cached_property
    def contacts(self) -> list[np.ndarray]:
        """Where the creature stands: groups of floor vertices seen from above, (k, 2) each.

        A group whose cells near the floor are never two cells deep is a sheet, not a foot:
        the hem of a robe reaching the floor."""
        h = self.height
        low = self.verts[self.verts[:, 2] < self.floor + FLOOR_BAND * h][:, :2]
        low = low[::max(1, len(low) // FLOOR_POINTS)]
        if len(low) < CONTACT_MIN:
            return []
        pairs = cKDTree(low).query_pairs(CONTACT_LINK * h, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(low), len(low)))
        _, label = connected_components(graph, directed=False)
        groups = [low[label == k] for k in np.unique(label) if (label == k).sum() >= CONTACT_MIN]
        near_floor = self.cells[:, 2] < self.floor + SHEET_BAND * h
        if not near_floor.any():
            return groups
        kd = cKDTree(self.cells[near_floor][:, :2])
        deep = self.cell_depth[near_floor]
        kept = []
        for g in groups:
            hits = kd.query_ball_point(g, SHEET_NEAR * h)
            idx = np.unique(np.concatenate([np.asarray(x, np.int64) for x in hits]))
            if not len(idx) or deep[idx].max() >= SHEET_DEPTH:
                kept.append(g)
        return kept

    def unreached(self, samples: np.ndarray) -> list[int]:
        """The contacts no bone reaches: no bone point low enough and close enough beside it."""
        h = self.height
        p = np.asarray(samples, float).reshape(-1, 3)
        p = p[p[:, 2] < self.floor + REACH_HIGH * h][:, :2]
        if not len(p):
            return list(range(len(self.contacts)))
        kd = cKDTree(p)
        return [k for k, g in enumerate(self.contacts) if kd.query(g)[0].min() > REACH_NEAR * h]
