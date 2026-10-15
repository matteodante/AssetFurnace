"""The asset a check looks at: a GLB, plus an optional reference and provenance.

Everything expensive (hashing, parsing, loading the mesh, welding) is computed once, on
first use, and shared by every check in the run.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from image_to_3dlab.checks import geometry

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
REPO = Path(__file__).resolve().parents[2]
VALIDATOR_ENV = "I2L_GLTF_VALIDATOR"


def find_validator(env: dict[str, str] | None = None, repo: Path = REPO) -> Path | None:
    """The glTF-Validator binary: `I2L_GLTF_VALIDATOR`, then `vendor/gltf-validator/`, then PATH."""
    env = os.environ if env is None else env
    explicit = env.get(VALIDATOR_ENV)
    if explicit:
        return Path(explicit) if Path(explicit).is_file() else None
    for name in ("gltf_validator.exe", "gltf_validator"):
        candidate = repo / "vendor" / "gltf-validator" / name
        if candidate.is_file():
            return candidate
    found = shutil.which("gltf_validator")
    return Path(found) if found else None


@dataclass(frozen=True)
class Asset:
    glb: Path
    key: str | None = None
    reference: Path | None = None              # a GLB, or a picture (concept or cut-out)
    reference_numbers: dict | None = None      # {"height": m, "width": m, "depth": m}
    provenance: Path | None = None             # explicit sidecar; otherwise found beside the GLB
    output_root: Path | None = None            # where generated runs live, for lineage
    validator: Path | None = field(default=None, compare=False)

    def __post_init__(self):
        for name in ("glb", "reference", "provenance", "output_root", "validator"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    # -- what is on disk ---------------------------------------------------------------

    @cached_property
    def sha256(self) -> str:
        from image_to_3dlab.provenance import sha256_file

        return sha256_file(self.glb)

    @cached_property
    def _parsed(self) -> tuple[dict, bytes]:
        from image_to_3dlab.glb_turn import parse_glb

        return parse_glb(self.glb.read_bytes())

    @property
    def document(self) -> dict:
        return self._parsed[0]

    @property
    def binary(self) -> bytes:
        return self._parsed[1]

    @cached_property
    def primitives(self) -> list:
        """One world-space trimesh per drawn primitive, with its UVs and material."""
        import trimesh

        scene = trimesh.load(self.glb, force="scene", process=False)
        return [m for m in scene.dump() if isinstance(m, trimesh.Trimesh) and len(m.faces)]

    @cached_property
    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        """All primitives as one (vertices, faces) pair, unwelded."""
        verts, faces, offset = [], [], 0
        for mesh in self.primitives:
            verts.append(np.asarray(mesh.vertices, dtype=np.float64))
            faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
            offset += len(mesh.vertices)
        if not verts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return np.vstack(verts), np.vstack(faces)

    @cached_property
    def welded(self) -> geometry.Welded:
        return geometry.weld(*self.positions)

    @cached_property
    def bounds(self) -> np.ndarray:
        vertices = self.positions[0]
        if len(vertices) == 0:
            return np.zeros((2, 3))
        return np.stack([vertices.min(axis=0), vertices.max(axis=0)])

    @property
    def reference_kind(self) -> str | None:
        if self.reference is None:
            return None
        return "image" if self.reference.suffix.lower() in IMAGE_SUFFIXES else "glb"

    @cached_property
    def reference_asset(self) -> Asset | None:
        return Asset(glb=self.reference) if self.reference_kind == "glb" else None

    @cached_property
    def validator_path(self) -> Path | None:
        return self.validator if self.validator and self.validator.is_file() else find_validator()

    @property
    def has_texture(self) -> bool:
        return any(base_colour(m) is not None for m in self.primitives)

    # -- what a check needs --------------------------------------------------------------

    def missing(self, needs: tuple[str, ...]) -> str | None:
        """Why a check with these `needs` cannot run, or None when it can."""
        # Lazy: deciding whether the validator can run must not load the mesh it validates.
        reasons = {
            "glb": lambda: None if self.glb.is_file() else f"{self.glb} does not exist",
            "reference_shape": lambda: None if self.reference_kind else "no reference model or picture given",
            "reference_height": lambda: None if (self.reference_kind == "glb" or (self.reference_numbers or {}).get("height"))
            else "no reference model or height given",
            "reference_footprint": lambda: None if (self.reference_kind == "glb" or any(
                (self.reference_numbers or {}).get(k) for k in ("width", "depth")))
            else "no reference model, width or depth given",
            "rig": lambda: None if self.document.get("skins") else "the model has no rig",
            "texture": lambda: None if self.has_texture else "the model has no texture",
            "gltf_validator": lambda: None if self.validator_path else
            "glTF-Validator is not installed (python scripts/bootstrap_gltf_validator.py)",
        }
        for need in ("glb", *needs):
            if need not in reasons:
                return f"unknown need {need!r}"
            reason = reasons[need]()
            if reason:
                return reason
        return None

    def reference_record(self) -> dict[str, Any] | None:
        if self.reference is not None:
            from image_to_3dlab.provenance import sha256_file

            return {"kind": self.reference_kind, "path": str(self.reference),
                    "sha256": sha256_file(self.reference) if self.reference.is_file() else None}
        if self.reference_numbers:
            return {"kind": "numbers", "values": dict(self.reference_numbers)}
        return None


def base_colour(mesh) -> Any:
    """The base-colour image of a trimesh primitive, or None."""
    material = getattr(getattr(mesh, "visual", None), "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    return image
