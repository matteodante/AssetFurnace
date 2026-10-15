"""Game-material checks from `image_to_3dlab.materials`, run by the checks registry.

Three of the four material checks: tint zones and painted light (direction, highlights).
The painted-occlusion check needs an AO bake the registry's Asset does not carry, so it
stays with `scripts/game_materials.py --ao`. One analysis per GLB serves all three.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from image_to_3dlab import materials
from image_to_3dlab.checks.registry import register, result

CHECKS = {
    "material.tint_zones": "tint_zones",
    "material.baked_light_direction": "baked_light_direction",
    "material.baked_light_highlights": "baked_light_highlights",
}


@lru_cache(maxsize=8)
def _records(glb: str, sha256: str) -> dict[str, dict]:
    """The materials records for one GLB's bytes; the hash keys the cache to its content."""
    return {r["check"]: r for r in materials.run_checks(Path(glb))}


def as_registry_result(name: str, record: dict) -> dict:
    """A materials record in the registry's shape, under the registry's name.

    The registry has no "skipped" result for a check whose inputs are present: one that
    could not measure is a warning, never a pass.
    """
    status, message = record["status"], record["message"]
    if status not in ("pass", "warn", "fail"):
        status, message = "warn", f"could not measure: {message}"
    return result(name, status, record["value"], record["threshold"], message, record.get("details"))


def _register(name: str, source: str) -> None:
    @register(name)
    def check(asset, config):
        return as_registry_result(name, _records(str(asset.glb), asset.sha256)[source])


for _name, _source in CHECKS.items():
    _register(_name, _source)
