#!/usr/bin/env python3
"""Make a generated GLB game-ready: tint mask, baked-light checks, normal, roughness and ORM.

Checks the albedo first and sends the asset back, with reasons, when light is painted
into it or it has no colour zones to tint. Otherwise writes the GLB with a normal map,
an ORM texture and a three-zone tint mask inside, the same textures beside it and a
``<asset>.materials.json`` sidecar holding the zones, the tint formula and the checks.

Exit code: 0 when every check passes or warns, 2 when one fails.

    python scripts/game_materials.py asset.glb out/ [--ao ao.png] [--zones materials]
    python scripts/game_materials.py asset.glb --check-only      # records as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.materials import CHANNELS, make_game_materials, run_checks, worst


def parse_zone_values(text: str | None) -> dict[str, float]:
    """``"R=0.3,B=0.9"`` -> ``{"R": 0.3, "B": 0.9}``."""
    values: dict[str, float] = {}
    for item in filter(None, (text or "").split(",")):
        channel, _, number = item.partition("=")
        channel = channel.strip().upper()
        if channel not in CHANNELS or not number:
            raise ValueError(f"expected R=, G= or B= values, got {item!r}")
        value = float(number)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{channel} must be in 0..1")
        values[channel] = value
    return values


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("asset", type=Path, help="GLB to make game-ready")
    p.add_argument("out_dir", type=Path, nargs="?", help="folder to write into")
    p.add_argument("--check-only", action="store_true", help="print the check records and stop")
    p.add_argument("--ao", type=Path, help="AO bake from blender_bake_ao.py, packed into ORM red")
    p.add_argument("--zones", choices=("colour", "materials"), default="colour",
                   help="zones from colour regions (default) or one per material slot")
    p.add_argument("--texture-prefix", help="texture file prefix (default T_<asset>)")
    p.add_argument("--roughness", help="base roughness per zone, e.g. R=0.35,G=0.8")
    p.add_argument("--metallic", help="metallic per zone, e.g. B=1")
    p.add_argument("--normal-strength", type=float, default=6.0,
                   help="relief strength per 1024 texels (default 6)")
    p.add_argument("--samples", type=int, default=60000, help="surface samples (default 60000)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.check_only:
        records = run_checks(args.asset, args.ao, zones_from=args.zones,
                             samples=args.samples, seed=args.seed)
        print(json.dumps(records, indent=2))
        return 2 if worst(r["status"] for r in records) == "fail" else 0
    if args.out_dir is None:
        p.error("out_dir is required unless --check-only")

    result = make_game_materials(
        args.asset, args.out_dir, ao=args.ao, zones_from=args.zones,
        texture_prefix=args.texture_prefix,
        roughness_by_zone=parse_zone_values(args.roughness),
        metallic_by_zone=parse_zone_values(args.metallic),
        normal_strength=args.normal_strength, samples=args.samples, seed=args.seed)
    for record in result.checks:
        print(f"{record['status']:>7}  {record['check']}: {record['message']}")
    for zone in result.zones:
        print(f"   zone {zone['channel']}  {zone['mean_srgb_hex']}  {zone['area_share'] * 100:.0f}%")
    if result.glb_path is None:
        print(f"sent back: see {result.sidecar_path}")
        return 2
    print(f"wrote {result.glb_path} and {result.sidecar_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
