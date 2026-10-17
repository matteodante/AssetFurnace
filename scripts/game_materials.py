#!/usr/bin/env python3
"""Pack a generated GLB's own maps as a game-ready ORM, with its sidecar and licence record.

    python scripts/game_materials.py asset.glb out/ [--ao ao.png] [--texture-prefix T_Name]
    python scripts/game_materials.py asset.glb out/ --zones --profile creature   # tint zones by material (SAM 3.1)
    python scripts/game_materials.py asset.glb out/ --zones-file zones.json --zones-mask zones.png

Roughness and metallic are the asset's own (Pixal3D writes them) or its factors; the normal
map is whatever the finish step baked from the dense mesh; occlusion is the asset's own or
the ``--ao`` bake from ``blender_bake_ao.py``. Nothing is derived from the colour. With
``--zones`` Qwen3.5-4B names the materials and SAM 2.1 regions it names (``sam2``, the default)
or SAM 3 (``sam3``, once its gated access is granted) find them in every view; the three largest
become the tint mask. Opt-in: see ``scripts/zones_trial.py`` and design section 10.
``--zones-file`` gives zones of your own instead: a JSON ``{"zones": {"1": "metal", "2":
"leather"}}`` with ``--zones-mask`` a PNG of those zone indices in UV space, or an earlier
run's ``.materials.json`` corrected by hand with its ``_Tint.png`` as the mask
(``material_zones.zones_from_file``). No model runs then.

Exit code: 0 when written, 1 when the asset cannot be packed without damage (the reason is
printed: compressed geometry, an unlit material, maps on another UV layout...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.materials import make_game_materials


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("asset", type=Path, help="GLB to pack")
    p.add_argument("out_dir", type=Path, help="folder to write into (not the asset's own)")
    p.add_argument("--ao", type=Path, help="AO bake from blender_bake_ao.py, packed into ORM red")
    p.add_argument("--texture-prefix", help="texture file prefix (default T_<asset>)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--zones", nargs="?", const="sam31", choices=("sam31", "sam3", "sam2"),
                        help="tint zones by material: sam31 (default), sam3, or sam2 without the gated SAM access")
    source.add_argument("--zones-file", type=Path,
                        help="zones of your own: a JSON {zone: material}, or a run's .materials.json corrected by hand")
    p.add_argument("--zones-mask", type=Path,
                   help="with --zones-file: the PNG of its zone indices in UV space, or that run's _Tint.png")
    p.add_argument("--materials", help="comma-separated materials instead of asking Qwen3.5 (e.g. leather,metal,cloth)")
    p.add_argument("--profile", help="asset profile; creature and equipment_part need 3 zones")
    p.add_argument("--work", type=Path, help="keep the views and the models' answers here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    zones = record = None
    try:
        if args.zones_file:
            from image_to_3dlab import material_zones as mz

            zones, record = mz.zones_from_file(args.asset, args.zones_file, args.zones_mask)
        elif args.zones:
            import tempfile

            from image_to_3dlab import material_zones as mz

            words = [w.strip() for w in args.materials.split(",")] if args.materials else None
            with tempfile.TemporaryDirectory(prefix="af-zones-") as scratch:
                zones, record = mz.find_zones(args.asset, args.work or Path(scratch), args.zones, words)
        result = make_game_materials(args.asset, args.out_dir, ao=args.ao, texture_prefix=args.texture_prefix,
                                     zones=zones, zone_record=record, profile=args.profile)
    except (ValueError, RuntimeError) as exc:
        print(f"refused: {exc}")
        return 1
    print(f"wrote {result.glb_path} ({', '.join(f'{k} {v}' for k, v in result.maps.items())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
