#!/usr/bin/env python3
"""Pack a generated GLB's own maps as a game-ready ORM, with its sidecar and licence record.

    python scripts/game_materials.py asset.glb out/ [--ao ao.png] [--texture-prefix T_Name]

Roughness and metallic are the asset's own (Pixal3D writes them) or its factors; the normal
map is whatever the finish step baked from the dense mesh; occlusion is the asset's own or
the ``--ao`` bake from ``blender_bake_ao.py``. Nothing is derived from the colour. Tint
zones by material and the learned checks come in later steps.

Exit code: 0 when written, 1 when the asset cannot be packed without damage (the reason is
printed: compressed geometry, an unlit material, maps on another UV layout...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.materials import make_game_materials


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("asset", type=Path, help="GLB to pack")
    p.add_argument("out_dir", type=Path, help="folder to write into (not the asset's own)")
    p.add_argument("--ao", type=Path, help="AO bake from blender_bake_ao.py, packed into ORM red")
    p.add_argument("--texture-prefix", help="texture file prefix (default T_<asset>)")
    args = p.parse_args(argv)
    try:
        result = make_game_materials(args.asset, args.out_dir, ao=args.ao, texture_prefix=args.texture_prefix)
    except ValueError as exc:
        print(f"refused: {exc}")
        return 1
    print(f"wrote {result.glb_path} ({', '.join(f'{k} {v}' for k, v in result.maps.items())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
