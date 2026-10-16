#!/usr/bin/env python3
"""Estimate roughness and metallic for an asset's materials that have none (Marigold-IID Appearance).

    python scripts/estimate_pbr.py output/model.glb --out output/pbr/
    python scripts/estimate_pbr.py output/model.glb --out output/pbr/ --force   # replace its own map too

A material with its own metallic-roughness map (Pixal3D writes one) keeps it unless --force.
Needs `python scripts/bootstrap_materials.py --only env,marigold-appearance` first; takes the
shared GPU lock only while Marigold runs. Not admitted yet: see scripts/pbr_trial.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import pbr_estimate  # noqa: E402
from image_to_3dlab.material_models import MissingModel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="folder for the result")
    parser.add_argument("--force", action="store_true", help="also replace a material's own map")
    parser.add_argument("--views", type=int, default=16, help="cameras around (plus top and bottom)")
    parser.add_argument("--keep-views", type=Path, help="keep the rendered views and answers here")
    args = parser.parse_args(argv)
    try:
        result = pbr_estimate.estimate_pbr(args.path, args.out, force=args.force, around=args.views,
                                           work=args.keep_views)
    except (MissingModel, ValueError, RuntimeError) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1
    if result.glb_path is None:
        print("every material has its own metallic-roughness map; nothing estimated (use --force)")
        return 0
    print(f"{result.glb_path}\n{result.sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
