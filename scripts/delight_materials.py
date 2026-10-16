#!/usr/bin/env python3
"""Take the light painted into an asset's texture, or into a concept, back out (Marigold-IID Lighting).

    python scripts/delight_materials.py asset output/model.glb --out output/delit/
    python scripts/delight_materials.py image output/images/concept.png --out output/images/concept_albedo.png

The asset route renders an orbit of unlit views in Blender, runs Marigold on each and carries
the result back to the texture; the image route runs it on the concept, before the 3D. Both
write a provenance record naming Marigold, its licence and its use restrictions. Needs
`python scripts/bootstrap_materials.py --only env,marigold-lighting` first. Takes the shared
GPU lock only while Marigold runs.

Not admitted yet: the step is used only when asked for, until it passes the test against the
user's labels (design section 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import delight  # noqa: E402
from image_to_3dlab.material_models import MissingModel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    asset = sub.add_parser("asset", help="a GLB: delight its base colour")
    asset.add_argument("path", type=Path)
    asset.add_argument("--out", type=Path, required=True, help="folder for the result")
    asset.add_argument("--views", type=int, default=16, help="cameras around (plus top and bottom)")
    asset.add_argument("--size", type=int, default=768, help="view size in pixels")
    asset.add_argument("--keep-views", type=Path, help="keep the rendered views and answers here")
    image = sub.add_parser("image", help="a concept picture, before the 3D")
    image.add_argument("path", type=Path)
    image.add_argument("--out", type=Path, required=True, help="the delit picture (.png)")
    args = parser.parse_args(argv)
    try:
        if args.kind == "asset":
            result = delight.delight_asset(args.path, args.out, size=args.size, around=args.views,
                                           work=args.keep_views)
            print(f"{result.glb_path}\n{result.sidecar_path}")
        else:
            print(delight.delight_image(args.path, args.out))
    except (MissingModel, ValueError, RuntimeError) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
