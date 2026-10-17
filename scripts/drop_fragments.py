#!/usr/bin/env python3
"""Drop the small bits floating away from a generated piece (image_to_3dlab/slot_parts/fragments.py).

    python scripts/drop_fragments.py piece.glb [out.glb]

Characters in pieces runs it on every piece made on its own, after its finish and winding:
a stray bit beside a glove hung in the air next to the hand. Only parts both small and apart
from the piece go; geometry, UVs and textures of the rest are untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_to_3dlab.slot_parts import fragments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="defaults to in-place")
    args = parser.parse_args()

    import trimesh

    mesh = trimesh.load(str(args.asset), force="mesh", process=False)
    keep, report = fragments.floating(mesh.vertices, mesh.faces)
    destination = args.output or args.asset
    if report["dropped"]:
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        destination.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(destination))
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(args.asset.read_bytes())
    print(f"fragments: dropped {report['dropped']} ({report['faces']} faces); wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
