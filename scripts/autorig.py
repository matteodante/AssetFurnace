#!/usr/bin/env python3
"""Give a model a skeleton, or play preset clips on a rigged one, as the studio's Animate tab does.

    python scripts/autorig.py rig output/finish/knight/knight_40k.glb
    python scripts/autorig.py animate output/animate/knight__rig__.../knight_rigged.glb --presets idle walk
    python scripts/autorig.py animate RIGGED.glb --presets punch --speed 1.3 --arm-spread 10
    python scripts/autorig.py presets

`rig` runs SkinTokens (holding the shared GPU lock) and the skin-weight clean-up, and
writes `<name>_rigged.glb` with a `.provenance.json` carrying the source's licence and
SkinTokens' (default folder `output/animate/<name>__rig__<time>/`, as the studio). The
creature family's test clips are `scripts/creature_clips.py --rig RIGGED.glb`.

`animate` plays shipped Kimodo clips (`image_to_3dlab/motions/`) with
`scripts/kimodo_retarget.py`, CPU only, into the rigged model's folder by default:
`<name>_<preset>.glb` each, with its provenance.

The same code as the studio and the batch steps `rig` and `animate`
(`image_to_3dlab/rigging.py`). Nothing downloads. Exit code 0 when written, 1 when refused.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import rigging  # noqa: E402

OUTPUT_ROOT = REPO / "output" / "animate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rig = sub.add_parser("rig", help="a skeleton and skin weights for a T-posed model")
    rig.add_argument("model", type=Path)
    rig.add_argument("--out", type=Path, help="folder (default output/animate/<name>__rig__<time>/)")
    rig.add_argument("--name", help="file name stem (default: the model's)")
    animate = sub.add_parser("animate", help="play preset clips on a rigged model")
    animate.add_argument("rigged", type=Path)
    animate.add_argument("--presets", nargs="+", required=True, help="preset ids; `presets` lists them")
    animate.add_argument("--speed", type=float, help=f"playback speed {rigging.SPEED_RANGE[0]:g} to "
                                                     f"{rigging.SPEED_RANGE[1]:g} (default: each preset's own)")
    animate.add_argument("--arm-spread", type=float, default=0.0,
                         help=f"arms out from the body, 0 to {rigging.MAX_ARM_SPREAD:g} degrees")
    animate.add_argument("--out", type=Path, help="folder (default: the rigged model's)")
    sub.add_parser("presets", help="the preset clips")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "presets":
            for preset in rigging.presets():
                print(f"{preset['id']:20} {preset.get('label', '')}")
            return 0
        if args.command == "rig":
            name = args.name or args.model.stem
            out = args.out or OUTPUT_ROOT / f"{name}__rig__{time.strftime('%Y%m%d-%H%M%S')}"
            rigged, _ = rigging.rig_model(args.model, out, name, origin=str(args.model),
                                          tool="scripts/autorig.py rig: SkinTokens auto-rig", log=print)
            print(rigged)
            return 0
        clips = rigging.animate_model(args.rigged, args.presets, args.out or args.rigged.parent, speed=args.speed,
                                      arm_spread=args.arm_spread, tool="scripts/autorig.py animate", log=print)
        for clip in clips:
            print(clip)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"autorig: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
