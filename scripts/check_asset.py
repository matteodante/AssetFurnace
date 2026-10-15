#!/usr/bin/env python3
"""Run the automatic checks on a GLB, write `<name>.checks.json` and put it in the review queue.

    python scripts/check_asset.py output/.../barrel.glb --profile prop
    python scripts/check_asset.py mob.glb --reference original.glb --key mob_0412 --profile creature
    python scripts/check_asset.py mob.glb --height 1.8 --set budget.faces.max=20000 --json

The checks are the same for every asset (see `image_to_3dlab/checks`): clean mesh, UVs,
budgets, size and silhouette against a reference, glTF validity, licences. Exit code 0
means pass or warn, 1 means at least one check failed, so a batch can stop on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_overrides(pairs: list[str]) -> dict:
    """`check.name.setting=value` pairs into the profile override shape.

    The check name is everything before the last dot: `budget.faces.max=20000` sets `max`
    on `budget.faces`. Values are read as JSON when they parse (numbers, true/false).
    """
    overrides: dict[str, dict] = {}
    for pair in pairs:
        name, _, raw = pair.partition("=")
        check, _, setting = name.rpartition(".")
        if not check or not setting or not _:
            raise ValueError(f"expected check.setting=value, got {pair!r}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        overrides.setdefault(check, {})[setting] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    from image_to_3dlab.checks import profile_names

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("glb", type=Path)
    parser.add_argument("--reference", type=Path, help="reference GLB or picture, for size and silhouette")
    parser.add_argument("--height", type=float, help="wanted height in metres")
    parser.add_argument("--width", type=float, help="wanted width (X) in metres")
    parser.add_argument("--depth", type=float, help="wanted depth (Z) in metres")
    parser.add_argument("--profile", default="prop", choices=profile_names())
    parser.add_argument("--key", help="the asset's key in the review queue (default: file name)")
    parser.add_argument("--provenance", type=Path, help="provenance record, if not beside the GLB")
    parser.add_argument("--set", action="append", default=[], metavar="CHECK.SETTING=VALUE",
                        help="override one threshold, e.g. budget.faces.max=20000")
    parser.add_argument("--no-queue", action="store_true", help="write the report only")
    parser.add_argument("--log", type=Path, help="review log (default output/review-log.jsonl)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(REPO))
    from image_to_3dlab.checks import Asset, review, run_checks, write_report

    args = build_parser().parse_args(argv)
    numbers = {k: v for k, v in (("height", args.height), ("width", args.width), ("depth", args.depth)) if v}
    asset = Asset(glb=args.glb, key=args.key, reference=args.reference, reference_numbers=numbers or None,
                  provenance=args.provenance)
    overrides = parse_overrides(args.set)
    if args.no_queue:
        report = run_checks(asset, profile=args.profile, overrides=overrides)
        path = write_report(report)
    else:
        report, path = review.check_and_enqueue(asset, log=args.log or review.LOG, profile=args.profile,
                                                overrides=overrides, key=args.key)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for found in report["results"]:
            print(f"{found['status']:4}  {found['check']:22} {found['message']}")
        for skipped in report["skipped"]:
            print(f"skip  {skipped['check']:22} {skipped['reason']}")
        print(f"{report['status'].upper()}: report in {path}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
