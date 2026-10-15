#!/usr/bin/env python3
"""Queue assets for the user's material labels, follow the progress, export what is complete.

    python scripts/material_label_set.py queue output/set/*.glb --kind creature
    python scripts/material_label_set.py progress
    python scripts/material_label_set.py export labels.json

``queue`` puts each GLB in the studio's review queue with source ``material-labels``; the
user opens it there and gives the three judgements (painted light, materials look, material
points). ``export`` writes the complete labels of the current versions, with each asset's
kind, as the set every material check is measured against (design v2, section 6).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import material_labels as ml
from image_to_3dlab.checks import review

SOURCE = "material-labels"
KINDS = ("creature", "prop", "equipment_part")


def key_for(path: Path, output: Path) -> str:
    """A key from the path under output/, so model.glb in two runs gets two keys."""
    relative = path.relative_to(output).with_suffix("")
    return review.key_for(Path("__".join(relative.parts) + ".glb"))


def _items(output: Path) -> dict:
    return {k: i for k, i in review.fold(review.load(output / "review-log.jsonl")).items() if i.get("source") == SOURCE}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=REPO / "output", help="the output folder (default: output/)")
    sub = p.add_subparsers(dest="command", required=True)
    queue = sub.add_parser("queue", help="queue GLBs under output/ for labelling")
    queue.add_argument("glbs", type=Path, nargs="+")
    queue.add_argument("--kind", choices=KINDS, required=True)
    sub.add_parser("progress", help="how many of the set are labelled")
    export = sub.add_parser("export", help="write the complete labels as JSON")
    export.add_argument("target", type=Path)
    for parser in (queue, sub.choices["progress"], export):
        parser.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    args = p.parse_args(argv)
    output = args.output.resolve()

    if args.command == "queue":
        queued = review.fold(review.load(output / "review-log.jsonl"))
        planned = {}
        for glb in args.glbs:
            path = glb.resolve()
            if not path.is_relative_to(output) or path.suffix.lower() != ".glb" or not path.is_file():
                print(f"refused: {glb} must be an existing .glb under {output}")
                return 1
            key = key_for(path, output)
            other = planned.get(key) or (Path(queued[key]["asset"]).resolve() if key in queued else None)
            if other is not None and other != path:
                print(f"refused: {glb} and {other} would share the key {key}")
                return 1
            planned[key] = path
        for key, path in planned.items():
            review.enqueue_for_review(path, key, source=SOURCE, kind=args.kind, log=output / "review-log.jsonl")
            print(f"queued {key} ({args.kind})")
        return 0

    items = _items(output)
    events = ml.load(output / "material-labels.jsonl")
    if args.command == "progress":
        done = ml.progress(events, {k: i.get("asset_sha256") for k, i in items.items()})
        print(f"{done['complete']} of {done['total']} complete, {done['started']} started, {done['missing']} not started")
        return 0

    labels = ml.fold(events)
    rows = []
    for key, item in items.items():
        found = labels.get(key)
        if found and found["asset_sha256"] == item.get("asset_sha256") and ml.is_complete(found):
            rows.append({**found, "kind": item.get("kind")})
    args.target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} complete labels to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
