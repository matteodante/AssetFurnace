#!/usr/bin/env python3
"""Test tint zones by material, and the roughness and metallic they imply (design section 6).

    python scripts/zones_trial.py human --labels labels.json --dir output/zones-trial [--segmenter sam3]
    python scripts/zones_trial.py pbr --dir output/pbr-trial [--segmenter sam3]

`human` finds the zones of every asset the user labelled (`scripts/material_label_set.py
export`) and counts how many of their clicked points land in the zone of their material: at
least 75% for the step to be admitted. `pbr` gives the Poly Haven trial assets
(`scripts/pbr_trial.py fetch`) roughness and metallic from their zones' materials
(`material_zones.DEFAULT_PBR`) and scores them against the true maps, as the Marigold
estimate was.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from image_to_3dlab import material_zones as mz  # noqa: E402
from image_to_3dlab import pbr_trial as pt  # noqa: E402
from image_to_3dlab.edit_glb import EditGLB  # noqa: E402
from image_to_3dlab.materials import surfaces  # noqa: E402


def human(folder: Path, labels: Path, segmenter: str) -> int:
    from image_to_3dlab.material_labels import still_labelled

    rows = [r for r in json.loads(labels.read_text(encoding="utf-8")) if r.get("points")]
    rows, changed = still_labelled(rows)
    for row in changed:
        print(f"  {row.get('key')}: the file is not the version labelled; left out", flush=True)
    per_asset, hits, total = [], 0, 0
    for row in rows:
        work = folder / "work" / row["key"].replace("/", "_")
        zones, record = mz.find_zones(Path(row["asset"]), work, segmenter)
        result = mz.point_agreement(zones, row["points"])
        hits += result["agree"]
        total += result["points"]
        per_asset.append({"key": row["key"], **result, "vocabulary": record["vocabulary"]})
        print(f"  {row['key']}: {result['agree']}/{result['points']}", flush=True)
    agreement = hits / total if total else None
    report = {"segmenter": segmenter, "assets": len(rows), "points": total, "agreement": agreement,
              "threshold": mz.POINT_AGREEMENT,
              "admitted": agreement is not None and agreement >= mz.POINT_AGREEMENT, "per_asset": per_asset,
              "left_out_changed": [r.get("key") for r in changed]}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"report-human-{segmenter}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_asset"}, indent=2))
    return 0


def pbr(folder: Path, segmenter: str) -> int:
    names = json.loads((folder / "selection.json").read_text(encoding="utf-8"))["assets"]
    rows = []
    for name in names:
        packed = (folder / "assets" / f"{name}.glb").read_bytes()
        work = folder / "zones-work" / name
        work.mkdir(parents=True, exist_ok=True)
        stripped = work / f"{name}.glb"
        stripped.write_bytes(pt.strip_maps(packed))
        try:
            zones, _ = mz.find_zones(stripped, work, segmenter)
            truth = pt.true_maps(packed)
            row = {}
            for index, surface in surfaces(EditGLB(stripped.read_bytes())).items():
                if index not in truth or index not in zones:
                    continue
                part = pt.score(*truth[index], *mz.default_maps(zones[index]), surface.coverage(grow=False))
                for key, value in part.items():
                    row[key] = ([a + b for a, b in zip(row[key], value)] if isinstance(value, list)
                                else row[key] + value) if key in row else value
            row["asset"] = name
            rows.append(row)
            print(f"  {name}: scored", flush=True)
        except Exception as problem:  # noqa: BLE001 - one bad asset must not stop the trial
            print(f"  {name}: {type(problem).__name__}: {problem}", flush=True)
    summary = pt.summarise([r for r in rows if r.get("texels")])
    (folder / f"report-zone-defaults-{segmenter}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("roughness", "metallic", "verdict")}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("human", "pbr"):
        command = sub.add_parser(step)
        command.add_argument("--dir", type=Path, required=True)
        command.add_argument("--segmenter", choices=mz.SEGMENTERS, default="sam2")
        if step == "human":
            command.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.step == "human":
        return human(args.dir, args.labels, args.segmenter)
    return pbr(args.dir, args.segmenter)


if __name__ == "__main__":
    raise SystemExit(main())
