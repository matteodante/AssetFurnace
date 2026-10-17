#!/usr/bin/env python3
"""Build the pictures of the material label sheet, and bring its labels back into the project's record.

    python scripts/label_sheet.py build assets.json --out output/label-sheet   # render every asset's pictures
    python scripts/label_sheet.py import labels.json --sheet output/label-sheet  # the sheet's labels -> record

The sheet is a page the user opens on a phone or a computer (design v2, section 6). For each
asset it shows four views of the albedo with no light, for "is light painted in?" and for the
points where the user taps a material, and the front under three lights, for "do the materials
look right?". `build` renders those pictures with Blender and keeps each view's UV pass, so
`import` can turn every tap into the texel and glTF material it landed on and append the labels
to `output/material-labels.jsonl` with the asset's hash, exactly as Studio's panel does.

`assets.json` lists ``{"key", "path", "kind"}`` (kind: humanoid, creature or prop); `labels.json`
is the sheet's ``labels`` collection, as `ArtifactData list` returns it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import material_labels as ml  # noqa: E402

WORDS = list(ml.MATERIALS)
SIZE = 640
# Where the three lights come from (the direction they travel, Blender's world axes): a key
# from the front left above, a hard side light, and a light from behind and above.
LIGHTS = {"front": (0.5, 0.7, -0.5), "side": (-0.95, 0.0, -0.3), "back": (0.0, -0.8, -0.6)}


def tap_to_point(uv_rgba: np.ndarray, x: float, y: float) -> dict | None:
    """The texel (glTF uv, v downward) and material under a tap at (x, y), each 0..1 of the picture."""
    height, width = uv_rgba.shape[:2]
    row = min(max(int(y * height), 0), height - 1)
    col = min(max(int(x * width), 0), width - 1)
    u, v, material, coverage = (float(c) for c in uv_rgba[row, col, :4])
    index = round(material) - 1
    if coverage < 0.5 or index < 0:
        return None
    return {"uv": [round(u % 1.0, 6), round(1.0 - (v % 1.0), 6)], "material": index}


def build(listing: Path, out: Path) -> int:
    from image_to_3dlab import multiview as mv

    assets = json.loads(listing.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"assets": []}
    for item in assets:
        path, key = Path(item["path"]), item["key"]
        folder = out / key
        views = mv.render_views(path, folder / "views", size=SIZE, around=4)
        entry = {"key": key, "path": str(path.resolve()), "kind": item.get("kind", "prop"),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "views": [], "lit": []}
        for view in views[:4]:
            picture = folder / f"view_{view.index:02d}.jpg"
            Image.fromarray(np.where((view.material >= 0)[..., None], view.color, 238).astype(np.uint8)).save(
                picture, quality=86)
            uv = folder / f"view_{view.index:02d}_uv.npz"
            uv.write_bytes((folder / "views" / f"view_{view.index:02d}_uv.npz").read_bytes())
            entry["views"].append({"index": view.index, "file": f"{key}/{picture.name}", "uv": f"{key}/{uv.name}"})
        for name, light in LIGHTS.items():
            # View 1 of a 4-view ring looks at the model's front (glTF +Z faces Blender's -Y).
            lit = mv.render_views(path, folder / f"lit_{name}", size=SIZE, around=4, lit=True, light=light)[1]
            picture = folder / f"lit_{name}.jpg"
            Image.fromarray(np.where((lit.material >= 0)[..., None], lit.lit, 238).astype(np.uint8)).save(
                picture, quality=86)
            entry["lit"].append({"light": name, "file": f"{key}/{picture.name}"})
        manifest["assets"].append(entry)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  {key}", flush=True)
    return 0


def import_labels(sheet: Path, labels: list[dict], log: Path) -> dict:
    """Append the sheet's labels to the record; returns what was imported and what was not."""
    manifest = {a["key"]: a for a in json.loads((sheet / "manifest.json").read_text(encoding="utf-8"))["assets"]}
    report = {"imported": 0, "points": 0, "dropped_points": 0, "unknown": [], "changed": []}
    for doc in labels:
        key, data = doc.get("id"), doc.get("data") or {}
        entry = manifest.get(key)
        if entry is None:
            report["unknown"].append(key)
            continue
        path = Path(entry["path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            report["changed"].append(key)                       # labels belong to the version shown
            continue
        views = {v["index"]: v for v in entry["views"]}
        points = []
        for tap in data.get("points") or []:
            if tap.get("label") not in WORDS:
                raise ValueError(f"{key}: a point's label must be one of {', '.join(WORDS)}, got {tap.get('label')!r}")
            view = views.get(tap.get("v"))
            point = tap_to_point(np.load(sheet / view["uv"])["rgba"], float(tap["x"]), float(tap["y"])) \
                if view else None
            if point is None:
                report["dropped_points"] += 1
                continue
            points.append({**point, "label": tap["label"]})
        fields = {k: data[k] for k in ("baked_light", "materials_look", "note") if data.get(k) is not None}
        if points:
            fields["points"] = points
        if not fields:
            continue
        ml.record(log, key, str(path), entry["sha256"], **fields)
        report["imported"] += 1
        report["points"] += len(points)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    make = sub.add_parser("build")
    make.add_argument("listing", type=Path)
    make.add_argument("--out", type=Path, required=True)
    bring = sub.add_parser("import")
    bring.add_argument("labels", type=Path)
    bring.add_argument("--sheet", type=Path, required=True)
    bring.add_argument("--log", type=Path, default=REPO / "output" / "material-labels.jsonl")
    args = parser.parse_args(argv)
    if args.step == "build":
        return build(args.listing, args.out)
    report = import_labels(args.sheet, json.loads(args.labels.read_text(encoding="utf-8")), args.log)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
