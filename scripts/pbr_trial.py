#!/usr/bin/env python3
"""Run the Poly Haven trial for estimated roughness and metallic (design section 6).

    python scripts/pbr_trial.py fetch --dir output/pbr-trial          # 30 CC0 models at 1k, asks first
    python scripts/pbr_trial.py run --dir output/pbr-trial            # estimate each, score it (resumes)
    python scripts/pbr_trial.py report --dir output/pbr-trial         # pooled numbers and the verdict

`fetch` picks the set (half tagged metal, no plants or rocks), downloads each glTF with its
textures from Poly Haven (CC0), checks every file against the md5 the API gives, and packs it
as one GLB. `run` takes each GLB's metallic-roughness map out, estimates it again with
Marigold-IID Appearance (GPU lock held per asset) and scores it against the original.
`report` pools the scores; the blind comparison stays with the user.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import pbr_estimate as pe  # noqa: E402
from image_to_3dlab import pbr_trial as pt  # noqa: E402
from image_to_3dlab.edit_glb import EditGLB  # noqa: E402
from image_to_3dlab.materials import _texture_image, surfaces  # noqa: E402

API = "https://api.polyhaven.com"
AGENT = {"User-Agent": "assetfurnace-pbr-trial"}
LICENCE = {"schema_version": 1, "classification": "commercial-clear", "models": [],
           "license": {"name": "CC0-1.0", "url": "https://polyhaven.com/license"},
           "source": "Poly Haven"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as response:
        return response.read()


def fetch(folder: Path, count: int, yes: bool, resolution: str = "1k") -> int:
    catalogue = json.loads(get(f"{API}/assets?t=models"))
    chosen = pt.select_assets(catalogue, count)
    plans = {}
    for name in chosen:
        files = json.loads(get(f"{API}/files/{name}"))["gltf"][resolution]["gltf"]
        plans[name] = [(files["url"], f"{name}.gltf", files["md5"], files["size"])] + [
            (item["url"], relative, item["md5"], item["size"]) for relative, item in files["include"].items()]
    total = sum(size for plan in plans.values() for *_, size in plan)
    print(f"Poly Haven (CC0): {len(chosen)} models at {resolution}, {total / 1024 ** 2:.0f} MB, into {folder}")
    if not yes and (not sys.stdin.isatty() or input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}):
        print("Nothing downloaded.")
        return 1
    for name, plan in plans.items():
        source = folder / "src" / name
        for url, relative, md5, _ in plan:
            target = source / relative
            if target.is_file() and hashlib.md5(target.read_bytes()).hexdigest() == md5:
                continue
            data = get(url)
            if hashlib.md5(data).hexdigest() != md5:
                raise SystemExit(f"{url} does not match the md5 Poly Haven gives; stopped")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        packed = folder / "assets" / f"{name}.glb"
        packed.parent.mkdir(parents=True, exist_ok=True)
        packed.write_bytes(pt.pack_gltf(source / f"{name}.gltf"))
        packed.with_suffix(".provenance.json").write_text(
            json.dumps({**LICENCE, "url": f"https://polyhaven.com/a/{name}"}, indent=2), encoding="utf-8")
        print(f"  {name}", flush=True)
    (folder / "selection.json").write_text(json.dumps({"resolution": resolution, "assets": chosen,
                                                       "license": "CC0-1.0"}, indent=2), encoding="utf-8")
    return 0


def estimated_maps(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    glb = EditGLB(path.read_bytes())
    out = {}
    for index in surfaces(glb):
        image = _texture_image(glb, glb.document["materials"][index]["pbrMetallicRoughness"]
                               .get("metallicRoughnessTexture"))
        if image is not None:
            out[index] = (image[..., 1] / 255.0, image[..., 2] / 255.0)
    return out


def score_asset(folder: Path, name: str) -> dict:
    """The asset's scores, from its packed original and the estimate the run left on disk."""
    truth = pt.true_maps((folder / "assets" / f"{name}.glb").read_bytes())
    work = folder / "work" / name
    estimate = estimated_maps(work / "out" / f"{name}.glb")
    glb = EditGLB((work / f"{name}.glb").read_bytes())
    row: dict = {}
    for index, surface in surfaces(glb).items():
        if index not in truth or index not in estimate:
            continue
        part = pt.score(*truth[index], *estimate[index], surface.coverage(grow=False))
        for key, value in part.items():
            row[key] = (np.add(row[key], value).tolist() if isinstance(value, list) else row[key] + value) \
                if key in row else value
    row["asset"] = name
    return row


def run(folder: Path, views: int) -> int:
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))["assets"]
    results = folder / "results.jsonl"
    done = {json.loads(line)["asset"] for line in results.read_text(encoding="utf-8").splitlines()} \
        if results.is_file() else set()
    for name in selection:
        if name in done:
            continue
        packed = (folder / "assets" / f"{name}.glb").read_bytes()
        work = folder / "work" / name
        work.mkdir(parents=True, exist_ok=True)
        stripped = work / f"{name}.glb"
        stripped.write_bytes(pt.strip_maps(packed))
        stripped.with_suffix(".provenance.json").write_text(json.dumps(LICENCE, indent=2), encoding="utf-8")
        try:
            pe.estimate_pbr(stripped, work / "out", around=views, work=work)
            row = score_asset(folder, name)
        except Exception as problem:  # noqa: BLE001 - one bad asset must not stop the trial
            row = {"asset": name, "error": f"{type(problem).__name__}: {problem}"}
        with results.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row) + "\n")
        print(f"  {name}: {row.get('error') or 'scored'}", flush=True)
    return 0


def report(folder: Path) -> int:
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    # Scored again from the files, so a newer measure applies to runs already done.
    scored = [score_asset(folder, r["asset"]) for r in rows if "error" not in r]
    scored = [r for r in scored if r.get("texels")]
    summary = pt.summarise(scored)
    summary["failed"] = [r for r in rows if "error" in r]
    (folder / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rough, metal = summary["roughness"], summary["metallic"]
    print(f"{summary['assets']} assets scored, {len(summary['failed'])} failed")
    print(f"roughness RMSE {rough['rmse']:.3f} (constant {rough['constant_rmse']:.3f}, "
          f"{(rough['improvement'] or 0) * 100:.0f}% better)")
    print(f"metallic F1 {metal['f1'] if metal['f1'] is None else round(metal['f1'], 3)}, "
          f"wrong on {metal['wrong_share'] * 100:.1f}% of the area (true metal {metal['true_share'] * 100:.1f}%); "
          f"best F1 in hindsight {metal['best_f1'] if metal['best_f1'] is None else round(metal['best_f1'], 3)} "
          f"at {metal['best_threshold']}")
    print(json.dumps(summary["verdict"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("fetch", "run", "report"):
        command = sub.add_parser(step)
        command.add_argument("--dir", type=Path, required=True)
        if step == "fetch":
            command.add_argument("--count", type=int, default=30)
            command.add_argument("--yes", action="store_true")
        if step == "run":
            command.add_argument("--views", type=int, default=16)
    args = parser.parse_args(argv)
    if args.step == "fetch":
        return fetch(args.dir, args.count, args.yes)
    if args.step == "run":
        return run(args.dir, args.views)
    return report(args.dir)


if __name__ == "__main__":
    raise SystemExit(main())
