#!/usr/bin/env python3
"""Build, train and test the learned baked-light check (design sections 3B and 6).

    python scripts/baked_light_trial.py fetch --dir output/baked-light --count 100   # CC0 models, asks first
    python scripts/baked_light_trial.py make --dir output/baked-light                # synthetic pairs + embeddings (CPU)
    python scripts/baked_light_trial.py train --dir output/baked-light               # head, threshold, synthetic test
    python scripts/baked_light_trial.py marigold --dir output/baked-light            # third candidate, GPU lock per batch
    python scripts/baked_light_trial.py qwen --dir output/baked-light                # second candidate, GPU lock per batch
    python scripts/baked_light_trial.py human --dir output/baked-light --labels labels.json

`make` renders each clean Poly Haven model from 8 + 2 views, unlit and under two suns fixed in
the world, paints each sun's shading into the albedo at a random strength, and embeds every
clean and baked view with SigLIP 2 on the processor. `train` fits two heads on half the assets,
one on the mean over the views and one on each view (an asset then scores the mean of its
views); the calibration quarter picks the head and the threshold, the last quarter tests. `marigold` scores the
same calibration and test samples by how much shading Marigold-IID Lighting finds, with no
training. `human` scores the assets the user labelled (`scripts/material_label_set.py
export`) and is the only verdict that can admit the check.
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
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import pbr_trial  # noqa: E402

from image_to_3dlab import baked_light as bl  # noqa: E402
from image_to_3dlab import material_models as mm  # noqa: E402
from image_to_3dlab import multiview as mv  # noqa: E402

SIZE, AROUND, LIGHTS = 384, 8, 2


def light_for(name: str, k: int) -> tuple[list[float], float]:
    """A sun from above, fixed per asset and k, and the strength its shading is painted at."""
    digest = hashlib.sha1(f"{name}:{k}".encode()).digest()
    azimuth = digest[0] / 255 * 2 * np.pi
    elevation = np.radians(25 + digest[1] / 255 * 45)
    strength = 0.35 + digest[2] / 255 * 0.65
    travel = [-np.cos(elevation) * np.cos(azimuth), -np.cos(elevation) * np.sin(azimuth), -np.sin(elevation)]
    return [round(float(v), 6) for v in travel], round(float(strength), 3)


def embedder():
    from image_to_3dlab.style.embed import load_embedder

    model = load_embedder("siglip2")
    return model.embed, model.record


def make(folder: Path) -> int:
    names = json.loads((folder / "selection.json").read_text(encoding="utf-8"))["assets"]
    split = bl.split_assets(names)
    keep_pictures = set(split["calibration"]) | set(split["test"])
    out = folder / "samples"
    out.mkdir(parents=True, exist_ok=True)
    embed, record = embedder()
    for name in names:
        target = out / f"{name}.npz"
        if target.is_file():
            continue
        asset = folder / "assets" / f"{name}.glb"
        rows = []
        try:
            for k in range(LIGHTS):
                light, strength = light_for(name, k)
                views = mv.render_views(asset, folder / "work" / name / f"light{k}", size=SIZE, around=AROUND,
                                        lit=True, light=light)
                normals = [bl.normal_picture(v) for v in views]
                normal_views = embed(normals)
                variants = [("baked", strength, bl.bake_views(views, strength))]
                if k == 0:
                    variants.insert(0, ("clean", 0.0, [v.color for v in views]))
                for kind, power, colours in variants:
                    pictures = [bl.picture(c, v.material) for c, v in zip(colours, views)]
                    albedo_views = embed(pictures)
                    rows.append({"kind": kind, "strength": power, "light": light,
                                 "views": np.concatenate([albedo_views, normal_views], axis=1)})
                    if name in keep_pictures:
                        keep = folder / "pictures" / name / f"{kind}{'' if kind == 'clean' else k}"
                        keep.mkdir(parents=True, exist_ok=True)
                        for v, p in zip(views, pictures):
                            if not (keep / f"view_{v.index:02d}.png").is_file():   # Marigold may be reading it
                                p.save(keep / f"view_{v.index:02d}.png")
        except Exception as problem:  # noqa: BLE001 - one bad asset must not stop the set
            print(f"  {name}: {type(problem).__name__}: {problem}", flush=True)
            continue
        np.savez(target, kind=[r["kind"] for r in rows], strength=[r["strength"] for r in rows],
                 light=[r["light"] for r in rows], views=np.stack([r["views"] for r in rows]))
        print(f"  {name}: {len(rows)} samples", flush=True)
    (folder / "embedder.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


def load_samples(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(per-view features S x V x D, label, asset, strength) per sample."""
    x, y, groups, strength = [], [], [], []
    for path in sorted((folder / "samples").glob("*.npz")):
        data = np.load(path)
        x.append(data["views"])
        y += [1.0 if k == "baked" else 0.0 for k in data["kind"]]
        strength += list(data["strength"])
        groups += [path.stem] * len(data["kind"])
    return np.vstack(x), np.array(y), np.array(groups), np.array(strength)


def pick_l2(x, y, groups, grid=(0.1, 1.0, 10.0, 100.0, 1000.0)) -> float:
    """The L2 strength with the best mean AUC over 4 folds of the training assets."""
    names = sorted(set(groups), key=lambda n: hashlib.sha1(n.encode()).hexdigest())
    folds = [set(names[i::4]) for i in range(4)]
    best, best_auc = grid[0], -1.0
    for l2 in grid:
        aucs = []
        for fold in folds:
            held = np.isin(groups, list(fold))
            head = bl.LogisticHead.fit(x[~held], y[~held], l2=l2)
            aucs.append(bl.roc_auc(y[held], head.score(x[held])))
        if np.nanmean(aucs) > best_auc:
            best, best_auc = l2, float(np.nanmean(aucs))
    return best


def evaluate(y_cal, s_cal, y_test, s_test, g_test, human: bool) -> dict:
    threshold = bl.choose_threshold(y_cal, s_cal)
    caught, flagged = bl.rates(y_test, s_test, threshold)
    auc = bl.roc_auc(y_test, s_test)
    low = bl.auc_lower_bound(y_test, s_test, g_test)
    return {"auc": auc, "auc_lower_bound": low, "threshold": threshold, "caught": caught, "flagged": flagged,
            "test_samples": len(y_test), "test_assets": len(set(g_test)),
            "verdict": bl.verdict(low, caught, flagged, human)}


def fit_variant(variant: str, views: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[bl.LogisticHead, float]:
    """A head on the mean over the views ("mean") or on every view ("views")."""
    if variant == "mean":
        x = views.mean(axis=1)
        l2 = pick_l2(x, y, groups)
        return bl.LogisticHead.fit(x, y, l2=l2), l2
    count = views.shape[1]
    x = views.reshape(-1, views.shape[2])
    l2 = pick_l2(x, np.repeat(y, count), np.repeat(groups, count))
    return bl.LogisticHead.fit(x, np.repeat(y, count), l2=l2), l2


def score_variant(variant: str, head: bl.LogisticHead, views: np.ndarray) -> np.ndarray:
    if variant == "mean":
        return head.score(views.mean(axis=1))
    return head.score(views.reshape(-1, views.shape[2])).reshape(views.shape[:2]).mean(axis=1)


def train(folder: Path) -> int:
    views, y, groups, strength = load_samples(folder)
    split = bl.split_assets(sorted(set(groups)))
    side = {name: key for key, names in split.items() for name in names}
    where = np.array([side[g] for g in groups])
    tr, cal, te = where == "train", where == "calibration", where == "test"
    heads = {v: fit_variant(v, views[tr], y[tr], groups[tr]) for v in ("mean", "views")}
    calibration = {v: bl.roc_auc(y[cal], score_variant(v, h, views[cal])) for v, (h, _) in heads.items()}
    variant = max(calibration, key=calibration.get)            # chosen on calibration, never on the test
    head, l2 = heads[variant]
    record = json.loads((folder / "embedder.json").read_text(encoding="utf-8"))
    s_cal, s_te = score_variant(variant, head, views[cal]), score_variant(variant, head, views[te])
    result = evaluate(y[cal], s_cal, y[te], s_te, groups[te], human=False)
    mild = (y[te] == 0) | (strength[te] < 0.6)
    result.update({"variant": variant, "calibration_auc": calibration, "l2": l2,
                   "train_assets": len(set(groups[tr])),
                   "auc_mild_bakes": bl.roc_auc(y[te][mild], s_te[mild]), "embedder": record,
                   "note": "synthetic test: says the head learned the synthetic difference, not that it is admitted"})
    bl.save_head(head, folder / "head.json", {"variant": variant, "l2": l2, "threshold": result["threshold"],
                                               "embedder": record, "trained_on": "synthetic pairs, Poly Haven CC0"})
    (folder / "report-synthetic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "embedder"}, indent=2))
    return 0


def marigold(folder: Path, batch: int = 12) -> int:
    """Third candidate, no training: how much shading Marigold-IID Lighting finds in the albedo."""
    from image_to_3dlab import delight

    pictures = sorted(p for p in (folder / "pictures").glob("*/*") if p.is_dir())
    scores_path = folder / "marigold-scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.is_file() else {}
    todo = [p for p in pictures if f"{p.parent.name}/{p.name}" not in scores]
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        images = [i for p in chunk for i in sorted(p.glob("view_*.png"))]
        outs = [i.with_name(i.stem + "_lighting.npz") for i in images]
        delight.run_intrinsics(images, outs, mm.MARIGOLD_LIGHTING, resolution=SIZE)
        for p in chunk:
            spreads = []
            for i in sorted(p.glob("view_*.png")):
                shading = np.load(i.with_name(i.stem + "_lighting.npz"))["shading"].astype(np.float64).mean(-1)
                shown = np.any(np.asarray(Image.open(i).convert("RGB")) != bl.GREY, axis=-1)
                if shown.sum() > 50:
                    spreads.append(float(np.std(np.log(np.maximum(shading[shown], 1e-4)))))
            scores[f"{p.parent.name}/{p.name}"] = float(np.median(spreads)) if spreads else 0.0
        scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"  {min(start + batch, len(todo))}/{len(todo)}", flush=True)
    names = sorted({k.split("/")[0] for k in scores})
    split = bl.split_assets(json.loads((folder / "selection.json").read_text(encoding="utf-8"))["assets"])

    def arrays(side):
        keys = [k for k in scores if k.split("/")[0] in set(split[side])]
        return (np.array([0.0 if k.endswith("clean") else 1.0 for k in keys]), np.array([scores[k] for k in keys]),
                np.array([k.split("/")[0] for k in keys]))

    y_cal, s_cal, _ = arrays("calibration")
    y_te, s_te, g_te = arrays("test")
    result = evaluate(y_cal, s_cal, y_te, s_te, g_te, human=False)
    result["assets"] = len(names)
    (folder / "report-marigold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def qwen(folder: Path, batch: int = 15) -> int:
    """Second candidate, no training: Qwen3.5-4B asked, view by view, whether light is painted in."""
    pictures = sorted(p for p in (folder / "pictures").glob("*/*") if p.is_dir())
    scores_path = folder / "qwen-scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.is_file() else {}
    todo = [p for p in pictures if f"{p.parent.name}/{p.name}" not in scores]
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        images = {p: sorted(p.glob("view_??.png")) for p in chunk}
        answers = bl.run_qwen("yes_no", [i for p in chunk for i in images[p]], bl.QUESTION, folder / "qwen-work")
        at = 0
        for p in chunk:
            mine = answers[at:at + len(images[p])]
            at += len(images[p])
            scores[f"{p.parent.name}/{p.name}"] = float(np.mean(mine)) if mine else 0.0
        scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"  {min(start + batch, len(todo))}/{len(todo)}", flush=True)
    return report_scores(folder, scores, "report-qwen.json")


def report_scores(folder: Path, scores: dict, name: str) -> int:
    """Calibration and test numbers for a score per sample ("<asset>/<clean|bakedN>")."""
    split = bl.split_assets(json.loads((folder / "selection.json").read_text(encoding="utf-8"))["assets"])

    def arrays(side):
        keys = [k for k in scores if k.split("/")[0] in set(split[side])]
        return (np.array([0.0 if k.endswith("clean") else 1.0 for k in keys]), np.array([scores[k] for k in keys]),
                np.array([k.split("/")[0] for k in keys]))

    y_cal, s_cal, _ = arrays("calibration")
    y_te, s_te, g_te = arrays("test")
    result = evaluate(y_cal, s_cal, y_te, s_te, g_te, human=False)
    (folder / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def human(folder: Path, labels: Path) -> int:
    """Score the user's labelled assets with the trained head; the only verdict that can admit it."""
    head, record = bl.load_head(folder / "head.json")
    variant = record.get("variant", "mean")
    rows = json.loads(labels.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("baked_light") is not None]
    embed, _ = embedder()
    names, y, s = [], [], []
    for row in rows:
        asset = Path(row["asset"])
        views = mv.render_views(asset, folder / "human" / row["key"].replace("/", "_"), size=SIZE, around=AROUND)
        per_view = np.concatenate([embed([bl.picture(v.color, v.material) for v in views]),
                                   embed([bl.normal_picture(v) for v in views])], axis=1)
        names.append(row["key"])
        y.append(1.0 if row["baked_light"] else 0.0)
        s.append(float(score_variant(variant, head, per_view[None])[0]))
    y, s, names = np.array(y), np.array(s), np.array(names)
    split = bl.split_assets(list(names))
    cal, te = np.isin(names, split["train"] + split["calibration"]), np.isin(names, split["test"])
    result = evaluate(y[cal], s[cal], y[te], s[te], names[te], human=True)
    result["labelled_assets"] = len(names)
    (folder / "report-human.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    for step in ("fetch", "make", "train", "marigold", "qwen", "human"):
        command = sub.add_parser(step)
        command.add_argument("--dir", type=Path, required=True)
        if step == "fetch":
            command.add_argument("--count", type=int, default=100)
            command.add_argument("--yes", action="store_true")
        if step == "human":
            command.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.step == "fetch":
        return pbr_trial.fetch(args.dir, args.count, args.yes)
    return {"make": make, "train": train, "marigold": marigold, "qwen": qwen}[args.step](args.dir) if args.step != "human" \
        else human(args.dir, args.labels)


if __name__ == "__main__":
    raise SystemExit(main())
