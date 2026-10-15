#!/usr/bin/env python3
"""Race the style judges on concepts the user has labelled, and say which ones may be used.

    python scripts/style_race.py <bible> labels.json                     # every judge
    python scripts/style_race.py <bible> labels.json --judges csd,siglip2 --out race.json

Judges: `csd` and `siglip2` (embeddings, each read two ways: nearest neighbours and CSLS)
and `vlm` (Qwen3-VL answering the bible's questions; needs the GPU). For each one it
prints the ROC area with a 95% interval, the threshold that stops at most 10% of the
approved concepts, how many sent-back concepts that catches, the time per concept, and
whether it meets the admission rule of the style bible design (area >= 0.90, >= 80%
caught). The labels file format is in `image_to_3dlab/style/race.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab.style import embed, race  # noqa: E402

JUDGES = ("csd", "siglip2", "vlm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bible", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--judges", default=",".join(JUDGES))
    parser.add_argument("--k", type=int, default=3, help="nearest approved pictures to average")
    parser.add_argument("--device", help="cpu (default) or cuda, for the embeddings")
    parser.add_argument("--out", type=Path, help="write the full report here as JSON")
    args = parser.parse_args(argv)
    chosen = [j.strip() for j in args.judges.split(",") if j.strip()]
    unknown = [j for j in chosen if j not in JUDGES]
    if unknown:
        print(f"error: unknown judge {', '.join(unknown)}; choose from {', '.join(JUDGES)}", file=sys.stderr)
        return 2

    rows, dropped = race.without_bible_pictures(args.bible, race.load_labels(args.labels))
    if dropped:
        print(f"left out {len(dropped)} concept(s) that are in the bible itself")
    results, scores, seconds = {}, {}, {}
    for name in chosen:
        if name == "vlm":
            from image_to_3dlab.style.judge import Judge, LlamaServer

            with LlamaServer(log=args.out.with_suffix(".llama.log") if args.out else None) as server:
                found, took = race.timed(race.judge_scores, Judge(server.post), args.bible, rows)
            scores["vlm"], seconds["vlm"] = found, took / len(rows)
        else:
            embedder = embed.load_embedder(name, device=args.device)
            found, took = race.timed(race.embedding_scores, embedder, args.bible, rows, k=args.k)
            for reading, values in found.items():
                scores[f"{name}.{reading}"], seconds[f"{name}.{reading}"] = values, took / len(rows)
    for name, values in scores.items():
        results[name] = {**race.report(values, rows), "seconds_per_concept": seconds[name]}

    print(f"{'judge':<14} {'ROC area':>9} {'95% interval':>14} {'caught':>7} {'good stopped':>13} {'s/concept':>10}  admitted")
    for name, r in results.items():
        low, high = r["auc_95"]
        print(f"{name:<14} {r['auc']:>9.3f} {f'{low:.2f}-{high:.2f}':>14} {r['caught']:>7.0%} "
              f"{r['good_stopped']:>13.0%} {r['seconds_per_concept']:>10.2f}  {'yes' if r['admitted'] else 'no'}")
    if args.out:
        payload = {"bible": str(args.bible), "labels": str(args.labels),
                   "concepts": [{"file": str(r["path"]), "approved": r["approved"], "kind": r["kind"],
                                 **{n: s[i] for n, s in scores.items()}} for i, r in enumerate(rows)],
                   "results": results}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
