"""The style race: each candidate judge scores the same labelled concepts against one bible.

A labels file names concepts and what the user decided about each:

    {"concepts": [{"file": "concepts/troll.png", "label": "approved", "kind": "creature"},
                  {"file": "concepts/wolf.png",  "label": "sent_back"}]}

Paths are relative to the labels file. A concept that is itself in the bible is left out:
it would score as a perfect match with itself and flatter the judge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from image_to_3dlab.style import evaluate, refs, score
from image_to_3dlab.style.bible import KINDS, load, sha256_file
from image_to_3dlab.style.embed import Embedder, embed_paths

LABELS = {"approved": True, "sent_back": False}


def load_labels(path: Path) -> list[dict]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data.get("concepts", []):
        label = item.get("label")
        if label not in LABELS:
            raise ValueError(f"{item.get('file')}: label must be 'approved' or 'sent_back', got {label!r}")
        kind = item.get("kind")
        if kind is not None and kind not in KINDS:
            raise ValueError(f"{item.get('file')}: unknown kind {kind!r}")
        concept = (path.parent / item["file"]).resolve()
        if not concept.is_file():
            raise ValueError(f"{concept} does not exist")
        rows.append({"path": concept, "approved": LABELS[label], "kind": kind})
    if not rows:
        raise ValueError(f"{path} lists no concepts")
    return rows


def without_bible_pictures(root: Path, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    data = load(root)
    inside = {e["sha256"] for key in ("images", "rejected") for e in data[key]}
    kept, dropped = [], []
    for row in rows:
        (dropped if sha256_file(row["path"]) in inside else kept).append(row)
    return kept, dropped


def _bible_paths(root: Path) -> list[Path]:
    return refs.ordered(root, None)


def embedding_scores(embedder: Embedder, root: Path, rows: list[dict], k: int = 3) -> dict[str, list[float]]:
    """`knn` and `csls` scores for every concept; vectors are cached in the bible's .cache/."""
    root = Path(root)
    cache = root / ".cache" / f"embeddings-{embedder.key}.npz"
    bible = embed_paths(embedder, _bible_paths(root), cache=cache)
    concepts = embed_paths(embedder, [r["path"] for r in rows], cache=cache)
    pool = np.vstack([bible, concepts])
    return {"knn": score.knn(concepts, bible, k).tolist(), "csls": score.csls(concepts, bible, pool, k).tolist()}


def judge_scores(judge, root: Path, rows: list[dict], criteria: list[str] | None = None) -> list[float]:
    """The vision-language judge's score per concept, with up to two sets of three references."""
    criteria = criteria if criteria is not None else load(root)["style"]["criteria"]
    out = []
    for row in rows:
        order = refs.ordered(root, row.get("kind"))
        sets = [s for s in (order[0:3], order[3:6]) if s]
        out.append(judge.judge(sets, row["path"], criteria)["score"])
    return out


def report(scores: list[float], rows: list[dict]) -> dict:
    return evaluate.evaluate(scores, [r["approved"] for r in rows])


def timed(function, *args, **kwargs):
    started = time.monotonic()
    value = function(*args, **kwargs)
    return value, time.monotonic() - started
