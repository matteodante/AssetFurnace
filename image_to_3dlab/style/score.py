"""How close a concept is to a bible, from embedding vectors. Pure numpy, no models.

Two readings of the same vectors:

* `knn`: the mean cosine similarity to the k nearest approved pictures. The nearest, not
  the centre: a bible holds knights and chests, and a new chest should look like the
  approved chests, not like an average of a knight and a chest.
* `csls`: the same, minus how "popular" those neighbours are (cross-domain similarity
  local scaling). A bible picture that is close to everything says little when it is
  close to the concept too. A 2026 audit of CSD found this reading separates styles
  better than raw cosine (arXiv 2605.09030). The popularity of a bible picture is its
  mean similarity to its k nearest pictures in a background pool (the other concepts
  seen), which is what CSLS measures.

Starting thresholds come from the bible itself: score every approved picture against the
others ("leave one out"). That says how alike good pictures are to each other.
"""

from __future__ import annotations

import numpy as np


def _topk_mean(similarity: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, similarity.shape[1]))
    return -np.sort(-similarity, axis=1)[:, :k].mean(axis=1)


def knn(queries: np.ndarray, bible: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean cosine similarity of each query row to its k nearest bible rows (rows are unit length)."""
    if len(bible) == 0:
        raise ValueError("the bible has no approved pictures to compare with")
    return _topk_mean(queries @ bible.T, k)


def hubness(bible: np.ndarray, pool: np.ndarray, k: int = 3) -> np.ndarray:
    """How close each bible picture is to its k nearest pictures in `pool` (exact copies left out)."""
    similarity = bible @ pool.T
    similarity[similarity > 0.9999] = -np.inf  # the same picture, when the pool contains it
    finite = np.where(np.isfinite(similarity), similarity, np.nan)
    k = max(1, min(k, pool.shape[0]))
    ordered = -np.sort(-np.nan_to_num(finite, nan=-np.inf), axis=1)[:, :k]
    ordered[~np.isfinite(ordered)] = np.nan
    return np.nanmean(ordered, axis=1)


def csls(queries: np.ndarray, bible: np.ndarray, pool: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean over the k nearest bible rows of 2*cos(q, b) - r(b), minus r(q); higher is closer.

    r(q) is the query's own mean similarity to its k nearest bible rows, r(b) the bible
    row's hubness in `pool`.
    """
    if len(bible) == 0:
        raise ValueError("the bible has no approved pictures to compare with")
    similarity = queries @ bible.T
    adjusted = 2 * similarity - hubness(bible, pool, k)[None, :]
    k_eff = max(1, min(k, bible.shape[0]))
    nearest = np.argsort(-similarity, axis=1)[:, :k_eff]
    picked = np.take_along_axis(adjusted, nearest, axis=1).mean(axis=1)
    return picked - _topk_mean(similarity, k)


def leave_one_out(bible: np.ndarray, k: int = 3, pool: np.ndarray | None = None,
                  method: str = "knn") -> np.ndarray:
    """Each approved picture scored against the others: the spread of good pictures."""
    scores = []
    for i in range(len(bible)):
        rest = np.delete(bible, i, axis=0)
        if method == "csls":
            scores.append(csls(bible[i:i + 1], rest, pool if pool is not None else rest, k)[0])
        else:
            scores.append(knn(bible[i:i + 1], rest, k)[0])
    return np.asarray(scores)


def starting_thresholds(loo: np.ndarray) -> dict[str, float]:
    """warn below the 10th percentile of the good pictures; fail that far again below it."""
    if len(loo) < 2:
        raise ValueError("at least two approved pictures are needed to set thresholds")
    p10, p50 = np.percentile(loo, [10, 50])
    return {"warn": float(p10), "fail": float(p10 - (p50 - p10))}
