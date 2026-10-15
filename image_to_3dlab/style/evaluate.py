"""Does a style score separate what the user approves from what they send back?

    report = evaluate(scores, labels)   # labels: True = approved, False = sent back

The area under the ROC curve is the chance that a random approved concept scores higher
than a random sent-back one: 1 is perfect, 0.5 is a coin. The operating point is the
highest threshold that stops at most `max_good_stopped` of the approved concepts; at it,
the report says how many sent-back concepts are caught. The admission rule of the design
(docs/plans/2026-10-15-style-bible-design.md, section 7) is area >= 0.90 and, at that
point, at least 80% caught with at most 10% of the good ones stopped.
"""

from __future__ import annotations

import numpy as np

MIN_AUC = 0.90
MIN_CAUGHT = 0.80
MAX_GOOD_STOPPED = 0.10


def roc_auc(scores, labels) -> float:
    """Mann-Whitney: P(score of an approved > score of a sent back), ties counted as half."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    good, bad = scores[labels], scores[~labels]
    if len(good) == 0 or len(bad) == 0:
        raise ValueError("need at least one approved and one sent-back concept")
    greater = (good[:, None] > bad[None, :]).sum()
    ties = (good[:, None] == bad[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(good) * len(bad)))


def operating_point(scores, labels, max_good_stopped: float = MAX_GOOD_STOPPED) -> dict:
    """The threshold (stop below it) that catches the most sent-back concepts within the budget."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    good, bad = scores[labels], scores[~labels]
    best = None
    for threshold in np.unique(scores):
        good_stopped = float((good < threshold).mean())
        if good_stopped > max_good_stopped + 1e-12:
            continue
        caught = float((bad < threshold).mean())
        if best is None or caught > best["caught"] or (caught == best["caught"] and threshold < best["threshold"]):
            best = {"threshold": float(threshold), "caught": caught, "good_stopped": good_stopped}
    return best or {"threshold": float(scores.min()), "caught": 0.0, "good_stopped": 0.0}


def auc_interval(scores, labels, rounds: int = 2000, seed: int = 0) -> tuple[float, float]:
    """A 95% bootstrap interval for the area: how much 40 labels can be trusted."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    good, bad = np.flatnonzero(labels), np.flatnonzero(~labels)
    values = []
    for _ in range(rounds):
        pick = np.concatenate([rng.choice(good, len(good)), rng.choice(bad, len(bad))])
        values.append(roc_auc(scores[pick], labels[pick]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def evaluate(scores, labels) -> dict:
    auc = roc_auc(scores, labels)
    point = operating_point(scores, labels)
    admitted = auc >= MIN_AUC and point["caught"] >= MIN_CAUGHT and point["good_stopped"] <= MAX_GOOD_STOPPED
    return {"auc": auc, "auc_95": auc_interval(scores, labels), **point, "admitted": admitted,
            "approved": int(np.sum(labels)), "sent_back": int(np.sum(~np.asarray(labels, dtype=bool)))}
