"""Style scores from vectors, and whether they separate the user's labels. No models."""

import numpy as np
import pytest

from image_to_3dlab.style import evaluate, score


def unit(*rows):
    array = np.asarray(rows, dtype=float)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


BIBLE = unit([1, 0, 0], [0.9, 0.1, 0], [0, 1, 0])


def test_knn_is_the_mean_similarity_to_the_nearest_approved_pictures():
    query = unit([1, 0, 0])
    assert score.knn(query, BIBLE, k=1)[0] == pytest.approx(1.0)
    expected = (1.0 + BIBLE[1] @ query[0]) / 2
    assert score.knn(query, BIBLE, k=2)[0] == pytest.approx(expected)
    assert score.knn(query, BIBLE, k=10)[0] == pytest.approx(np.mean(BIBLE @ query[0]))  # k capped
    with pytest.raises(ValueError, match="no approved"):
        score.knn(query, np.zeros((0, 3)))


def test_csls_lowers_a_match_with_a_picture_that_is_close_to_everything():
    hub = unit([1, 1, 1])
    plain = unit([1, 0, 0])
    pool = unit([1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1], [0, 0, 1])
    query = unit([1, 0.5, 0.5])
    with_hub = score.csls(query, np.vstack([hub]), pool, k=1)[0]
    with_plain = score.csls(query, np.vstack([plain]), pool, k=1)[0]
    raw_hub, raw_plain = (query @ hub.T)[0, 0], (query @ plain.T)[0, 0]
    # The raw cosine prefers the hub; CSLS discounts it for being close to the whole pool.
    assert raw_hub > raw_plain
    assert with_hub - with_plain < raw_hub - raw_plain


def test_hubness_ignores_the_picture_itself_in_the_pool():
    pool = unit([1, 0, 0], [0, 1, 0])
    assert score.hubness(unit([1, 0, 0]), pool, k=1)[0] == pytest.approx(0.0)


def test_leave_one_out_and_the_starting_thresholds():
    loo = score.leave_one_out(BIBLE, k=1)
    assert loo[0] == pytest.approx(BIBLE[0] @ BIBLE[1])
    assert loo[2] == pytest.approx(BIBLE[2] @ BIBLE[1])
    found = score.starting_thresholds(np.array([0.5, 0.6, 0.7, 0.8, 0.9]))
    p10, p50 = np.percentile([0.5, 0.6, 0.7, 0.8, 0.9], [10, 50])
    assert found == {"warn": pytest.approx(p10), "fail": pytest.approx(p10 - (p50 - p10))}
    with pytest.raises(ValueError):
        score.starting_thresholds(np.array([0.5]))


def test_roc_auc_is_one_for_perfect_separation_and_a_half_for_ties():
    labels = [True, True, False, False]
    assert evaluate.roc_auc([0.9, 0.8, 0.2, 0.1], labels) == 1.0
    assert evaluate.roc_auc([0.1, 0.2, 0.8, 0.9], labels) == 0.0
    assert evaluate.roc_auc([0.5, 0.5, 0.5, 0.5], labels) == 0.5
    with pytest.raises(ValueError):
        evaluate.roc_auc([0.1], [True])


def test_the_operating_point_respects_the_budget_for_good_concepts():
    scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.3, 0.2, 0.4, 0.62]
    labels = [True] * 10 + [False] * 4
    point = evaluate.operating_point(scores, labels, max_good_stopped=0.1)
    assert point["good_stopped"] <= 0.1
    # 0.45 and 0.5 both catch 3 of 4; 0.45 stops no good concept, so it wins.
    assert point["threshold"] == pytest.approx(0.45)
    assert point["caught"] == pytest.approx(0.75) and point["good_stopped"] == 0.0


def test_evaluate_admits_only_what_meets_the_rule():
    labels = [True] * 10 + [False] * 10
    good = evaluate.evaluate(list(np.linspace(0.6, 1, 10)) + list(np.linspace(0, 0.4, 10)), labels)
    assert good["auc"] == 1.0 and good["admitted"] and good["approved"] == 10
    low, high = good["auc_95"]
    assert low <= good["auc"] <= high
    coin = evaluate.evaluate(list(np.random.default_rng(1).random(20)), labels)
    assert not coin["admitted"]
