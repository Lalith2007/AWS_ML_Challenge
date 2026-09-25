"""
Unit tests for official competition macro-F0.5 scorer.
Amazon ML Challenge 2026 — E1.
"""

import pytest
from e1.scorer import score_predictions, score_single_entity


def test_official_competition_example():
    """
    Official example:
    Truth: [S2-00047, S3-00812]
    Prediction: [S2-00047, S2-00193, S3-00812]
    Expected: TP=2, m=2, n=3 -> F0.5 = 1.25 * 2 / (0.25 * 2 + 3) = 2.5 / 3.5 = 5/7 ≈ 0.7142857
    """
    truth = ["S2-00047", "S3-00812"]
    pred = ["S2-00047", "S2-00193", "S3-00812"]

    score, tp, fp, fn, m, n = score_single_entity(truth, pred)

    assert tp == 2
    assert fp == 1
    assert fn == 0
    assert m == 2
    assert n == 3
    assert abs(score - (5.0 / 7.0)) < 1e-7


def test_correct_empty_singleton():
    """True: [], Pred: [] -> Score: 1.0"""
    score, tp, fp, fn, m, n = score_single_entity([], [])
    assert score == 1.0
    assert tp == 0
    assert fp == 0
    assert fn == 0


def test_false_positive_on_singleton():
    """True: [], Pred: [S2-1] -> Score: 0.0"""
    score, tp, fp, fn, m, n = score_single_entity([], ["S2-1"])
    assert score == 0.0
    assert tp == 0
    assert fp == 1


def test_missed_non_singleton():
    """True: [S2-1], Pred: [] -> Score: 0.0"""
    score, tp, fp, fn, m, n = score_single_entity(["S2-1"], [])
    assert score == 0.0
    assert tp == 0
    assert fn == 1


def test_perfect_one_match():
    """True: [S2-1], Pred: [S2-1] -> Score: 1.0"""
    score, tp, fp, fn, m, n = score_single_entity(["S2-1"], ["S2-1"])
    assert score == 1.0
    assert tp == 1
    assert fp == 0
    assert fn == 0


def test_partial_match_two_of_three():
    """
    True: [S2-1, S2-2, S2-3] (m=3)
    Pred: [S2-1, S2-2] (n=2, tp=2)
    F0.5 = 1.25 * 2 / (0.25 * 3 + 2) = 2.5 / 2.75 = 10 / 11 ≈ 0.9090909
    """
    truth = ["S2-1", "S2-2", "S2-3"]
    pred = ["S2-1", "S2-2"]
    score, tp, fp, fn, m, n = score_single_entity(truth, pred)
    assert tp == 2
    assert m == 3
    assert n == 2
    assert abs(score - (10.0 / 11.0)) < 1e-7


def test_duplicate_prediction_ids_rejected():
    """Duplicate prediction IDs within an S1 entity must raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate prediction IDs"):
        score_single_entity(["S2-1"], ["S2-1", "S2-1"])


def test_mixed_s2_s3_ids():
    """Mixed S2 and S3 IDs evaluate properly."""
    truth = {"S2-100", "S3-200"}
    pred = {"S2-100", "S3-300"}
    score, tp, fp, fn, m, n = score_single_entity(truth, pred)
    assert tp == 1
    assert fp == 1
    assert fn == 1
    # F0.5 = 1.25 * 1 / (0.25 * 2 + 2) = 1.25 / 2.5 = 0.5
    assert abs(score - 0.5) < 1e-7


def test_macro_average_predictions():
    """Test macro-averaging over multiple S1 entities."""
    truth = {
        "S1-1": {"S2-1"},       # perfect match -> 1.0
        "S1-2": set(),          # singleton correct -> 1.0
        "S1-3": {"S2-2"},       # missed match -> 0.0
        "S1-4": set(),          # FP on singleton -> 0.0
    }
    preds = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
        "S1-3": set(),
        "S1-4": {"S2-3"},
    }
    res = score_predictions(truth, preds)
    assert abs(res["macro_f0_5"] - 0.5) < 1e-7
    assert res["num_entities"] == 4
    assert res["num_singletons"] == 2
    assert res["singleton_accuracy"] == 0.5
    assert res["empty_predictions"] == 2
    assert res["non_empty_predictions"] == 2
    assert res["exact_set_accuracy"] == 0.5
