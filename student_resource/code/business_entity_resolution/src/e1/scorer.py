"""
Official Competition Macro-F0.5 Scorer for E1.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
F0.5 is the official competition objective and is entity-level macro-averaged, so all experiments
(blocking evaluation, candidate ranking, LightGBM feature selection, threshold calibration, and
greedy decision tuning) must evaluate against the exact same canonical implementation without
deviation or metric drift.
"""

from statistics import mean, median
from typing import Any, Collection, Dict, List, Optional, Set, Tuple, Union


def score_single_entity(
    true_items: Union[Set[str], Collection[str]],
    pred_items: Union[Set[str], Collection[str]],
    check_duplicates: bool = True,
) -> Tuple[float, int, int, int, int, int]:
    """
    Compute F0.5 for a single Source-1 entity according to official competition rules.

    Args:
        true_items: Collection or set of ground-truth matched entity IDs.
        pred_items: Collection or set of predicted matched entity IDs.
        check_duplicates: If True and pred_items is a sequence/list, reject duplicate IDs.

    Returns:
        (f0_5, tp, fp, fn, m, n)
        where:
          f0_5: computed score in [0.0, 1.0]
          tp: true positives
          fp: false positives
          fn: false negatives
          m: true count |true_set|
          n: predicted count |pred_set|
    """
    if check_duplicates and not isinstance(pred_items, set):
        pred_list = list(pred_items)
        if len(pred_list) != len(set(pred_list)):
            duplicates = [x for x in set(pred_list) if pred_list.count(x) > 1]
            raise ValueError(
                f"Duplicate prediction IDs are strictly prohibited in competition output. Found duplicates: {duplicates}"
            )

    true_set = set(true_items)
    pred_set = set(pred_items)

    m = len(true_set)
    n = len(pred_set)

    # Special Case A: m == 0 and n == 0 -> perfect singleton prediction
    if m == 0 and n == 0:
        return (1.0, 0, 0, 0, 0, 0)

    # Special Case B: m == 0 and n > 0 -> false positive on singleton
    if m == 0 and n > 0:
        return (0.0, 0, n, 0, 0, n)

    # Special Case C: m > 0 and n == 0 -> missed non-singleton
    if m > 0 and n == 0:
        return (0.0, 0, 0, m, m, 0)

    # Special Case D: m > 0 and n > 0 -> standard F0.5 formula
    tp = len(true_set & pred_set)
    fp = n - tp
    fn = m - tp

    # F0.5 = (1 + 0.5^2) * TP / (0.5^2 * m + n) = 1.25 * TP / (0.25 * m + n)
    denominator = 0.25 * m + n
    f0_5 = (1.25 * tp) / denominator if denominator > 0 else 0.0

    return (f0_5, tp, fp, fn, m, n)


def score_predictions(
    truth_by_s1: Dict[str, Union[Set[str], Collection[str]]],
    predictions_by_s1: Dict[str, Union[Set[str], Collection[str]]],
    return_per_entity: bool = True,
    check_duplicates: bool = True,
) -> Dict[str, Any]:
    """
    Compute official macro-averaged F0.5 score across all Source-1 entities.

    Every S1 entity in truth_by_s1 must have an entry in predictions_by_s1 (or defaults to empty set).
    Any prediction for an S1 entity not present in truth_by_s1 is flagged.

    Returns:
        {
            "macro_f0_5": float,
            "num_entities": int,
            "num_singletons": int,
            "singleton_accuracy": float,
            "empty_predictions": int,
            "non_empty_predictions": int,
            "mean_true_matches": float,
            "mean_predicted_matches": float,
            "median_predicted_matches": float,
            "overprediction_rate": float,
            "underprediction_rate": float,
            "exact_set_accuracy": float,
            "aggregate_diagnostics": {
                "pooled_tp": int,
                "pooled_fp": int,
                "pooled_fn": int,
                "pooled_precision": float,
                "pooled_recall": float,
            },
            "per_entity": List[Dict[str, Any]] (optional),
        }
    """
    num_entities = len(truth_by_s1)
    if num_entities == 0:
        return {
            "macro_f0_5": 0.0,
            "num_entities": 0,
            "aggregate_diagnostics": {},
            "per_entity": [],
        }

    scores: List[float] = []
    per_entity_records: List[Dict[str, Any]] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true = 0
    total_pred = 0

    num_singletons = 0
    singleton_correct = 0
    empty_preds = 0
    non_empty_preds = 0
    over_preds = 0
    under_preds = 0
    exact_set_matches = 0

    pred_counts: List[int] = []

    for s1_id, true_items in truth_by_s1.items():
        pred_items = predictions_by_s1.get(s1_id, set())

        f0_5, tp, fp, fn, m, n = score_single_entity(
            true_items=true_items,
            pred_items=pred_items,
            check_duplicates=check_duplicates,
        )

        scores.append(f0_5)
        pred_counts.append(n)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true += m
        total_pred += n

        if m == 0:
            num_singletons += 1
            if n == 0:
                singleton_correct += 1

        if n == 0:
            empty_preds += 1
        else:
            non_empty_preds += 1

        if n > m:
            over_preds += 1
        elif n < m:
            under_preds += 1

        if set(true_items) == set(pred_items):
            exact_set_matches += 1

        if return_per_entity:
            per_entity_records.append({
                "source1_entity_id": s1_id,
                "true_count": m,
                "predicted_count": n,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "f0_5": f0_5,
            })

    macro_f0_5 = mean(scores) if scores else 0.0
    singleton_acc = (singleton_correct / num_singletons) if num_singletons > 0 else 1.0
    pooled_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    pooled_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0

    return {
        "macro_f0_5": macro_f0_5,
        "num_entities": num_entities,
        "num_singletons": num_singletons,
        "singleton_accuracy": singleton_acc,
        "empty_predictions": empty_preds,
        "non_empty_predictions": non_empty_preds,
        "mean_true_matches": total_true / num_entities,
        "mean_predicted_matches": total_pred / num_entities,
        "median_predicted_matches": median(pred_counts) if pred_counts else 0.0,
        "overprediction_rate": over_preds / num_entities,
        "underprediction_rate": under_preds / num_entities,
        "exact_set_accuracy": exact_set_matches / num_entities,
        "aggregate_diagnostics": {
            "pooled_tp": total_tp,
            "pooled_fp": total_fp,
            "pooled_fn": total_fn,
            "pooled_precision": pooled_prec,
            "pooled_recall": pooled_rec,
        },
        "per_entity": per_entity_records if return_per_entity else [],
    }
