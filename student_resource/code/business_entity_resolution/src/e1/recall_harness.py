"""
Blocking-Recall Evaluation Harness Interface for E1.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Blocking (L2) determines the mathematical upper bound on achievable end-to-end recall.
Every candidate generation lane (K1-K9) and blocking configuration must be evaluated through
this canonical harness to verify retained ground-truth edges, candidate volume, and reduction ratio
before passing candidate pairs to downstream pairwise feature engineering (L3) and LightGBM (L4).
"""

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Collection, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .gt_utils import get_multiplicity_bucket


def validate_candidate_set(
    candidate_by_s1: Dict[str, Union[Set[str], Collection[str]]],
    allowed_s1_ids: Optional[Set[str]] = None,
    matched_by_s1: Optional[Dict[str, Union[Set[str], Collection[str]]]] = None,
) -> Dict[str, Any]:
    """
    Validates candidate set formatting and structural constraints:
    1. All candidate S1 IDs belong to the allowed S1 set (if provided).
    2. All candidate targets begin with 'S2-' or 'S3-'.
    3. No duplicate IDs exist within any single S1 candidate list.
    4. If matched predictions are provided, verifies that matched targets are a subset of candidates.

    Returns:
        {"valid": bool, "issues": List[str]}
    """
    issues: List[str] = []

    for s1_id, cands in candidate_by_s1.items():
        if allowed_s1_ids is not None and s1_id not in allowed_s1_ids:
            issues.append(f"Candidate S1 ID {s1_id} is not in the allowed S1 set for this split.")
            break

        cand_list = list(cands)
        if len(cand_list) != len(set(cand_list)):
            issues.append(f"Candidate list for {s1_id} contains duplicate IDs.")
            break

        for target in cand_list:
            if not target.startswith(("S2-", "S3-")):
                issues.append(f"Invalid target ID prefix in candidates for {s1_id}: '{target}'")
                break
        if issues:
            break

        if matched_by_s1 is not None and s1_id in matched_by_s1:
            matched_set = set(matched_by_s1[s1_id])
            cand_set = set(cand_list)
            missing = matched_set - cand_set
            if missing:
                issues.append(
                    f"Matched targets are not a subset of candidates for {s1_id}: {len(missing)} missing (e.g. {list(missing)[:3]})"
                )
                break

    return {"valid": len(issues) == 0, "issues": issues}


def evaluate_blocking_recall(
    truth_by_s1: Dict[str, Union[Set[str], Collection[str]]],
    candidate_by_s1: Dict[str, Union[Set[str], Collection[str]]],
    metadata_by_s1: Optional[Dict[str, Dict[str, Any]]] = None,
    num_s2_records: int = 5_034_616,
    num_s3_records: int = 5_285_603,
) -> Dict[str, Any]:
    """
    Evaluates blocking recall, coverage, and reduction ratio.

    Args:
        truth_by_s1: {s1_id: set_of_gt_target_ids}
        candidate_by_s1: {s1_id: set_of_candidate_target_ids}
        metadata_by_s1: Optional {s1_id: {"country": ..., "multiplicity_bucket": ..., ...}}
        num_s2_records: Total count of records in Source 2 (for reduction ratio).
        num_s3_records: Total count of records in Source 3 (for reduction ratio).

    Returns:
        Structured dictionary with:
        - edge_recall: global, S2, S3
        - entity_coverage: at least one, all recovered
        - candidate_distribution: total, mean, median, p90, p95, p99, max, min
        - reduction_ratio: naive comparison reduction
        - breakdown: by country, multiplicity bucket, etc.
    """
    total_gt_edges = 0
    recovered_gt_edges = 0

    s2_gt_edges = 0
    s2_recovered_edges = 0
    s3_gt_edges = 0
    s3_recovered_edges = 0

    num_entities_with_gt = 0
    num_singletons = 0
    entities_at_least_one = 0
    entities_all_recovered = 0

    candidate_lengths: List[int] = []

    # Breakdowns
    country_breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    bucket_breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())

    for s1_id, true_items in truth_by_s1.items():
        true_set = set(true_items)
        cand_set = set(candidate_by_s1.get(s1_id, set()))

        n_cand = len(cand_set)
        candidate_lengths.append(n_cand)

        m = len(true_set)
        if m == 0:
            num_singletons += 1
        else:
            num_entities_with_gt += 1

        recovered = true_set & cand_set
        n_rec = len(recovered)

        total_gt_edges += m
        recovered_gt_edges += n_rec

        if m > 0:
            if n_rec > 0:
                entities_at_least_one += 1
            if n_rec == m:
                entities_all_recovered += 1

        # S2 vs S3 breakdown
        for t in true_set:
            if t.startswith("S2-"):
                s2_gt_edges += 1
                if t in cand_set:
                    s2_recovered_edges += 1
            elif t.startswith("S3-"):
                s3_gt_edges += 1
                if t in cand_set:
                    s3_recovered_edges += 1

        # Metadata breakdown
        if metadata_by_s1 is not None and s1_id in metadata_by_s1:
            meta = metadata_by_s1[s1_id]
            c = meta.get("country", "Unknown")
            country_breakdown[c]["total_gt_edges"] += m
            country_breakdown[c]["recovered_gt_edges"] += n_rec
            country_breakdown[c]["total_candidates"] += n_cand

            b = meta.get("multiplicity_bucket", get_multiplicity_bucket(m))
            bucket_breakdown[b]["total_gt_edges"] += m
            bucket_breakdown[b]["recovered_gt_edges"] += n_rec
            bucket_breakdown[b]["total_candidates"] += n_cand
        else:
            b = get_multiplicity_bucket(m)
            bucket_breakdown[b]["total_gt_edges"] += m
            bucket_breakdown[b]["recovered_gt_edges"] += n_rec
            bucket_breakdown[b]["total_candidates"] += n_cand

    num_s1 = len(truth_by_s1)
    total_candidate_edges = sum(candidate_lengths)

    # Candidate lengths distribution
    if candidate_lengths:
        cand_arr = np.array(candidate_lengths)
        cand_mean = float(np.mean(cand_arr))
        cand_median = float(np.median(cand_arr))
        cand_p90 = float(np.percentile(cand_arr, 90))
        cand_p95 = float(np.percentile(cand_arr, 95))
        cand_p99 = float(np.percentile(cand_arr, 99))
        cand_max = int(np.max(cand_arr))
        cand_min = int(np.min(cand_arr))
    else:
        cand_mean = cand_median = cand_p90 = cand_p95 = cand_p99 = 0.0
        cand_max = cand_min = 0

    # Reduction ratio
    naive_pairs = num_s1 * (num_s2_records + num_s3_records)
    reduction_ratio = (
        1.0 - (total_candidate_edges / naive_pairs) if naive_pairs > 0 else 1.0
    )

    # Format breakdowns
    country_results = {}
    for c, stats in sorted(country_breakdown.items()):
        gt_e = stats["total_gt_edges"]
        rec_e = stats["recovered_gt_edges"]
        country_results[c] = {
            "total_gt_edges": gt_e,
            "recovered_gt_edges": rec_e,
            "edge_recall": (rec_e / gt_e) if gt_e > 0 else 1.0,
            "total_candidates": stats["total_candidates"],
        }

    bucket_results = {}
    for b, stats in sorted(bucket_breakdown.items()):
        gt_e = stats["total_gt_edges"]
        rec_e = stats["recovered_gt_edges"]
        bucket_results[b] = {
            "total_gt_edges": gt_e,
            "recovered_gt_edges": rec_e,
            "edge_recall": (rec_e / gt_e) if gt_e > 0 else 1.0,
            "total_candidates": stats["total_candidates"],
        }

    return {
        "edge_recall": {
            "global_edge_recall": (recovered_gt_edges / total_gt_edges) if total_gt_edges > 0 else 1.0,
            "total_gt_edges": total_gt_edges,
            "recovered_gt_edges": recovered_gt_edges,
            "s2_edge_recall": (s2_recovered_edges / s2_gt_edges) if s2_gt_edges > 0 else 1.0,
            "s2_gt_edges": s2_gt_edges,
            "s2_recovered_edges": s2_recovered_edges,
            "s3_edge_recall": (s3_recovered_edges / s3_gt_edges) if s3_gt_edges > 0 else 1.0,
            "s3_gt_edges": s3_gt_edges,
            "s3_recovered_edges": s3_recovered_edges,
        },
        "entity_coverage": {
            "num_entities": num_s1,
            "num_entities_with_gt": num_entities_with_gt,
            "num_singletons": num_singletons,
            "at_least_one_recovered_count": entities_at_least_one,
            "at_least_one_recovered_rate": (
                (entities_at_least_one / num_entities_with_gt) if num_entities_with_gt > 0 else 1.0
            ),
            "all_recovered_count": entities_all_recovered,
            "all_recovered_rate": (
                (entities_all_recovered / num_entities_with_gt) if num_entities_with_gt > 0 else 1.0
            ),
        },
        "candidate_distribution": {
            "total_candidate_edges": total_candidate_edges,
            "mean_candidates_per_s1": cand_mean,
            "median_candidates_per_s1": cand_median,
            "p90_candidates_per_s1": cand_p90,
            "p95_candidates_per_s1": cand_p95,
            "p99_candidates_per_s1": cand_p99,
            "max_candidates_per_s1": cand_max,
            "min_candidates_per_s1": cand_min,
        },
        "reduction_ratio": {
            "naive_search_space": naive_pairs,
            "candidate_edges": total_candidate_edges,
            "reduction_ratio": reduction_ratio,
            "reduction_percentage": reduction_ratio * 100,
        },
        "country_breakdown": country_results,
        "multiplicity_breakdown": bucket_results,
    }
