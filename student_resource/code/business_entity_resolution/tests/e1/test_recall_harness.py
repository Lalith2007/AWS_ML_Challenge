"""
Unit tests for blocking recall harness and candidate validation.
Amazon ML Challenge 2026 — E1.
"""

from e1.recall_harness import evaluate_blocking_recall, validate_candidate_set


def test_candidate_set_validation():
    """Verify candidate set integrity validation rules."""
    # 1. Valid candidate set
    valid_cands = {"S1-1": {"S2-10", "S3-20"}, "S1-2": set()}
    res = validate_candidate_set(valid_cands, allowed_s1_ids={"S1-1", "S1-2"})
    assert res["valid"] is True

    # 2. Invalid target prefix
    bad_prefix = {"S1-1": {"S1-10"}}
    res = validate_candidate_set(bad_prefix)
    assert res["valid"] is False
    assert "Invalid target ID prefix" in res["issues"][0]

    # 3. Duplicate IDs in candidate list
    dup_list = {"S1-1": ["S2-10", "S2-10"]}
    res = validate_candidate_set(dup_list)
    assert res["valid"] is False
    assert "duplicate IDs" in res["issues"][0]

    # 4. Subset property: matched targets must be in candidates
    cands = {"S1-1": {"S2-10"}}
    matched = {"S1-1": {"S2-10", "S3-20"}}  # S3-20 not in candidates
    res = validate_candidate_set(cands, matched_by_s1=matched)
    assert res["valid"] is False
    assert "not a subset" in res["issues"][0]


def test_blocking_recall_metrics():
    """Verify edge recall, entity coverage, and reduction ratio calculations."""
    truth = {
        "S1-1": {"S2-10", "S3-20"},  # 2 GT targets
        "S1-2": {"S2-30"},           # 1 GT target
        "S1-3": set(),               # 0 GT targets (singleton)
    }
    # Candidate recovers S2-10 (S1-1) and S2-30 (S1-2), misses S3-20, adds 1 FP
    cands = {
        "S1-1": {"S2-10", "S2-99"},  # recovered 1/2
        "S1-2": {"S2-30"},           # recovered 1/1
        "S1-3": set(),
    }
    meta = {
        "S1-1": {"country": "US", "multiplicity_bucket": "bucket_2_4"},
        "S1-2": {"country": "India", "multiplicity_bucket": "bucket_1"},
        "S1-3": {"country": "US", "multiplicity_bucket": "bucket_0"},
    }

    res = evaluate_blocking_recall(
        truth_by_s1=truth,
        candidate_by_s1=cands,
        metadata_by_s1=meta,
        num_s2_records=100,
        num_s3_records=100,
    )

    # Edge recall: 2 recovered / 3 total GT edges = 2/3
    assert abs(res["edge_recall"]["global_edge_recall"] - (2.0 / 3.0)) < 1e-7
    assert res["edge_recall"]["s2_recovered_edges"] == 2
    assert res["edge_recall"]["s2_gt_edges"] == 2
    assert res["edge_recall"]["s2_edge_recall"] == 1.0  # 2/2
    assert res["edge_recall"]["s3_recovered_edges"] == 0
    assert res["edge_recall"]["s3_gt_edges"] == 1
    assert res["edge_recall"]["s3_edge_recall"] == 0.0  # 0/1

    # Entity coverage: entities with GT = 2 (S1-1, S1-2)
    # at least one recovered: S1-1 (1/2), S1-2 (1/1) -> 2/2 = 1.0
    # all recovered: only S1-2 -> 1/2 = 0.5
    assert res["entity_coverage"]["at_least_one_recovered_rate"] == 1.0
    assert res["entity_coverage"]["all_recovered_rate"] == 0.5

    # Reduction ratio: naive = 3 S1 * (100 + 100) = 600 pairs. candidates = 3 edges (2 + 1 + 0)
    # reduction ratio = 1 - 3/600 = 1 - 0.005 = 0.995 (99.5%)
    assert abs(res["reduction_ratio"]["reduction_ratio"] - 0.995) < 1e-7
