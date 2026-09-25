"""
Unit tests for E2 Target Exclusivity Scan.
"""

from pathlib import Path
import pytest
from e2.target_exclusivity import (
    save_target_exclusivity_summary,
    save_target_indegree_distribution,
    save_target_overlap_examples,
    scan_target_exclusivity,
)


@pytest.fixture
def synthetic_exclusive_gt(tmp_path: Path):
    """Synthetic GT where every target ID appears under exactly one S1 entity."""
    gt_path = tmp_path / "gt.tsv"
    gt_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-101,S3-201\n"
        "S1-2\tS2-102\n"
        "S1-3\tS3-202,S3-203\n"
        "S1-4\t\n"  # Singleton S1
    )
    return gt_path


@pytest.fixture
def synthetic_multi_parent_gt(tmp_path: Path):
    """Synthetic GT where target S2-101 is claimed by both S1-1 and S1-2."""
    gt_path = tmp_path / "gt.tsv"
    gt_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-101,S3-201\n"
        "S1-2\tS2-101,S2-102\n"  # S2-101 is duplicate parented!
        "S1-3\tS3-202\n"
    )
    return gt_path


def test_one_to_one_target_mapping(synthetic_exclusive_gt, tmp_path: Path):
    country_map = {"S1-1": "US", "S1-2": "India", "S1-3": "US", "S1-4": "India"}
    result = scan_target_exclusivity(synthetic_exclusive_gt, s1_country_map=country_map)

    assert result.total_gt_edges == 5
    assert result.total_distinct_targets == 5
    assert result.total_duplicate_references == 0
    assert result.is_g4_structurally_cleared is True
    assert result.architectural_status == "LEGALLY PLAUSIBLE FOR EXPERIMENTATION"
    assert len(result.multi_parent_examples) == 0

    # Check S2 stats
    assert result.s2_stats is not None
    assert result.s2_stats.total_edges == 2
    assert result.s2_stats.distinct_targets == 2
    assert result.s2_stats.single_parent_targets == 2
    assert result.s2_stats.multi_parent_targets == 0
    assert result.s2_stats.max_parent_count == 1
    assert result.s2_stats.indegree_distribution == {1: 2}

    # Check S3 stats
    assert result.s3_stats is not None
    assert result.s3_stats.total_edges == 3
    assert result.s3_stats.distinct_targets == 3
    assert result.s3_stats.single_parent_targets == 3
    assert result.s3_stats.multi_parent_targets == 0
    assert result.s3_stats.max_parent_count == 1
    assert result.s3_stats.indegree_distribution == {1: 3}

    # Check output saving
    summary_path = tmp_path / "te_summary.csv"
    save_target_exclusivity_summary(result, summary_path)
    assert summary_path.is_file()
    assert "S2,2,2,2,0,1,0,100.0" in summary_path.read_text()
    assert "OVERALL,5,5,5,0,1,0,100.0" in summary_path.read_text()

    dist_path = tmp_path / "dist.csv"
    save_target_indegree_distribution(result, dist_path)
    assert dist_path.is_file()
    assert "OVERALL,1,5,100.0" in dist_path.read_text()

    overlap_path = tmp_path / "overlap.tsv"
    save_target_overlap_examples(result, overlap_path)
    assert overlap_path.is_file()
    lines = overlap_path.read_text().strip().split("\n")
    assert len(lines) == 1  # Header only


def test_multi_parent_target_detection(synthetic_multi_parent_gt, tmp_path: Path):
    country_map = {"S1-1": "US", "S1-2": "India", "S1-3": "US"}
    result = scan_target_exclusivity(synthetic_multi_parent_gt, s1_country_map=country_map)

    assert result.total_gt_edges == 5
    assert result.total_distinct_targets == 4  # S2-101, S3-201, S2-102, S3-202
    assert result.total_duplicate_references == 1
    assert result.is_g4_structurally_cleared is False
    assert result.architectural_status == "DO NOT ENABLE"

    # S2 has 1 multi-parent target
    assert result.s2_stats.multi_parent_targets == 1
    assert result.s2_stats.single_parent_targets == 1
    assert result.s2_stats.max_parent_count == 2
    assert result.s2_stats.indegree_distribution == {1: 1, 2: 1}

    # Verify example
    assert len(result.multi_parent_examples) == 1
    ex = result.multi_parent_examples[0]
    assert ex["target_entity_id"] == "S2-101"
    assert ex["num_parents"] == 2
    assert "S1-1" in ex["parent_s1_ids"]
    assert "S1-2" in ex["parent_s1_ids"]
    assert "US" in ex["parent_countries"]
    assert "India" in ex["parent_countries"]
