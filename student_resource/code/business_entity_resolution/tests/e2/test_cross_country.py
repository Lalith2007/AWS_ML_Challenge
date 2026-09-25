"""
Unit tests for E2 Cross-Country Ground-Truth Edge Scan.
"""

from pathlib import Path
import pytest
from e2.cross_country import (
    CrossCountryScanResult,
    save_cross_country_examples,
    save_cross_country_summary,
    scan_cross_country_edges,
)


@pytest.fixture
def synthetic_same_country_data(tmp_path: Path):
    """Create synthetic data where all match edges are same-country."""
    s1_path = tmp_path / "s1.tsv"
    s1_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tUS Store A\t123 Main St\tUS\n"
        "S1-2\tIndia Shop B\t456 MG Road\tIndia\n"
        "S1-3\tUS Store C\t789 Elm St\tUS\n"
    )

    s2_path = tmp_path / "s2.tsv"
    s2_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-101\tUS Store A Match\t123 Main St\tUS\n"
        "S2-102\tIndia Shop B Match\t456 MG Road\tIndia\n"
    )

    s3_path = tmp_path / "s3.tsv"
    s3_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-201\tUS Store A Match 2\t123 Main St Ste 1\tUS\n"
        "S3-202\tUS Store C Match\t789 Elm St\tUS\n"
    )

    gt_path = tmp_path / "gt.tsv"
    gt_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-101,S3-201\n"
        "S1-2\tS2-102\n"
        "S1-3\tS3-202\n"
    )

    return s1_path, s2_path, s3_path, gt_path


@pytest.fixture
def synthetic_cross_country_data(tmp_path: Path):
    """Create synthetic data with cross-country edges."""
    s1_path = tmp_path / "s1.tsv"
    s1_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tGlobal Store\t123 Broadway\tUS\n"
        "S1-2\tDelhi Hub\t10 Connaught Place\tIndia\n"
    )

    s2_path = tmp_path / "s2.tsv"
    s2_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-101\tGlobal Branch\t123 Broadway\tIndia\n"  # Cross-country! (S1=US, S2=India)
        "S2-102\tDelhi Branch\t10 Connaught Place\tIndia\n"  # Same-country
    )

    s3_path = tmp_path / "s3.tsv"
    s3_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-201\tUS Direct\t123 Broadway\tUS\n"  # Same-country
    )

    gt_path = tmp_path / "gt.tsv"
    gt_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-101,S3-201\n"
        "S1-2\tS2-102\n"
    )

    return s1_path, s2_path, s3_path, gt_path


def test_all_same_country_edges(synthetic_same_country_data, tmp_path: Path):
    s1_path, s2_path, s3_path, gt_path = synthetic_same_country_data
    result = scan_cross_country_edges(s1_path, s2_path, s3_path, gt_path)

    assert result.total_edges == 4
    assert result.same_country_edges == 4
    assert result.cross_country_edges == 0
    assert result.same_country_pct == 100.0
    assert result.cross_country_pct == 0.0
    assert result.is_safe_for_hard_partitioning is True
    assert result.architectural_status == "SAFE FOR HARD PARTITIONING ON TRAINING GT"
    assert len(result.cross_country_examples) == 0

    # Test file saving
    summary_path = tmp_path / "summary.csv"
    save_cross_country_summary(result, summary_path)
    assert summary_path.is_file()
    assert "OVERALL,ALL,ALL,4,4,0,100.0,0.0" in summary_path.read_text()

    examples_path = tmp_path / "examples.tsv"
    save_cross_country_examples(result, examples_path)
    assert examples_path.is_file()
    lines = examples_path.read_text().strip().split("\n")
    assert len(lines) == 1  # Header only


def test_cross_country_edge_detection(synthetic_cross_country_data, tmp_path: Path):
    s1_path, s2_path, s3_path, gt_path = synthetic_cross_country_data
    result = scan_cross_country_edges(s1_path, s2_path, s3_path, gt_path)

    assert result.total_edges == 3
    assert result.same_country_edges == 2
    assert result.cross_country_edges == 1
    assert result.cross_country_pct == pytest.approx(33.333333, abs=1e-4)
    assert result.is_safe_for_hard_partitioning is False
    assert result.architectural_status == "HARD COUNTRY PARTITION WOULD CREATE A MEASURED RECALL RISK"

    assert len(result.cross_country_examples) == 1
    ex = result.cross_country_examples[0]
    assert ex["source1_entity_id"] == "S1-1"
    assert ex["source1_country"] == "US"
    assert ex["target_entity_id"] == "S2-101"
    assert ex["target_country"] == "India"
    assert ex["target_source"] == "S2"


def test_country_and_source_breakdown(synthetic_same_country_data):
    s1_path, s2_path, s3_path, gt_path = synthetic_same_country_data
    result = scan_cross_country_edges(s1_path, s2_path, s3_path, gt_path)

    # Check S2 slice
    assert result.s2_slice is not None
    assert result.s2_slice.total_edges == 2
    assert result.s2_slice.same_country_edges == 2
    assert result.s2_slice.cross_country_edges == 0

    # Check S3 slice
    assert result.s3_slice is not None
    assert result.s3_slice.total_edges == 2
    assert result.s3_slice.same_country_edges == 2
    assert result.s3_slice.cross_country_edges == 0

    # Check country slices
    assert "US" in result.country_slices
    assert "India" in result.country_slices
    assert result.country_slices["US"].total_edges == 3
    assert result.country_slices["India"].total_edges == 1
