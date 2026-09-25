"""
Unit tests for E2 Runner and Property Audit Coordinator.
"""

from pathlib import Path
import json
import pytest

from e2.gt_property_scans import (
    EXPECTED_COUNTS,
    run_full_gt_property_audit,
    verify_gt_integrity_and_reconciliation,
)
from e2.runner import run_e2_pipeline


@pytest.fixture
def synthetic_complete_dataset(tmp_path: Path):
    """Create complete consistent synthetic dataset."""
    s1_path = tmp_path / "train_source1.tsv"
    s1_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAcme Corp\t100 Main St\tUS\n"
        "S1-2\tBeta LLC\t200 MG Rd\tIndia\n"
    )

    s2_path = tmp_path / "train_source2.tsv"
    s2_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-101\tAcme\t100 Main St\tUS\n"
        "S2-102\tBeta\t200 MG Rd\tIndia\n"
    )

    s3_path = tmp_path / "train_source3.tsv"
    s3_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-201\tAcme Co\t100 Main St # 1\tUS\n"
        "S3-202\tBeta Ltd\t200 MG Rd Ste 2\tIndia\n"
    )

    gt_path = tmp_path / "train_ground_truth.tsv"
    gt_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-101,S3-201\n"
        "S1-2\tS2-102,S3-202\n"
    )

    return s1_path, s2_path, s3_path, gt_path


def test_runner_synthetic_pipeline(synthetic_complete_dataset, tmp_path: Path):
    s1_path, s2_path, s3_path, gt_path = synthetic_complete_dataset
    out_dir = tmp_path / "e2_output"

    manifest = run_e2_pipeline(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        gt_path=gt_path,
        output_dir=out_dir,
    )

    # Verify all 8 files exist
    expected_files = [
        "e2_cross_country_summary.csv",
        "e2_cross_country_examples.tsv",
        "e2_target_exclusivity_summary.csv",
        "e2_target_overlap_examples.tsv",
        "e2_target_indegree_distribution.csv",
        "e2_report.json",
        "e2_report.md",
        "e2_manifest.json",
    ]
    for fname in expected_files:
        p = out_dir / fname
        assert p.is_file(), f"Expected artifact {fname} missing"
        assert p.stat().st_size > 0

    # Verify manifest content
    assert manifest["cross_country"]["total_edges"] == 4
    assert manifest["cross_country"]["same_country_edges"] == 4
    assert manifest["cross_country"]["cross_country_edges"] == 0
    assert manifest["cross_country"]["decision"] == "SAFE FOR HARD PARTITIONING ON TRAINING GT"

    assert manifest["target_exclusivity"]["total_distinct_targets"] == 4
    assert manifest["target_exclusivity"]["total_duplicate_references"] == 0
    assert manifest["target_exclusivity"]["decision"] == "LEGALLY PLAUSIBLE FOR EXPERIMENTATION"

    assert manifest["e1_documentation_correction"] == "precision weighting = 2x, not 4x"

    # Check report JSON
    report_json_path = out_dir / "e2_report.json"
    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["cross_country"]["total_edges"] == 4
    assert data["e1_documentation_correction"] == "precision weighting = 2x, not 4x"

    # Check report MD
    report_md_path = out_dir / "e2_report.md"
    md_text = report_md_path.read_text()
    assert "# E2 Report" in md_text
    assert "SAFE FOR HARD PARTITIONING ON TRAINING GT" in md_text
    assert "LEGALLY PLAUSIBLE FOR EXPERIMENTATION" in md_text
    assert "precision weighting = 2x, not 4x" in md_text


def test_gt_count_reconciliation_mismatch(synthetic_complete_dataset):
    s1_path, s2_path, s3_path, gt_path = synthetic_complete_dataset
    report = run_full_gt_property_audit(s1_path, s2_path, s3_path, gt_path)

    # In synthetic dataset, count does not match full competition expected count
    assert report.integrity.counts_reconciled is False
    assert len(report.integrity.errors) > 0
