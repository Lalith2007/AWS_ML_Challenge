"""
Unit tests for ground truth utilities.
Amazon ML Challenge 2026 — E1.
"""

import tempfile
from pathlib import Path

import pytest
from e1.gt_utils import get_multiplicity_bucket, load_ground_truth, validate_ground_truth_integrity


def test_multiplicity_buckets():
    """Verify locked architecture multiplicity buckets."""
    assert get_multiplicity_bucket(0) == "bucket_0"
    assert get_multiplicity_bucket(1) == "bucket_1"
    assert get_multiplicity_bucket(2) == "bucket_2_4"
    assert get_multiplicity_bucket(3) == "bucket_2_4"
    assert get_multiplicity_bucket(4) == "bucket_2_4"
    assert get_multiplicity_bucket(5) == "bucket_5_plus"
    assert get_multiplicity_bucket(10) == "bucket_5_plus"


def test_load_ground_truth_clean():
    """Load valid ground truth file with singletons and matches."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-1\tS2-10,S3-20\n")
        f.write("S1-2\t\n")
        f.write("S1-3\tS3-30\n")
        temp_path = Path(f.name)

    try:
        truth = load_ground_truth(temp_path)
        assert len(truth) == 3
        assert truth["S1-1"] == {"S2-10", "S3-20"}
        assert truth["S1-2"] == set()
        assert truth["S1-3"] == {"S3-30"}
    finally:
        temp_path.unlink()


def test_load_ground_truth_duplicate_in_list_raises():
    """Duplicate target IDs in a single GT line must raise ValueError."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-1\tS2-10,S2-10\n")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Duplicate target ID found"):
            load_ground_truth(temp_path, validate_duplicates=True)
    finally:
        temp_path.unlink()


def test_validate_ground_truth_integrity():
    """Verify ground truth validation with source file existence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        s1_file = tmp_path / "train_source1.tsv"
        s2_file = tmp_path / "train_source2.tsv"
        s3_file = tmp_path / "train_source3.tsv"

        with open(s1_file, "w") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tA\tAddr\tUS\nS1-2\tB\tAddr\tUS\n")
        with open(s2_file, "w") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-10\tA2\tAddr\tUS\n")
        with open(s3_file, "w") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-20\tA3\tAddr\tUS\n")

        valid_truth = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": set(),
        }
        res = validate_ground_truth_integrity(valid_truth, s1_file, s2_file, s3_file)
        assert res["integrity_passed"] is True
        assert res["total_gt_s1_entities"] == 2
        assert res["total_gt_edges"] == 2

        # Invalid prefix raises
        bad_prefix_truth = {"S1-1": {"X2-10"}}
        with pytest.raises(ValueError):
            validate_ground_truth_integrity(bad_prefix_truth, s1_file, s2_file, s3_file)
