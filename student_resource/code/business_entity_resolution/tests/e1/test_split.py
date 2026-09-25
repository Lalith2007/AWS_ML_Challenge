"""
Unit tests for deterministic S1 train/validation split.
Amazon ML Challenge 2026 — E1.
"""

import tempfile
from pathlib import Path

from e1.validation_split import check_target_id_leakage, create_deterministic_s1_split


def test_stratified_split_properties():
    """Verify split properties: disjoint sets, full coverage, and determinism."""
    # Synthetic dataset with 2 countries and various multiplicities
    truth = {}
    countries = {}
    for i in range(100):
        s1_id = f"S1-{i}"
        country = "US" if i < 60 else "India"
        countries[s1_id] = country
        if i % 5 == 0:
            truth[s1_id] = set()  # bucket_0
        elif i % 5 == 1:
            truth[s1_id] = {f"S2-{i}"}  # bucket_1
        elif i % 5 in (2, 3):
            truth[s1_id] = {f"S2-{i}", f"S3-{i}"}  # bucket_2_4
        else:
            truth[s1_id] = {f"S2-{i}_1", f"S2-{i}_2", f"S3-{i}_1", f"S3-{i}_2", f"S3-{i}_3"}  # bucket_5_plus

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        train_ids_1, val_ids_1, summary_1, manifest_1 = create_deterministic_s1_split(
            truth_by_s1=truth,
            country_by_s1=countries,
            val_fraction=0.10,
            seed=42,
            output_dir=tmp_path,
        )

        # Disjoint
        assert len(train_ids_1 & val_ids_1) == 0
        # Union
        assert (train_ids_1 | val_ids_1) == set(countries.keys())
        # Approximately 10%
        assert 8 <= len(val_ids_1) <= 12
        assert len(train_ids_1) + len(val_ids_1) == 100

        # Determinism: re-run with same seed yields identical sets
        train_ids_2, val_ids_2, summary_2, manifest_2 = create_deterministic_s1_split(
            truth_by_s1=truth,
            country_by_s1=countries,
            val_fraction=0.10,
            seed=42,
            output_dir=tmp_path / "run2",
        )
        assert train_ids_1 == train_ids_2
        assert val_ids_1 == val_ids_2


def test_target_leakage_diagnostic():
    """Verify target leakage diagnostic correctly detects shared and disjoint targets."""
    truth = {
        "S1-1": {"S2-A", "S3-A"},
        "S1-2": {"S2-B"},
        "S1-3": {"S2-A", "S3-C"},  # S2-A shared with S1-1
    }
    train_ids = {"S1-1", "S1-2"}
    val_ids = {"S1-3"}

    with tempfile.TemporaryDirectory() as tmpdir:
        report, summary = check_target_id_leakage(truth, train_ids, val_ids, output_dir=Path(tmpdir))

        assert summary["shared_s2_target_count"] == 1  # S2-A shared across train and val
        assert summary["shared_s3_target_count"] == 0
        assert summary["warning"] is not None
        assert "Potential target-level leakage exists" in summary["warning"]
