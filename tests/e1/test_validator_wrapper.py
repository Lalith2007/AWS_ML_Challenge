"""
Unit tests for submission validator integration wrapper.
Amazon ML Challenge 2026 — E1.
"""

import tempfile
from pathlib import Path

from e1.submission_validator import run_submission_validator


def test_validator_wrapper_pass():
    """Valid matching file on minimal synthetic test set passes with status PASS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        # Create synthetic test_source1.tsv
        with open(test_dir / "test_source1.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tFoo\t1 St\tUS\nS1-2\tBar\t2 St\tUS\n")

        # Create valid matching_results.tsv
        matching_file = tmp_path / "matching_results.tsv"
        with open(matching_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-10\nS1-2\t\n")

        res = run_submission_validator(
            matching_path=matching_file,
            test_dir=test_dir,
            working_dir=tmp_path,
            log_path=tmp_path / "val.log",
        )

        assert res["status"] == "PASS"
        assert res["exit_code"] == 0
        assert len(res["errors"]) == 0
        assert (tmp_path / "val.log").is_file()


def test_validator_wrapper_fail_on_malformed_header():
    """Comma-separated header or invalid columns fail with status FAIL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        with open(test_dir / "test_source1.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tFoo\t1 St\tUS\n")

        # Comma-separated (common mistake)
        bad_file = tmp_path / "bad_matching.tsv"
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id,matched_entity_ids\nS1-1,S2-10\n")

        res = run_submission_validator(
            matching_path=bad_file,
            test_dir=test_dir,
            working_dir=tmp_path,
            log_path=tmp_path / "val_fail.log",
        )

        assert res["status"] == "FAIL"
        assert res["exit_code"] == 1
        assert len(res["errors"]) > 0


def test_validator_wrapper_fail_on_missing_required_s1():
    """Missing an S1 entity from test_source1 fails validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_dir = tmp_path / "test"
        test_dir.mkdir(parents=True)

        with open(test_dir / "test_source1.tsv", "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tFoo\t1 St\tUS\nS1-2\tBar\t2 St\tUS\n")

        # Only includes S1-1, omitting S1-2
        incomplete_file = tmp_path / "incomplete.tsv"
        with open(incomplete_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-10\n")

        res = run_submission_validator(
            matching_path=incomplete_file,
            test_dir=test_dir,
            working_dir=tmp_path,
            log_path=tmp_path / "val_incomp.log",
        )

        assert res["status"] == "FAIL"
        assert res["exit_code"] == 1
        assert any("required S1 entity" in err for err in res["errors"])
