"""
Unit tests for Candidate Generation, Inverted Indexing, and Partition Validation.

Tests covering requirements:
1. Country partitioning
11. Oversized block protection
12. Candidate deduplication
13. Source separation
14. Deterministic candidate generation
15. A synthetic case with zero matches
16. A synthetic case with multiple true matches for one S1
"""

import csv
from pathlib import Path
import pytest

from e3.blocking_lanes import BlockingConfig
from e3.candidate_generator import run_partition_blocking
from e3.inverted_index import PartitionInvertedIndex
from e3.runner import run_e3_pipeline, verify_country_partition_integrity


@pytest.fixture
def synthetic_e3_data(tmp_path: Path):
    """Generate isolated synthetic multi-source dataset."""
    s1_path = tmp_path / "train_source1.tsv"
    s1_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tAcme Superstore\t100 Main St, Austin, TX\tUS\n"
        "S1-2\tBeta Solutions\t200 MG Rd, Bangalore\tIndia\n"
        "S1-3\tOmega Lone Entity\t999 Desert Rd, Phoenix, AZ\tUS\n"
    )

    s2_path = tmp_path / "train_source2.tsv"
    s2_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-101\tAcme Superstore\t100 Main St, Austin, TX\tUS\n"
        "S2-102\tAcme Store Inc\t100 Main St Ste 5, Austin, TX\tUS\n"
        "S2-201\tBeta Solutions Pvt Ltd\t200 MG Road, Bangalore\tIndia\n"
        "S2-999\tForeign Target\t555 Cross Way\tIndia\n"
    )

    s3_path = tmp_path / "train_source3.tsv"
    s3_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-301\tAcme Superstore [Austin]\t100 Main St, Austin, TX\tUS\n"
        "S3-401\tBeta Tech Group\t200 MG Rd, Bangalore\tIndia\n"
    )

    gt_path = tmp_path / "train_ground_truth.tsv"
    gt_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-101,S2-102,S3-301\n"
        "S1-2\tS2-201,S3-401\n"
    )

    return {
        "s1": s1_path,
        "s2": s2_path,
        "s3": s3_path,
        "gt": gt_path,
        "tmp_path": tmp_path,
    }


def test_country_partitioning(synthetic_e3_data):
    """Requirement 1: Verify that candidates are generated strictly within matching country."""
    output_dir = synthetic_e3_data["tmp_path"] / "out_country"
    manifest = run_e3_pipeline(
        s1_path=synthetic_e3_data["s1"],
        s2_path=synthetic_e3_data["s2"],
        s3_path=synthetic_e3_data["s3"],
        gt_path=synthetic_e3_data["gt"],
        output_dir=output_dir,
    )

    cand_path = output_dir / "candidate_pairs.tsv"
    assert cand_path.is_file()

    # S2-999 is in India. S1-1 is in US. S1-1 must NEVER generate S2-999.
    with open(cand_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            s1, target, src = row[0], row[1], row[2]
            if s1 == "S1-1":
                assert target != "S2-999"


def test_oversized_block_protection():
    """Requirement 11: Verify that posting lists exceeding max_block_size are suppressed and logged."""
    config = BlockingConfig(max_block_size=5, max_token_doc_freq=5)
    index = PartitionInvertedIndex(country="US", source="S2", config=config)

    # Add 10 records with the exact same common token "universal"
    for i in range(10):
        index.add_record(f"S2-{i}", f"Universal Enterprise {i}", f"{i} Market St")

    stats = index.freeze_and_audit()
    assert "k1_universal" in index.oversized_keys["K1"]
    assert stats["lane_stats"]["K1"]["oversized_keys_count"] >= 1

    # Query with S1 having "universal"
    s1_keys = {"K1": ["k1_universal"], "K2": [], "K3": [], "K4": [], "K5": [], "K6": [], "K7": []}
    cands, oversized_events = index.query(s1_keys)
    # The oversized key should NOT return 10 records; it should be suppressed
    assert len(cands["K1"]) == 0
    assert oversized_events == 1


def test_candidate_deduplication(synthetic_e3_data):
    """Requirement 12: Ensure union candidate set has zero duplicate (s1_id, target_id) pairs."""
    output_dir = synthetic_e3_data["tmp_path"] / "out_dedup"
    run_e3_pipeline(
        s1_path=synthetic_e3_data["s1"],
        s2_path=synthetic_e3_data["s2"],
        s3_path=synthetic_e3_data["s3"],
        gt_path=synthetic_e3_data["gt"],
        output_dir=output_dir,
    )

    cand_path = output_dir / "candidate_pairs.tsv"
    seen_pairs = set()
    with open(cand_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            pair = (row[0], row[1])
            assert pair not in seen_pairs, f"Duplicate candidate pair detected: {pair}"
            seen_pairs.add(pair)


def test_source_separation(synthetic_e3_data):
    """Requirement 13: Ensure S2 candidates are tagged as S2 and S3 candidates as S3, never mixed."""
    output_dir = synthetic_e3_data["tmp_path"] / "out_source_sep"
    run_e3_pipeline(
        s1_path=synthetic_e3_data["s1"],
        s2_path=synthetic_e3_data["s2"],
        s3_path=synthetic_e3_data["s3"],
        gt_path=synthetic_e3_data["gt"],
        output_dir=output_dir,
    )

    cand_path = output_dir / "candidate_pairs.tsv"
    with open(cand_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            s1, target, source = row[0], row[1], row[2]
            if target.startswith("S2-"):
                assert source == "S2", f"Target {target} emitted with incorrect source {source}"
            elif target.startswith("S3-"):
                assert source == "S3", f"Target {target} emitted with incorrect source {source}"


def test_deterministic_candidate_generation(synthetic_e3_data):
    """Requirement 14: Ensure two runs with identical inputs produce bitwise identical candidate files."""
    out1 = synthetic_e3_data["tmp_path"] / "out_det1"
    out2 = synthetic_e3_data["tmp_path"] / "out_det2"

    run_e3_pipeline(
        s1_path=synthetic_e3_data["s1"],
        s2_path=synthetic_e3_data["s2"],
        s3_path=synthetic_e3_data["s3"],
        gt_path=synthetic_e3_data["gt"],
        output_dir=out1,
    )
    run_e3_pipeline(
        s1_path=synthetic_e3_data["s1"],
        s2_path=synthetic_e3_data["s2"],
        s3_path=synthetic_e3_data["s3"],
        gt_path=synthetic_e3_data["gt"],
        output_dir=out2,
    )

    cand1 = (out1 / "candidate_pairs.tsv").read_text()
    cand2 = (out2 / "candidate_pairs.tsv").read_text()
    assert cand1 == cand2, "Candidate files across repeated runs are not identical."


def test_synthetic_case_zero_matches(synthetic_e3_data):
    """Requirement 15: S1-3 has 0 ground truth matches in S2/S3. Ensure metrics handle singletons safely."""
    output_dir = synthetic_e3_data["tmp_path"] / "out_zero_matches"
    manifest = run_e3_pipeline(
        s1_path=synthetic_e3_data["s1"],
        s2_path=synthetic_e3_data["s2"],
        s3_path=synthetic_e3_data["s3"],
        gt_path=synthetic_e3_data["gt"],
        output_dir=output_dir,
    )

    # Recall should be 100% on entities that have GT
    assert manifest["overall_metrics"]["overall_recall"] == 1.0


def test_synthetic_case_multiple_true_matches(synthetic_e3_data):
    """Requirement 16: S1-1 has 3 true matches (S2-101, S2-102, S3-301). Ensure all are recovered."""
    output_dir = synthetic_e3_data["tmp_path"] / "out_multi_match"
    run_e3_pipeline(
        s1_path=synthetic_e3_data["s1"],
        s2_path=synthetic_e3_data["s2"],
        s3_path=synthetic_e3_data["s3"],
        gt_path=synthetic_e3_data["gt"],
        output_dir=output_dir,
    )

    cand_path = output_dir / "candidate_pairs.tsv"
    s1_1_cands = set()
    with open(cand_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row[0] == "S1-1":
                s1_1_cands.add(row[1])

    expected = {"S2-101", "S2-102", "S3-301"}
    assert expected.issubset(s1_1_cands), f"Expected {expected} to be in candidates, got {s1_1_cands}"
