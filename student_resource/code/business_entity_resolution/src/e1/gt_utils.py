"""
Ground Truth Utilities for E1.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Ground truth parsing and validation ensures data integrity, consistent mapping, and validates
entity ID existence without modifying raw competition files. Later stages (L2 blocking, L4 training,
and L5 evaluation) rely on this exact, vetted mapping as their authoritative ground truth.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .io_utils import TRAIN_FILES


def get_multiplicity_bucket(m: int) -> str:
    """
    Classify entity match multiplicity into the locked architecture's exact buckets:
    - bucket_0: m == 0 (singletons / non-matching S1)
    - bucket_1: m == 1
    - bucket_2_4: 2 <= m <= 4
    - bucket_5_plus: m >= 5
    """
    if m == 0:
        return "bucket_0"
    elif m == 1:
        return "bucket_1"
    elif 2 <= m <= 4:
        return "bucket_2_4"
    else:
        return "bucket_5_plus"


def load_ground_truth(
    gt_path: Optional[Path] = None,
    validate_duplicates: bool = True,
) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping from TSV file:
    source1_entity_id -> set of matched_entity_ids (S2-*, S3-*).

    Empty value in matched_entity_ids indicates zero matches (singleton S1).
    Validates that no duplicate IDs exist inside a single S1 entity's ground-truth list.
    """
    if gt_path is None:
        gt_path = TRAIN_FILES["GT"]

    truth_by_s1: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or header[:2] != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(
                f"Invalid ground truth header in {gt_path}: expected ['source1_entity_id', 'matched_entity_ids'], got {header}"
            )

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            s1_id = row[0].strip()
            target_str = row[1].strip() if len(row) > 1 else ""

            if not target_str:
                truth_by_s1[s1_id] = set()
                continue

            raw_targets = [t.strip() for t in target_str.split(",") if t.strip()]
            if validate_duplicates and len(raw_targets) != len(set(raw_targets)):
                duplicates = [t for t in set(raw_targets) if raw_targets.count(t) > 1]
                raise ValueError(
                    f"Duplicate target ID found in ground truth at line {line_num} for {s1_id}: {duplicates}"
                )

            truth_by_s1[s1_id] = set(raw_targets)

    return truth_by_s1


def validate_ground_truth_integrity(
    truth_by_s1: Dict[str, Set[str]],
    s1_source_path: Optional[Path] = None,
    s2_source_path: Optional[Path] = None,
    s3_source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate that:
    1. All S1 IDs in ground truth exist in train Source 1.
    2. Every target ID starts with 'S2-' or 'S3-'.
    3. All S2 target IDs exist in train Source 2.
    4. All S3 target IDs exist in train Source 3.
    5. Track any cross-S1 target reuse (for leakage diagnostics).
    """
    if s1_source_path is None:
        s1_source_path = TRAIN_FILES["S1"]
    if s2_source_path is None:
        s2_source_path = TRAIN_FILES["S2"]
    if s3_source_path is None:
        s3_source_path = TRAIN_FILES["S3"]

    total_gt_s1 = len(truth_by_s1)
    total_edges = 0
    s2_edges = 0
    s3_edges = 0
    invalid_prefixes = set()

    # Track target occurrence frequency
    s2_target_counts: Dict[str, int] = {}
    s3_target_counts: Dict[str, int] = {}

    for s1_id, targets in truth_by_s1.items():
        for t in targets:
            total_edges += 1
            if t.startswith("S2-"):
                s2_edges += 1
                s2_target_counts[t] = s2_target_counts.get(t, 0) + 1
            elif t.startswith("S3-"):
                s3_edges += 1
                s3_target_counts[t] = s3_target_counts.get(t, 0) + 1
            else:
                invalid_prefixes.add(t)

    # 1. Read S1 IDs
    s1_source_ids: Set[str] = set()
    with open(s1_source_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                s1_source_ids.add(row[0].strip())

    missing_s1_in_source = set(truth_by_s1.keys()) - s1_source_ids

    # 2. Read S2 IDs and verify
    s2_source_ids: Set[str] = set()
    with open(s2_source_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                s2_source_ids.add(row[0].strip())

    missing_s2_in_source = set(s2_target_counts.keys()) - s2_source_ids

    # 3. Read S3 IDs and verify
    s3_source_ids: Set[str] = set()
    with open(s3_source_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                s3_source_ids.add(row[0].strip())

    missing_s3_in_source = set(s3_target_counts.keys()) - s3_source_ids

    # Check for cross-S1 target multiplicity
    s2_multi_s1 = {t: c for t, c in s2_target_counts.items() if c > 1}
    s3_multi_s1 = {t: c for t, c in s3_target_counts.items() if c > 1}

    results = {
        "total_gt_s1_entities": total_gt_s1,
        "total_gt_edges": total_edges,
        "total_s2_edges": s2_edges,
        "total_s3_edges": s3_edges,
        "unique_s2_targets": len(s2_target_counts),
        "unique_s3_targets": len(s3_target_counts),
        "invalid_prefix_count": len(invalid_prefixes),
        "missing_s1_in_source_count": len(missing_s1_in_source),
        "missing_s2_in_source_count": len(missing_s2_in_source),
        "missing_s3_in_source_count": len(missing_s3_in_source),
        "s2_targets_under_multiple_s1_count": len(s2_multi_s1),
        "s3_targets_under_multiple_s1_count": len(s3_multi_s1),
        "integrity_passed": (
            len(invalid_prefixes) == 0
            and len(missing_s1_in_source) == 0
            and len(missing_s2_in_source) == 0
            and len(missing_s3_in_source) == 0
        ),
    }

    if not results["integrity_passed"]:
        error_msg = f"Ground truth integrity validation failed: {results}"
        raise ValueError(error_msg)

    return results
