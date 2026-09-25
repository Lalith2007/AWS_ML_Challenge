"""
Deterministic Stratified S1 Train/Validation Split for E1.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Pair-level random splitting would leak entity-specific patterns across train and validation;
splitting strictly on Source-1 entities preserves the intended evaluation structure and prevents
data leakage. All subsequent pipeline stages (L2 blocking evaluation, L4 LightGBM training, and
L5 threshold tuning) must strictly train on train_s1_ids and evaluate on val_s1_ids.
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .gt_utils import get_multiplicity_bucket, load_ground_truth
from .io_utils import E1_SPLITS_DIR, TRAIN_FILES, save_csv_from_dicts, save_json, stream_s1_countries


def create_deterministic_s1_split(
    truth_by_s1: Dict[str, Set[str]],
    country_by_s1: Dict[str, str],
    val_fraction: float = 0.10,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[Set[str], Set[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Creates a deterministic, stratified 90/10 train/val split at the Source-1 ENTITY level.

    Stratification Key: (country, multiplicity_bucket)
    Buckets:
      bucket_0: m == 0
      bucket_1: m == 1
      bucket_2_4: 2 <= m <= 4
      bucket_5_plus: m >= 5

    Guarantees:
      - train_s1_ids & val_s1_ids is empty.
      - train_s1_ids | val_s1_ids == all S1 entities.
      - 100% reproducible given the seed.
    """
    if output_dir is None:
        output_dir = E1_SPLITS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group entities by stratum
    strata: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for s1_id, country in country_by_s1.items():
        targets = truth_by_s1.get(s1_id, set())
        m = len(targets)
        m_bucket = get_multiplicity_bucket(m)
        strata[(country, m_bucket)].append(s1_id)

    rng = random.Random(seed)

    train_ids: Set[str] = set()
    val_ids: Set[str] = set()

    summary_records: List[Dict[str, Any]] = []

    country_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    stratum_counts: Dict[str, Dict[str, int]] = {}

    for (country, m_bucket), s1_list in sorted(strata.items()):
        # Sort first for deterministic ordering before shuffle
        s1_list.sort()
        rng.shuffle(s1_list)

        n_total = len(s1_list)
        n_val = round(n_total * val_fraction)
        # Ensure at least 1 in val if stratum >= 10, but allow 0 if n_total is tiny
        if n_total > 0 and n_val == 0 and val_fraction > 0 and n_total >= 5:
            n_val = 1

        val_subset = s1_list[:n_val]
        train_subset = s1_list[n_val:]

        val_ids.update(val_subset)
        train_ids.update(train_subset)

        stratum_name = f"{country}__{m_bucket}"
        country_counts[country] += n_total
        bucket_counts[m_bucket] += n_total
        stratum_counts[stratum_name] = {
            "country": country,
            "multiplicity_bucket": m_bucket,
            "total_s1": n_total,
            "train_s1": len(train_subset),
            "val_s1": len(val_subset),
            "val_percentage": (len(val_subset) / n_total * 100) if n_total > 0 else 0.0,
        }

        summary_records.append({
            "country": country,
            "multiplicity_bucket": m_bucket,
            "total_s1": n_total,
            "train_s1": len(train_subset),
            "val_s1": len(val_subset),
            "val_percentage": (len(val_subset) / n_total * 100) if n_total > 0 else 0.0,
        })

    # Total check
    all_s1 = set(country_by_s1.keys())
    assert len(train_ids & val_ids) == 0, "Train and validation S1 IDs must be completely disjoint!"
    assert (train_ids | val_ids) == all_s1, "Union of train and val S1 IDs must equal all Source 1 entities!"

    # Save ID text files (sorted for reproducibility)
    train_ids_path = output_dir / "train_s1_ids.txt"
    val_ids_path = output_dir / "val_s1_ids.txt"

    with open(train_ids_path, "w", encoding="utf-8") as f:
        for s1_id in sorted(train_ids):
            f.write(f"{s1_id}\n")

    with open(val_ids_path, "w", encoding="utf-8") as f:
        for s1_id in sorted(val_ids):
            f.write(f"{s1_id}\n")

    # Save summary CSV
    summary_csv_path = output_dir / "split_summary.csv"
    save_csv_from_dicts(summary_records, summary_csv_path)

    # Save manifest JSON
    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "total_s1_entities": len(all_s1),
        "train_s1_entities": len(train_ids),
        "val_s1_entities": len(val_ids),
        "actual_val_percentage": (len(val_ids) / len(all_s1) * 100) if all_s1 else 0.0,
        "stratification_fields": ["country", "multiplicity_bucket"],
        "counts_by_country": dict(country_counts),
        "counts_by_multiplicity_bucket": dict(bucket_counts),
        "counts_by_stratum": stratum_counts,
    }
    manifest_path = output_dir / "split_manifest.json"
    save_json(manifest, manifest_path)

    return train_ids, val_ids, summary_records, manifest


def check_target_id_leakage(
    truth_by_s1: Dict[str, Set[str]],
    train_s1_ids: Set[str],
    val_s1_ids: Set[str],
    output_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Inspect ground-truth target IDs (S2 and S3) for cross-S1 and train/val overlap.
    Diagnostic ONLY — does not alter split.
    """
    if output_dir is None:
        output_dir = E1_SPLITS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_s2_targets: Counter = Counter()
    train_s3_targets: Counter = Counter()
    val_s2_targets: Counter = Counter()
    val_s3_targets: Counter = Counter()

    for s1_id in train_s1_ids:
        targets = truth_by_s1.get(s1_id, set())
        for t in targets:
            if t.startswith("S2-"):
                train_s2_targets[t] += 1
            elif t.startswith("S3-"):
                train_s3_targets[t] += 1

    for s1_id in val_s1_ids:
        targets = truth_by_s1.get(s1_id, set())
        for t in targets:
            if t.startswith("S2-"):
                val_s2_targets[t] += 1
            elif t.startswith("S3-"):
                val_s3_targets[t] += 1

    # Overlaps
    shared_s2 = set(train_s2_targets.keys()) & set(val_s2_targets.keys())
    shared_s3 = set(train_s3_targets.keys()) & set(val_s3_targets.keys())

    s2_multi_train = sum(1 for c in train_s2_targets.values() if c > 1)
    s2_multi_val = sum(1 for c in val_s2_targets.values() if c > 1)
    s3_multi_train = sum(1 for c in train_s3_targets.values() if c > 1)
    s3_multi_val = sum(1 for c in val_s3_targets.values() if c > 1)

    report_records = [
        {
            "category": "S2 Targets in Train",
            "unique_targets": len(train_s2_targets),
            "multi_s1_occurrences": s2_multi_train,
            "train_val_shared_count": len(shared_s2),
        },
        {
            "category": "S2 Targets in Val",
            "unique_targets": len(val_s2_targets),
            "multi_s1_occurrences": s2_multi_val,
            "train_val_shared_count": len(shared_s2),
        },
        {
            "category": "S3 Targets in Train",
            "unique_targets": len(train_s3_targets),
            "multi_s1_occurrences": s3_multi_train,
            "train_val_shared_count": len(shared_s3),
        },
        {
            "category": "S3 Targets in Val",
            "unique_targets": len(val_s3_targets),
            "multi_s1_occurrences": s3_multi_val,
            "train_val_shared_count": len(shared_s3),
        },
    ]

    report_csv = output_dir / "target_overlap_report.csv"
    save_csv_from_dicts(report_records, report_csv)

    warning_text = None
    if len(shared_s2) > 0 or len(shared_s3) > 0:
        warning_text = (
            "Potential target-level leakage exists; E2 exclusivity scan is required "
            "before interpreting cross-S1 target reuse."
        )

    summary_json = {
        "train_s1_count": len(train_s1_ids),
        "val_s1_count": len(val_s1_ids),
        "train_s2_target_count": len(train_s2_targets),
        "val_s2_target_count": len(val_s2_targets),
        "train_s3_target_count": len(train_s3_targets),
        "val_s3_target_count": len(val_s3_targets),
        "shared_s2_target_count": len(shared_s2),
        "shared_s3_target_count": len(shared_s3),
        "s2_targets_under_multiple_s1": s2_multi_train + s2_multi_val,
        "s3_targets_under_multiple_s1": s3_multi_train + s3_multi_val,
        "warning": warning_text,
    }

    summary_json_path = output_dir / "target_overlap_summary.json"
    save_json(summary_json, summary_json_path)

    return report_records, summary_json
