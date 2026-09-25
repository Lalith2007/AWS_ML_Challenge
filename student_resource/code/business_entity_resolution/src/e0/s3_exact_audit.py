"""
S3 Exact Match Reconfirmation Audit for E0.
Evaluates exact normalized agreement across the FULL train ground truth for S1 <-> S3 edges.
Amazon ML Challenge 2026.
"""

import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .common import E0_RESULTS_DIR, TRAIN_FILES, normalize_text


def run_s3_exact_audit(
    output_dir: Optional[Path] = None,
    max_examples: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates exact normalized agreement for ALL true S1 <-> S3 edges in train_ground_truth.tsv.

    Memory strategy:
    1. Reads train_source1.tsv into memory: storing normalized name, normalized address,
       raw name, raw address, country (~450 MB RAM for 2.2M rows).
    2. Streams train_ground_truth.tsv: maps each S3 ID to (s1_id, multiplicity) (~150 MB RAM).
    3. Streams train_source3.tsv: matches S3 records on the fly, evaluates exact agreement,
       and discards chunks immediately. Memory stays well under 1 GB total.

    Writes:
      - s3_exact_summary.csv
      - s3_exact_by_country.csv
      - s3_both_exact_examples.tsv
    Returns:
      (df_summary, df_country, df_examples, full_stats_dict)
    """
    if output_dir is None:
        output_dir = E0_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("OBJECTIVE 2: RUNNING S3 FULL GROUND-TRUTH EXACT AUDIT")
    print("=" * 90)

    # Step 1: Load S1 records
    t0 = time.time()
    print("Step 1/3: Loading Source 1 records and precomputing conservative normalization...")
    s1_records: Dict[str, Tuple[str, str, str, str, str]] = {}

    with open(TRAIN_FILES["S1"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            raw_name = row[1] if len(row) > 1 else ""
            raw_addr = row[2] if len(row) > 2 else ""
            country = row[3] if len(row) > 3 else ""

            norm_name = normalize_text(raw_name)
            norm_addr = normalize_text(raw_addr)

            s1_records[s1_id] = (raw_name, raw_addr, norm_name, norm_addr, country)

    print(f"  Loaded {len(s1_records):,} Source 1 records in {time.time() - t0:.2f}s.")

    # Step 2: Parse Ground Truth S3 edges
    t1 = time.time()
    print("Step 2/3: Parsing Ground Truth for true S1 <-> S3 edges...")
    s3_to_s1: Dict[str, Tuple[str, int]] = {}
    total_gt_s1_rows = 0
    total_gt_s2_edges = 0
    total_gt_s3_edges = 0

    with open(TRAIN_FILES["GT"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            total_gt_s1_rows += 1
            s1_id = row[0]
            target_str = row[1] if len(row) > 1 else ""
            if not target_str:
                continue

            targets = [t.strip() for t in target_str.split(",") if t.strip()]
            mult = len(targets)

            for target_id in targets:
                if target_id.startswith("S2-"):
                    total_gt_s2_edges += 1
                elif target_id.startswith("S3-"):
                    total_gt_s3_edges += 1
                    s3_to_s1[target_id] = (s1_id, mult)

    print(
        f"  Parsed Ground Truth in {time.time() - t1:.2f}s: "
        f"{total_gt_s1_rows:,} S1 rows, {total_gt_s2_edges:,} S2 edges, "
        f"{total_gt_s3_edges:,} S3 edges."
    )

    # Step 3: Stream Source 3 and evaluate exact match agreement
    t2 = time.time()
    print("Step 3/3: Streaming Source 3 and evaluating exact normalized agreement...")

    total_s3_evaluated = 0
    name_exact = 0
    addr_exact = 0
    both_exact = 0
    name_only = 0
    addr_only = 0
    neither_exact = 0

    country_stats = defaultdict(lambda: Counter())
    mult_stats = defaultdict(lambda: Counter())
    addr_status_counts = Counter()

    both_exact_examples: List[Dict[str, Any]] = []

    with open(TRAIN_FILES["S3"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s3_id = row[0]

            if s3_id not in s3_to_s1:
                continue

            total_s3_evaluated += 1
            s1_id, mult = s3_to_s1[s3_id]

            s1_data = s1_records.get(s1_id)
            if s1_data is None:
                raise ValueError(f"Ground truth S1 ID {s1_id} not found in Source 1!")

            s1_raw_name, s1_raw_addr, s1_name_norm, s1_addr_norm, s1_country = s1_data

            s3_raw_name = row[1] if len(row) > 1 else ""
            s3_raw_addr = row[2] if len(row) > 2 else ""
            s3_country = row[3] if len(row) > 3 else ""

            s3_name_norm = normalize_text(s3_raw_name)
            s3_addr_norm = normalize_text(s3_raw_addr)

            # Missing-address status
            s1_has_addr = bool(s1_addr_norm)
            s3_has_addr = bool(s3_addr_norm)
            if s1_has_addr and s3_has_addr:
                addr_status = "both_present"
            elif not s1_has_addr and s3_has_addr:
                addr_status = "s1_missing_s3_present"
            elif s1_has_addr and not s3_has_addr:
                addr_status = "s1_present_s3_missing"
            else:
                addr_status = "both_missing"
            addr_status_counts[addr_status] += 1

            # Exact matches (must be non-empty)
            n_match = bool(s1_name_norm) and (s1_name_norm == s3_name_norm)
            a_match = bool(s1_addr_norm) and (s1_addr_norm == s3_addr_norm)
            b_match = n_match and a_match

            if n_match:
                name_exact += 1
            if a_match:
                addr_exact += 1
            if b_match:
                both_exact += 1
                if len(both_exact_examples) < max_examples:
                    both_exact_examples.append({
                        "source1_entity_id": s1_id,
                        "source3_entity_id": s3_id,
                        "country": s1_country,
                        "multiplicity": mult,
                        "s1_name_raw": s1_raw_name,
                        "s3_name_raw": s3_raw_name,
                        "s1_address_raw": s1_raw_addr,
                        "s3_address_raw": s3_raw_addr,
                        "normalized_name": s1_name_norm,
                        "normalized_address": s1_addr_norm,
                    })
            elif n_match:
                name_only += 1
            elif a_match:
                addr_only += 1
            else:
                neither_exact += 1

            c = s1_country if s1_country else "Unknown"
            country_stats[c]["total"] += 1
            if n_match:
                country_stats[c]["name_exact"] += 1
            if a_match:
                country_stats[c]["addr_exact"] += 1
            if b_match:
                country_stats[c]["both_exact"] += 1
            elif n_match:
                country_stats[c]["name_only"] += 1
            elif a_match:
                country_stats[c]["addr_only"] += 1
            else:
                country_stats[c]["neither_exact"] += 1

            m_bucket = str(mult) if mult <= 5 else "6+"
            mult_stats[m_bucket]["total"] += 1
            if n_match:
                mult_stats[m_bucket]["name_exact"] += 1
            if a_match:
                mult_stats[m_bucket]["addr_exact"] += 1
            if b_match:
                mult_stats[m_bucket]["both_exact"] += 1

    eval_time = time.time() - t2
    print(f"  Streaming evaluation completed in {eval_time:.2f}s.")

    # Reconcile counts
    if total_s3_evaluated != total_gt_s3_edges:
        raise RuntimeError(
            f"Ground truth S3 edges count mismatch: parsed {total_gt_s3_edges:,} from GT "
            f"but evaluated {total_s3_evaluated:,} against Source 3!"
        )

    # 1. Summary DataFrame
    summary_data = [{
        "metric": "Total S3 True Edges Evaluated",
        "count": total_s3_evaluated,
        "percentage": 100.0,
    }, {
        "metric": "NAME_EXACT (normalized)",
        "count": name_exact,
        "percentage": name_exact / total_s3_evaluated * 100,
    }, {
        "metric": "ADDR_EXACT (normalized)",
        "count": addr_exact,
        "percentage": addr_exact / total_s3_evaluated * 100,
    }, {
        "metric": "BOTH_EXACT (name & addr)",
        "count": both_exact,
        "percentage": both_exact / total_s3_evaluated * 100,
    }, {
        "metric": "Name Exact Only",
        "count": name_only,
        "percentage": name_only / total_s3_evaluated * 100,
    }, {
        "metric": "Address Exact Only",
        "count": addr_only,
        "percentage": addr_only / total_s3_evaluated * 100,
    }, {
        "metric": "Neither Exact",
        "count": neither_exact,
        "percentage": neither_exact / total_s3_evaluated * 100,
    }, {
        "metric": "Address Status: Both Present",
        "count": addr_status_counts["both_present"],
        "percentage": addr_status_counts["both_present"] / total_s3_evaluated * 100,
    }, {
        "metric": "Address Status: S1 Present, S3 Missing",
        "count": addr_status_counts["s1_present_s3_missing"],
        "percentage": addr_status_counts["s1_present_s3_missing"] / total_s3_evaluated * 100,
    }, {
        "metric": "Address Status: S1 Missing, S3 Present",
        "count": addr_status_counts["s1_missing_s3_present"],
        "percentage": addr_status_counts["s1_missing_s3_present"] / total_s3_evaluated * 100,
    }, {
        "metric": "Address Status: Both Missing",
        "count": addr_status_counts["both_missing"],
        "percentage": addr_status_counts["both_missing"] / total_s3_evaluated * 100,
    }]
    df_summary = pd.DataFrame(summary_data)

    # 2. Country breakdown DataFrame
    country_rows = []
    for c in sorted(country_stats.keys()):
        stats = country_stats[c]
        c_total = stats["total"]
        country_rows.append({
            "country": c,
            "total_edges": c_total,
            "name_exact": stats["name_exact"],
            "name_exact_pct": stats["name_exact"] / c_total * 100,
            "addr_exact": stats["addr_exact"],
            "addr_exact_pct": stats["addr_exact"] / c_total * 100,
            "both_exact": stats["both_exact"],
            "both_exact_pct": stats["both_exact"] / c_total * 100,
            "name_only": stats["name_only"],
            "name_only_pct": stats["name_only"] / c_total * 100,
            "addr_only": stats["addr_only"],
            "addr_only_pct": stats["addr_only"] / c_total * 100,
            "neither_exact": stats["neither_exact"],
            "neither_exact_pct": stats["neither_exact"] / c_total * 100,
        })
    df_country = pd.DataFrame(country_rows)

    # 3. Examples DataFrame
    df_examples = pd.DataFrame(both_exact_examples)

    # Output paths
    summary_path = output_dir / "s3_exact_summary.csv"
    country_path = output_dir / "s3_exact_by_country.csv"
    examples_path = output_dir / "s3_both_exact_examples.tsv"

    df_summary.to_csv(summary_path, index=False)
    df_country.to_csv(country_path, index=False)
    df_examples.to_csv(examples_path, sep="\t", index=False)

    print(f"\nWrote S3 exact summary to: {summary_path}")
    print(f"Wrote S3 country breakdown to: {country_path}")
    print(f"Wrote S3 both-exact examples to: {examples_path}")

    full_stats = {
        "total_s3_evaluated": total_s3_evaluated,
        "name_exact": name_exact,
        "name_exact_pct": name_exact / total_s3_evaluated * 100,
        "addr_exact": addr_exact,
        "addr_exact_pct": addr_exact / total_s3_evaluated * 100,
        "both_exact": both_exact,
        "both_exact_pct": both_exact / total_s3_evaluated * 100,
        "name_only": name_only,
        "name_only_pct": name_only / total_s3_evaluated * 100,
        "addr_only": addr_only,
        "addr_only_pct": addr_only / total_s3_evaluated * 100,
        "neither_exact": neither_exact,
        "neither_exact_pct": neither_exact / total_s3_evaluated * 100,
        "addr_status_counts": dict(addr_status_counts),
        "country_stats": {k: dict(v) for k, v in country_stats.items()},
        "multiplicity_stats": {k: dict(v) for k, v in mult_stats.items()},
        "total_both_exact_found": len(both_exact_examples),
    }

    return df_summary, df_country, df_examples, full_stats
