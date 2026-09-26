"""
Production Runner and CLI for E3 Blocking and Candidate Validation.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
This module serves as the primary reproducible entry point for Sprint E3.
It coordinates country-partitioned inverted indexing across K1-K7, streams candidates to
candidate_pairs.tsv, evaluates blocking recall, entity coverage, candidate volume,
and per-lane contributions against ground truth, and writes all certified artifacts and manifests.
"""

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from e1.io_utils import EXPERIMENTS_ROOT, STUDENT_RESOURCE_ROOT, TRAIN_FILES
from .blocking_lanes import BlockingConfig
from .candidate_generator import PartitionRunResult, run_partition_blocking
from .evaluator import E3EvaluationSummary, analyze_failure_buckets, categorize_missed_pair

DEFAULT_E3_OUTPUT_DIR = EXPERIMENTS_ROOT / "e3"


def compute_file_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def count_lines(filepath: Path) -> int:
    """Count lines in text file."""
    cnt = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            cnt += 1
    return cnt


def load_ground_truth(
    gt_path: Path, allowed_s1_ids: Optional[Set[str]] = None
) -> Tuple[Dict[str, Set[str]], int]:
    """
    Load ground truth mapping: s1_id -> set(target_ids).
    Returns (truth_by_s1, total_gt_edges).
    """
    truth_by_s1: Dict[str, Set[str]] = {}
    total_edges = 0
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            s1_id = row[0].strip()
            if allowed_s1_ids is not None and s1_id not in allowed_s1_ids:
                continue
            targets = [t.strip() for t in row[1].split(",") if t.strip()]
            if targets:
                truth_by_s1[s1_id] = set(targets)
                total_edges += len(targets)
    return truth_by_s1, total_edges


def verify_country_partition_integrity(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    gt_path: Path,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validation check: verify that every ground truth edge belongs to the same country partition.
    """
    print("[Integrity] Verifying country partitioning on ground truth...")
    s1_country: Dict[str, str] = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                s1_country[row[0].strip()] = row[3].strip() if len(row) > 3 else "Unknown"

    target_country: Dict[str, str] = {}
    for p in (s2_path, s3_path):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    target_country[row[0].strip()] = row[3].strip() if len(row) > 3 else "Unknown"

    cross_country_violations = 0
    total_checked = 0
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            s1 = row[0].strip()
            c1 = s1_country.get(s1)
            targets = [t.strip() for t in row[1].split(",") if t.strip()]
            for t in targets:
                c_target = target_country.get(t)
                total_checked += 1
                if c1 and c_target and c1 != c_target:
                    cross_country_violations += 1
            if sample_limit and total_checked >= sample_limit:
                break

    print(
        f"[Integrity] Verified {total_checked:,} GT edges. "
        f"Cross-country violations: {cross_country_violations}"
    )
    return {
        "total_gt_checked": total_checked,
        "cross_country_violations": cross_country_violations,
        "is_safe": cross_country_violations == 0,
    }


def sample_missed_records(
    missed_edges: List[Tuple[str, str]],
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    max_samples: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Fetch record details for missed GT pairs and run failure classification.
    """
    if not missed_edges:
        return []

    sample = missed_edges[:max_samples]
    needed_s1 = {pair[0] for pair in sample}
    needed_s2 = {pair[1] for pair in sample if pair[1].startswith("S2-")}
    needed_s3 = {pair[1] for pair in sample if pair[1].startswith("S3-")}

    s1_data: Dict[str, Tuple[str, str, str]] = {}
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row and row[0].strip() in needed_s1:
                eid = row[0].strip()
                name = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                country = row[3].strip() if len(row) > 3 else "US"
                s1_data[eid] = (name, addr, country)

    target_data: Dict[str, Tuple[str, str]] = {}
    for p, target_set in [(s2_path, needed_s2), (s3_path, needed_s3)]:
        if not target_set:
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row and row[0].strip() in target_set:
                    eid = row[0].strip()
                    name = row[1].strip() if len(row) > 1 else ""
                    addr = row[2].strip() if len(row) > 2 else ""
                    target_data[eid] = (name, addr)

    categorized = []
    for s1_id, target_id in sample:
        if s1_id in s1_data and target_id in target_data:
            s1_name, s1_addr, country = s1_data[s1_id]
            target_name, target_addr = target_data[target_id]
            source = "S2" if target_id.startswith("S2-") else "S3"
            cat = categorize_missed_pair(
                s1_name=s1_name,
                s1_addr=s1_addr,
                target_name=target_name,
                target_addr=target_addr,
                country=country,
                source=source,
            )
            cat["s1_id"] = s1_id
            cat["target_id"] = target_id
            categorized.append(cat)

    return categorized


def verify_candidate_file_integrity(
    candidate_path: Path,
    expected_pairs: int,
    recovered_gt_edges: int,
    truth_by_s1: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Any]:
    """
    Streamingly verify candidate_pairs.tsv integrity:
    1. Exact header matching: source1_entity_id\tsource2_or_source3_entity_id\tsource
    2. Column count == 3
    3. Valid IDs and correct source prefixes (S2- for S2, S3- for S3)
    4. Zero duplicates within S1 candidate blocks
    5. Exact count matches expected_pairs
    6. All recovered GT edges exist in candidate_pairs.tsv
    """
    print("[Verification] Verifying candidate_pairs.tsv integrity...")
    total_pairs = 0
    duplicate_pairs = 0
    malformed_rows = 0
    source_mismatches = 0
    gt_recovered_count = 0
    header_valid = False

    current_s1: Optional[str] = None
    seen_targets_for_s1: Set[str] = set()

    with open(candidate_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if header == "source1_entity_id\tsource2_or_source3_entity_id\tsource\n":
            header_valid = True

        for line in f:
            total_pairs += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                malformed_rows += 1
                continue
            s1_id, target_id, source = parts[0], parts[1], parts[2]

            # Source prefix check
            expected_prefix = f"{source}-"
            if not target_id.startswith(expected_prefix):
                source_mismatches += 1

            # Duplicate check within S1 query blocks
            if s1_id == current_s1:
                if target_id in seen_targets_for_s1:
                    duplicate_pairs += 1
                else:
                    seen_targets_for_s1.add(target_id)
            else:
                current_s1 = s1_id
                seen_targets_for_s1 = {target_id}

            # Check if this candidate pair is a recovered GT edge
            if truth_by_s1 and s1_id in truth_by_s1 and target_id in truth_by_s1[s1_id]:
                gt_recovered_count += 1

    print(
        f"[Verification] Candidate integrity check complete: "
        f"{total_pairs:,} candidate pairs verified. "
        f"Header valid: {header_valid}. "
        f"Duplicates: {duplicate_pairs}. "
        f"Source mismatches: {source_mismatches}. "
        f"Malformed rows: {malformed_rows}."
    )
    if truth_by_s1:
        print(
            f"[Verification] GT edges recovered in candidate file: {gt_recovered_count:,} / {recovered_gt_edges:,} expected."
        )

    is_valid = (
        header_valid
        and total_pairs == expected_pairs
        and duplicate_pairs == 0
        and malformed_rows == 0
        and source_mismatches == 0
        and (gt_recovered_count == recovered_gt_edges if truth_by_s1 else True)
    )

    return {
        "is_valid": is_valid,
        "header_valid": header_valid,
        "total_candidate_pairs": total_pairs,
        "expected_candidate_pairs": expected_pairs,
        "duplicate_pairs": duplicate_pairs,
        "malformed_rows": malformed_rows,
        "source_mismatches": source_mismatches,
        "gt_edges_in_candidate_file": gt_recovered_count,
        "expected_gt_recovered": recovered_gt_edges,
    }


def run_e3_pipeline(
    s1_path: Path = TRAIN_FILES["S1"],
    s2_path: Path = TRAIN_FILES["S2"],
    s3_path: Path = TRAIN_FILES["S3"],
    gt_path: Path = TRAIN_FILES["GT"],
    output_dir: Path = DEFAULT_E3_OUTPUT_DIR,
    sample_size: Optional[int] = None,
    config: Optional[BlockingConfig] = None,
) -> Dict[str, Any]:
    """
    Execute full E3 symbolic blocking pipeline and produce all certified artifacts.
    """
    if config is None:
        config = BlockingConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 75)
    print("STARTING SPRINT E3 — L2 SYMBOLIC BLOCKING & CANDIDATE VALIDATION")
    print(f"Output Directory: {output_dir}")
    print(f"Config: {config.to_dict()}")
    if sample_size:
        print(f"Running on deterministic sample: {sample_size:,} S1 entities")
    print("=" * 75)

    # 1. Deterministic S1 ID filtering if sample_size is specified
    allowed_s1_ids: Optional[Set[str]] = None
    if sample_size:
        allowed_s1_ids = set()
        with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    allowed_s1_ids.add(row[0].strip())
                    if len(allowed_s1_ids) >= sample_size:
                        break

    # 2. Ground Truth Integrity and Loading
    gt_integrity = verify_country_partition_integrity(
        s1_path, s2_path, s3_path, gt_path, sample_limit=sample_size * 4 if sample_size else None
    )
    truth_by_s1, total_gt_edges = load_ground_truth(gt_path, allowed_s1_ids=allowed_s1_ids)
    print(f"[GT] Loaded {len(truth_by_s1):,} S1 entities with {total_gt_edges:,} total GT edges.")

    # 3. Initialize candidate_pairs.tsv
    candidate_path = output_dir / "candidate_pairs.tsv"
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tsource2_or_source3_entity_id\tsource\n")

    # 4. Partition Execution (Executed in isolated subprocesses for zero memory leak)
    partitions = [
        ("India", "S2", s2_path),
        ("India", "S3", s3_path),
        ("US", "S2", s2_path),
        ("US", "S3", s3_path),
    ]

    partition_results: List[PartitionRunResult] = []
    all_missed_edges: List[Tuple[str, str]] = []

    for country, source, path in partitions:
        result_json_path = output_dir / f"res_{country}_{source}.json"
        cmd = [
            sys.executable,
            "-m", "e3.partition_worker",
            "--country", country,
            "--target-source", source,
            "--target-path", str(path),
            "--s1-path", str(s1_path),
            "--gt-path", str(gt_path),
            "--candidate-path", str(candidate_path),
            "--result-path", str(result_json_path),
            "--config-json", json.dumps(config.to_dict()),
        ]
        if sample_size:
            cmd.extend(["--sample-size", str(sample_size)])

        # Run isolated partition worker with guaranteed PYTHONPATH
        worker_env = os.environ.copy()
        src_dir = str(Path(__file__).resolve().parent.parent)
        worker_env["PYTHONPATH"] = (
            f"{src_dir}:{worker_env['PYTHONPATH']}"
            if "PYTHONPATH" in worker_env
            else src_dir
        )
        subprocess.run(cmd, env=worker_env, check=True)

        with open(result_json_path, "r", encoding="utf-8") as f:
            res_dict = json.load(f)

        p_res = PartitionRunResult(
            country=res_dict["country"],
            target_source=res_dict["target_source"],
            total_s1_queried=res_dict["total_s1_queried"],
            total_target_records_indexed=res_dict["total_target_records_indexed"],
            total_candidate_pairs=res_dict["total_candidate_pairs"],
            total_gt_edges=res_dict["total_gt_edges"],
            recovered_gt_edges=res_dict["recovered_gt_edges"],
            entities_with_gt=res_dict["entities_with_gt"],
            entities_fully_covered=res_dict["entities_fully_covered"],
            entities_partially_covered=res_dict["entities_partially_covered"],
            entities_uncovered=res_dict["entities_uncovered"],
            s1_recovered_counts=res_dict.get("s1_recovered_counts", {}),
            s1_recovered_counts_cap150=res_dict.get("s1_recovered_counts_cap150", {}),
            lane_metrics=res_dict["lane_metrics"],
            candidate_lengths=res_dict.get("candidate_lengths", []),
            candidate_histogram=Counter({int(k): v for k, v in res_dict.get("candidate_histogram", {}).items()}),
            candidate_histogram_cap150=Counter({int(k): v for k, v in res_dict.get("candidate_histogram_cap150", {}).items()}),
            queries_hitting_cap_150=res_dict.get("queries_hitting_cap_150", 0),
            gt_edges_lost_to_cap_150=res_dict.get("gt_edges_lost_to_cap_150", 0),
            recovered_gt_edges_cap150=res_dict.get("recovered_gt_edges_cap150", 0),
            total_candidate_pairs_cap150=res_dict.get("total_candidate_pairs_cap150", 0),
            k4_diagnostics=Counter(res_dict.get("k4_diagnostics", {})),
            total_oversized_query_events=res_dict["total_oversized_query_events"],
            missed_gt_edges=[tuple(pair) for pair in res_dict["missed_gt_edges"]],
        )
        partition_results.append(p_res)
        all_missed_edges.extend(p_res.missed_gt_edges)
        result_json_path.unlink(missing_ok=True)

    # 5. Failure Analysis
    print(f"\n[Failure Analysis] Analyzing {len(all_missed_edges):,} missed GT edges...")
    missed_records = sample_missed_records(
        missed_edges=all_missed_edges,
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        max_samples=5000,
    )
    failure_analysis = analyze_failure_buckets(missed_records)

    duration = time.time() - t0

    # 6. Evaluation Summary
    total_s1_entities = count_lines(s1_path) - 1
    if sample_size and sample_size < total_s1_entities:
        total_s1_entities = sample_size

    summary = E3EvaluationSummary(
        partition_results=partition_results,
        failure_analysis=failure_analysis,
        config=config.to_dict(),
        duration_seconds=duration,
        truth_by_s1=truth_by_s1,
        total_s1_entities=total_s1_entities,
    )

    # 7. Candidate file integrity verification
    integrity_result = verify_candidate_file_integrity(
        candidate_path=candidate_path,
        expected_pairs=summary.total_candidate_pairs,
        recovered_gt_edges=summary.recovered_gt_edges,
        truth_by_s1=truth_by_s1,
    )
    assert integrity_result["is_valid"], f"Candidate file integrity validation failed: {integrity_result}"

    # 8. Write Artifacts
    # A. JSON Report
    report_json_path = output_dir / "e3_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    # B. Markdown Report
    report_md_path = output_dir / "e3_report.md"
    run_cmd = "PYTHONPATH=student_resource/code/business_entity_resolution/src python3 -m e3.runner"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(summary.generate_markdown_report(run_command=run_cmd))

    # C. CSV Summaries
    metrics_csv_path = output_dir / "e3_metrics_overall.csv"
    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.to_dict()["overall_metrics"].items():
            writer.writerow([k, v])
        for src, data in summary.to_dict()["source_breakdown"].items():
            for k, v in data.items():
                writer.writerow([f"{src}_{k}", v])

    lane_csv_path = output_dir / "e3_lane_contributions.csv"
    with open(lane_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lane",
            "candidate_pairs",
            "standalone_gt_recovered",
            "marginal_gt_recovered",
            "cumulative_gt_recovered",
            "marginal_recall_pct",
            "oversized_keys",
        ])
        for lane, stats in summary.lane_summary.items():
            writer.writerow([
                lane,
                stats["candidate_pairs"],
                stats.get("standalone_gt_recovered", 0),
                stats.get("marginal_gt_recovered", 0),
                stats.get("cumulative_gt_recovered", 0),
                stats.get("marginal_recall_pct", 0.0),
                stats.get("oversized_keys", 0),
            ])

    failure_csv_path = output_dir / "e3_failure_analysis.csv"
    with open(failure_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "category", "count", "percentage"])
        for dim, cats in failure_analysis.get("dimensions", {}).items():
            for cat, data in cats.items():
                writer.writerow([dim, cat, data["count"], data["percentage"]])

    ab_csv_path = output_dir / "e3_ab_cap_comparison.csv"
    with open(ab_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration",
            "recovered_gt_edges",
            "recall_pct",
            "total_candidate_pairs",
            "candidate_to_gt_ratio",
            "queries_hitting_cap",
            "queries_hitting_cap_pct",
            "gt_edges_lost_to_cap",
            "fully_covered_entities",
            "fully_covered_pct",
        ])
        for cfg_key in ("config_a_cap_150", "config_b_untruncated"):
            cdata = summary.ab_comparison[cfg_key]
            writer.writerow([
                cdata["name"],
                cdata["recovered_gt_edges"],
                cdata["recall_pct"],
                cdata["total_candidate_pairs"],
                cdata["candidate_to_gt_ratio"],
                cdata["queries_hitting_cap"],
                cdata["queries_hitting_cap_pct"],
                cdata["gt_edges_lost_to_cap"],
                cdata["fully_covered_entities"],
                cdata["fully_covered_pct"],
            ])

    k4_csv_path = output_dir / "e3_k4_diagnostics.csv"
    with open(k4_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_mode", "count", "percentage"])
        total_k4_diag = sum(summary.k4_diagnostic_summary.values())
        for mode, count in summary.k4_diagnostic_summary.most_common():
            pct = round((count / total_k4_diag * 100), 2) if total_k4_diag > 0 else 0.0
            writer.writerow([mode, count, pct])

    # 9. Build Manifest
    artifact_files = [
        candidate_path,
        report_json_path,
        report_md_path,
        metrics_csv_path,
        lane_csv_path,
        failure_csv_path,
        ab_csv_path,
        k4_csv_path,
    ]
    manifest_artifacts = {}
    for p in artifact_files:
        manifest_artifacts[p.name] = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "line_count": count_lines(p),
            "sha256": compute_file_sha256(p),
        }

    manifest = {
        "stage": "E3",
        "description": "L2 Symbolic Multi-Lane Blocking & Candidate Validation",
        "status": summary.determine_status(),
        "duration_seconds": round(duration, 2),
        "gt_integrity": gt_integrity,
        "candidate_integrity": integrity_result,
        "configuration": config.to_dict(),
        "overall_metrics": summary.to_dict()["overall_metrics"],
        "source_breakdown": summary.to_dict()["source_breakdown"],
        "country_breakdown": summary.to_dict()["country_breakdown"],
        "entity_coverage": summary.to_dict()["entity_coverage"],
        "candidate_volume_stats": summary.to_dict()["candidate_volume_stats"],
        "ab_cap_comparison": summary.ab_comparison,
        "k4_diagnostics": dict(summary.k4_diagnostic_summary),
        "lane_contributions": summary.lane_summary,
        "failure_analysis": failure_analysis,
        "artifacts": manifest_artifacts,
    }

    manifest_path = output_dir / "e3_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Console Summary
    print("\n" + "=" * 75)
    print("SPRINT E3 EXECUTION COMPLETE")
    print(f"Status: {summary.determine_status()}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Overall Recall: {summary.overall_recall*100:.2f}% ({summary.recovered_gt_edges:,} / {summary.total_gt_edges:,})")
    print(f"Source 2 Recall: {summary.s2_recall*100:.2f}% ({summary.s2_rec:,} / {summary.s2_gt:,})")
    print(f"Source 3 Recall: {summary.s3_recall*100:.2f}% ({summary.s3_rec:,} / {summary.s3_gt:,})")
    print(f"India Recall: {summary.india_recall*100:.2f}% | US Recall: {summary.us_recall*100:.2f}%")
    print(f"Total Candidates: {summary.total_candidate_pairs:,} (Mean per S1: {summary.cand_mean:.2f})")
    print(f"Candidate / GT Multiplier: {summary.total_candidate_pairs / summary.total_gt_edges:.2f}x")
    print("=" * 75 + "\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Sprint E3 Blocking Evaluation Runner")
    parser.add_argument("--data-dir", type=Path, default=STUDENT_RESOURCE_ROOT / "dataset" / "train")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_E3_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size of S1 entities to evaluate")
    parser.add_argument("--max-block-size", type=int, default=500, help="Max posting list size")
    parser.add_argument(
        "--max-candidates-per-query",
        type=lambda x: None if str(x).lower() in ("none", "null", "") else int(x),
        default=None,
        help="Max candidates per S1 query (default: None for untruncated candidate generation bounded by max_block_size)",
    )
    parser.add_argument("--max-token-doc-freq", type=int, default=2000, help="Max token document frequency")
    parser.add_argument("--enabled-lanes", type=str, default="K1,K2,K3,K4,K5,K6,K7")

    args = parser.parse_args()

    s1_path = args.data_dir / f"{args.split}_source1.tsv"
    s2_path = args.data_dir / f"{args.split}_source2.tsv"
    s3_path = args.data_dir / f"{args.split}_source3.tsv"
    gt_path = args.gt if args.gt else (args.data_dir / f"{args.split}_ground_truth.tsv")

    config = BlockingConfig(
        enabled_lanes=[l.strip() for l in args.enabled_lanes.split(",") if l.strip()],
        max_block_size=args.max_block_size,
        max_candidates_per_query=args.max_candidates_per_query,
        max_token_doc_freq=args.max_token_doc_freq,
    )

    run_e3_pipeline(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        gt_path=gt_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        config=config,
    )


if __name__ == "__main__":
    main()
