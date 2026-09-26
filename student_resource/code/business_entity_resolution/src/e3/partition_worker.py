"""
Isolated Subprocess Worker for Single (Country, Source) Partition Blocking.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
The full training dataset contains ~10.3M target records and ~2.2M S1 records.
Executing each country-source partition in an isolated process guarantees:
1. 100% of memory is reclaimed by the OS immediately upon partition completion.
2. Zero memory leak or heap fragmentation across partitions.
3. Strict country and source isolation as required by the locked architecture.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from .blocking_lanes import BlockingConfig
from .candidate_generator import PartitionRunResult, run_partition_blocking


def load_ground_truth_for_partition(
    gt_path: Path,
    target_prefix: str,
    allowed_s1_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Set[str]], int]:
    """Load ground truth filtered to target source prefix (e.g. 'S2-')."""
    truth_by_s1: Dict[str, Set[str]] = {}
    total_edges = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            s1_id = row[0].strip()
            if allowed_s1_ids is not None and s1_id not in allowed_s1_ids:
                continue
            targets = {t.strip() for t in row[1].split(",") if t.strip().startswith(target_prefix)}
            if targets:
                truth_by_s1[s1_id] = targets
                total_edges += len(targets)

    return truth_by_s1, total_edges


def main():
    parser = argparse.ArgumentParser(description="Partition Blocking Worker")
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--target-source", type=str, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--s1-path", type=Path, required=True)
    parser.add_argument("--gt-path", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--config-json", type=str, default=None)

    args = parser.parse_args()

    config = BlockingConfig()
    if args.config_json:
        cfg_dict = json.loads(args.config_json)
        config.enabled_lanes = cfg_dict.get("enabled_lanes", config.enabled_lanes)
        config.max_block_size = cfg_dict.get("max_block_size", config.max_block_size)
        config.max_candidates_per_query = cfg_dict.get("max_candidates_per_query", config.max_candidates_per_query)
        config.max_token_doc_freq = cfg_dict.get("max_token_doc_freq", config.max_token_doc_freq)

    allowed_s1_ids: Optional[Set[str]] = None
    if args.sample_size:
        allowed_s1_ids = set()
        with open(args.s1_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if row:
                    allowed_s1_ids.add(row[0].strip())
                    if len(allowed_s1_ids) >= args.sample_size:
                        break

    prefix = f"{args.target_source}-"
    truth_by_s1, total_gt = load_ground_truth_for_partition(
        args.gt_path, target_prefix=prefix, allowed_s1_ids=allowed_s1_ids
    )

    # Ensure candidate file exists with header
    if not args.candidate_path.is_file() or args.candidate_path.stat().st_size == 0:
        with open(args.candidate_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tsource2_or_source3_entity_id\tsource\n")

    t0 = time.time()
    with open(args.candidate_path, "a", encoding="utf-8") as cand_file:
        res = run_partition_blocking(
            country=args.country,
            target_source=args.target_source,
            target_path=args.target_path,
            s1_path=args.s1_path,
            truth_by_s1=truth_by_s1,
            allowed_s1_ids=allowed_s1_ids,
            candidate_file_handle=cand_file,
            config=config,
            collect_missed_edges=True,
        )

    duration = time.time() - t0

    # Save partition result JSON
    res_dict = {
        "country": res.country,
        "target_source": res.target_source,
        "duration_seconds": round(duration, 2),
        "total_s1_queried": res.total_s1_queried,
        "total_target_records_indexed": res.total_target_records_indexed,
        "total_candidate_pairs": res.total_candidate_pairs,
        "total_gt_edges": res.total_gt_edges,
        "recovered_gt_edges": res.recovered_gt_edges,
        "recall": round(res.recall, 6),
        "entities_with_gt": res.entities_with_gt,
        "entities_fully_covered": res.entities_fully_covered,
        "entities_partially_covered": res.entities_partially_covered,
        "entities_uncovered": res.entities_uncovered,
        "lane_metrics": res.lane_metrics,
        "candidate_lengths": res.candidate_lengths[:5000],
        "candidate_histogram": dict(res.candidate_histogram),
        "candidate_histogram_cap150": dict(res.candidate_histogram_cap150),
        "queries_hitting_cap_150": res.queries_hitting_cap_150,
        "gt_edges_lost_to_cap_150": res.gt_edges_lost_to_cap_150,
        "recovered_gt_edges_cap150": res.recovered_gt_edges_cap150,
        "total_candidate_pairs_cap150": res.total_candidate_pairs_cap150,
        "s1_recovered_counts": res.s1_recovered_counts,
        "s1_recovered_counts_cap150": res.s1_recovered_counts_cap150,
        "k4_diagnostics": dict(res.k4_diagnostics),
        "total_oversized_query_events": res.total_oversized_query_events,
        "missed_gt_edges": res.missed_gt_edges[:5000],  # sample up to 5k for failure analysis
    }

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(res_dict, f)

    sys.exit(0)


if __name__ == "__main__":
    main()
