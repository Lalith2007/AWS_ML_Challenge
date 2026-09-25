"""
Candidate Generator and Partition Orchestrator for E3 Blocking.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
This module generates the official L2 candidate set (candidate_pairs.tsv)
by evaluating S1 records against S2 and S3 partitioned strictly by country.
It coordinates index building, per-lane candidate retrieval, deterministic deduplication,
safety capping, and streaming disk output without exceeding system RAM limits.
"""

import csv
import gc
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from e1.io_utils import TRAIN_FILES
from .blocking_lanes import BlockingConfig, generate_lane_keys
from .inverted_index import PartitionInvertedIndex


@dataclass
class PartitionRunResult:
    """Metrics and diagnostics for a single (country, source) partition."""
    country: str
    target_source: str
    total_s1_queried: int = 0
    total_target_records_indexed: int = 0
    total_candidate_pairs: int = 0
    total_gt_edges: int = 0
    recovered_gt_edges: int = 0

    # Coverage metrics
    entities_with_gt: int = 0
    entities_fully_covered: int = 0
    entities_partially_covered: int = 0
    entities_uncovered: int = 0

    # Per-lane metrics: lane -> {"pairs": int, "recovered": int, "marginal_recovered": int, "oversized_keys": int}
    lane_metrics: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Candidate count histogram: count -> frequency
    candidate_lengths: List[int] = field(default_factory=list)
    candidate_histogram: Counter = field(default_factory=Counter)

    # Oversized block events encountered during querying
    total_oversized_query_events: int = 0

    # Missed GT edge records for failure analysis: List of (s1_id, target_id)
    missed_gt_edges: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return (self.recovered_gt_edges / self.total_gt_edges) if self.total_gt_edges > 0 else 1.0


def stream_partition_targets(
    source_path: Path, country: str
) -> Iterator[Tuple[str, str, str]]:
    """Stream (entity_id, name, address) for records matching a specific country."""
    with open(source_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            rec_country = row[3].strip() if len(row) > 3 else "Unknown"
            if rec_country == country:
                eid = row[0].strip()
                name = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                yield eid, name, addr


def stream_partition_s1(
    s1_path: Path, country: str, allowed_s1_ids: Optional[Set[str]] = None
) -> Iterator[Tuple[str, str, str]]:
    """Stream S1 (entity_id, name, address) for records matching a specific country and allowed set."""
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            rec_country = row[3].strip() if len(row) > 3 else "Unknown"
            if rec_country == country:
                eid = row[0].strip()
                if allowed_s1_ids is not None and eid not in allowed_s1_ids:
                    continue
                name = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                yield eid, name, addr


def run_partition_blocking(
    country: str,
    target_source: str,
    target_path: Path,
    s1_path: Path,
    truth_by_s1: Dict[str, Set[str]],
    allowed_s1_ids: Optional[Set[str]] = None,
    candidate_file_handle: Optional[Any] = None,
    config: Optional[BlockingConfig] = None,
    collect_missed_edges: bool = True,
) -> PartitionRunResult:
    """
    Execute blocking on a single (country, target_source) partition:
    1. Build PartitionInvertedIndex on target table.
    2. Stream S1 entities in this country.
    3. Query index, evaluate per-lane and cumulative recall.
    4. Stream candidate pairs to file.
    5. Clean up memory.
    """
    if config is None:
        config = BlockingConfig()

    print(f"[{country} | {target_source}] Building inverted index...")
    index = PartitionInvertedIndex(country=country, source=target_source, config=config)

    for eid, name, addr in stream_partition_targets(target_path, country):
        index.add_record(eid, name, addr)

    audit_stats = index.freeze_and_audit()
    print(
        f"[{country} | {target_source}] Indexed {len(index.target_ids):,} records. "
        f"K1 oversized keys: {audit_stats['lane_stats'].get('K1', {}).get('oversized_keys_count', 0)}"
    )

    result = PartitionRunResult(
        country=country,
        target_source=target_source,
        total_target_records_indexed=len(index.target_ids),
    )

    # Initialize per-lane metric tracking
    for lane in config.enabled_lanes:
        result.lane_metrics[lane] = {
            "candidate_pairs": 0,
            "unique_gt_recovered": 0,
            "marginal_gt_recovered": 0,
            "oversized_keys": audit_stats["lane_stats"].get(lane, {}).get("oversized_keys_count", 0),
        }

    seen_marginal_gt: Set[Tuple[str, str]] = set()

    print(f"[{country} | {target_source}] Querying S1 records...")
    s1_count = 0
    buffer: List[str] = []

    for s1_id, s1_name, s1_addr in stream_partition_s1(s1_path, country, allowed_s1_ids):
        s1_count += 1
        s1_lane_keys = generate_lane_keys(
            entity_id=s1_id,
            name=s1_name,
            address=s1_addr,
            country=country,
            source="S1",
            config=config,
        )

        lane_cands, oversized_events = index.query(s1_lane_keys)
        result.total_oversized_query_events += oversized_events

        # Ground truth targets for this S1 in this specific target source
        all_true_targets = truth_by_s1.get(s1_id, set())
        prefix = f"{target_source}-"
        true_source_targets = {t for t in all_true_targets if t.startswith(prefix)}
        num_true = len(true_source_targets)

        if num_true > 0:
            result.entities_with_gt += 1
            result.total_gt_edges += num_true

        # Track per-lane candidates and multi-lane agreement
        cand_lane_counts: Counter[str] = Counter()

        for lane in config.enabled_lanes:
            cands = lane_cands.get(lane, set())
            result.lane_metrics[lane]["candidate_pairs"] += len(cands)
            for c in cands:
                cand_lane_counts[c] += 1

            if num_true > 0:
                rec_lane = cands & true_source_targets
                result.lane_metrics[lane]["unique_gt_recovered"] += len(rec_lane)

                # Marginal recall: recovered by this lane that wasn't previously seen in order
                for t in rec_lane:
                    pair = (s1_id, t)
                    if pair not in seen_marginal_gt:
                        seen_marginal_gt.add(pair)
                        result.lane_metrics[lane]["marginal_gt_recovered"] += 1

        # Multi-lane agreement ranking: prioritize candidates retrieved by multiple lanes, tie-break by ID
        sorted_union = sorted(
            cand_lane_counts.keys(),
            key=lambda cid: (-cand_lane_counts[cid], cid)
        )
        if len(sorted_union) > config.max_candidates_per_query:
            sorted_union = sorted_union[:config.max_candidates_per_query]

        num_cands = len(sorted_union)
        if len(result.candidate_lengths) < 10000:
            result.candidate_lengths.append(num_cands)
        result.candidate_histogram[num_cands] += 1
        result.total_candidate_pairs += num_cands

        # Evaluate cumulative recall for this S1
        if num_true > 0:
            recovered = set(sorted_union) & true_source_targets
            n_rec = len(recovered)
            result.recovered_gt_edges += n_rec

            if n_rec == num_true:
                result.entities_fully_covered += 1
            elif n_rec > 0:
                result.entities_partially_covered += 1
            else:
                result.entities_uncovered += 1

            if collect_missed_edges and len(result.missed_gt_edges) < 10000:
                missed = true_source_targets - set(sorted_union)
                for m_target in missed:
                    if len(result.missed_gt_edges) < 10000:
                        result.missed_gt_edges.append((s1_id, m_target))

        # Stream candidate pairs to disk
        if candidate_file_handle is not None:
            for cand_id in sorted_union:
                buffer.append(f"{s1_id}\t{cand_id}\t{target_source}\n")
                if len(buffer) >= 20000:
                    candidate_file_handle.writelines(buffer)
                    buffer.clear()

    # Flush remaining buffer
    if candidate_file_handle is not None and buffer:
        candidate_file_handle.writelines(buffer)
        buffer.clear()

    result.total_s1_queried = s1_count
    print(
        f"[{country} | {target_source}] Finished {s1_count:,} S1 queries. "
        f"Recall: {result.recovered_gt_edges:,}/{result.total_gt_edges:,} ({result.recall*100:.2f}%) | "
        f"Total Candidates: {result.total_candidate_pairs:,}"
    )

    del index
    gc.collect()

    return result
