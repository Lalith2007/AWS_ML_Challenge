"""
Full S2/S3 Target Exclusivity Scan for E2.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
The target exclusivity scan determines whether a later assignment post-processor
(G4) is structurally admissible. If true target entities in S2 and S3 map to at
most one Source-1 entity (in-degree == 1), the ground truth represents a strict
partitioning problem where target entities are mutually exclusive. Under this condition,
1-to-1 matching and competitive assignment post-processing (e.g., Jonker-Volgenant or greedy
bipartite matching) are structurally aligned with the true data-generating process.
If targets frequently map to multiple S1 parents, enforcing 1-to-1 exclusivity would
artificially cap recall and G4 assignment must not be enabled.
"""

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TargetSourceExclusivityStats:
    """Statistics for a single target source (S2 or S3)."""
    target_source: str
    total_edges: int = 0
    distinct_targets: int = 0
    single_parent_targets: int = 0
    multi_parent_targets: int = 0
    max_parent_count: int = 0
    duplicate_references: int = 0
    indegree_distribution: Dict[int, int] = field(default_factory=dict)

    @property
    def exclusivity_rate_pct(self) -> float:
        if self.distinct_targets == 0:
            return 0.0
        return round((self.single_parent_targets / self.distinct_targets) * 100.0, 6)

    @property
    def is_perfectly_exclusive(self) -> bool:
        return self.multi_parent_targets == 0 and self.distinct_targets > 0

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "target_source": self.target_source,
            "total_edges": self.total_edges,
            "distinct_targets": self.distinct_targets,
            "single_parent_targets": self.single_parent_targets,
            "multi_parent_targets": self.multi_parent_targets,
            "max_parent_count": self.max_parent_count,
            "duplicate_references": self.duplicate_references,
            "exclusivity_rate_pct": self.exclusivity_rate_pct,
        }


@dataclass
class TargetExclusivityScanResult:
    """Complete aggregated results of the full target exclusivity scan."""
    total_gt_edges: int = 0
    total_distinct_targets: int = 0
    total_duplicate_references: int = 0

    s2_stats: Optional[TargetSourceExclusivityStats] = None
    s3_stats: Optional[TargetSourceExclusivityStats] = None

    combined_indegree_distribution: Dict[int, int] = field(default_factory=dict)
    multi_parent_examples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_g4_structurally_cleared(self) -> bool:
        """Determines if G4 assignment is legally plausible for experimentation."""
        s2_ok = self.s2_stats is not None and self.s2_stats.is_perfectly_exclusive
        s3_ok = self.s3_stats is not None and self.s3_stats.is_perfectly_exclusive
        return s2_ok and s3_ok

    @property
    def architectural_status(self) -> str:
        if self.is_g4_structurally_cleared:
            return "LEGALLY PLAUSIBLE FOR EXPERIMENTATION"
        else:
            return "DO NOT ENABLE"

    def get_summary_records(self) -> List[Dict[str, Any]]:
        """Return summary records for CSV export."""
        records: List[Dict[str, Any]] = []
        if self.s2_stats:
            records.append(self.s2_stats.to_summary_dict())
        if self.s3_stats:
            records.append(self.s3_stats.to_summary_dict())

        # Overall row
        single_all = (
            (self.s2_stats.single_parent_targets if self.s2_stats else 0)
            + (self.s3_stats.single_parent_targets if self.s3_stats else 0)
        )
        multi_all = (
            (self.s2_stats.multi_parent_targets if self.s2_stats else 0)
            + (self.s3_stats.multi_parent_targets if self.s3_stats else 0)
        )
        max_parents_all = max(
            (self.s2_stats.max_parent_count if self.s2_stats else 0),
            (self.s3_stats.max_parent_count if self.s3_stats else 0),
        )
        overall_exclusivity = (
            round((single_all / self.total_distinct_targets) * 100.0, 6)
            if self.total_distinct_targets > 0
            else 0.0
        )

        records.append({
            "target_source": "OVERALL",
            "total_edges": self.total_gt_edges,
            "distinct_targets": self.total_distinct_targets,
            "single_parent_targets": single_all,
            "multi_parent_targets": multi_all,
            "max_parent_count": max_parents_all,
            "duplicate_references": self.total_duplicate_references,
            "exclusivity_rate_pct": overall_exclusivity,
        })
        return records

    def get_indegree_distribution_records(self) -> List[Dict[str, Any]]:
        """Return distribution records for CSV export."""
        records: List[Dict[str, Any]] = []

        if self.s2_stats:
            for deg in sorted(self.s2_stats.indegree_distribution.keys()):
                cnt = self.s2_stats.indegree_distribution[deg]
                pct = round((cnt / self.s2_stats.distinct_targets) * 100.0, 6) if self.s2_stats.distinct_targets > 0 else 0.0
                records.append({
                    "target_source": "S2",
                    "in_degree": deg,
                    "target_count": cnt,
                    "percentage": pct,
                })

        if self.s3_stats:
            for deg in sorted(self.s3_stats.indegree_distribution.keys()):
                cnt = self.s3_stats.indegree_distribution[deg]
                pct = round((cnt / self.s3_stats.distinct_targets) * 100.0, 6) if self.s3_stats.distinct_targets > 0 else 0.0
                records.append({
                    "target_source": "S3",
                    "in_degree": deg,
                    "target_count": cnt,
                    "percentage": pct,
                })

        for deg in sorted(self.combined_indegree_distribution.keys()):
            cnt = self.combined_indegree_distribution[deg]
            pct = round((cnt / self.total_distinct_targets) * 100.0, 6) if self.total_distinct_targets > 0 else 0.0
            records.append({
                "target_source": "OVERALL",
                "in_degree": deg,
                "target_count": cnt,
                "percentage": pct,
            })

        return records


def scan_target_exclusivity(
    gt_path: Path,
    s1_country_map: Optional[Dict[str, str]] = None,
    max_examples: int = 100,
) -> TargetExclusivityScanResult:
    """
    Perform an authoritative full ground-truth target exclusivity scan.

    Streams the full training ground truth line-by-line:
    - Tracks first parent S1 ID for each target.
    - If a target is seen again under another S1 ID, records duplicate in multi_parents dict.
    - Captures in-degree distribution and detailed multi-parent examples.
    """
    s2_first_parent: Dict[str, str] = {}
    s2_multi_parents: Dict[str, List[str]] = {}
    total_s2_edges = 0

    s3_first_parent: Dict[str, str] = {}
    s3_multi_parents: Dict[str, List[str]] = {}
    total_s3_edges = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Skip header
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            raw_targets = row[1].strip().split(",") if len(row) > 1 and row[1].strip() else []

            for t in raw_targets:
                t = t.strip()
                if not t:
                    continue
                if t.startswith("S2-"):
                    total_s2_edges += 1
                    if t not in s2_first_parent:
                        s2_first_parent[t] = s1_id
                    else:
                        if t not in s2_multi_parents:
                            s2_multi_parents[t] = [s2_first_parent[t], s1_id]
                        else:
                            s2_multi_parents[t].append(s1_id)
                elif t.startswith("S3-"):
                    total_s3_edges += 1
                    if t not in s3_first_parent:
                        s3_first_parent[t] = s1_id
                    else:
                        if t not in s3_multi_parents:
                            s3_multi_parents[t] = [s3_first_parent[t], s1_id]
                        else:
                            s3_multi_parents[t].append(s1_id)

    # Compute S2 Statistics
    distinct_s2 = len(s2_first_parent)
    multi_s2_cnt = len(s2_multi_parents)
    single_s2_cnt = distinct_s2 - multi_s2_cnt
    dup_s2_refs = total_s2_edges - distinct_s2

    s2_indegree_dist: Counter = Counter()
    s2_indegree_dist[1] = single_s2_cnt
    for parents in s2_multi_parents.values():
        s2_indegree_dist[len(parents)] += 1

    max_s2_parents = 1 if multi_s2_cnt == 0 else max(len(p) for p in s2_multi_parents.values())

    s2_stats = TargetSourceExclusivityStats(
        target_source="S2",
        total_edges=total_s2_edges,
        distinct_targets=distinct_s2,
        single_parent_targets=single_s2_cnt,
        multi_parent_targets=multi_s2_cnt,
        max_parent_count=max_s2_parents,
        duplicate_references=dup_s2_refs,
        indegree_distribution=dict(s2_indegree_dist),
    )

    # Compute S3 Statistics
    distinct_s3 = len(s3_first_parent)
    multi_s3_cnt = len(s3_multi_parents)
    single_s3_cnt = distinct_s3 - multi_s3_cnt
    dup_s3_refs = total_s3_edges - distinct_s3

    s3_indegree_dist: Counter = Counter()
    s3_indegree_dist[1] = single_s3_cnt
    for parents in s3_multi_parents.values():
        s3_indegree_dist[len(parents)] += 1

    max_s3_parents = 1 if multi_s3_cnt == 0 else max(len(p) for p in s3_multi_parents.values())

    s3_stats = TargetSourceExclusivityStats(
        target_source="S3",
        total_edges=total_s3_edges,
        distinct_targets=distinct_s3,
        single_parent_targets=single_s3_cnt,
        multi_parent_targets=multi_s3_cnt,
        max_parent_count=max_s3_parents,
        duplicate_references=dup_s3_refs,
        indegree_distribution=dict(s3_indegree_dist),
    )

    # Combined metrics
    total_gt_edges = total_s2_edges + total_s3_edges
    total_distinct_targets = distinct_s2 + distinct_s3
    total_duplicates = dup_s2_refs + dup_s3_refs

    combined_indegree: Counter = Counter()
    for deg, cnt in s2_indegree_dist.items():
        combined_indegree[deg] += cnt
    for deg, cnt in s3_indegree_dist.items():
        combined_indegree[deg] += cnt

    # Format multi-parent examples
    multi_examples: List[Dict[str, Any]] = []
    # Collect from S2
    for target_id, parents in list(s2_multi_parents.items())[:max_examples]:
        parent_countries = (
            [s1_country_map.get(p, "Unknown") for p in parents]
            if s1_country_map
            else ["Unknown"] * len(parents)
        )
        multi_examples.append({
            "target_entity_id": target_id,
            "target_source": "S2",
            "num_parents": len(parents),
            "parent_s1_ids": ",".join(parents),
            "parent_countries": ",".join(parent_countries),
        })
    # Collect from S3
    remaining_slots = max_examples - len(multi_examples)
    if remaining_slots > 0:
        for target_id, parents in list(s3_multi_parents.items())[:remaining_slots]:
            parent_countries = (
                [s1_country_map.get(p, "Unknown") for p in parents]
                if s1_country_map
                else ["Unknown"] * len(parents)
            )
            multi_examples.append({
                "target_entity_id": target_id,
                "target_source": "S3",
                "num_parents": len(parents),
                "parent_s1_ids": ",".join(parents),
                "parent_countries": ",".join(parent_countries),
            })

    return TargetExclusivityScanResult(
        total_gt_edges=total_gt_edges,
        total_distinct_targets=total_distinct_targets,
        total_duplicate_references=total_duplicates,
        s2_stats=s2_stats,
        s3_stats=s3_stats,
        combined_indegree_distribution=dict(combined_indegree),
        multi_parent_examples=multi_examples,
    )


def save_target_exclusivity_summary(result: TargetExclusivityScanResult, output_path: Path) -> None:
    """Save target exclusivity summary CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = result.get_summary_records()
    fieldnames = [
        "target_source",
        "total_edges",
        "distinct_targets",
        "single_parent_targets",
        "multi_parent_targets",
        "max_parent_count",
        "duplicate_references",
        "exclusivity_rate_pct",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_target_indegree_distribution(result: TargetExclusivityScanResult, output_path: Path) -> None:
    """Save target in-degree distribution CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = result.get_indegree_distribution_records()
    fieldnames = ["target_source", "in_degree", "target_count", "percentage"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_target_overlap_examples(result: TargetExclusivityScanResult, output_path: Path) -> None:
    """Save multi-parent target examples TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_entity_id",
        "target_source",
        "num_parents",
        "parent_s1_ids",
        "parent_countries",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(result.multi_parent_examples)
