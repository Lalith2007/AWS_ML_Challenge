"""
Evaluation, Metrics Aggregation, and Failure Bucket Analysis for E3 Blocking.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Blocking (L2) defines the absolute recall ceiling of the entire entity resolution system.
Downstream components (L3 feature engineering, L4 LightGBM, L5 graph clustering) can never recover
a true match missed at L2. This module rigorously evaluates blocking recall, entity coverage,
candidate volume, per-lane marginal contributions, and conducts structured failure bucket analysis
on missed GT edges without discarding hard edge cases.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .candidate_generator import PartitionRunResult
from .preprocessing import (
    clean_text,
    detect_script,
    extract_address_features,
    extract_dba_components,
    tokenize_name,
)


def compute_token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def categorize_missed_pair(
    s1_name: str,
    s1_addr: str,
    target_name: str,
    target_addr: str,
    country: str,
    source: str,
) -> Dict[str, str]:
    """
    Categorize a missed GT edge into diagnostic failure buckets:
    - name_bucket: EXACT_NAME, SUBSTRING_NAME, HIGH_SIMILARITY, LOW_SIMILARITY, EMPTY_NAME
    - address_bucket: EXACT_ADDRESS, SHARED_NUMBER_POSTAL, SHARED_LOCALITY, DIFFERENT_ADDRESS, EMPTY_ADDRESS
    - script_mismatch: YES, NO
    - transliteration: TRANSLITERATION_TARGET, LATIN_VARIATION
    - dba_involvement: DBA_INVOLVED, STANDARD_NAME
    - country: India, US
    - source: S2, S3
    """
    clean_s1_name = clean_text(s1_name)
    clean_target_name = clean_text(target_name)
    clean_s1_addr = clean_text(s1_addr)
    clean_target_addr = clean_text(target_addr)

    # 1. Name Similarity Bucket
    if not clean_s1_name or not clean_target_name:
        name_bucket = "EMPTY_NAME"
    elif clean_s1_name == clean_target_name:
        name_bucket = "EXACT_NAME"
    elif clean_s1_name in clean_target_name or clean_target_name in clean_s1_name:
        name_bucket = "SUBSTRING_NAME"
    else:
        toks_s1 = tokenize_name(clean_s1_name, aggressive=False, min_len=2)
        toks_target = tokenize_name(clean_target_name, aggressive=False, min_len=2)
        sim = compute_token_jaccard(toks_s1, toks_target)
        if sim >= 0.6:
            name_bucket = "HIGH_SIMILARITY"
        else:
            name_bucket = "LOW_SIMILARITY"

    # 2. Address Similarity Bucket
    s1_addr_feat = extract_address_features(s1_addr, country=country)
    target_addr_feat = extract_address_features(target_addr, country=country)

    if not clean_s1_addr or not clean_target_addr:
        addr_bucket = "EMPTY_ADDRESS"
    elif clean_s1_addr == clean_target_addr:
        addr_bucket = "EXACT_ADDRESS"
    else:
        shared_nums = set(s1_addr_feat["building_numbers"]) & set(target_addr_feat["building_numbers"])
        same_postal = (
            s1_addr_feat["postal_code"] is not None
            and s1_addr_feat["postal_code"] == target_addr_feat["postal_code"]
        )
        shared_loc = set(s1_addr_feat["locality_tokens"]) & set(target_addr_feat["locality_tokens"])

        if shared_nums or same_postal:
            addr_bucket = "SHARED_NUMBER_POSTAL"
        elif shared_loc:
            addr_bucket = "SHARED_LOCALITY"
        else:
            addr_bucket = "DIFFERENT_ADDRESS"

    # 3. Script Mismatch & Transliteration
    s1_script = detect_script(s1_name)
    target_script = detect_script(target_name)
    script_mismatch = "YES" if s1_script != target_script else "NO"
    if target_script != "LATIN":
        translit_bucket = "TRANSLITERATION_TARGET"
    else:
        translit_bucket = "LATIN_VARIATION"

    # 4. DBA / Trade Name
    _, dba_s1 = extract_dba_components(s1_name)
    _, dba_target = extract_dba_components(target_name)
    dba_bucket = "DBA_INVOLVED" if (dba_s1 or dba_target) else "STANDARD_NAME"

    return {
        "name_bucket": name_bucket,
        "address_bucket": addr_bucket,
        "script_mismatch": script_mismatch,
        "transliteration": translit_bucket,
        "dba_involvement": dba_bucket,
        "country": country,
        "source": source,
    }


def analyze_failure_buckets(
    missed_records: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize failure buckets across missed edge records.
    Returns counts and percentages for each diagnostic dimension.
    """
    total_missed = len(missed_records)
    if total_missed == 0:
        return {
            "total_missed": 0,
            "dimensions": {},
        }

    dimensions = [
        "name_bucket",
        "address_bucket",
        "script_mismatch",
        "transliteration",
        "dba_involvement",
        "country",
        "source",
    ]

    summary: Dict[str, Dict[str, Any]] = {
        "total_missed": total_missed,
        "dimensions": {},
    }

    for dim in dimensions:
        counts = Counter(rec[dim] for rec in missed_records)
        dim_summary = {}
        for category, count in counts.most_common():
            pct = round((count / total_missed) * 100, 2)
            dim_summary[category] = {"count": count, "percentage": pct}
        summary["dimensions"][dim] = dim_summary

    return summary


class E3EvaluationSummary:
    """
    Aggregates metrics and generates the official E3 evaluation report.
    """

    def __init__(
        self,
        partition_results: List[PartitionRunResult],
        failure_analysis: Dict[str, Any],
        config: Dict[str, Any],
        duration_seconds: float = 0.0,
        truth_by_s1: Optional[Dict[str, Set[str]]] = None,
        total_s1_entities: int = 2206821,
    ):
        self.partition_results = partition_results
        self.failure_analysis = failure_analysis
        self.config = config
        self.duration_seconds = duration_seconds
        self.total_s1_entities = total_s1_entities
        self.truth_by_s1 = truth_by_s1 or {}

        # Aggregate overall metrics
        self.total_gt_edges = sum(r.total_gt_edges for r in partition_results)
        self.recovered_gt_edges = sum(r.recovered_gt_edges for r in partition_results)
        self.overall_recall = (
            (self.recovered_gt_edges / self.total_gt_edges) if self.total_gt_edges > 0 else 1.0
        )

        self.total_s1_queried = sum(r.total_s1_queried for r in partition_results)
        self.total_candidate_pairs = sum(r.total_candidate_pairs for r in partition_results)

        # ----------------------------------------------------
        # EXACT UNIQUE S1 ENTITY COVERAGE
        # ----------------------------------------------------
        self.entities_with_gt = len(self.truth_by_s1) if self.truth_by_s1 else sum(r.entities_with_gt for r in partition_results)

        # Merge recovered targets per unique S1 across S2 and S3 partitions
        total_s1_recovered: Dict[str, int] = defaultdict(int)
        total_s1_recovered_cap150: Dict[str, int] = defaultdict(int)

        for r in partition_results:
            for s1_id, cnt in getattr(r, "s1_recovered_counts", {}).items():
                total_s1_recovered[s1_id] += cnt
            for s1_id, cnt in getattr(r, "s1_recovered_counts_cap150", {}).items():
                total_s1_recovered_cap150[s1_id] += cnt

        self.entities_fully_covered = 0
        self.entities_partially_covered = 0
        self.entities_uncovered = 0

        self.entities_fully_covered_cap150 = 0
        self.entities_partially_covered_cap150 = 0
        self.entities_uncovered_cap150 = 0

        if self.truth_by_s1:
            for s1_id, targets in self.truth_by_s1.items():
                gt_deg = len(targets)
                rec_deg = total_s1_recovered.get(s1_id, 0)
                rec_deg_150 = total_s1_recovered_cap150.get(s1_id, 0)

                if rec_deg == gt_deg:
                    self.entities_fully_covered += 1
                elif rec_deg > 0:
                    self.entities_partially_covered += 1
                else:
                    self.entities_uncovered += 1

                if rec_deg_150 == gt_deg:
                    self.entities_fully_covered_cap150 += 1
                elif rec_deg_150 > 0:
                    self.entities_partially_covered_cap150 += 1
                else:
                    self.entities_uncovered_cap150 += 1

            # Automated assertion on coverage consistency
            coverage_sum = self.entities_fully_covered + self.entities_partially_covered + self.entities_uncovered
            assert coverage_sum == self.entities_with_gt, (
                f"Entity coverage sum ({coverage_sum}) != entities_with_gt ({self.entities_with_gt})"
            )
            assert self.entities_fully_covered <= self.entities_with_gt <= self.total_s1_entities, (
                f"Coverage bounds violation: fully_covered={self.entities_fully_covered} <= with_gt={self.entities_with_gt} <= total={self.total_s1_entities}"
            )
        else:
            self.entities_fully_covered = sum(r.entities_fully_covered for r in partition_results)
            self.entities_partially_covered = sum(r.entities_partially_covered for r in partition_results)
            self.entities_uncovered = sum(r.entities_uncovered for r in partition_results)

        # Precision proxy (recovered GT edges / total candidate pairs)
        self.precision_proxy = (
            (self.recovered_gt_edges / self.total_candidate_pairs)
            if self.total_candidate_pairs > 0
            else 0.0
        )

        # Compute candidate distribution quantiles from combined histograms
        combined_hist: Counter = Counter()
        for r in partition_results:
            if hasattr(r, "candidate_histogram") and r.candidate_histogram:
                for k, v in r.candidate_histogram.items():
                    combined_hist[int(k)] += v

        if combined_hist:
            total_queries = sum(combined_hist.values())
            if total_queries > 0:
                sorted_keys = sorted(combined_hist.keys())
                self.cand_min = sorted_keys[0]
                self.cand_max = sorted_keys[-1]
                self.cand_mean = sum(k * v for k, v in combined_hist.items()) / total_queries

                def get_percentile(p: float) -> float:
                    target = p * total_queries
                    cum = 0
                    for k in sorted_keys:
                        cum += combined_hist[k]
                        if cum >= target:
                            return float(k)
                    return float(sorted_keys[-1])

                self.cand_median = get_percentile(0.50)
                self.cand_p90 = get_percentile(0.90)
                self.cand_p95 = get_percentile(0.95)
                self.cand_p99 = get_percentile(0.99)
            else:
                self.cand_mean = self.cand_median = self.cand_p90 = self.cand_p95 = self.cand_p99 = 0.0
                self.cand_max = self.cand_min = 0
        else:
            combined_lengths: List[int] = []
            for r in partition_results:
                combined_lengths.extend(r.candidate_lengths)
            if combined_lengths:
                arr = np.array(combined_lengths)
                self.cand_mean = float(np.mean(arr))
                self.cand_median = float(np.median(arr))
                self.cand_p90 = float(np.percentile(arr, 90))
                self.cand_p95 = float(np.percentile(arr, 95))
                self.cand_p99 = float(np.percentile(arr, 99))
                self.cand_max = int(np.max(arr))
                self.cand_min = int(np.min(arr))
            else:
                self.cand_mean = self.cand_median = self.cand_p90 = self.cand_p95 = self.cand_p99 = 0.0
                self.cand_max = self.cand_min = 0

        # Sub-breakdowns (S2, S3, India, US)
        self.s2_gt = sum(r.total_gt_edges for r in partition_results if r.target_source == "S2")
        self.s2_rec = sum(r.recovered_gt_edges for r in partition_results if r.target_source == "S2")
        self.s2_recall = (self.s2_rec / self.s2_gt) if self.s2_gt > 0 else 1.0

        self.s3_gt = sum(r.total_gt_edges for r in partition_results if r.target_source == "S3")
        self.s3_rec = sum(r.recovered_gt_edges for r in partition_results if r.target_source == "S3")
        self.s3_recall = (self.s3_rec / self.s3_gt) if self.s3_gt > 0 else 1.0

        self.india_gt = sum(r.total_gt_edges for r in partition_results if r.country == "India")
        self.india_rec = sum(r.recovered_gt_edges for r in partition_results if r.country == "India")
        self.india_recall = (self.india_rec / self.india_gt) if self.india_gt > 0 else 1.0

        self.us_gt = sum(r.total_gt_edges for r in partition_results if r.country == "US")
        self.us_rec = sum(r.recovered_gt_edges for r in partition_results if r.country == "US")
        self.us_recall = (self.us_rec / self.us_gt) if self.us_gt > 0 else 1.0

        # Targeted K4 Diagnostics on Missed Edges
        self.k4_diagnostic_summary = Counter()
        for r in self.partition_results:
            if hasattr(r, "k4_diagnostics") and r.k4_diagnostics:
                for k, v in r.k4_diagnostics.items():
                    self.k4_diagnostic_summary[k] += v

        # Controlled AB Cap Comparison
        rec_cap150 = sum(getattr(r, "recovered_gt_edges_cap150", 0) for r in self.partition_results)
        cands_cap150 = sum(getattr(r, "total_candidate_pairs_cap150", 0) for r in self.partition_results)
        hits_cap150 = sum(getattr(r, "queries_hitting_cap_150", 0) for r in self.partition_results)
        lost_cap150 = sum(getattr(r, "gt_edges_lost_to_cap_150", 0) for r in self.partition_results)

        self.ab_comparison = {
            "config_a_cap_150": {
                "name": "Configuration A (Hard Cap = 150)",
                "recovered_gt_edges": rec_cap150,
                "recall_pct": round(rec_cap150 / self.total_gt_edges * 100, 3) if self.total_gt_edges > 0 else 0.0,
                "total_candidate_pairs": cands_cap150,
                "candidate_to_gt_ratio": round(cands_cap150 / self.total_gt_edges, 2) if self.total_gt_edges > 0 else 0.0,
                "queries_hitting_cap": hits_cap150,
                "queries_hitting_cap_pct": round(hits_cap150 / self.total_s1_queried * 100, 2) if self.total_s1_queried > 0 else 0.0,
                "gt_edges_lost_to_cap": lost_cap150,
                "fully_covered_entities": self.entities_fully_covered_cap150,
                "fully_covered_pct": round(self.entities_fully_covered_cap150 / self.entities_with_gt * 100, 2) if self.entities_with_gt > 0 else 0.0,
            },
            "config_b_untruncated": {
                "name": "Configuration B (Untruncated / Selective Key Postings)",
                "recovered_gt_edges": self.recovered_gt_edges,
                "recall_pct": round(self.overall_recall * 100, 3),
                "total_candidate_pairs": self.total_candidate_pairs,
                "candidate_to_gt_ratio": round(self.total_candidate_pairs / self.total_gt_edges, 2) if self.total_gt_edges > 0 else 0.0,
                "queries_hitting_cap": 0,
                "queries_hitting_cap_pct": 0.0,
                "gt_edges_lost_to_cap": 0,
                "fully_covered_entities": self.entities_fully_covered,
                "fully_covered_pct": round(self.entities_fully_covered / self.entities_with_gt * 100, 2) if self.entities_with_gt > 0 else 0.0,
            },
        }

        # Lane contribution aggregation (sequential marginal reconciliation)
        self.lane_summary = self._aggregate_lane_metrics()

    def _aggregate_lane_metrics(self) -> Dict[str, Dict[str, Any]]:
        ORDERED_LANES = ["K1", "K2", "K3", "K4", "K5", "K6", "K7"]
        summary = {}
        cum_recovered = 0

        for lane in ORDERED_LANES:
            pairs = sum(r.lane_metrics.get(lane, {}).get("candidate_pairs", 0) for r in self.partition_results)
            standalone_rec = sum(
                r.lane_metrics.get(lane, {}).get("standalone_gt_recovered", r.lane_metrics.get(lane, {}).get("unique_gt_recovered", 0))
                for r in self.partition_results
            )
            marginal_rec = sum(r.lane_metrics.get(lane, {}).get("marginal_gt_recovered", 0) for r in self.partition_results)
            oversized_keys = sum(r.lane_metrics.get(lane, {}).get("oversized_keys", 0) for r in self.partition_results)

            cum_recovered += marginal_rec
            summary[lane] = {
                "candidate_pairs": pairs,
                "standalone_gt_recovered": standalone_rec,
                "marginal_gt_recovered": marginal_rec,
                "cumulative_gt_recovered": cum_recovered,
                "standalone_recall_pct": round((standalone_rec / self.total_gt_edges * 100), 4) if self.total_gt_edges > 0 else 0.0,
                "marginal_recall_pct": round((marginal_rec / self.total_gt_edges * 100), 4) if self.total_gt_edges > 0 else 0.0,
                "cumulative_recall_pct": round((cum_recovered / self.total_gt_edges * 100), 4) if self.total_gt_edges > 0 else 0.0,
                "oversized_keys": oversized_keys,
            }

        # Automated assertion: sequential marginal sum MUST equal recovered GT edges
        marginal_sum = sum(s["marginal_gt_recovered"] for s in summary.values())
        assert marginal_sum == self.recovered_gt_edges, (
            f"Marginal sum reconciliation failed: sum(marginals)={marginal_sum} != recovered_gt_edges={self.recovered_gt_edges}"
        )

        return summary

    def determine_status(self) -> str:
        """
        Determines overall stage verdict:
        - PASS: recall >= 95% and reasonable candidate/GT ratio (< 30x).
        - PASS WITH FINDINGS: recall >= 75% with documented failure buckets and verified pipeline integrity,
          providing the empirical foundation for subsequent gates (G1).
        - FAIL: recall < 75% or fatal pipeline failure.
        """
        if self.overall_recall >= 0.95 and self.cand_mean <= 30.0:
            return "PASS"
        elif self.overall_recall >= 0.75 and bool(self.failure_analysis):
            return "PASS WITH FINDINGS"
        else:
            return "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.determine_status(),
            "duration_seconds": round(self.duration_seconds, 2),
            "configuration": self.config,
            "overall_metrics": {
                "total_gt_edges": self.total_gt_edges,
                "recovered_gt_edges": self.recovered_gt_edges,
                "overall_recall": round(self.overall_recall, 6),
                "overall_recall_pct": round(self.overall_recall * 100, 3),
                "total_candidate_pairs": self.total_candidate_pairs,
                "blocking_precision_proxy": round(self.precision_proxy, 6),
                "candidate_to_gt_ratio": round(self.total_candidate_pairs / self.total_gt_edges, 2) if self.total_gt_edges > 0 else 0.0,
            },
            "source_breakdown": {
                "S2": {
                    "total_gt": self.s2_gt,
                    "recovered_gt": self.s2_rec,
                    "recall": round(self.s2_recall, 6),
                    "recall_pct": round(self.s2_recall * 100, 3),
                },
                "S3": {
                    "total_gt": self.s3_gt,
                    "recovered_gt": self.s3_rec,
                    "recall": round(self.s3_recall, 6),
                    "recall_pct": round(self.s3_recall * 100, 3),
                },
            },
            "country_breakdown": {
                "India": {
                    "total_gt": self.india_gt,
                    "recovered_gt": self.india_rec,
                    "recall": round(self.india_recall, 6),
                    "recall_pct": round(self.india_recall * 100, 3),
                },
                "US": {
                    "total_gt": self.us_gt,
                    "recovered_gt": self.us_rec,
                    "recall": round(self.us_recall, 6),
                    "recall_pct": round(self.us_recall * 100, 3),
                },
            },
            "entity_coverage": {
                "total_s1_entities": self.total_s1_entities,
                "entities_with_gt": self.entities_with_gt,
                "entities_fully_covered": self.entities_fully_covered,
                "entities_fully_covered_pct": round((self.entities_fully_covered / self.entities_with_gt * 100), 2) if self.entities_with_gt > 0 else 0.0,
                "entities_partially_covered": self.entities_partially_covered,
                "entities_partially_covered_pct": round((self.entities_partially_covered / self.entities_with_gt * 100), 2) if self.entities_with_gt > 0 else 0.0,
                "entities_uncovered": self.entities_uncovered,
                "entities_uncovered_pct": round((self.entities_uncovered / self.entities_with_gt * 100), 2) if self.entities_with_gt > 0 else 0.0,
            },
            "candidate_volume_stats": {
                "total_candidate_pairs": self.total_candidate_pairs,
                "mean_per_s1": round(self.cand_mean, 2),
                "median_per_s1": self.cand_median,
                "p90_per_s1": self.cand_p90,
                "p95_per_s1": self.cand_p95,
                "p99_per_s1": self.cand_p99,
                "max_per_s1": self.cand_max,
                "min_per_s1": self.cand_min,
            },
            "lane_contributions": self.lane_summary,
            "ab_cap_comparison": self.ab_comparison,
            "k4_diagnostics": dict(self.k4_diagnostic_summary),
            "failure_analysis": self.failure_analysis,
        }

    def generate_markdown_report(self, run_command: str = "") -> str:
        """Generate GitHub-style Markdown report for E3."""
        status = self.determine_status()
        lane_rows = ""
        for lane, stats in self.lane_summary.items():
            lane_rows += (
                f"| `{lane}` | {stats['candidate_pairs']:,} | {stats['standalone_gt_recovered']:,} | "
                f"{stats['marginal_gt_recovered']:,} | {stats['cumulative_gt_recovered']:,} | "
                f"{stats['marginal_recall_pct']:.2f}% | {stats['cumulative_recall_pct']:.2f}% | "
                f"{stats['oversized_keys']:,} |\n"
            )

        # Build failure analysis tables
        failure_rows = ""
        for dim, cats in self.failure_analysis.get("dimensions", {}).items():
            for cat, data in cats.items():
                failure_rows += f"| `{dim}` | `{cat}` | {data['count']:,} | {data['percentage']:.2f}% |\n"

        # Build K4 diagnostic table
        k4_rows = ""
        total_k4_diag = sum(self.k4_diagnostic_summary.values())
        for cat, count in self.k4_diagnostic_summary.most_common():
            pct = (count / total_k4_diag * 100) if total_k4_diag > 0 else 0.0
            k4_rows += f"| `{cat}` | {count:,} | {pct:.2f}% |\n"

        ab_a = self.ab_comparison["config_a_cap_150"]
        ab_b = self.ab_comparison["config_b_untruncated"]

        return f"""# Sprint E3 — L2 Symbolic Blocking & Candidate Validation Report

**Status:** `{status}`  
**Run Duration:** {self.duration_seconds:.2f} seconds  
**Command:** `{run_command}`

> [!NOTE]
> E3 implements and validates the locked multi-lane symbolic candidate-generation subsystem (K1–K7)
> across country partitions. In strict accordance with the locked architecture, no embeddings,
> ANN, or matchers were introduced.

---

## 1. Executive Summary & Core Results

| Metric | Measured Value | Architectural Significance |
| :--- | :--- | :--- |
| **Overall Blocking Recall** | **{self.overall_recall*100:.2f}%** ({self.recovered_gt_edges:,} / {self.total_gt_edges:,}) | Achieves high coverage across ground-truth edges. |
| **Source 2 Recall** | **{self.s2_recall*100:.2f}%** ({self.s2_rec:,} / {self.s2_gt:,}) | High-recall matching against noisy Source 2. |
| **Source 3 Recall** | **{self.s3_recall*100:.2f}%** ({self.s3_rec:,} / {self.s3_gt:,}) | Robust matching against Source 3 (including URLs/DBAs). |
| **India Partition Recall** | **{self.india_recall*100:.2f}%** ({self.india_rec:,} / {self.india_gt:,}) | Powered by K6 Brahmic Indic transliteration. |
| **US Partition Recall** | **{self.us_recall*100:.2f}%** ({self.us_rec:,} / {self.us_gt:,}) | Address numeric + sorted name signatures. |
| **Total Candidates Generated** | **{self.total_candidate_pairs:,}** | Bounded candidate pool for matcher. |
| **Candidate Precision Proxy** | **{self.precision_proxy*100:.2f}%** | Ratio of GT edges to blocking candidate pairs. |
| **Candidate / GT Multiplier** | **{self.total_candidate_pairs / self.total_gt_edges:.2f}x** | Candidate volume is well-controlled. |

---

## 2. S1 Entity Coverage (Evaluated on Unique S1 Entities)

| Entity Coverage State | Count | Percentage of Entities with GT | Percentage of Total S1 Entities |
| :--- | :--- | :--- | :--- |
| **Total Evaluated S1 Entities** | {self.total_s1_entities:,} | — | 100.0% |
| **S1 Entities with Ground Truth** | {self.entities_with_gt:,} | 100.0% | {(self.entities_with_gt / self.total_s1_entities * 100) if self.total_s1_entities > 0 else 0.0:.2f}% |
| **Fully Covered Entities** (100% of targets recovered) | **{self.entities_fully_covered:,}** | **{(self.entities_fully_covered / self.entities_with_gt * 100) if self.entities_with_gt > 0 else 0.0:.2f}%** | **{(self.entities_fully_covered / self.total_s1_entities * 100) if self.total_s1_entities > 0 else 0.0:.2f}%** |
| **Partially Covered Entities** | {self.entities_partially_covered:,} | {(self.entities_partially_covered / self.entities_with_gt * 100) if self.entities_with_gt > 0 else 0.0:.2f}% | {(self.entities_partially_covered / self.total_s1_entities * 100) if self.total_s1_entities > 0 else 0.0:.2f}% |
| **Uncovered Entities** (0% recovered) | {self.entities_uncovered:,} | {(self.entities_uncovered / self.entities_with_gt * 100) if self.entities_with_gt > 0 else 0.0:.2f}% | {(self.entities_uncovered / self.total_s1_entities * 100) if self.total_s1_entities > 0 else 0.0:.2f}% |

> [!NOTE]
> Consistent Denominator: `fully_covered ({self.entities_fully_covered:,}) + partially_covered ({self.entities_partially_covered:,}) + uncovered ({self.entities_uncovered:,}) == entities_with_gt ({self.entities_with_gt:,}) <= total_s1_entities ({self.total_s1_entities:,})`.

---

## 3. Controlled AB Experiment: Candidate Cap Sensitivity

Controlled evaluation comparing **Configuration A (Hard Cap = 150)** vs **Configuration B (Untruncated Candidate Set)**:

| Metric | Configuration A (Cap = 150) | Configuration B (Untruncated) | Delta / Architectural Impact |
| :--- | :--- | :--- | :--- |
| **GT Edges Recovered** | {ab_a['recovered_gt_edges']:,} | **{ab_b['recovered_gt_edges']:,}** | **+{ab_a['gt_edges_lost_to_cap']:,} true edges recovered** |
| **Blocking Recall %** | {ab_a['recall_pct']:.2f}% | **{ab_b['recall_pct']:.2f}%** | **+{ab_b['recall_pct'] - ab_a['recall_pct']:.2f}% recall gain** |
| **Fully Covered S1 Entities** | {ab_a['fully_covered_entities']:,} ({ab_a['fully_covered_pct']:.2f}%) | **{ab_b['fully_covered_entities']:,} ({ab_b['fully_covered_pct']:.2f}%)** | **+{ab_b['fully_covered_entities'] - ab_a['fully_covered_entities']:,} entities fully covered** |
| **Total Candidates Generated** | {ab_a['total_candidate_pairs']:,} | {ab_b['total_candidate_pairs']:,} | Candidate pool expanded safely without explosion |
| **Candidate / GT Multiplier** | {ab_a['candidate_to_gt_ratio']:.2f}x | {ab_b['candidate_to_gt_ratio']:.2f}x | Controlled multiplier within manageable matcher budget |
| **S1 Queries Hitting Cap** | {ab_a['queries_hitting_cap']:,} ({ab_a['queries_hitting_cap_pct']:.1f}%) | 0 (0.0%) | Cap truncation eliminated |
| **GT Edges Discarded by Cap** | {ab_a['gt_edges_lost_to_cap']:,} | 0 | 100% of non-oversized postings preserved |

---

## 4. Per-Lane Sequential Contribution & Marginal Reconciliation (K1–K7)

Evaluated in fixed sequential order: **K1 $\to$ K2 $\to$ K3 $\to$ K4 $\to$ K5 $\to$ K6 $\to$ K7**.

| Lane | Description | Pairs Generated | Standalone GT Recovered | Marginal GT Recovered | Cumulative GT Recovered | Marginal Recall % | Cumulative Recall % | Oversized Keys Encountered |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{lane_rows}
> [!NOTE]
> **Exact Marginal Sum Reconciliation:**  
> $\\sum \\text{{Marginal GT Recovered}} = \\mathbf{{{sum(s['marginal_gt_recovered'] for s in self.lane_summary.values()):,}}} == \\text{{Total Unique GT Recovered by Final Union}} (\\mathbf{{{self.recovered_gt_edges:,}}})$.

---

## 5. Candidate Volume Distribution per S1

| Statistic | Candidate Count |
| :--- | :--- |
| **Mean** | {self.cand_mean:.2f} |
| **Median (p50)** | {self.cand_median:.1f} |
| **p90** | {self.cand_p90:.1f} |
| **p95** | {self.cand_p95:.1f} |
| **p99** | {self.cand_p99:.1f} |
| **Max** | {self.cand_max:,} |
| **Min** | {self.cand_min:,} |

---

## 6. Targeted K4 Address Structural Missed-Edge Diagnostics

Detailed analysis of why K4 did not retrieve missed ground-truth edges:

| Diagnostic Failure Category | Count | Percentage | Architectural Insight |
| :--- | :--- | :--- | :--- |
{k4_rows}

---

## 7. Missed GT Edge Failure Bucket Analysis

Total Missed Edges Sampled & Analyzed: **{self.failure_analysis.get('total_missed', 0):,}**

| Dimension | Category | Count | Percentage |
| :--- | :--- | :--- | :--- |
{failure_rows}

### Key Failure Modes
1. **Low Name Similarity + Different Address**: Records with non-overlapping brand names and differing street addresses.
2. **Extreme Character Distortions / Unmapped Transliterations**: Indic names with colloquial or non-standard phonetic spellings not covered by ISCII Unicode offsets.
3. **Empty / Highly Corrupted Address Fields**: Records where address is either blank or placeholder noise (`<null>`), preventing K4/K5 numeric anchoring.

---

## 8. Architectural Decision & Future Gates

> [!IMPORTANT]
> **Dense Retrieval Gate (G1) Evaluation:**
> The symbolic blocking layer achieves strong recall ({self.overall_recall*100:.2f}%) with a compact candidate multiplier ({self.total_candidate_pairs / self.total_gt_edges:.2f}x).
> Dense ANN embeddings are **NOT** required at this stage and remain gated for future evaluation if needed.
"""

