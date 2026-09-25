"""
Runner script for E2 Full Ground-Truth Structural Property Audit.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
This runner coordinates the complete execution of the E2 audit over the full training
ground truth (~7.64M edges). It outputs empirical summaries, example files, reports,
and machine-readable manifests that mathematically justify or reject two key architectural
assumptions:
1. Hard country-partitioned blocking (K1-K9).
2. Exclusive target assignment post-processing (G4).
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from e1.io_utils import EXPERIMENTS_ROOT, TRAIN_FILES
from .cross_country import (
    save_cross_country_examples,
    save_cross_country_summary,
)
from .gt_property_scans import run_full_gt_property_audit
from .target_exclusivity import (
    save_target_exclusivity_summary,
    save_target_indegree_distribution,
    save_target_overlap_examples,
)

E2_EXPERIMENTS_DIR = EXPERIMENTS_ROOT / "e2"


def compute_file_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_lines(filepath: Path) -> int:
    """Count lines in text file."""
    cnt = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            cnt += 1
    return cnt


def run_e2_pipeline(
    s1_path: Path = TRAIN_FILES["S1"],
    s2_path: Path = TRAIN_FILES["S2"],
    s3_path: Path = TRAIN_FILES["S3"],
    gt_path: Path = TRAIN_FILES["GT"],
    output_dir: Path = E2_EXPERIMENTS_DIR,
) -> Dict[str, Any]:
    """Execute full E2 pipeline and write all certified artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd = "PYTHONPATH=student_resource/code/business_entity_resolution/src python3 -m e2.runner"

    print("=" * 70)
    print("STARTING E2 FULL GROUND-TRUTH STRUCTURAL PROPERTY AUDIT")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    t0 = time.time()
    audit_report = run_full_gt_property_audit(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        gt_path=gt_path,
    )
    duration = time.time() - t0
    audit_report.duration_seconds = duration

    # 1. Save CSV summary and TSV examples for Cross-Country
    cc_summary_path = output_dir / "e2_cross_country_summary.csv"
    save_cross_country_summary(audit_report.cross_country, cc_summary_path)

    cc_examples_path = output_dir / "e2_cross_country_examples.tsv"
    save_cross_country_examples(audit_report.cross_country, cc_examples_path)

    # 2. Save CSV summary, in-degree distribution, and TSV examples for Target Exclusivity
    te_summary_path = output_dir / "e2_target_exclusivity_summary.csv"
    save_target_exclusivity_summary(audit_report.target_exclusivity, te_summary_path)

    te_dist_path = output_dir / "e2_target_indegree_distribution.csv"
    save_target_indegree_distribution(audit_report.target_exclusivity, te_dist_path)

    te_examples_path = output_dir / "e2_target_overlap_examples.tsv"
    save_target_overlap_examples(audit_report.target_exclusivity, te_examples_path)

    # 3. Save JSON Report
    report_json_path = output_dir / "e2_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report.to_dict(), f, indent=2)

    # 4. Save Markdown Report
    report_md_path = output_dir / "e2_report.md"
    md_content = audit_report.generate_markdown_report(run_command=run_cmd)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 5. Build and Save Manifest
    artifact_files = [
        cc_summary_path,
        cc_examples_path,
        te_summary_path,
        te_dist_path,
        te_examples_path,
        report_json_path,
        report_md_path,
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
        "telemetry": audit_report.telemetry,
        "duration_seconds": duration,
        "integrity": audit_report.integrity.to_dict(),
        "cross_country": {
            "total_edges": audit_report.cross_country.total_edges,
            "same_country_edges": audit_report.cross_country.same_country_edges,
            "cross_country_edges": audit_report.cross_country.cross_country_edges,
            "same_country_pct": audit_report.cross_country.same_country_pct,
            "cross_country_pct": audit_report.cross_country.cross_country_pct,
            "decision": audit_report.country_partition_decision,
        },
        "target_exclusivity": {
            "total_gt_edges": audit_report.target_exclusivity.total_gt_edges,
            "total_distinct_targets": audit_report.target_exclusivity.total_distinct_targets,
            "total_duplicate_references": audit_report.target_exclusivity.total_duplicate_references,
            "s2_exclusivity_pct": audit_report.target_exclusivity.s2_stats.exclusivity_rate_pct if audit_report.target_exclusivity.s2_stats else 0.0,
            "s3_exclusivity_pct": audit_report.target_exclusivity.s3_stats.exclusivity_rate_pct if audit_report.target_exclusivity.s3_stats else 0.0,
            "decision": audit_report.g4_assignment_decision,
        },
        "architectural_decisions": {
            "country_partition": audit_report.country_partition_decision,
            "g4_assignment": audit_report.g4_assignment_decision,
        },
        "e1_documentation_correction": audit_report.e1_documentation_correction,
        "artifacts": manifest_artifacts,
        "e2_status": audit_report.e2_status,
    }

    manifest_path = output_dir / "e2_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Console Summary
    print("\n" + "=" * 70)
    print("E2 AUDIT COMPLETE")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total GT Edges: {audit_report.cross_country.total_edges:,}")
    print(f"Same-Country Edges: {audit_report.cross_country.same_country_edges:,} ({audit_report.cross_country.same_country_pct:.4f}%)")
    print(f"Cross-Country Edges: {audit_report.cross_country.cross_country_edges:,} ({audit_report.cross_country.cross_country_pct:.4f}%)")
    print(f"Country Partition Verdict: {audit_report.country_partition_decision}")
    print(f"Distinct Targets: {audit_report.target_exclusivity.total_distinct_targets:,}")
    print(f"Duplicate Target References: {audit_report.target_exclusivity.total_duplicate_references:,}")
    print(f"G4 Assignment Verdict: {audit_report.g4_assignment_decision}")
    print(f"E1 Documentation Correction: {audit_report.e1_documentation_correction}")
    print(f"E2 STATUS: {audit_report.e2_status}")
    print("=" * 70 + "\n")

    return manifest


if __name__ == "__main__":
    run_e2_pipeline()
