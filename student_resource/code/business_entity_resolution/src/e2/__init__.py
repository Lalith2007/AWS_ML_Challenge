"""
E2: Full Ground-Truth Structural Property Audit.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Architecture Version 2.0 locks country-partitioned blocking (K1-K9) and anticipates
an assignment post-processor (G4). E2 performs the authoritative, full ground-truth
empirical audit to determine whether:
1. True matching edges ever cross country boundaries (validating hard country partitioning).
2. True target entities in S2 and S3 map exclusively to a single S1 parent (validating G4 admissibility).
"""

from .cross_country import CrossCountryScanResult, scan_cross_country_edges
from .gt_property_scans import (
    FullGTAuditReport,
    run_full_gt_property_audit,
    verify_gt_integrity_and_reconciliation,
)
from .target_exclusivity import (
    TargetExclusivityScanResult,
    scan_target_exclusivity,
)

__all__ = [
    "CrossCountryScanResult",
    "TargetExclusivityScanResult",
    "FullGTAuditReport",
    "scan_cross_country_edges",
    "scan_target_exclusivity",
    "verify_gt_integrity_and_reconciliation",
    "run_full_gt_property_audit",
]
