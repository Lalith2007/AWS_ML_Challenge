# E2 Report
**Amazon ML Challenge 2026 — Business Entity Resolution**

**Status:** **PASS WITH FINDINGS**

---

## 1. Execution
- **Exact Command:** `PYTHONPATH=student_resource/code/business_entity_resolution/src python3 -m e2.runner`
- **Duration:** 27.79 seconds
- **Environment:**
  - Python: `3.14.6`
  - Platform: `macOS-27.0-arm64-arm-64bit-Mach-O`
  - CPU Cores: `8`
  - Total System RAM: `8.0 GB`

---

## 2. GT Integrity
- **S1 Resolution:** PASS (100.0% resolved, 0 missing)
- **S2 Resolution:** PASS (100.0% resolved, 0 missing)
- **S3 Resolution:** PASS (100.0% resolved, 0 missing)
- **Edge Count Reconciliation:**
  - Total S2 GT Edges: `3,693,619` (Expected: `3,693,619`) — MATCH
  - Total S3 GT Edges: `3,944,746` (Expected: `3,944,746`) — MATCH
  - Total Combined GT Edges: `7,638,365` (Expected: `7,638,365`) — MATCH

---

## 3. Cross-Country Scan
- **Total GT Edges:** `7,638,365`
- **Same-Country Edges:** `7,638,365` (100.0000%)
- **Cross-Country Edges:** `0` (0.0000%)
- **Architectural Verdict:** `SAFE FOR HARD PARTITIONING ON TRAINING GT`

### Breakdown Table

| Slice Type | S1 Country | Target Source | Total Edges | Same Country | Cross Country | Same % | Cross % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OVERALL | ALL | ALL | 7,638,365 | 7,638,365 | 0 | 100.0000% | 0.0000% |
| BY_TARGET_SOURCE | ALL | S2 | 3,693,619 | 3,693,619 | 0 | 100.0000% | 0.0000% |
| BY_TARGET_SOURCE | ALL | S3 | 3,944,746 | 3,944,746 | 0 | 100.0000% | 0.0000% |
| BY_S1_COUNTRY | India | ALL | 3,059,843 | 3,059,843 | 0 | 100.0000% | 0.0000% |
| BY_S1_COUNTRY | US | ALL | 4,578,522 | 4,578,522 | 0 | 100.0000% | 0.0000% |
| BY_COUNTRY_X_SOURCE | India | S2 | 1,480,545 | 1,480,545 | 0 | 100.0000% | 0.0000% |
| BY_COUNTRY_X_SOURCE | India | S3 | 1,579,298 | 1,579,298 | 0 | 100.0000% | 0.0000% |
| BY_COUNTRY_X_SOURCE | US | S2 | 2,213,074 | 2,213,074 | 0 | 100.0000% | 0.0000% |
| BY_COUNTRY_X_SOURCE | US | S3 | 2,365,448 | 2,365,448 | 0 | 100.0000% | 0.0000% |

- **Cross-Country Examples:** None. Exactly zero cross-country edges exist in the training ground truth.

---

## 4. Target Exclusivity
- **Total Distinct Target IDs:** `7,638,365`
- **Total Duplicate Target References:** `0`
- **S2 Targets Referenced:** `3,693,619` | Single-Parent: `3,693,619` (100.0000%) | Multi-Parent: `0` | Max Parents: `1`
- **S3 Targets Referenced:** `3,944,746` | Single-Parent: `3,944,746` (100.0000%) | Multi-Parent: `0` | Max Parents: `1`
- **Architectural Verdict:** `LEGALLY PLAUSIBLE FOR EXPERIMENTATION`

### In-Degree Distribution

| Target Source | In-Degree (Parents) | Target Count | Percentage |
| --- | --- | --- | --- |
| S2 | 1 | 3,693,619 | 100.0000% |
| S3 | 1 | 3,944,746 | 100.0000% |
| OVERALL | 1 | 7,638,365 | 100.0000% |

- **Duplicate-Parent Examples:** None. Every single S2 and S3 target entity in ground truth maps to exactly one Source-1 entity.

---

## 5. Architectural Decisions
### Country Partition
**STATUS: SAFE FOR HARD PARTITIONING ON TRAINING GT**

> Empirical audit of all 7,638,365 training ground-truth match edges confirmed that 100.0000% of matches share identical country labels between Source 1 and Source 2/3. Partitioning L2 candidate generation strictly within each country creates 0.00% recall risk on training data while cutting candidate search space in half.

### G4 Assignment
**STATUS: LEGALLY PLAUSIBLE FOR EXPERIMENTATION**

> Empirical audit confirmed that all 3,693,619 S2 targets and all 3,944,746 S3 targets map to exactly one S1 entity (in-degree == 1). A 1-to-1 target exclusivity assumption is structurally consistent with the data-generating process. However, per locked architecture rules, G4 assignment is NOT automatically deployed; it is cleared as a legal candidate for future validation gating in E5.

---

## 6. E1 Documentation Correction
- **E1 DOCUMENTATION CORRECTION:** `precision weighting = 2x, not 4x`.
- Description: F0.5 formula is $F_{0.5} = (1 + 0.5^2) \frac{P \cdot R}{0.5^2 P + R}$. Because $\beta = 0.5$, $\frac{1}{\beta} = 2$, placing **2x** more weight on precision than recall. The E1 scorer implementation (`scorer.py`) was already correct; the documentation statement in `e1_report.md` and `src/e1/runner.py` was updated accordingly.

---

## 7. E2 Status
**E2 STATUS: PASS WITH FINDINGS**

All full ground-truth property scans completed successfully with 100.0% ID resolution, exact edge count reconciliation, zero cross-country edges, and zero multi-parent targets.

---

## 8. E3 Consumption
### What E3 Now Knows:
1. **Country Partitioning is Structurally Safe:** E3 blocking implementations (K1-K9) can safely enforce hard country partitions without incurring true-match recall loss on the training distribution.
2. **Target Mutual Exclusivity is Structurally Sound:** True targets are strictly 1-to-1 disjoint with respect to S1 entities in ground truth.
3. **Ground Truth Counts are Certified:** Exact training dataset size is locked at 2,206,821 S1 entities, 3,693,619 S2 edges, and 3,944,746 S3 edges.

### What Remains Unknown:
1. **Test Set Country Consistency:** While training ground truth exhibits 0 cross-country matches, test set properties cannot be observed directly.
2. **Blocking Recall Limits:** The upper-bound candidate recall achievable by union of rule-based blocking keys (K1-K9) within compute limits remains to be measured in E3.
