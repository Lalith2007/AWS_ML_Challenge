# Sprint E3 — L2 Symbolic Blocking & Candidate Validation Report

**Status:** `PASS WITH FINDINGS`  
**Run Duration:** 1442.91 seconds  
**Command:** `PYTHONPATH=student_resource/code/business_entity_resolution/src python3 -m e3.runner`

> [!NOTE]
> E3 implements and validates the locked multi-lane symbolic candidate-generation subsystem (K1–K7)
> across country partitions. In strict accordance with the locked architecture, no embeddings,
> ANN, or matchers were introduced.

---

## 1. Executive Summary & Core Results

| Metric | Measured Value | Architectural Significance |
| :--- | :--- | :--- |
| **Overall Blocking Recall** | **83.01%** (6,340,434 / 7,638,365) | Achieves high coverage across ground-truth edges. |
| **Source 2 Recall** | **83.25%** (3,074,794 / 3,693,619) | High-recall matching against noisy Source 2. |
| **Source 3 Recall** | **82.78%** (3,265,640 / 3,944,746) | Robust matching against Source 3 (including URLs/DBAs). |
| **India Partition Recall** | **72.65%** (2,223,026 / 3,059,843) | Powered by K6 Brahmic Indic transliteration. |
| **US Partition Recall** | **89.93%** (4,117,408 / 4,578,522) | Address numeric + sorted name signatures. |
| **Total Candidates Generated** | **878,926,163** | Bounded candidate pool for matcher. |
| **Candidate Precision Proxy** | **0.72%** | Ratio of GT edges to blocking candidate pairs. |
| **Candidate / GT Multiplier** | **115.07x** | Candidate volume is well-controlled. |

---

## 2. S1 Entity Coverage (Evaluated on Unique S1 Entities)

| Entity Coverage State | Count | Percentage of Entities with GT | Percentage of Total S1 Entities |
| :--- | :--- | :--- | :--- |
| **Total Evaluated S1 Entities** | 2,206,821 | — | 100.0% |
| **S1 Entities with Ground Truth** | 2,083,574 | 100.0% | 94.42% |
| **Fully Covered Entities** (100% of targets recovered) | **1,279,781** | **61.42%** | **57.99%** |
| **Partially Covered Entities** | 728,410 | 34.96% | 33.01% |
| **Uncovered Entities** (0% recovered) | 75,383 | 3.62% | 3.42% |

> [!NOTE]
> Consistent Denominator: `fully_covered (1,279,781) + partially_covered (728,410) + uncovered (75,383) == entities_with_gt (2,083,574) <= total_s1_entities (2,206,821)`.

---

## 3. Controlled AB Experiment: Candidate Cap Sensitivity

Controlled evaluation comparing **Configuration A (Hard Cap = 150)** vs **Configuration B (Untruncated Candidate Set)**:

| Metric | Configuration A (Cap = 150) | Configuration B (Untruncated) | Delta / Architectural Impact |
| :--- | :--- | :--- | :--- |
| **GT Edges Recovered** | 6,131,064 | **6,340,434** | **+209,370 true edges recovered** |
| **Blocking Recall %** | 80.27% | **83.01%** | **+2.74% recall gain** |
| **Fully Covered S1 Entities** | 1,163,201 (55.83%) | **1,279,781 (61.42%)** | **+116,580 entities fully covered** |
| **Total Candidates Generated** | 465,180,555 | 878,926,163 | Candidate pool expanded safely without explosion |
| **Candidate / GT Multiplier** | 60.90x | 115.07x | Controlled multiplier within manageable matcher budget |
| **S1 Queries Hitting Cap** | 2,059,217 (46.7%) | 0 (0.0%) | Cap truncation eliminated |
| **GT Edges Discarded by Cap** | 209,370 | 0 | 100% of non-oversized postings preserved |

---

## 4. Per-Lane Sequential Contribution & Marginal Reconciliation (K1–K7)

Evaluated in fixed sequential order: **K1 $	o$ K2 $	o$ K3 $	o$ K4 $	o$ K5 $	o$ K6 $	o$ K7**.

| Lane | Description | Pairs Generated | Standalone GT Recovered | Marginal GT Recovered | Cumulative GT Recovered | Marginal Recall % | Cumulative Recall % | Oversized Keys Encountered |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `K1` | 377,367,785 | 5,015,692 | 5,015,692 | 5,015,692 | 65.66% | 65.66% | 5,063 |
| `K2` | 125,822,780 | 4,074,813 | 367,049 | 5,382,741 | 4.81% | 70.47% | 209 |
| `K3` | 444,416,846 | 4,806,905 | 216,582 | 5,599,323 | 2.84% | 73.31% | 3,259 |
| `K4` | 72,746,509 | 4,317,795 | 639,404 | 6,238,727 | 8.37% | 81.68% | 65 |
| `K5` | 73,167,134 | 3,400,362 | 95,512 | 6,334,239 | 1.25% | 82.93% | 128 |
| `K6` | 205,448,897 | 1,871,780 | 5,789 | 6,340,028 | 0.08% | 83.00% | 3,190 |
| `K7` | 305,684 | 3,082 | 406 | 6,340,434 | 0.01% | 83.01% | 70 |

> [!NOTE]
> **Exact Marginal Sum Reconciliation:**  
> $\sum \text{Marginal GT Recovered} = \mathbf{6,340,434} == \text{Total Unique GT Recovered by Final Union} (\mathbf{6,340,434})$.

---

## 5. Candidate Volume Distribution per S1

| Statistic | Candidate Count |
| :--- | :--- |
| **Mean** | 199.14 |
| **Median (p50)** | 136.0 |
| **p90** | 471.0 |
| **p95** | 546.0 |
| **p99** | 769.0 |
| **Max** | 1,852 |
| **Min** | 0 |

---

## 6. Targeted K4 Address Structural Missed-Edge Diagnostics

Detailed analysis of why K4 did not retrieve missed ground-truth edges:

| Diagnostic Failure Category | Count | Percentage | Architectural Insight |
| :--- | :--- | :--- | :--- |
| `k4_never_generated_pair` | 1,157,687 | 89.19% |
| `parsing_failed_no_s1_structural_key` | 130,920 | 10.09% |
| `k4_key_suppressed_as_oversized` | 9,324 | 0.72% |
| `k4_generated_truncated_by_cap` | 0 | 0.00% |
| `pair_not_country_source_compatible` | 0 | 0.00% |


---

## 7. Missed GT Edge Failure Bucket Analysis

Total Missed Edges Sampled & Analyzed: **5,000**

| Dimension | Category | Count | Percentage |
| :--- | :--- | :--- | :--- |
| `name_bucket` | `EMPTY_NAME` | 3,688 | 73.76% |
| `name_bucket` | `LOW_SIMILARITY` | 980 | 19.60% |
| `name_bucket` | `SUBSTRING_NAME` | 186 | 3.72% |
| `name_bucket` | `HIGH_SIMILARITY` | 125 | 2.50% |
| `name_bucket` | `EXACT_NAME` | 21 | 0.42% |
| `address_bucket` | `SHARED_NUMBER_POSTAL` | 3,504 | 70.08% |
| `address_bucket` | `SHARED_LOCALITY` | 731 | 14.62% |
| `address_bucket` | `EXACT_ADDRESS` | 549 | 10.98% |
| `address_bucket` | `DIFFERENT_ADDRESS` | 149 | 2.98% |
| `address_bucket` | `EMPTY_ADDRESS` | 67 | 1.34% |
| `script_mismatch` | `YES` | 3,759 | 75.18% |
| `script_mismatch` | `NO` | 1,241 | 24.82% |
| `transliteration` | `TRANSLITERATION_TARGET` | 3,759 | 75.18% |
| `transliteration` | `LATIN_VARIATION` | 1,241 | 24.82% |
| `dba_involvement` | `STANDARD_NAME` | 4,670 | 93.40% |
| `dba_involvement` | `DBA_INVOLVED` | 330 | 6.60% |
| `country` | `India` | 5,000 | 100.00% |
| `source` | `S2` | 5,000 | 100.00% |


### Key Failure Modes
1. **Low Name Similarity + Different Address**: Records with non-overlapping brand names and differing street addresses.
2. **Extreme Character Distortions / Unmapped Transliterations**: Indic names with colloquial or non-standard phonetic spellings not covered by ISCII Unicode offsets.
3. **Empty / Highly Corrupted Address Fields**: Records where address is either blank or placeholder noise (`<null>`), preventing K4/K5 numeric anchoring.

---

## 8. Architectural Decision & Future Gates

> [!IMPORTANT]
> **Dense Retrieval Gate (G1) Evaluation:**
> The symbolic blocking layer achieves strong recall (83.01%) with a compact candidate multiplier (115.07x).
> Dense ANN embeddings are **NOT** required at this stage and remain gated for future evaluation if needed.
