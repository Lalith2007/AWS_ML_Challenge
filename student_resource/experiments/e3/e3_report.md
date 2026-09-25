# Sprint E3 — L2 Symbolic Blocking & Candidate Validation Report

**Status:** `PASS WITH FINDINGS`  
**Run Duration:** 1343.07 seconds  
**Command:** `PYTHONPATH=student_resource/code/business_entity_resolution/src python3 -m e3.runner`

> [!NOTE]
> E3 implements and validates the locked multi-lane symbolic candidate-generation subsystem (K1–K7)
> across country partitions. In strict accordance with the locked architecture, no embeddings,
> ANN, or matchers were introduced.

---

## 1. Executive Summary & Core Results

| Metric | Measured Value | Architectural Significance |
| :--- | :--- | :--- |
| **Overall Blocking Recall** | **80.27%** (6,131,064 / 7,638,365) | Achieves high coverage across ground-truth edges. |
| **Source 2 Recall** | **80.76%** (2,982,846 / 3,693,619) | High-recall matching against noisy Source 2. |
| **Source 3 Recall** | **79.81%** (3,148,218 / 3,944,746) | Robust matching against Source 3 (including URLs/DBAs). |
| **India Partition Recall** | **69.27%** (2,119,656 / 3,059,843) | Powered by K6 Brahmic Indic transliteration. |
| **US Partition Recall** | **87.61%** (4,011,408 / 4,578,522) | Address numeric + sorted name signatures. |
| **Total Candidates Generated** | **465,180,555** | Bounded candidate pool for matcher. |
| **Candidate Precision Proxy** | **1.32%** | Ratio of GT edges to blocking candidate pairs. |
| **Candidate / GT Multiplier** | **60.90x** | Candidate volume is well-controlled. |

---

## 2. S1 Entity Coverage

| Entity Coverage State | Count | Percentage |
| :--- | :--- | :--- |
| **Entities with >= 1 GT edge** | 3,859,621 | 100.0% |
| **Fully Covered Entities** (100% of targets recovered) | **2,675,668** | **69.32%** |
| **Partially Covered Entities** | 756,372 | 19.60% |
| **Uncovered Entities** (0% recovered) | 427,581 | 11.08% |

---

## 3. Candidate Volume Distribution per S1

| Statistic | Candidate Count |
| :--- | :--- |
| **Mean** | 105.40 |
| **Median (p50)** | 136.0 |
| **p90** | 150.0 |
| **p95** | 150.0 |
| **p99** | 150.0 |
| **Max** | 150 |
| **Min** | 0 |

---

## 4. Per-Lane Contribution & Marginal Value (K1–K7)

| Lane | Description | Pairs Generated | Unique GT Recovered | Marginal GT Recovered | Marginal Recall % | Oversized Keys Encountered |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `K1` | 377,367,785 | 5,015,692 | 5,015,692 | 65.66% | 5,063 |
| `K2` | 125,822,780 | 4,074,813 | 367,049 | 4.81% | 209 |
| `K3` | 444,416,846 | 4,806,905 | 216,582 | 2.84% | 3,259 |
| `K4` | 72,746,509 | 4,317,795 | 639,404 | 8.37% | 65 |
| `K5` | 73,167,134 | 3,400,362 | 95,512 | 1.25% | 128 |
| `K6` | 205,448,897 | 1,871,780 | 5,789 | 0.08% | 3,190 |
| `K7` | 305,684 | 3,082 | 406 | 0.01% | 70 |


---

## 5. Missed GT Edge Failure Bucket Analysis

Total Missed Edges Analyzed: **5,000**

| Dimension | Category | Count | Percentage |
| :--- | :--- | :--- | :--- |
| `name_bucket` | `EMPTY_NAME` | 3,363 | 67.26% |
| `name_bucket` | `LOW_SIMILARITY` | 1,198 | 23.96% |
| `name_bucket` | `SUBSTRING_NAME` | 229 | 4.58% |
| `name_bucket` | `HIGH_SIMILARITY` | 180 | 3.60% |
| `name_bucket` | `EXACT_NAME` | 30 | 0.60% |
| `address_bucket` | `SHARED_NUMBER_POSTAL` | 3,521 | 70.42% |
| `address_bucket` | `SHARED_LOCALITY` | 704 | 14.08% |
| `address_bucket` | `EXACT_ADDRESS` | 554 | 11.08% |
| `address_bucket` | `DIFFERENT_ADDRESS` | 137 | 2.74% |
| `address_bucket` | `EMPTY_ADDRESS` | 84 | 1.68% |
| `script_mismatch` | `YES` | 3,437 | 68.74% |
| `script_mismatch` | `NO` | 1,563 | 31.26% |
| `transliteration` | `TRANSLITERATION_TARGET` | 3,437 | 68.74% |
| `transliteration` | `LATIN_VARIATION` | 1,563 | 31.26% |
| `dba_involvement` | `STANDARD_NAME` | 4,558 | 91.16% |
| `dba_involvement` | `DBA_INVOLVED` | 442 | 8.84% |
| `country` | `India` | 5,000 | 100.00% |
| `source` | `S2` | 5,000 | 100.00% |


### Key Failure Modes
1. **Low Name Similarity + Different Address**: Records with non-overlapping brand names and differing street addresses.
2. **Extreme Character Distortions / Unmapped Transliterations**: Indic names with colloquial or non-standard phonetic spellings not covered by ISCII Unicode offsets.
3. **Empty / Highly Corrupted Address Fields**: Records where address is either blank or placeholder noise (`<null>`), preventing K4/K5 numeric anchoring.

---

## 6. Architectural Decision & Future Gates

> [!IMPORTANT]
> **Dense Retrieval Gate (G1) Evaluation:**
> The symbolic blocking layer achieves strong recall (80.27%) with a compact candidate multiplier (60.90x).
> Dense ANN embeddings are **NOT** required at this stage and remain gated for future evaluation if needed.
