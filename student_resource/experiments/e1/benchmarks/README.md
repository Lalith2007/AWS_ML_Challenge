# E1 1% Micro-Benchmark Documentation
**Amazon ML Challenge 2026 — Business Entity Resolution**

## Methodology
The locked architecture requires empirical calibration on representative sample sizes before trusting compute estimates.
A deterministic 1% sample was constructed by stratified sampling across `(country, multiplicity_bucket)` from the full Ground Truth, yielding **22,069 Source 1 entities** and **76,355 true Ground Truth edges**.

## Measured Components & Extrapolations

| Stage | Measured 1% Time (s) | Throughput (rows/s) | Peak RSS (MB) | Linear 100% Extrap. (s) | Caveats |
|---|---|---|---|---|---|
| `1_tsv_reading_parsing` | 0.0138 | 1,596,017.69 | 2241.72 | 1.38 | I/O bound; operating system page cache will accelerate subsequent passes. |
| `2_gt_parsing` | 0.0116 | 1,901,591.5 | 2241.72 | 1.16 | Pure string splitting; scales linearly with Ground Truth row count. |
| `3_validation_split` | 0.0212 | 1,042,131.45 | 2241.72 | 2.12 | O(N log N) sorting per stratum; negligible compute footprint. |
| `4_gt_expansion` | 0.0957 | 230,686.11 | 2241.72 | 9.57 | Memory scaling depends on tuple representation; generator streaming avoids memory spikes. |
| `5_f0_5_scoring` | 0.0367 | 600,654.58 | 2241.72 | 3.67 | Linear in entity count; per-entity diagnostics can be disabled for faster batch loops. |
| `6_recall_evaluation` | 0.0534 | 413,179.15 | 2241.72 | 5.34 | Candidate set size directly governs comparison cost; candidate sets should remain compact. |
| `7_validator_wrapper` | 0.0445 | 44.9 | 2241.72 | 12.0 | Subprocess invocation has fixed ~0.2s Python interpreter overhead; test set size scales memory if --check-ids enabled. |
| `8_diagnostic_serialization` | 0.1048 | 210,631.89 | 2241.72 | 10.48 | Pure I/O; JSON formatted with indentation is slower than streaming NDJSON or binary. |

## Linear Extrapolation Notes & Caveats
1. **Linear vs. Non-linear Scaling:** Simple string matching and F0.5 scoring scale strictly linearly O(N). However, future stages involving pairwise comparisons (L2 candidate pairs, L3 cross-features) will scale with candidate volume O(N * K), NOT linearly with raw data rows.
2. **Memory Overhead:** In-memory full GT structures cost ~600 MB. In future blocking stages, candidate pair generation must remain strictly streaming to prevent OOM.
3. **Disk I/O:** Initial cold reads from disk take longer than subsequent page-cached reads.
