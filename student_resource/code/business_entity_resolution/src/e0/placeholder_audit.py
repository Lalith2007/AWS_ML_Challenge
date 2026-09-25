"""
Placeholder and NULL Audit for E0.
Analyzes business_address fields across all train and test source files.
Amazon ML Challenge 2026.
"""

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .common import (
    ALL_SOURCES,
    E0_RESULTS_DIR,
    PANDAS_DEFAULT_NA_VALUES,
    PLACEHOLDER_SET,
    PLACEHOLDER_TOKENS,
)

PUNCT_STRIP_CHARS = " ,.<>()[]{}!?:;\"'\t\r\n"


def audit_single_source_file(
    file_path: Path,
    dataset_split: str,
    source_name: str,
    chunk_size: int = 250_000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Audits a single TSV source file at both field-level and component-level.
    Streams line-by-line / row-by-row to maintain a tiny memory footprint.
    """
    total_rows = 0
    pandas_nan_count = 0
    empty_string_count = 0
    whitespace_only_count = 0
    whole_field_placeholder_count = 0
    effectively_missing_count = 0

    rows_with_component_placeholder_cleaned = 0
    rows_with_component_placeholder_raw_ws = 0
    rows_clean_nonempty = 0

    # Token-level accumulators
    whole_field_token_counts = Counter()
    comp_cleaned_token_counts = Counter()
    comp_raw_ws_token_counts = Counter()
    comp_cleaned_row_counts = Counter()
    comp_raw_ws_row_counts = Counter()

    # Casing and punctuation context per token
    token_casing = defaultdict(lambda: Counter())  # token -> {"upper": int, "lower": int, ...}
    token_punct = defaultdict(lambda: Counter())   # token -> {"punct_surrounded": int, "ws_isolated": int}

    # Open file with standard CSV reader over TSV
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty TSV file: {file_path}")

        # business_address is column index 2
        for row in reader:
            total_rows += 1

            if len(row) <= 2:
                # Column completely absent
                pandas_nan_count += 1
                empty_string_count += 1
                effectively_missing_count += 1
                continue

            raw_addr = row[2]

            # 1. Check exact empty string
            if raw_addr == "":
                empty_string_count += 1
                pandas_nan_count += 1
                effectively_missing_count += 1
                continue

            # 2. Check whitespace-only
            stripped = raw_addr.strip()
            if not stripped:
                whitespace_only_count += 1
                # pandas keeps whitespace-only as string by default, not NaN
                effectively_missing_count += 1
                continue

            stripped_low = stripped.lower()

            # Check if whole field is an exact placeholder or enclosed placeholder like <NULL>
            is_whole_field = False
            matched_whole_token = None

            if stripped_low in PLACEHOLDER_SET:
                is_whole_field = True
                matched_whole_token = stripped_low
            elif (
                (stripped.startswith("<") and stripped.endswith(">"))
                or (stripped.startswith("[") and stripped.endswith("]"))
                or (stripped.startswith("(") and stripped.endswith(")"))
            ):
                inner = stripped[1:-1].strip().lower()
                if inner in PLACEHOLDER_SET:
                    is_whole_field = True
                    matched_whole_token = inner

            # Check pandas NaN behavior with default keep_default_na=True
            if stripped_low in PANDAS_DEFAULT_NA_VALUES:
                pandas_nan_count += 1

            if is_whole_field:
                whole_field_placeholder_count += 1
                effectively_missing_count += 1
                whole_field_token_counts[matched_whole_token] += 1
                continue

            # 3. Address is non-empty and not a whole-field placeholder
            ws_tokens = stripped.split()

            row_cleaned_tokens = set()
            row_raw_ws_tokens = set()

            for t in ws_tokens:
                t_low = t.lower()
                # Raw whitespace token match (Cell 11 reproduction)
                if t_low in PLACEHOLDER_SET:
                    comp_raw_ws_token_counts[t_low] += 1
                    row_raw_ws_tokens.add(t_low)

                # Cleaned token match (stripping surrounding punctuation like commas, quotes, brackets)
                cl = t.strip(PUNCT_STRIP_CHARS)
                cl_low = cl.lower()

                if cl_low in PLACEHOLDER_SET:
                    comp_cleaned_token_counts[cl_low] += 1
                    row_cleaned_tokens.add(cl_low)

                    # Track casing of the token
                    if cl.isupper():
                        token_casing[cl_low]["upper"] += 1
                    elif cl.islower():
                        token_casing[cl_low]["lower"] += 1
                    elif cl.istitle():
                        token_casing[cl_low]["title"] += 1
                    else:
                        token_casing[cl_low]["other"] += 1

                    # Track punctuation context
                    if t == cl:
                        token_punct[cl_low]["ws_isolated"] += 1
                    else:
                        token_punct[cl_low]["punct_surrounded"] += 1

            for tok in row_raw_ws_tokens:
                comp_raw_ws_row_counts[tok] += 1

            for tok in row_cleaned_tokens:
                comp_cleaned_row_counts[tok] += 1

            if row_cleaned_tokens:
                rows_with_component_placeholder_cleaned += 1
            else:
                rows_clean_nonempty += 1

            if row_raw_ws_tokens:
                rows_with_component_placeholder_raw_ws += 1

    nonempty_count = total_rows - effectively_missing_count

    summary_record = {
        "dataset": dataset_split,
        "source": source_name,
        "file_name": file_path.name,
        "total_rows": total_rows,
        "pandas_nan_count": pandas_nan_count,
        "empty_string_count": empty_string_count,
        "whitespace_only_count": whitespace_only_count,
        "whole_field_placeholder_count": whole_field_placeholder_count,
        "effectively_missing_count": effectively_missing_count,
        "effectively_missing_pct": (
            (effectively_missing_count / total_rows * 100) if total_rows else 0.0
        ),
        "nonempty_address_count": nonempty_count,
        "rows_with_component_placeholder_cleaned": rows_with_component_placeholder_cleaned,
        "rows_with_component_placeholder_cleaned_pct": (
            (rows_with_component_placeholder_cleaned / nonempty_count * 100)
            if nonempty_count
            else 0.0
        ),
        "rows_with_component_placeholder_raw_ws": rows_with_component_placeholder_raw_ws,
        "rows_clean_nonempty": rows_clean_nonempty,
        "rows_clean_nonempty_pct": (
            (rows_clean_nonempty / nonempty_count * 100) if nonempty_count else 0.0
        ),
    }

    # Per-token records
    token_records = []
    for tok in PLACEHOLDER_TOKENS:
        token_records.append({
            "dataset": dataset_split,
            "source": source_name,
            "token": tok,
            "whole_field_rows": whole_field_token_counts[tok],
            "component_token_occurrences_cleaned": comp_cleaned_token_counts[tok],
            "component_token_occurrences_raw_ws": comp_raw_ws_token_counts[tok],
            "component_rows_cleaned": comp_cleaned_row_counts[tok],
            "component_rows_raw_ws": comp_raw_ws_row_counts[tok],
            "case_uppercase": token_casing[tok]["upper"],
            "case_lowercase": token_casing[tok]["lower"],
            "case_titlecase": token_casing[tok]["title"],
            "case_other": token_casing[tok]["other"],
            "punct_surrounded": token_punct[tok]["punct_surrounded"],
            "whitespace_isolated": token_punct[tok]["ws_isolated"],
        })

    return summary_record, token_records


def run_placeholder_audit(
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the placeholder audit across all 6 train and test source files.
    Saves:
      - placeholder_summary.csv
      - placeholder_token_summary.csv
    Returns the two DataFrames.
    """
    if output_dir is None:
        output_dir = E0_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []
    all_token_records = []

    print("=" * 90)
    print("OBJECTIVE 1: RUNNING PLACEHOLDER / NULL AUDIT")
    print("=" * 90)

    for split, src_name, file_path in ALL_SOURCES:
        print(f"\nProcessing {split} {src_name} ({file_path.name})...")
        summary_rec, token_recs = audit_single_source_file(
            file_path=file_path,
            dataset_split=split,
            source_name=src_name,
        )
        summary_records.append(summary_rec)
        all_token_records.extend(token_recs)

        print(
            f"  Total Rows: {summary_rec['total_rows']:,} | "
            f"Effectively Missing: {summary_rec['effectively_missing_count']:,} "
            f"({summary_rec['effectively_missing_pct']:.2f}%) | "
            f"Non-empty with placeholder: {summary_rec['rows_with_component_placeholder_cleaned']:,} "
            f"({summary_rec['rows_with_component_placeholder_cleaned_pct']:.2f}%)"
        )

    df_summary = pd.DataFrame(summary_records)
    df_tokens = pd.DataFrame(all_token_records)

    summary_csv = output_dir / "placeholder_summary.csv"
    tokens_csv = output_dir / "placeholder_token_summary.csv"

    df_summary.to_csv(summary_csv, index=False)
    df_tokens.to_csv(tokens_csv, index=False)

    print(f"\nWrote placeholder summary to: {summary_csv}")
    print(f"Wrote placeholder token summary to: {tokens_csv}")

    return df_summary, df_tokens
