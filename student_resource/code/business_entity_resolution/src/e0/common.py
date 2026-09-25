"""
Common utilities, path configuration, and deterministic normalization for E0.
Amazon ML Challenge 2026 — Business Entity Resolution.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Locate project and data roots robustly
CURRENT_FILE = Path(__file__).resolve()
# src/e0 -> src -> business_entity_resolution -> code -> student_resource -> AWS_Challenge
if (CURRENT_FILE.parents[4] / "dataset").is_dir():
    STUDENT_RESOURCE_ROOT = CURRENT_FILE.parents[4]
    PROJECT_ROOT = CURRENT_FILE.parents[5]
elif (Path.cwd() / "student_resource" / "dataset").is_dir():
    STUDENT_RESOURCE_ROOT = Path.cwd() / "student_resource"
    PROJECT_ROOT = Path.cwd()
elif (Path.cwd() / "dataset").is_dir():
    STUDENT_RESOURCE_ROOT = Path.cwd()
    PROJECT_ROOT = Path.cwd().parent
else:
    STUDENT_RESOURCE_ROOT = CURRENT_FILE.parents[4]
    PROJECT_ROOT = CURRENT_FILE.parents[5]

DATA_ROOT = STUDENT_RESOURCE_ROOT / "dataset"
EXPERIMENTS_ROOT = STUDENT_RESOURCE_ROOT / "experiments"
E0_RESULTS_DIR = EXPERIMENTS_ROOT / "e0"

# Source file definitions
TRAIN_FILES = {
    "S1": DATA_ROOT / "train" / "train_source1.tsv",
    "S2": DATA_ROOT / "train" / "train_source2.tsv",
    "S3": DATA_ROOT / "train" / "train_source3.tsv",
    "GT": DATA_ROOT / "train" / "train_ground_truth.tsv",
}

TEST_FILES = {
    "S1": DATA_ROOT / "test" / "test_source1.tsv",
    "S2": DATA_ROOT / "test" / "test_source2.tsv",
    "S3": DATA_ROOT / "test" / "test_source3.tsv",
}

ALL_SOURCES = [
    ("train", "S1", TRAIN_FILES["S1"]),
    ("train", "S2", TRAIN_FILES["S2"]),
    ("train", "S3", TRAIN_FILES["S3"]),
    ("test", "S1", TEST_FILES["S1"]),
    ("test", "S2", TEST_FILES["S2"]),
    ("test", "S3", TEST_FILES["S3"]),
]

EXPECTED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLS = ["source1_entity_id", "matched_entity_ids"]

PLACEHOLDER_TOKENS = [
    "null",
    "n/a",
    "na",
    "nan",
    "none",
    "unknown",
    "undefined",
]

PLACEHOLDER_SET = set(PLACEHOLDER_TOKENS)

# Set of values that pandas converts to NaN by default when keep_default_na=True
PANDAS_DEFAULT_NA_VALUES = {
    "",
    "#n/a",
    "#n/a n/a",
    "#na",
    "-1.#ind",
    "-1.#qnan",
    "-nan",
    "-nan",
    "1.#ind",
    "1.#qnan",
    "<na>",
    "n/a",
    "na",
    "null",
    "nan",
    "none",
}


def normalize_text(value: Optional[str]) -> str:
    """
    Deterministic conservative normalization for business_name and business_address.

    Specification:
    1. Returns empty string "" if value is None, empty, or whitespace-only.
    2. Applies Unicode NFKC normalization (canonical decomposition + compatibility).
    3. Converts to lowercase.
    4. Replaces all non-alphanumeric, non-whitespace characters with space
       using re.UNICODE to preserve foreign script letters and digits (e.g. Hindi, French).
    5. Collapses multiple whitespace characters to a single space and strips ends.

    Guarantees:
    - Deterministic and pure function.
    - No semantic rewriting.
    - No external API or geocoding calls.
    - No country-specific dictionary lookups.
    """
    if value is None:
        return ""
    str_val = str(value)
    if not str_val:
        return ""

    # Step 1: Unicode normalization (NFKC)
    norm = unicodedata.normalize("NFKC", str_val)

    # Step 2: Case normalization
    norm = norm.lower()

    # Step 3: Replace punctuation with space, preserving unicode alphanumeric letters
    norm = re.sub(r"[^\w\s]", " ", norm, flags=re.UNICODE)

    # Step 4: Collapse whitespace and strip
    return re.sub(r"\s+", " ", norm).strip()


def verify_source_files() -> Dict[str, bool]:
    """
    Verify that all expected dataset files exist and have correct TSV headers.
    """
    results = {}
    for name, path in TRAIN_FILES.items():
        if not path.is_file():
            results[f"train_{name}_exists"] = False
            continue
        results[f"train_{name}_exists"] = True

        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            expected = EXPECTED_GT_COLS if name == "GT" else EXPECTED_SOURCE_COLS
            results[f"train_{name}_header_valid"] = (header == expected)

    for name, path in TEST_FILES.items():
        if not path.is_file():
            results[f"test_{name}_exists"] = False
            continue
        results[f"test_{name}_exists"] = True

        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            results[f"test_{name}_header_valid"] = (header == EXPECTED_SOURCE_COLS)

    return results
