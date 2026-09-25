"""
I/O and Environment Utilities for E1.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
I/O and environment utilities provide standardized, memory-safe paths, streaming file readers,
and runtime environment telemetry so that all pipeline stages operate on consistent file definitions
without hard-coded paths and stay within strict memory and compute limits.
"""

import csv
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import psutil
except ImportError:
    psutil = None

# Locate project and data roots robustly
CURRENT_FILE = Path(__file__).resolve()
# src/e1 -> src -> business_entity_resolution -> code -> student_resource -> AWS_Challenge
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
E1_EXPERIMENTS_DIR = EXPERIMENTS_ROOT / "e1"
E1_SPLITS_DIR = E1_EXPERIMENTS_DIR / "splits"
E1_VALIDATOR_DIR = E1_EXPERIMENTS_DIR / "validator"
E1_BENCHMARKS_DIR = E1_EXPERIMENTS_DIR / "benchmarks"

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

VALIDATOR_SCRIPT = STUDENT_RESOURCE_ROOT / "utils" / "validate_submission.py"


def get_environment_telemetry() -> Dict[str, Any]:
    """Capture hardware and execution environment details for reproducibility."""
    telemetry: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    if psutil is not None:
        vm = psutil.virtual_memory()
        telemetry["total_ram_gb"] = round(vm.total / (1024 ** 3), 2)
        telemetry["available_ram_gb"] = round(vm.available / (1024 ** 3), 2)
    else:
        telemetry["total_ram_gb"] = None
        telemetry["available_ram_gb"] = None
    return telemetry


def stream_s1_countries(path: Optional[Path] = None) -> Dict[str, str]:
    """
    Stream Source 1 file and extract mapping of {entity_id: country}.
    Lightweight, minimal memory footprint.
    """
    if path is None:
        path = TRAIN_FILES["S1"]
    country_by_s1: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            country = row[3] if len(row) > 3 else "Unknown"
            country_by_s1[s1_id] = country
    return country_by_s1


def save_json(data: Any, path: Path) -> None:
    """Save serializable data to JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv_from_dicts(records: List[Dict[str, Any]], path: Path) -> None:
    """Save list of dict records to CSV using dict keys as headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
