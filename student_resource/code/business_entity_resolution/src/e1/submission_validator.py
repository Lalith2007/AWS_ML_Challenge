"""
Official Submission Validator Integration Wrapper for E1.

WHY DOES THIS EXIST IN THE LOCKED ARCHITECTURE?
Candidate and output compliance is part of the architecture and must be tested continuously
using the official validator without rewriting it. This wrapper provides a programmatic Python API
to invoke utils/validate_submission.py, record logs, and enforce competition constraints
across all experimental runs before submission packaging.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import E1_VALIDATOR_DIR, STUDENT_RESOURCE_ROOT, VALIDATOR_SCRIPT


def run_submission_validator(
    matching_path: Path,
    candidate_path: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    check_ids: bool = False,
    working_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Programmatic wrapper around student_resource/utils/validate_submission.py.

    Executes the official validator subprocess, captures stdout/stderr,
    extracts warnings/errors, and writes an execution log.

    Args:
        matching_path: Path to matching_results.tsv to validate.
        candidate_path: Optional path to candidate_pairs.tsv.
        test_dir: Folder containing test_source1/2/3.tsv (defaults to student_resource/dataset/test).
        check_ids: If True, passes --check-ids (memory heavy on full test set).
        working_dir: Directory from which to run (defaults to student_resource).
        log_path: Path to save the validator run log.

    Returns:
        {
            "status": "PASS" | "FAIL",
            "exit_code": int,
            "stdout": str,
            "stderr": str,
            "warnings": List[str],
            "errors": List[str],
            "log_path": str,
        }
    """
    if working_dir is None:
        working_dir = STUDENT_RESOURCE_ROOT
    if test_dir is None:
        test_dir = STUDENT_RESOURCE_ROOT / "dataset" / "test"
    if log_path is None:
        log_path = E1_VALIDATOR_DIR / "validator_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not VALIDATOR_SCRIPT.is_file():
        raise FileNotFoundError(f"Official validator script not found at: {VALIDATOR_SCRIPT}")

    cmd = [
        sys.executable,
        str(VALIDATOR_SCRIPT),
        "--matching",
        str(matching_path),
        "--test-dir",
        str(test_dir),
    ]

    if candidate_path is not None:
        cmd.extend(["--candidate", str(candidate_path)])

    if check_ids:
        cmd.append("--check-ids")

    result = subprocess.run(
        cmd,
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    stdout = result.stdout
    stderr = result.stderr

    warnings: List[str] = []
    errors: List[str] = []

    in_fail_section = False
    for line in stdout.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        if line_clean.startswith("WARNING:"):
            warnings.append(line_clean)
        elif line_clean.startswith("FAIL"):
            in_fail_section = True
            errors.append(line_clean)
        elif in_fail_section:
            errors.append(line_clean)
        elif any(k in line_clean.lower() for k in ("malformed", "unexpected header", "cannot read")):
            errors.append(line_clean)

    status = "PASS" if result.returncode == 0 else "FAIL"

    # Save log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"OFFICIAL VALIDATOR RUN LOG — Status: {status} (Exit code: {result.returncode})\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Working Dir: {working_dir}\n")
        f.write("=" * 80 + "\n\n")
        f.write("--- STDOUT ---\n")
        f.write(stdout + "\n\n")
        if stderr:
            f.write("--- STDERR ---\n")
            f.write(stderr + "\n")

    return {
        "status": status,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "warnings": warnings,
        "errors": errors,
        "log_path": str(log_path),
    }
