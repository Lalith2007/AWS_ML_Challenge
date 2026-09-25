"""
E1: Validation and Measurement Foundation Package.
Amazon ML Challenge 2026 — Business Entity Resolution.
"""

from .benchmark import run_e1_microbenchmark
from .gt_utils import get_multiplicity_bucket, load_ground_truth, validate_ground_truth_integrity
from .io_utils import (
    DATA_ROOT,
    E1_BENCHMARKS_DIR,
    E1_EXPERIMENTS_DIR,
    E1_SPLITS_DIR,
    E1_VALIDATOR_DIR,
    PROJECT_ROOT,
    STUDENT_RESOURCE_ROOT,
    TEST_FILES,
    TRAIN_FILES,
    VALIDATOR_SCRIPT,
    get_environment_telemetry,
    stream_s1_countries,
)
from .recall_harness import evaluate_blocking_recall, validate_candidate_set
from .scorer import score_predictions, score_single_entity
from .submission_validator import run_submission_validator
from .validation_split import check_target_id_leakage, create_deterministic_s1_split

__all__ = [
    "DATA_ROOT",
    "E1_BENCHMARKS_DIR",
    "E1_EXPERIMENTS_DIR",
    "E1_SPLITS_DIR",
    "E1_VALIDATOR_DIR",
    "PROJECT_ROOT",
    "STUDENT_RESOURCE_ROOT",
    "TEST_FILES",
    "TRAIN_FILES",
    "VALIDATOR_SCRIPT",
    "check_target_id_leakage",
    "create_deterministic_s1_split",
    "evaluate_blocking_recall",
    "get_environment_telemetry",
    "get_multiplicity_bucket",
    "load_ground_truth",
    "run_e1_microbenchmark",
    "run_submission_validator",
    "score_predictions",
    "score_single_entity",
    "stream_s1_countries",
    "validate_candidate_set",
    "validate_ground_truth_integrity",
]
