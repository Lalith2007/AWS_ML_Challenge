"""
Sprint E3 — L2 Symbolic Blocking and Candidate Generation Subsystem.
"""

from .blocking_lanes import BlockingConfig, generate_lane_keys
from .candidate_generator import PartitionRunResult, run_partition_blocking
from .evaluator import E3EvaluationSummary, analyze_failure_buckets, categorize_missed_pair
from .inverted_index import PartitionInvertedIndex
from .preprocessing import (
    clean_text,
    detect_script,
    extract_address_features,
    extract_dba_components,
    tokenize_name,
    transliterate_indic,
)
from .runner import run_e3_pipeline

__all__ = [
    "BlockingConfig",
    "generate_lane_keys",
    "PartitionInvertedIndex",
    "PartitionRunResult",
    "run_partition_blocking",
    "E3EvaluationSummary",
    "analyze_failure_buckets",
    "categorize_missed_pair",
    "clean_text",
    "detect_script",
    "extract_address_features",
    "extract_dba_components",
    "tokenize_name",
    "transliterate_indic",
    "run_e3_pipeline",
]
