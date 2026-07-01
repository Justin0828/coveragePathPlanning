"""Candidate-scan methods used by segmentation ablation experiments."""

from .candidate_scan import (
    SUPPORTED_CANDIDATE_SCAN_STRATEGIES,
    CandidateSelection,
    select_candidates,
)

__all__ = [
    "SUPPORTED_CANDIDATE_SCAN_STRATEGIES",
    "CandidateSelection",
    "select_candidates",
]
