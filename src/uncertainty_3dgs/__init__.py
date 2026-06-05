"""Lightweight utilities for uncertainty-aware 3DGS experiments."""

from .metrics import (
    average_precision_score,
    evaluate_uncertainty,
    gaussian_nll,
    reliability_diagram,
    risk_coverage_curve,
    roc_auc_score,
    sparsification_summary,
    spearman_correlation,
)
from .splits import (
    SplitPlan,
    generate_split_plan,
    load_frame_ids,
    write_split_plan,
)

__all__ = [
    "SplitPlan",
    "average_precision_score",
    "evaluate_uncertainty",
    "gaussian_nll",
    "generate_split_plan",
    "load_frame_ids",
    "reliability_diagram",
    "risk_coverage_curve",
    "roc_auc_score",
    "sparsification_summary",
    "spearman_correlation",
    "write_split_plan",
]
