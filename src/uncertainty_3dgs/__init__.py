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
    uncertainty_error_bins,
)
from .splits import (
    SplitPlan,
    active_pose_novelty_order,
    generate_split_plan,
    load_frame_ids,
    load_frame_positions,
    write_split_plan,
)

__all__ = [
    "SplitPlan",
    "active_pose_novelty_order",
    "average_precision_score",
    "evaluate_uncertainty",
    "gaussian_nll",
    "generate_split_plan",
    "load_frame_ids",
    "load_frame_positions",
    "reliability_diagram",
    "risk_coverage_curve",
    "roc_auc_score",
    "sparsification_summary",
    "spearman_correlation",
    "uncertainty_error_bins",
    "write_split_plan",
]
