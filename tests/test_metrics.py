from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uncertainty_3dgs.metrics import (
    average_precision_score,
    evaluate_uncertainty,
    gaussian_nll,
    reliability_diagram,
    roc_auc_score,
    sparsification_summary,
    spearman_correlation,
)


class MetricTests(unittest.TestCase):
    def test_spearman_correlation_perfect_ordering(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([0.1, 0.2, 0.9, 1.0], [1.0, 2.0, 9.0, 10.0]),
            1.0,
        )

    def test_failure_detection_metrics_for_perfect_scores(self) -> None:
        scores = [0.1, 0.8, 0.2, 0.9]
        labels = [0, 1, 0, 1]

        self.assertAlmostEqual(roc_auc_score(scores, labels), 1.0)
        self.assertAlmostEqual(average_precision_score(scores, labels), 1.0)

    def test_reliability_diagram_ece(self) -> None:
        result = reliability_diagram([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], num_bins=2)

        self.assertAlmostEqual(result["ece"], 0.15)
        self.assertEqual(result["bins"][0]["count"], 2)
        self.assertEqual(result["bins"][1]["count"], 2)

    def test_sparsification_is_zero_gap_when_uncertainty_matches_error_order(self) -> None:
        result = sparsification_summary(
            error=[0.1, 0.2, 0.9, 1.5],
            uncertainty=[0.1, 0.2, 0.9, 1.5],
            fractions=[0.0, 0.25, 0.5],
        )

        self.assertAlmostEqual(result["ause"], 0.0)

    def test_gaussian_nll_matches_unit_normal_at_zero_residual(self) -> None:
        value = gaussian_nll(target=[0.0, 0.0], mean=[0.0, 0.0], variance_or_stddev=[1.0, 1.0])

        self.assertAlmostEqual(value, 0.5 * math.log(2.0 * math.pi))

    def test_evaluate_uncertainty_summary_with_mask(self) -> None:
        summary = evaluate_uncertainty(
            uncertainty=[[0.1, 0.8], [0.2, 0.9]],
            error=[[0.0, 1.0], [0.1, 2.0]],
            mask=[[True, True], [False, True]],
            bad_threshold=1.0,
        )

        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["auroc"], 1.0)
        self.assertAlmostEqual(summary["auprc"], 1.0)
        self.assertIn("sparsification", summary)

    def test_non_finite_values_are_filtered_without_shifting_pairs(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [1.0, float("nan"), 2.0]),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_correlation([1.0, float("nan"), 3.0], [1.0, 100.0, 2.0]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
