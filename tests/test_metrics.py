from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
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
    uncertainty_error_bins,
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

    def test_uncertainty_error_bins_track_observed_error(self) -> None:
        bins = uncertainty_error_bins(
            uncertainty=[0.1, 0.2, 0.8, 0.9],
            error=[0.0, 0.2, 1.0, 2.0],
            bad_threshold=0.5,
            num_bins=2,
        )

        self.assertEqual(len(bins), 2)
        self.assertEqual(bins[0]["count"], 2)
        self.assertAlmostEqual(bins[0]["mean_error"], 0.1)
        self.assertAlmostEqual(bins[0]["bad_fraction"], 0.0)
        self.assertAlmostEqual(bins[1]["mean_error"], 1.5)
        self.assertAlmostEqual(bins[1]["bad_fraction"], 1.0)

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
        self.assertIn("uncertainty_bins", summary)

    def test_compute_uncertainty_metrics_cli_accepts_reliability_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metric_input = root / "metric_input.json"
            output = root / "summary.json"
            metric_input.write_text(
                json.dumps(
                    {
                        "uncertainty": [0.1, 0.2, 0.8, 0.9],
                        "error": [0.0, 0.2, 1.0, 2.0],
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compute_uncertainty_metrics.py"),
                    "--input",
                    str(metric_input),
                    "--bad-threshold",
                    "0.5",
                    "--reliability-bins",
                    "2",
                    "--output",
                    str(output),
                ],
                check=True,
            )

            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["uncertainty_bins"]), 2)
            self.assertAlmostEqual(summary["uncertainty_bins"][1]["mean_error"], 1.5)
            self.assertAlmostEqual(summary["uncertainty_bins"][1]["bad_fraction"], 1.0)

    def test_non_finite_values_are_filtered_without_shifting_pairs(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [1.0, float("nan"), 2.0]),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_correlation([1.0, float("nan"), 3.0], [1.0, 100.0, 2.0]),
            1.0,
        )

    def test_frame_uncertainty_cli_evaluates_camera_distance_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = root / "transforms.json"
            split = root / "split.json"
            scores = root / "scores.json"
            output = root / "report.json"

            transforms.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "file_path": f"./images/frame_{index:03d}.png",
                                "transform_matrix": [
                                    [1, 0, 0, float(index)],
                                    [0, 1, 0, 0.0],
                                    [0, 0, 1, 0.0],
                                    [0, 0, 0, 1],
                                ],
                            }
                            for index in range(4)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            split.write_text(
                json.dumps(
                    {
                        "splits": {
                            "2": {
                                "train": [
                                    "./images/frame_000.png",
                                    "./images/frame_001.png",
                                ],
                                "val": [],
                                "test": ["./images/frame_002.png"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            scores.write_text(
                json.dumps(
                    {
                        "scores": [
                            {
                                "file_path": "images/frame_002.png",
                                "lpips": 0.2,
                                "mean_transmittance": 0.1,
                            },
                            {
                                "file_path": "images/frame_003.png",
                                "lpips": 0.9,
                                "mean_transmittance": 0.9,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_frame_uncertainty.py"),
                    "--frames",
                    str(transforms),
                    "--split-json",
                    str(split),
                    "--budget",
                    "2",
                    "--scores",
                    str(scores),
                    "--signals",
                    "nearest-train-distance",
                    "temporal-index-distance",
                    "uniform",
                    "--score-signal-fields",
                    "mean_transmittance",
                    "--bad-threshold",
                    "0.5",
                    "--output",
                    str(output),
                ],
                check=True,
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            nearest = report["signals"]["nearest-train-distance"]
            temporal = report["signals"]["temporal-index-distance"]
            transmittance = report["signals"]["mean_transmittance"]

            self.assertEqual(report["metadata"]["count"], 2)
            self.assertAlmostEqual(nearest["spearman"], 1.0)
            self.assertAlmostEqual(nearest["auroc"], 1.0)
            self.assertAlmostEqual(temporal["spearman"], 1.0)
            self.assertAlmostEqual(transmittance["spearman"], 1.0)
            self.assertAlmostEqual(transmittance["auroc"], 1.0)
            self.assertTrue(math.isnan(report["signals"]["uniform"]["spearman"]))
            self.assertEqual(
                [row["signals"]["nearest-train-distance"] for row in report["frames"]],
                [1.0, 2.0],
            )


if __name__ == "__main__":
    unittest.main()
