from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ActiveMetricSummaryTests(unittest.TestCase):
    def test_summarizes_grouped_active_metric_deltas(self) -> None:
        rows = [
            _row("random_v1", "050", 20.0, 0.70, 0.30, 10.0),
            _row("active_v1", "050", 21.0, 0.75, 0.25, 9.0),
            _row("random_v2", "050", 24.0, 0.80, 0.20, 11.0),
            _row("active_v2", "050", 25.5, 0.82, 0.19, 12.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "metrics.json"
            output_path = root / "summary.json"
            input_path.write_text(json.dumps(rows), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_active_metrics.py"),
                    "--input",
                    str(input_path),
                    "--budget",
                    "50",
                    "--pair",
                    "demo:v1:random_v1:active_v1",
                    "--pair",
                    "demo:v2:random_v2:active_v2",
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(output_path.read_text(encoding="utf-8"))
            group = summary["groups"][0]
            self.assertEqual(group["group"], "demo")
            self.assertEqual(group["count"], 2)
            self.assertAlmostEqual(group["mean_delta"]["psnr"], 1.25)
            self.assertAlmostEqual(group["mean_delta"]["ssim"], 0.035)
            self.assertAlmostEqual(group["mean_delta"]["lpips"], -0.03)
            self.assertAlmostEqual(group["mean_delta"]["fps"], 0.0)

    def test_parses_noisy_modal_metrics_output(self) -> None:
        row_a = _row("random_v1", "budget_050", 20.0, 0.70, 0.30, 10.0)
        row_b = _row("active_v1", "050", 21.0, 0.75, 0.25, 9.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "modal.log"
            input_path.write_text(
                "\n".join(
                    [
                        "Initialized Modal app",
                        json.dumps(row_a, indent=2),
                        "Created function collect_metric_rows.",
                        json.dumps(row_b, indent=2),
                        "Stopping app - local entrypoint completed.",
                    ]
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_active_metrics.py"),
                    "--input",
                    str(input_path),
                    "--budget",
                    "050",
                    "--pair",
                    "demo:v1:random_v1:active_v1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(completed.stdout)
            self.assertEqual(summary["pairs"][0]["delta"]["psnr"], 1.0)
            self.assertEqual(summary["pairs"][0]["budget"], "050")

    def test_accepts_pairs_file(self) -> None:
        rows = [
            _row("random_v1", "050", 20.0, 0.70, 0.30, 10.0),
            _row("active_v1", "050", 21.0, 0.75, 0.25, 9.0),
        ]
        pairs = {
            "budget": "50",
            "pairs": [
                {
                    "group": "demo",
                    "seed": "v1",
                    "baseline_scene": "random_v1",
                    "active_scene": "active_v1",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "metrics.json"
            pairs_path = root / "pairs.json"
            input_path.write_text(json.dumps(rows), encoding="utf-8")
            pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_active_metrics.py"),
                    "--input",
                    str(input_path),
                    "--pairs-file",
                    str(pairs_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(completed.stdout)
            self.assertEqual(summary["budget"], "050")
            self.assertEqual(summary["groups"][0]["count"], 1)
            self.assertAlmostEqual(summary["groups"][0]["mean_delta"]["lpips"], -0.05)


def _row(scene: str, budget: str, psnr: float, ssim: float, lpips: float, fps: float) -> dict[str, object]:
    return {
        "scene": scene,
        "budget": budget,
        "checkpoint": f"/tmp/{scene}.ckpt",
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "fps": fps,
        "metrics_path": f"/tmp/{scene}.json",
    }


if __name__ == "__main__":
    unittest.main()
