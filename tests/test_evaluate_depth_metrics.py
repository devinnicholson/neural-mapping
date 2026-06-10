from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_depth_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_depth_metrics", SCRIPT_PATH)
evaluate_depth_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = evaluate_depth_metrics
SPEC.loader.exec_module(evaluate_depth_metrics)

TORCH_SPEC = importlib.util.find_spec("torch")


@unittest.skipUnless(TORCH_SPEC is not None, "torch is required for depth metric tests")
class EvaluateDepthMetricsTest(unittest.TestCase):
    def test_depth_metrics_known_values(self) -> None:
        import torch

        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 4.0, 2.0])

        metrics = evaluate_depth_metrics.depth_metrics(prediction, target)

        self.assertEqual(metrics["count"], 3.0)
        self.assertAlmostEqual(metrics["abs_rel"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["sq_rel"], 0.5)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(metrics["delta1"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["delta2"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["delta3"], 2.0 / 3.0)

    def test_median_align_scales_prediction_to_target_median(self) -> None:
        import torch

        prediction = torch.tensor([1.0, 2.0, 4.0])
        target = torch.tensor([2.0, 4.0, 8.0])

        aligned, scale = evaluate_depth_metrics.median_align(prediction, target)

        self.assertAlmostEqual(scale, 2.0)
        self.assertTrue(torch.equal(aligned, target))

    def test_batch_depth_applies_depth_scale_to_unscaled_uint_depth(self) -> None:
        import torch

        batch = {"depth_image": torch.tensor([[5000.0, 0.0], [1000.0, 2000.0]])}

        depth = evaluate_depth_metrics._batch_depth(batch, depth_unit_scale=0.0002)

        self.assertIsNotNone(depth)
        self.assertTrue(torch.allclose(depth, torch.tensor([[1.0, 0.0], [0.2, 0.4]])))

    def test_test_frames_resolves_test_filenames(self) -> None:
        payload = {
            "frames": [
                {"file_path": "./images/a.png", "depth_file_path": "./depth/a.png"},
                {"file_path": "./images/b.png", "depth_file_path": "./depth/b.png"},
            ],
            "test_filenames": ["images/b.png"],
        }

        frames = evaluate_depth_metrics._test_frames(payload)

        self.assertEqual(frames, [{"file_path": "./images/b.png", "depth_file_path": "./depth/b.png"}])


if __name__ == "__main__":
    unittest.main()
