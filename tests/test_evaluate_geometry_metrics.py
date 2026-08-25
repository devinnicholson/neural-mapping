from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_geometry_metrics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_geometry_metrics", SCRIPT_PATH)
evaluate_geometry_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC.loader.exec_module(evaluate_geometry_metrics)


@unittest.skipIf(np is None, "NumPy numeric extras are not installed")
class EvaluateGeometryMetricsTests(unittest.TestCase):
    def test_backprojection_uses_opengl_negative_z(self) -> None:
        depth = np.asarray([[2.0]], dtype=np.float32)
        transform = np.eye(4)
        transform[:3, 3] = (1.0, 2.0, 3.0)

        points = evaluate_geometry_metrics.backproject_depth(
            depth,
            np.asarray([0]),
            transform,
            {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        )

        np.testing.assert_allclose(points, [[1.0, 2.0, 1.0]])

    def test_intrinsics_are_scaled_to_render_shape(self) -> None:
        intrinsics = evaluate_geometry_metrics.intrinsics_for_shape(
            {"w": 640, "h": 480, "fl_x": 480.0, "fl_y": 480.0, "cx": 320.0, "cy": 240.0},
            (240, 320),
        )
        self.assertEqual(intrinsics, {"fx": 240.0, "fy": 240.0, "cx": 160.0, "cy": 120.0})

    def test_geometry_metrics_are_symmetric_and_thresholded(self) -> None:
        reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        predicted = reference + np.asarray([0.02, 0.0, 0.0])

        metrics = evaluate_geometry_metrics.geometry_metrics(predicted, reference, (0.01, 0.05))

        self.assertAlmostEqual(metrics["accuracy_mean_m"], 0.02)
        self.assertAlmostEqual(metrics["completeness_mean_m"], 0.02)
        self.assertEqual(metrics["fscore_01cm"], 0.0)
        self.assertEqual(metrics["fscore_05cm"], 1.0)


if __name__ == "__main__":
    unittest.main()
