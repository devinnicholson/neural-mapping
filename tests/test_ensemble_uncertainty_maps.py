from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_ensemble_uncertainty_maps.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_ensemble_uncertainty_maps", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EnsembleUncertaintyMapTests(unittest.TestCase):
    def test_uncertainty_summary_fields_include_tail_scores(self) -> None:
        module = _load_script_module()

        summary = module._uncertainty_summary_fields([0.0, 1.0, 2.0, 3.0])

        self.assertAlmostEqual(summary["mean_uncertainty"], 1.5)
        self.assertAlmostEqual(summary["uncertainty_std"], math.sqrt(1.25))
        self.assertAlmostEqual(summary["p90_uncertainty"], 3.0)
        self.assertAlmostEqual(summary["p95_uncertainty"], 3.0)
        self.assertAlmostEqual(summary["top_decile_mean_uncertainty"], 3.0)


if __name__ == "__main__":
    unittest.main()
