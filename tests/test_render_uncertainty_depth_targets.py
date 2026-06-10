from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_render_uncertainty_maps.py"
SPEC = importlib.util.spec_from_file_location("evaluate_render_uncertainty_maps", SCRIPT_PATH)
evaluate_render_uncertainty_maps = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = evaluate_render_uncertainty_maps
SPEC.loader.exec_module(evaluate_render_uncertainty_maps)


class RenderUncertaintyDepthTargetsTest(unittest.TestCase):
    def test_candidate_frames_resolves_manifest_paths(self) -> None:
        payload = {
            "frames": [
                {"file_path": "./rgb/a.png", "depth_file_path": "./depth/a.png"},
                {"file_path": "./rgb/b.png", "depth_file_path": "./depth/b.png"},
            ]
        }

        frames = evaluate_render_uncertainty_maps._candidate_frames(payload, ["rgb/b.png"])

        self.assertEqual(frames, [{"file_path": "./rgb/b.png", "depth_file_path": "./depth/b.png"}])

    def test_downsample_global_samples_preserves_alignment(self) -> None:
        uncertainty = {"a": list(range(10)), "b": list(range(10, 20))}
        error = list(range(100, 110))

        sampled_uncertainty, sampled_error = evaluate_render_uncertainty_maps._downsample_global_samples(
            uncertainty,
            error,
            max_pixels=4,
        )

        self.assertEqual(sampled_uncertainty, {"a": [0, 2, 4, 6], "b": [10, 12, 14, 16]})
        self.assertEqual(sampled_error, [100, 102, 104, 106])

    def test_uncertainty_summary_fields_include_top_decile(self) -> None:
        fields = evaluate_render_uncertainty_maps._uncertainty_summary_fields([1, 2, 3, 4, 5])

        self.assertEqual(fields["mean_uncertainty"], 3.0)
        self.assertEqual(fields["p90_uncertainty"], 5.0)
        self.assertEqual(fields["top_decile_mean_uncertainty"], 5.0)


if __name__ == "__main__":
    unittest.main()
