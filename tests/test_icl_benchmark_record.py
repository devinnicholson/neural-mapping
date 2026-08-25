from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_icl_benchmark_record.py"
SPEC = importlib.util.spec_from_file_location("build_icl_benchmark_record", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IclBenchmarkRecordTests(unittest.TestCase):
    def test_cluster_bootstrap_preserves_trajectory_clusters(self) -> None:
        interval = MODULE._cluster_bootstrap_interval(
            {"kt0": [0.0, 2.0], "kt1": [4.0, 6.0]}
        )
        self.assertEqual(interval, [1.0, 3.0])

    def test_favorable_direction_is_metric_specific(self) -> None:
        self.assertTrue(MODULE._favorable("psnr", 0.1))
        self.assertFalse(MODULE._favorable("psnr", -0.1))
        self.assertTrue(MODULE._favorable("lpips", -0.1))
        self.assertFalse(MODULE._favorable("lpips", 0.1))

    def test_split_audit_rejects_leakage(self) -> None:
        path = Path(self.id().replace(".", "_") + ".json")
        payload = {
            "budget": 1,
            "train": ["rgb/0.png"],
            "val": ["rgb/0.png", *[f"rgb/v{i}.png" for i in range(9)]],
            "test": [f"rgb/t{i}.png" for i in range(20)],
        }
        payload["transforms_frames"] = payload["train"] + payload["val"] + payload["test"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "leakage"):
                MODULE._audit_split_manifest(path, 1)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
