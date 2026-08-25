from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_icl_nuim.py"
SPEC = importlib.util.spec_from_file_location("prepare_icl_nuim", SCRIPT_PATH)
prepare_icl_nuim = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC.loader.exec_module(prepare_icl_nuim)


class PrepareIclNuimTests(unittest.TestCase):
    def test_official_trajectory_metadata_and_intrinsics(self) -> None:
        self.assertEqual(set(prepare_icl_nuim.TRAJECTORIES), {"lr_kt0", "lr_kt1", "lr_kt2", "lr_kt3"})
        self.assertEqual(prepare_icl_nuim.INTRINSICS["fl_x"], 481.20)
        self.assertEqual(prepare_icl_nuim.INTRINSICS["fl_y"], 480.00)
        self.assertIn("traj2n", prepare_icl_nuim.TRAJECTORIES["lr_kt2"]["noisy_url"])

    def test_indexed_pairing_excludes_unmatched_and_non_numeric_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "nested" / "sequence"
            rgb = root / "rgb"
            depth = root / "depth"
            rgb.mkdir(parents=True)
            depth.mkdir(parents=True)
            for name in ("0.png", "1.png", "2.png", "note.png"):
                (rgb / name).touch()
            for name in ("1.png", "2.png", "3.png"):
                (depth / name).touch()

            pairs = prepare_icl_nuim.indexed_image_pairs(Path(directory))

            self.assertEqual([pair[0] for pair in pairs], [1, 2])
            self.assertEqual([pair[1].name for pair in pairs], ["1.png", "2.png"])

    def test_pose_loader_keys_freiburg_rows_by_frame_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "livingRoom0.gt.freiburg"
            path.write_text(
                "# frame tx ty tz qx qy qz qw\n"
                "1 0 0 -2.25 0 0 0 1\n"
                "2 0.1 0.2 -2.0 0 0 0 1\n",
                encoding="utf-8",
            )

            poses = prepare_icl_nuim.load_indexed_poses(path)

            self.assertEqual(sorted(poses), [1, 2])
            self.assertEqual(poses[1], (0.0, 0.0, -2.25, 0.0, 0.0, 0.0, 1.0))

    def test_uniform_subsample_keeps_full_trajectory_extent(self) -> None:
        self.assertEqual(prepare_icl_nuim.uniform_subsample(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(prepare_icl_nuim.uniform_subsample([1, 2], 4), [1, 2])


if __name__ == "__main__":
    unittest.main()
