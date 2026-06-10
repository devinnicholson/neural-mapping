from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_tum_rgbd.py"
SPEC = importlib.util.spec_from_file_location("prepare_tum_rgbd", SCRIPT_PATH)
prepare_tum_rgbd = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_tum_rgbd)


class PrepareTumRgbdTest(unittest.TestCase):
    def test_pose_to_matrix_flips_tum_camera_axes_by_default(self) -> None:
        matrix = prepare_tum_rgbd.pose_to_matrix((1, 2, 3, 0, 0, 0, 1))

        self.assertEqual(matrix[0], [1.0, -0.0, -0.0, 1.0])
        self.assertEqual(matrix[1], [0.0, -1.0, -0.0, 2.0])
        self.assertEqual(matrix[2], [0.0, -0.0, -1.0, 3.0])
        self.assertEqual(matrix[3], [0.0, 0.0, 0.0, 1.0])

    def test_build_associations_matches_nearest_depth_and_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "rgb.txt").write_text(
                "\n".join(
                    [
                        "# timestamp path",
                        "1.000 rgb/1.png",
                        "2.000 rgb/2.png",
                        "3.000 rgb/3.png",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "depth.txt").write_text(
                "\n".join(
                    [
                        "1.010 depth/1.png",
                        "2.030 depth/2.png",
                        "2.990 depth/3.png",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "groundtruth.txt").write_text(
                "\n".join(
                    [
                        "0.995 0 0 0 0 0 0 1",
                        "2.000 1 2 3 0 0 0 1",
                        "3.100 1 2 3 0 0 0 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            associations = prepare_tum_rgbd.build_associations(root, tolerance=0.02)

        self.assertEqual(len(associations), 1)
        self.assertEqual(associations[0]["rgb"], (1.0, "rgb/1.png"))
        self.assertEqual(associations[0]["depth"], (1.01, "depth/1.png"))
        self.assertEqual(associations[0]["groundtruth"][0], 0.995)


if __name__ == "__main__":
    unittest.main()
