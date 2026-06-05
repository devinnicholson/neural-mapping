from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uncertainty_3dgs.splits import generate_split_plan, load_frame_ids


class SplitGenerationTests(unittest.TestCase):
    def test_split_plan_has_fixed_holdouts_and_nested_train_sets(self) -> None:
        frames = [f"frame_{index:03d}.png" for index in range(20)]

        plan = generate_split_plan(
            frames,
            [5, 10],
            seed=7,
            val_count=3,
            test_count=4,
        )

        split_5 = plan.splits["5"]
        split_10 = plan.splits["10"]
        self.assertEqual(len(split_5["train"]), 5)
        self.assertEqual(len(split_10["train"]), 10)
        self.assertEqual(split_5["val"], split_10["val"])
        self.assertEqual(split_5["test"], split_10["test"])
        self.assertTrue(set(split_5["train"]).issubset(split_10["train"]))
        self.assertFalse(set(split_10["train"]) & set(split_10["val"]))
        self.assertFalse(set(split_10["train"]) & set(split_10["test"]))
        self.assertFalse(set(split_10["val"]) & set(split_10["test"]))

    def test_split_plan_is_reproducible(self) -> None:
        frames = [str(index) for index in range(30)]

        first = generate_split_plan(frames, [8], seed=11, val_count=2, test_count=5)
        second = generate_split_plan(frames, [8], seed=11, val_count=2, test_count=5)

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_duplicate_frame_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            generate_split_plan(["a.png", "b.png", "a.png"], [1], val_count=0, test_count=1)

    def test_load_frame_ids_from_nerfstudio_style_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transforms.json"
            path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"file_path": "images/000.png"},
                            {"file_path": "images/001.png"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_frame_ids(path),
                ["images/000.png", "images/001.png"],
            )

    def test_load_frame_ids_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            (root / "rgb" / "000.png").write_text("", encoding="utf-8")
            (root / "rgb" / "001.jpg").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(
                load_frame_ids(root),
                ["rgb/000.png", "rgb/001.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
