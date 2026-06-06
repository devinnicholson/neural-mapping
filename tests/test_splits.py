from __future__ import annotations

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

from uncertainty_3dgs.splits import (
    active_pose_novelty_order,
    generate_split_plan,
    load_frame_ids,
    load_frame_positions,
)


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

    def test_farthest_index_selection_is_nested_and_uses_trajectory_coverage(self) -> None:
        frames = [f"frame_{index:03d}.png" for index in range(10)]

        plan = generate_split_plan(
            frames,
            [2, 4],
            val_count=0,
            test_count=0,
            shuffle=False,
            selection_method="farthest-index",
        )

        split_2 = plan.splits["2"]
        split_4 = plan.splits["4"]
        self.assertEqual(plan.selection_method, "farthest-index")
        self.assertEqual(split_2["train"], ["frame_000.png", "frame_009.png"])
        self.assertEqual(
            split_4["train"],
            ["frame_000.png", "frame_002.png", "frame_004.png", "frame_009.png"],
        )
        self.assertTrue(set(split_2["train"]).issubset(split_4["train"]))

    def test_farthest_index_selection_keeps_random_holdouts_fixed(self) -> None:
        frames = [f"frame_{index:03d}.png" for index in range(30)]

        random_plan = generate_split_plan(
            frames,
            [8],
            seed=13,
            val_count=3,
            test_count=5,
        )
        coverage_plan = generate_split_plan(
            frames,
            [8],
            seed=13,
            val_count=3,
            test_count=5,
            selection_method="farthest-index",
        )

        self.assertEqual(random_plan.splits["8"]["val"], coverage_plan.splits["8"]["val"])
        self.assertEqual(random_plan.splits["8"]["test"], coverage_plan.splits["8"]["test"])
        self.assertNotEqual(
            random_plan.splits["8"]["train"],
            coverage_plan.splits["8"]["train"],
        )

    def test_unknown_selection_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection_method"):
            generate_split_plan(["a.png", "b.png"], [1], selection_method="bad")

    def test_farthest_pose_selection_uses_camera_centers(self) -> None:
        frames = [f"frame_{index:03d}.png" for index in range(5)]
        positions = {
            "frame_000.png": (0.0, 0.0, 0.0),
            "frame_001.png": (10.0, 0.0, 0.0),
            "frame_002.png": (5.0, 0.0, 0.0),
            "frame_003.png": (1.0, 0.0, 0.0),
            "frame_004.png": (9.0, 0.0, 0.0),
        }

        plan = generate_split_plan(
            frames,
            [2, 3],
            val_count=0,
            test_count=0,
            shuffle=False,
            selection_method="farthest-pose",
            frame_positions=positions,
        )

        split_2 = plan.splits["2"]
        split_3 = plan.splits["3"]
        self.assertEqual(plan.selection_method, "farthest-pose")
        self.assertEqual(split_2["train"], ["frame_000.png", "frame_001.png"])
        self.assertEqual(
            split_3["train"],
            ["frame_000.png", "frame_001.png", "frame_002.png"],
        )
        self.assertTrue(set(split_2["train"]).issubset(split_3["train"]))

    def test_farthest_pose_requires_positions(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_positions"):
            generate_split_plan(
                ["a.png", "b.png"],
                [1],
                selection_method="farthest-pose",
            )

    def test_active_pose_novelty_expands_from_seed_set(self) -> None:
        frames = [f"frame_{index:03d}.png" for index in range(6)]
        original_order = {frame: index for index, frame in enumerate(frames)}
        positions = {
            "frame_000.png": (0.0, 0.0, 0.0),
            "frame_001.png": (1.0, 0.0, 0.0),
            "frame_002.png": (2.0, 0.0, 0.0),
            "frame_003.png": (3.0, 0.0, 0.0),
            "frame_004.png": (9.0, 0.0, 0.0),
            "frame_005.png": (10.0, 0.0, 0.0),
        }

        order = active_pose_novelty_order(
            ["frame_002.png", "frame_003.png", "frame_004.png", "frame_005.png"],
            ["frame_000.png", "frame_001.png"],
            positions,
            original_order,
        )

        self.assertEqual(order[:2], ["frame_005.png", "frame_003.png"])

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

    def test_load_frame_positions_from_nerfstudio_style_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transforms.json"
            path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "file_path": "images/000.png",
                                "transform_matrix": [
                                    [1, 0, 0, 1.5],
                                    [0, 1, 0, -2.0],
                                    [0, 0, 1, 3.25],
                                    [0, 0, 0, 1],
                                ],
                            },
                            {
                                "file_path": "images/001.png",
                                "transform_matrix": [
                                    [1, 0, 0, 4.0],
                                    [0, 1, 0, 5.0],
                                    [0, 0, 1, 6.0],
                                    [0, 0, 0, 1],
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_frame_positions(path),
                {
                    "images/000.png": (1.5, -2.0, 3.25),
                    "images/001.png": (4.0, 5.0, 6.0),
                },
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

    def test_generate_active_split_cli_preserves_seed_and_holdouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = root / "transforms.json"
            frames = [
                {
                    "file_path": f"./images/frame_{index:03d}.png",
                    "transform_matrix": [
                        [1, 0, 0, float(index)],
                        [0, 1, 0, 0.0],
                        [0, 0, 1, 0.0],
                        [0, 0, 0, 1],
                    ],
                }
                for index in range(8)
            ]
            transforms.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            base_split = root / "base.json"
            base_split.write_text(
                json.dumps(
                    {
                        "scene": "example",
                        "seed": 3,
                        "splits": {
                            "2": {
                                "train": [
                                    "./images/frame_000.png",
                                    "./images/frame_001.png",
                                ],
                                "val": ["./images/frame_002.png"],
                                "test": ["./images/frame_003.png"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "active.json"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_active_split.py"),
                    "--frames",
                    str(transforms),
                    "--base-split",
                    str(base_split),
                    "--base-budget",
                    "2",
                    "--target-budget",
                    "4",
                    "--scene",
                    "example_active",
                    "--output",
                    str(output),
                ],
                check=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            split_2 = payload["splits"]["2"]
            split_4 = payload["splits"]["4"]
            self.assertEqual(payload["selection_method"], "active-pose-novelty")
            self.assertEqual(split_2["train"], ["./images/frame_000.png", "./images/frame_001.png"])
            self.assertEqual(split_2["val"], split_4["val"])
            self.assertEqual(split_2["test"], split_4["test"])
            self.assertTrue(set(split_2["train"]).issubset(split_4["train"]))
            self.assertEqual(len(split_4["train"]), 4)

    def test_generate_active_split_cli_accepts_score_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transforms = root / "transforms.json"
            transforms.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"file_path": f"frame_{index:03d}.png"}
                            for index in range(6)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            base_split = root / "base.json"
            base_split.write_text(
                json.dumps(
                    {
                        "splits": {
                            "2": {
                                "train": ["frame_000.png", "frame_001.png"],
                                "val": ["frame_002.png"],
                                "test": ["frame_003.png"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            scores = root / "scores.json"
            scores.write_text(
                json.dumps(
                    {
                        "metadata": {"score_metric": "lpips"},
                        "scores": [
                            {"file_path": "frame_004.png", "score": 0.1},
                            {"file_path": "frame_005.png", "score": 0.9},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "active.json"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_active_split.py"),
                    "--frames",
                    str(transforms),
                    "--base-split",
                    str(base_split),
                    "--base-budget",
                    "2",
                    "--target-budget",
                    "3",
                    "--strategy",
                    "score-desc",
                    "--scores",
                    str(scores),
                    "--output",
                    str(output),
                ],
                check=True,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("frame_005.png", payload["splits"]["3"]["train"])
            self.assertNotIn("frame_004.png", payload["splits"]["3"]["train"])

    def test_materialize_nerfstudio_split_cli_writes_explicit_split_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "images").mkdir(parents=True)
            for index in range(6):
                (source / "images" / f"frame_{index:03d}.png").write_text("", encoding="utf-8")
            (source / "transforms.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {"file_path": f"./images/frame_{index:03d}.png"}
                            for index in range(6)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "splits": {
                            "2": {
                                "train": [
                                    "./images/frame_000.png",
                                    "./images/frame_001.png",
                                ],
                                "val": ["./images/frame_002.png"],
                                "test": ["./images/frame_003.png"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            output_root = root / "materialized"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_nerfstudio_split.py"),
                    "--source-dir",
                    str(source),
                    "--split-json",
                    str(split),
                    "--output-root",
                    str(output_root),
                    "--budgets",
                    "2",
                ],
                check=True,
            )

            output = output_root / "budget_002"
            transforms = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [frame["file_path"] for frame in transforms["frames"]],
                [
                    "./images/frame_000.png",
                    "./images/frame_001.png",
                    "./images/frame_002.png",
                    "./images/frame_003.png",
                ],
            )
            self.assertEqual(
                transforms["train_filenames"],
                [
                    "./images/frame_000.png",
                    "./images/frame_001.png",
                    "./images/frame_002.png",
                ],
            )
            self.assertEqual(transforms["val_filenames"], ["./images/frame_002.png"])
            self.assertEqual(transforms["test_filenames"], ["./images/frame_003.png"])

    def test_materialize_candidate_eval_cli_uses_remaining_frames_as_eval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "images").mkdir(parents=True)
            for index in range(6):
                (source / "images" / f"frame_{index:03d}.png").write_text("", encoding="utf-8")
            (source / "transforms.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {"file_path": f"./images/frame_{index:03d}.png"}
                            for index in range(6)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            base_split = root / "base.json"
            base_split.write_text(
                json.dumps(
                    {
                        "splits": {
                            "2": {
                                "train": [
                                    "./images/frame_000.png",
                                    "./images/frame_001.png",
                                ],
                                "val": ["./images/frame_002.png"],
                                "test": ["./images/frame_003.png"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "candidate_eval"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "materialize_candidate_eval.py"),
                    "--source-dir",
                    str(source),
                    "--base-split",
                    str(base_split),
                    "--base-budget",
                    "2",
                    "--output-dir",
                    str(output),
                ],
                check=True,
            )

            transforms = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "candidate_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [frame["file_path"] for frame in transforms["frames"]],
                [
                    "./images/frame_000.png",
                    "./images/frame_001.png",
                    "./images/frame_002.png",
                    "./images/frame_004.png",
                    "./images/frame_005.png",
                ],
            )
            self.assertEqual(
                transforms["test_filenames"],
                ["./images/frame_004.png", "./images/frame_005.png"],
            )
            self.assertEqual(
                transforms["train_filenames"],
                [
                    "./images/frame_000.png",
                    "./images/frame_001.png",
                    "./images/frame_002.png",
                ],
            )
            self.assertEqual(transforms["val_filenames"], ["./images/frame_002.png"])
            self.assertEqual(manifest["candidates"], transforms["test_filenames"])


if __name__ == "__main__":
    unittest.main()
