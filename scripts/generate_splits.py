#!/usr/bin/env python3
"""Generate deterministic dataset split JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uncertainty_3dgs.splits import generate_split_plan, load_frame_ids, write_split_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        required=True,
        help="Directory, JSON manifest, CSV, or text file containing frame ids.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        required=True,
        help="Training frame budgets, for example: 25 50 100 200.",
    )
    parser.add_argument("--output", required=True, help="Output split JSON path.")
    parser.add_argument("--scene", default=None, help="Optional scene name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Validation/calibration holdout fraction when --val-count is omitted.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Test holdout fraction when --test-count is omitted.",
    )
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use input ordering instead of seeded randomization.",
    )
    parser.add_argument(
        "--selection-method",
        choices=("random", "farthest-index"),
        default="random",
        help="Training frame selection policy after validation/test holdouts are fixed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_ids = load_frame_ids(args.frames)
    plan = generate_split_plan(
        frame_ids,
        args.budgets,
        scene=args.scene,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_count=args.val_count,
        test_count=args.test_count,
        shuffle=not args.no_shuffle,
        selection_method=args.selection_method,
    )
    write_split_plan(plan, args.output)
    print(
        "wrote "
        f"{args.output} with {plan.total_frames} frames, "
        f"budgets={list(plan.train_counts)}, "
        f"val={plan.val_count}, test={plan.test_count}, "
        f"selection={plan.selection_method}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
