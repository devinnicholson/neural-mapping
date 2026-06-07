#!/usr/bin/env python3
"""Generate an active-expansion split from an existing seed split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uncertainty_3dgs.splits import (
    SplitPlan,
    active_pose_novelty_order,
    active_score_pose_hybrid_order,
    load_frame_ids,
    load_frame_positions,
    write_split_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        required=True,
        help="Full frame source, usually the filtered Nerfstudio transforms.json.",
    )
    parser.add_argument("--base-split", required=True, help="Existing split JSON.")
    parser.add_argument("--base-budget", type=int, required=True)
    parser.add_argument("--target-budget", type=int, required=True)
    parser.add_argument("--output", required=True, help="Output active split JSON.")
    parser.add_argument("--scene", default=None, help="Optional scene name.")
    parser.add_argument(
        "--strategy",
        choices=("pose-novelty", "score-desc", "score-pose-hybrid"),
        default="pose-novelty",
        help=(
            "How to select extra frames. pose-novelty expands from the seed set by "
            "camera-center novelty. score-desc selects highest scored candidates. "
            "score-pose-hybrid mixes candidate score with camera-center novelty."
        ),
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="JSON/CSV frame score file required for --strategy score-desc.",
    )
    parser.add_argument(
        "--score-key",
        default="score",
        help="Score field for JSON/CSV files used by score-based strategies.",
    )
    parser.add_argument(
        "--score-weight",
        type=float,
        default=0.65,
        help="Score weight for --strategy score-pose-hybrid; pose weight is 1-score_weight.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = load_frame_ids(args.frames)
    original_order = {frame: index for index, frame in enumerate(frames)}
    base_payload = _load_object_json(Path(args.base_split))
    base_seed = int(args.base_budget)
    target_budget = int(args.target_budget)

    base_splits = base_payload.get("splits")
    if not isinstance(base_splits, dict):
        raise SystemExit("Base split JSON must contain an object-valued 'splits' key.")
    base_entry = base_splits.get(str(base_seed))
    if not isinstance(base_entry, dict):
        raise SystemExit(f"Base split JSON is missing budget {base_seed}.")

    seed_train = _split_frames(base_entry, "train")
    val = _split_frames(base_entry, "val")
    test = _split_frames(base_entry, "test")
    if target_budget <= len(seed_train):
        raise SystemExit(
            f"--target-budget must exceed base train count {len(seed_train)}; "
            f"got {target_budget}."
        )

    known = set(seed_train) | set(val) | set(test)
    candidates = [frame for frame in frames if frame not in known]
    add_count = target_budget - len(seed_train)
    if add_count > len(candidates):
        raise SystemExit(
            f"Need {add_count} active frames, but only {len(candidates)} candidates are available."
        )

    if args.strategy == "pose-novelty":
        positions = load_frame_positions(args.frames)
        if not positions:
            raise SystemExit("--strategy pose-novelty requires transform_matrix camera poses.")
        extra_order = active_pose_novelty_order(candidates, seed_train, positions, original_order)
    elif args.strategy == "score-desc":
        if args.scores is None:
            raise SystemExit("--strategy score-desc requires --scores.")
        scores = _load_scores(Path(args.scores), args.score_key)
        missing = [frame for frame in candidates if frame not in scores]
        if missing:
            raise SystemExit(f"Score file is missing candidate frames: {missing[:10]}")
        extra_order = sorted(
            candidates,
            key=lambda frame: (-float(scores[frame]), original_order[frame]),
        )
    else:
        if args.scores is None:
            raise SystemExit("--strategy score-pose-hybrid requires --scores.")
        positions = load_frame_positions(args.frames)
        if not positions:
            raise SystemExit("--strategy score-pose-hybrid requires transform_matrix camera poses.")
        scores = _load_scores(Path(args.scores), args.score_key)
        try:
            extra_order = active_score_pose_hybrid_order(
                candidates,
                seed_train,
                positions,
                scores,
                original_order,
                score_weight=args.score_weight,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error

    selected_extra = extra_order[:add_count]
    active_train = _order_like_input([*seed_train, *selected_extra], original_order)
    if len(active_train) != target_budget:
        raise SystemExit(
            f"Internal error: expected {target_budget} train frames, got {len(active_train)}."
        )

    scene = args.scene if args.scene is not None else base_payload.get("scene")
    seed = int(base_payload.get("seed", 0))
    plan = SplitPlan(
        scene=str(scene) if scene is not None else None,
        seed=seed,
        selection_method=f"active-{args.strategy}",
        total_frames=len(frames),
        val_count=len(val),
        test_count=len(test),
        train_counts=(base_seed, target_budget),
        splits={
            str(base_seed): {
                "train": _order_like_input(seed_train, original_order),
                "val": _order_like_input(val, original_order),
                "test": _order_like_input(test, original_order),
            },
            str(target_budget): {
                "train": active_train,
                "val": _order_like_input(val, original_order),
                "test": _order_like_input(test, original_order),
            },
        },
    )
    write_split_plan(plan, args.output)
    print(
        "wrote "
        f"{args.output} with seed_budget={base_seed}, "
        f"target_budget={target_budget}, added={add_count}, "
        f"strategy={args.strategy}"
    )
    return 0


def _load_object_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON must contain an object: {path}")
    return payload


def _split_frames(split_entry: dict[str, object], key: str) -> list[str]:
    values = split_entry.get(key)
    if not isinstance(values, list):
        raise SystemExit(f"Split entry must contain list-valued '{key}'.")
    return [str(value) for value in values]


def _order_like_input(frames: list[str], original_order: dict[str, int]) -> list[str]:
    missing = [frame for frame in frames if frame not in original_order]
    if missing:
        raise SystemExit(f"Split references frames missing from --frames: {missing[:10]}")
    return sorted(frames, key=lambda frame: original_order[frame])


def _load_scores(path: Path, score_key: str) -> dict[str, float]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return {}
            frame_key = next(
                (
                    key
                    for key in ("file_path", "frame", "frame_id", "image_path", "path")
                    if key in reader.fieldnames
                ),
                reader.fieldnames[0],
            )
            return {
                str(row[frame_key]): float(row[score_key])
                for row in reader
                if row.get(frame_key) and row.get(score_key)
            }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "scores" in payload and isinstance(payload["scores"], list):
            payload = payload["scores"]
        elif "frames" in payload and isinstance(payload["frames"], list):
            payload = payload["frames"]
        else:
            return {str(key): float(value) for key, value in payload.items()}

    if isinstance(payload, list):
        scores: dict[str, float] = {}
        for item in payload:
            if not isinstance(item, dict):
                raise SystemExit("Score list entries must be objects.")
            frame = item.get("file_path") or item.get("frame") or item.get("frame_id")
            if frame is None or score_key not in item:
                raise SystemExit(
                    "Score entries must contain a frame id and the requested score key."
                )
            scores[str(frame)] = float(item[score_key])
        return scores

    raise SystemExit("Score file must be a JSON object, JSON list, or CSV file.")


if __name__ == "__main__":
    raise SystemExit(main())
