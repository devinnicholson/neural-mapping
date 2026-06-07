#!/usr/bin/env python3
"""Evaluate frame-level uncertainty baselines against held-out frame errors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uncertainty_3dgs.metrics import evaluate_uncertainty
from uncertainty_3dgs.splits import load_frame_ids, load_frame_positions


SIGNALS = (
    "nearest-train-distance",
    "mean-train-distance",
    "temporal-index-distance",
    "uniform",
)
ERROR_METRICS = ("lpips", "negative-psnr", "one-minus-ssim", "score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        required=True,
        help="Nerfstudio-style transforms.json containing frame poses.",
    )
    parser.add_argument(
        "--split-json",
        required=True,
        help="Split JSON containing the seed train frames.",
    )
    parser.add_argument("--budget", type=int, required=True, help="Seed train budget.")
    parser.add_argument(
        "--scores",
        required=True,
        help="JSON score file from score_candidate_frames.py or a compatible row list.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--error-metric",
        choices=ERROR_METRICS,
        default="lpips",
        help="Observed error target. Higher values must mean worse frames.",
    )
    parser.add_argument(
        "--signals",
        choices=SIGNALS,
        nargs="+",
        default=("nearest-train-distance", "temporal-index-distance", "uniform"),
        help="Frame-level uncertainty signals to evaluate.",
    )
    parser.add_argument(
        "--score-signal-fields",
        nargs="*",
        default=(),
        help=(
            "Numeric fields from each score row to evaluate as additional "
            "frame-level uncertainty signals, for example mean_transmittance."
        ),
    )
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=None,
        help="Observed-error threshold for bad-frame AUROC/AUPRC.",
    )
    parser.add_argument(
        "--bad-quantile",
        type=float,
        default=None,
        help="Use this observed-error quantile as the bad-frame threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bad_threshold is not None and args.bad_quantile is not None:
        raise SystemExit("Use either --bad-threshold or --bad-quantile, not both.")

    frame_ids = load_frame_ids(args.frames)
    original_order = _normalized_order(frame_ids)
    positions = _normalized_positions(load_frame_positions(args.frames))
    train_frames = _load_train_frames(Path(args.split_json), args.budget)
    score_rows = _load_score_rows(Path(args.scores))
    signal_names = _combined_signal_names(args.signals, args.score_signal_fields)

    normalized_train = [_normalize_path(frame) for frame in train_frames]
    missing_train = [frame for frame in normalized_train if frame not in original_order]
    if missing_train:
        raise SystemExit(f"Train frames missing from frame manifest: {missing_train[:5]}")

    rows = _frame_rows(
        score_rows,
        error_metric=args.error_metric,
        built_in_signals=args.signals,
        score_signal_fields=args.score_signal_fields,
        original_order=original_order,
        positions=positions,
        train_frames=normalized_train,
    )
    if not rows:
        raise SystemExit("No scored rows matched frames in the manifest.")

    errors = [float(row["error"]) for row in rows]
    bad_threshold = args.bad_threshold
    if args.bad_quantile is not None:
        bad_threshold = _quantile(errors, args.bad_quantile)

    signal_summaries: dict[str, object] = {}
    for signal in signal_names:
        uncertainty = [float(row["signals"][signal]) for row in rows]
        signal_summaries[signal] = evaluate_uncertainty(
            uncertainty,
            errors,
            bad_threshold=bad_threshold,
        )

    output = {
        "metadata": {
            "frames": str(args.frames),
            "split_json": str(args.split_json),
            "scores": str(args.scores),
            "budget": args.budget,
            "error_metric": args.error_metric,
            "signals": signal_names,
            "built_in_signals": list(args.signals),
            "score_signal_fields": list(args.score_signal_fields),
            "count": len(rows),
            "bad_threshold": bad_threshold,
        },
        "signals": signal_summaries,
        "frames": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} with {len(rows)} scored frames")
    return 0


def _load_train_frames(path: Path, budget: int) -> list[str]:
    payload = _load_object_json(path)
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise SystemExit(f"Split JSON is missing object key 'splits': {path}")
    split = splits.get(str(budget))
    if not isinstance(split, dict):
        raise SystemExit(f"Split JSON is missing budget {budget}: {path}")
    train = split.get("train")
    if not isinstance(train, list):
        raise SystemExit(f"Split budget {budget} is missing list key 'train': {path}")
    return [str(frame) for frame in train]


def _load_score_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_object_json(path)
    rows = payload.get("scores", payload.get("frames"))
    if not isinstance(rows, list):
        raise SystemExit(f"Score JSON must contain a list key 'scores' or 'frames': {path}")
    output = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit(f"Score rows must be objects: {path}")
        output.append(row)
    return output


def _frame_rows(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    error_metric: str,
    built_in_signals: Sequence[str],
    score_signal_fields: Sequence[str],
    original_order: Mapping[str, int],
    positions: Mapping[str, Sequence[float]],
    train_frames: Sequence[str],
) -> list[dict[str, object]]:
    needs_positions = any(signal in {"nearest-train-distance", "mean-train-distance"} for signal in built_in_signals)
    if needs_positions:
        missing_positions = [frame for frame in train_frames if frame not in positions]
        if missing_positions:
            raise SystemExit(f"Train frames missing camera poses: {missing_positions[:5]}")

    rows: list[dict[str, object]] = []
    for row in score_rows:
        file_path = _row_file_path(row)
        normalized = _normalize_path(file_path)
        if normalized not in original_order:
            continue
        if needs_positions and normalized not in positions:
            continue
        error = _row_error(row, error_metric)
        row_signals = {
            signal: _signal_value(
                signal,
                normalized,
                train_frames=train_frames,
                original_order=original_order,
                positions=positions,
            )
            for signal in built_in_signals
        }
        score_signals = _score_signal_values(row, score_signal_fields)
        if score_signals is None:
            continue
        row_signals.update(score_signals)
        rows.append(
            {
                "file_path": file_path,
                "error": error,
                "signals": row_signals,
            }
        )
    return rows


def _signal_value(
    signal: str,
    frame: str,
    *,
    train_frames: Sequence[str],
    original_order: Mapping[str, int],
    positions: Mapping[str, Sequence[float]],
) -> float:
    if signal == "nearest-train-distance":
        return min(_euclidean_distance(positions[frame], positions[train]) for train in train_frames)
    if signal == "mean-train-distance":
        return _mean(_euclidean_distance(positions[frame], positions[train]) for train in train_frames)
    if signal == "temporal-index-distance":
        frame_index = original_order[frame]
        return float(min(abs(frame_index - original_order[train]) for train in train_frames))
    if signal == "uniform":
        return 1.0
    raise ValueError(f"Unknown signal: {signal}")


def _score_signal_values(
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for field in fields:
        if field not in row:
            return None
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values[field] = value
    return values


def _row_file_path(row: Mapping[str, Any]) -> str:
    for key in ("file_path", "path", "frame_id", "name"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise SystemExit("Score row is missing one of: file_path, path, frame_id, name")


def _row_error(row: Mapping[str, Any], error_metric: str) -> float:
    if error_metric == "lpips":
        return float(row["lpips"])
    if error_metric == "negative-psnr":
        return -float(row["psnr"])
    if error_metric == "one-minus-ssim":
        return 1.0 - float(row["ssim"])
    if error_metric == "score":
        return float(row["score"])
    raise ValueError(f"Unknown error metric: {error_metric}")


def _normalized_order(frame_ids: Iterable[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for index, frame_id in enumerate(frame_ids):
        normalized = _normalize_path(frame_id)
        if normalized in output:
            raise SystemExit(f"Duplicate normalized frame path: {normalized}")
        output[normalized] = index
    return output


def _combined_signal_names(
    built_in_signals: Sequence[str],
    score_signal_fields: Sequence[str],
) -> list[str]:
    names = [*built_in_signals, *score_signal_fields]
    output: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise SystemExit(f"Duplicate uncertainty signal name: {name}")
        seen.add(name)
        output.append(str(name))
    return output


def _normalized_positions(
    positions: Mapping[str, Sequence[float]],
) -> dict[str, Sequence[float]]:
    output: dict[str, Sequence[float]] = {}
    for frame, position in positions.items():
        normalized = _normalize_path(frame)
        if normalized in output:
            raise SystemExit(f"Duplicate normalized frame pose path: {normalized}")
        output[normalized] = position
    return output


def _normalize_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise SystemExit("--bad-quantile must be between 0 and 1.")
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("Frame positions must be 3D coordinates.")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return math.nan
    return sum(items) / len(items)


def _load_object_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON must contain an object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
