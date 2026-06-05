#!/usr/bin/env python3
"""Materialize budgeted Nerfstudio dataset directories from a split manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Source Nerfstudio dataset.")
    parser.add_argument("--split-json", required=True, help="Split manifest JSON.")
    parser.add_argument("--output-root", required=True, help="Output root for subsets.")
    parser.add_argument(
        "--budgets",
        nargs="*",
        default=None,
        help="Budgets to materialize. Defaults to every budget in split JSON.",
    )
    parser.add_argument(
        "--include-val-in-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include validation frames in transforms.json train frames.",
    )
    parser.add_argument(
        "--symlink-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink asset directories/files instead of copying them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir).resolve()
    split_path = Path(args.split_json).resolve()
    output_root = Path(args.output_root).resolve()

    payload = _load_json(source / "transforms.json")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit("Source transforms.json must contain a list-valued 'frames' key.")

    frame_by_path = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        file_path = frame.get("file_path")
        if isinstance(file_path, str):
            frame_by_path[_normalize_frame_path(file_path)] = frame

    split = _load_json(split_path)
    split_sets = split.get("splits")
    if not isinstance(split_sets, dict):
        raise SystemExit("Split JSON must contain an object-valued 'splits' key.")

    budgets = args.budgets or sorted(split_sets.keys(), key=lambda item: int(item))
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for budget in budgets:
        if str(budget) not in split_sets:
            raise SystemExit(f"Budget {budget!r} is missing from split JSON.")
        split_entry = split_sets[str(budget)]
        if not isinstance(split_entry, dict):
            raise SystemExit(f"Split entry for budget {budget!r} must be an object.")

        train_paths = _split_paths(split_entry, "train")
        val_paths = _split_paths(split_entry, "val")
        test_paths = _split_paths(split_entry, "test")
        included_train = list(train_paths)
        if args.include_val_in_train:
            included_train.extend(path for path in val_paths if path not in included_train)

        subset_dir = output_root / f"budget_{int(budget):03d}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy_assets(source, subset_dir, symlink=args.symlink_assets)

        train_frames = _frames_for_paths(frame_by_path, included_train, source)
        eval_frames = _frames_for_paths(frame_by_path, test_paths, source)
        source_metadata = payload.get("metadata")
        metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
        metadata.update(
            {
                "source_dir": str(source),
                "split_json": str(split_path),
                "budget": int(budget),
                "train_count": len(train_paths),
                "val_count": len(val_paths),
                "test_count": len(test_paths),
                "frames_in_transforms": len(train_frames),
                "include_val_in_train": args.include_val_in_train,
            }
        )

        subset_payload = dict(payload)
        subset_payload["frames"] = train_frames
        subset_payload["eval_filenames"] = test_paths
        subset_payload["metadata"] = metadata

        _write_json(subset_dir / "transforms.json", subset_payload)
        _write_json(
            subset_dir / "split_manifest.json",
            {
                "budget": int(budget),
                "train": train_paths,
                "val": val_paths,
                "test": test_paths,
                "transforms_frames": [frame["file_path"] for frame in train_frames],
                "eval_frames": [frame["file_path"] for frame in eval_frames],
            },
        )
        summaries.append(
            {
                "budget": int(budget),
                "output_dir": str(subset_dir),
                "train_count": len(train_paths),
                "val_count": len(val_paths),
                "test_count": len(test_paths),
                "transforms_frame_count": len(train_frames),
            }
        )

    _write_json(output_root / "materialization_summary.json", {"subsets": summaries})
    print(f"materialized {len(summaries)} budgeted datasets under {output_root}")
    for summary in summaries:
        print(
            f"budget {summary['budget']}: "
            f"{summary['transforms_frame_count']} transform frames, "
            f"test={summary['test_count']} -> {summary['output_dir']}"
        )
    return 0


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _split_paths(split_entry: dict[str, object], key: str) -> list[str]:
    values = split_entry.get(key)
    if not isinstance(values, list):
        raise SystemExit(f"Split entry must contain list-valued '{key}'.")
    return [str(value) for value in values]


def _frames_for_paths(
    frame_by_path: dict[str, dict[str, object]],
    paths: Iterable[str],
    source: Path,
) -> list[dict[str, object]]:
    selected = []
    missing = []
    for path in paths:
        normalized = _normalize_frame_path(path)
        frame = frame_by_path.get(normalized)
        if frame is None:
            missing.append(path)
            continue
        if not (source / normalized).exists():
            missing.append(path)
            continue
        selected.append(dict(frame))
    if missing:
        raise SystemExit(f"Split references missing frames: {missing[:10]}")
    return selected


def _normalize_frame_path(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def _link_or_copy_assets(source: Path, output: Path, *, symlink: bool) -> None:
    for child in source.iterdir():
        if child.name in {"transforms.json", "split_manifest.json", "materialization_summary.json"}:
            continue
        target = output / child.name
        if target.exists() or target.is_symlink():
            continue
        if symlink:
            os.symlink(child, target, target_is_directory=child.is_dir())
        elif child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


if __name__ == "__main__":
    raise SystemExit(main())
