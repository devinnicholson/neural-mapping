#!/usr/bin/env python3
"""Materialize a Nerfstudio dataset for scoring active-selection candidates."""

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
    parser.add_argument("--base-split", required=True, help="Seed split JSON.")
    parser.add_argument("--base-budget", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-val-in-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the training materialization by including validation frames in transforms.json.",
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
    base_split_path = Path(args.base_split).resolve()
    output = Path(args.output_dir).resolve()

    payload = _load_json(source / "transforms.json")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit("Source transforms.json must contain a list-valued 'frames' key.")

    frame_by_path = {}
    frame_paths = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        file_path = frame.get("file_path")
        if isinstance(file_path, str):
            normalized = _normalize_frame_path(file_path)
            frame_by_path[normalized] = frame
            frame_paths.append(file_path)

    split = _load_json(base_split_path)
    split_sets = split.get("splits")
    if not isinstance(split_sets, dict):
        raise SystemExit("Base split JSON must contain an object-valued 'splits' key.")
    split_entry = split_sets.get(str(args.base_budget))
    if not isinstance(split_entry, dict):
        raise SystemExit(f"Base split JSON is missing budget {args.base_budget}.")

    train_paths = _split_paths(split_entry, "train")
    val_paths = _split_paths(split_entry, "val")
    test_paths = _split_paths(split_entry, "test")
    included_train = list(train_paths)
    if args.include_val_in_train:
        included_train.extend(path for path in val_paths if path not in included_train)

    excluded = set(train_paths) | set(val_paths) | set(test_paths)
    candidates = [path for path in frame_paths if path not in excluded]
    if not candidates:
        raise SystemExit("No candidate frames remain after excluding train/val/test frames.")

    output.mkdir(parents=True, exist_ok=True)
    _link_or_copy_assets(source, output, symlink=args.symlink_assets)

    transform_paths = _unique_preserving_order(
        [*included_train, *val_paths, *candidates]
    )
    transform_frames = _frames_for_paths(frame_by_path, transform_paths, source)
    source_metadata = payload.get("metadata")
    metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    metadata.update(
        {
            "source_dir": str(source),
            "base_split": str(base_split_path),
            "base_budget": args.base_budget,
            "candidate_count": len(candidates),
            "train_count": len(train_paths),
            "val_count": len(val_paths),
            "test_count": len(test_paths),
            "frames_in_transforms": len(transform_frames),
            "include_val_in_train": args.include_val_in_train,
        }
    )

    subset_payload = dict(payload)
    subset_payload["frames"] = transform_frames
    subset_payload["train_filenames"] = included_train
    subset_payload["val_filenames"] = val_paths
    subset_payload["test_filenames"] = candidates
    subset_payload["metadata"] = metadata
    _write_json(output / "transforms.json", subset_payload)
    _write_json(
        output / "candidate_manifest.json",
        {
            "source_dir": str(source),
            "base_split": str(base_split_path),
            "base_budget": args.base_budget,
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
            "transforms_frames": [frame["file_path"] for frame in transform_frames],
            "candidates": candidates,
        },
    )
    print(
        f"materialized {len(candidates)} candidate eval frames under {output}; "
        f"frames in transforms={len(transform_frames)}"
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


def _unique_preserving_order(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        output.append(path)
    return output


def _link_or_copy_assets(source: Path, output: Path, *, symlink: bool) -> None:
    for child in source.iterdir():
        if child.name in {
            "transforms.json",
            "split_manifest.json",
            "materialization_summary.json",
            "candidate_manifest.json",
        }:
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
