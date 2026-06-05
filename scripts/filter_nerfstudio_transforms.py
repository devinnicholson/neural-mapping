#!/usr/bin/env python3
"""Create a Nerfstudio dataset view with only frames whose images exist."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Source Nerfstudio dataset.")
    parser.add_argument("--output-dir", required=True, help="Filtered dataset output.")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=8,
        help="Fail if fewer than this many frames remain.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy top-level assets instead of symlinking them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input_dir).resolve()
    output = Path(args.output_dir).resolve()
    transforms_path = source / "transforms.json"

    if not transforms_path.exists():
        raise SystemExit(f"Missing transforms.json: {transforms_path}")

    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit("transforms.json must contain a list-valued 'frames' key.")

    kept = []
    missing = []
    for frame in frames:
        if not isinstance(frame, dict):
            missing.append({"reason": "frame entry is not an object", "frame": frame})
            continue
        file_path = frame.get("file_path")
        if not isinstance(file_path, str):
            missing.append({"reason": "missing file_path", "frame": frame})
            continue
        resolved = (source / file_path).resolve()
        if resolved.exists():
            kept.append(frame)
        else:
            missing.append({"file_path": file_path})

    if len(kept) < args.min_frames:
        raise SystemExit(
            f"Only {len(kept)} frames remain after filtering; "
            f"minimum required is {args.min_frames}."
        )

    for child in source.iterdir():
        if child.name == "transforms.json":
            continue
        target = output / child.name
        if target.exists() or target.is_symlink():
            continue
        if args.copy:
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        else:
            os.symlink(child, target, target_is_directory=child.is_dir())

    filtered_payload = dict(payload)
    filtered_payload["frames"] = kept
    (output / "transforms.json").write_text(
        json.dumps(filtered_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "missing_frames.json").write_text(
        json.dumps(
            {
                "input_dir": str(source),
                "output_dir": str(output),
                "source_frame_count": len(frames),
                "kept_frame_count": len(kept),
                "missing_frame_count": len(missing),
                "missing": missing,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"wrote filtered dataset to {output} "
        f"with {len(kept)}/{len(frames)} frames kept; "
        f"missing={len(missing)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

