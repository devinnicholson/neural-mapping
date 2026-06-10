#!/usr/bin/env python3
"""Download and convert a TUM RGB-D sequence to Nerfstudio format."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Sequence


TUM_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"
SEQUENCE_URLS = {
    "freiburg1_desk": f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "fr1_desk": f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "freiburg1_xyz": f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
    "fr1_xyz": f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
    "freiburg1_room": f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_room.tgz",
    "fr1_room": f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_room.tgz",
}

FR1_INTRINSICS = {
    "w": 640,
    "h": 480,
    "fl_x": 517.3,
    "fl_y": 516.5,
    "cx": 318.6,
    "cy": 255.3,
    "k1": 0.2624,
    "k2": -0.9531,
    "p1": -0.0054,
    "p2": 0.0026,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        default="freiburg1_desk",
        help="TUM sequence key. Known: freiburg1_desk, freiburg1_xyz, freiburg1_room.",
    )
    parser.add_argument("--url", default=None, help="Override download URL.")
    parser.add_argument("--raw-root", required=True, help="Directory for downloaded/extracted TUM data.")
    parser.add_argument("--output-dir", required=True, help="Nerfstudio dataset output directory.")
    parser.add_argument("--scene-name", default=None, help="Scene name written to metadata.")
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--match-tolerance", type=float, default=0.02)
    parser.add_argument("--copy-assets", action="store_true", help="Copy rgb/depth dirs instead of symlinking.")
    parser.add_argument(
        "--no-opengl-conversion",
        action="store_true",
        help="Keep TUM camera axes instead of flipping y/z for Nerfstudio/OpenGL convention.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_key = args.sequence.lower()
    url = args.url or SEQUENCE_URLS.get(sequence_key)
    if url is None:
        raise SystemExit(
            f"Unknown sequence {args.sequence!r}; pass --url or use one of "
            f"{sorted(SEQUENCE_URLS)}."
        )

    raw_root = Path(args.raw_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = raw_root / Path(url).name
    if not archive_path.exists():
        _download(url, archive_path)

    sequence_dir = _extract_if_needed(archive_path, raw_root)
    pairs = build_associations(sequence_dir, tolerance=args.match_tolerance)
    if args.frame_stride > 1:
        pairs = pairs[:: args.frame_stride]
    if args.max_frames is not None and args.max_frames > 0:
        pairs = pairs[: args.max_frames]
    if len(pairs) < 8:
        raise SystemExit(f"Only {len(pairs)} associated RGB-D pose frames found.")

    _link_or_copy(sequence_dir / "rgb", output_dir / "rgb", copy=args.copy_assets)
    _link_or_copy(sequence_dir / "depth", output_dir / "depth", copy=args.copy_assets)

    frames = []
    for pair in pairs:
        matrix = pose_to_matrix(
            pair["groundtruth"][1:],
            opengl=not args.no_opengl_conversion,
        )
        frames.append(
            {
                "file_path": pair["rgb"][1],
                "depth_file_path": pair["depth"][1],
                "transform_matrix": matrix,
                "timestamp": pair["rgb"][0],
                "depth_timestamp": pair["depth"][0],
                "pose_timestamp": pair["groundtruth"][0],
            }
        )

    payload = {
        "camera_model": "OPENCV",
        **FR1_INTRINSICS,
        "depth_unit_scale_factor": 1.0 / 5000.0,
        "frames": frames,
        "metadata": {
            "dataset": "tum_rgbd",
            "sequence": args.sequence,
            "scene_name": args.scene_name or args.sequence,
            "source_url": url,
            "raw_sequence_dir": str(sequence_dir),
            "associated_frame_count": len(pairs),
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "match_tolerance": args.match_tolerance,
            "pose_convention": "tum_rgbd_with_yz_flip" if not args.no_opengl_conversion else "tum_rgbd_raw",
        },
    }
    (output_dir / "transforms.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "association_summary.json").write_text(
        json.dumps(
            {
                "sequence": args.sequence,
                "source_url": url,
                "raw_sequence_dir": str(sequence_dir),
                "output_dir": str(output_dir),
                "associated_frame_count": len(pairs),
                "first_frame": frames[0]["file_path"],
                "last_frame": frames[-1]["file_path"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"wrote {output_dir / 'transforms.json'} with {len(frames)} RGB-D frames "
        f"from {sequence_dir}"
    )
    return 0


def build_associations(sequence_dir: Path, *, tolerance: float) -> list[dict[str, tuple]]:
    rgb = _read_timestamp_file(sequence_dir / "rgb.txt")
    depth = _read_timestamp_file(sequence_dir / "depth.txt")
    groundtruth = _read_timestamp_file(sequence_dir / "groundtruth.txt", min_fields=8)

    associations = []
    for rgb_row in rgb:
        depth_row = _nearest_row(rgb_row[0], depth, tolerance=tolerance)
        pose_row = _nearest_row(rgb_row[0], groundtruth, tolerance=tolerance)
        if depth_row is None or pose_row is None:
            continue
        associations.append(
            {
                "rgb": (rgb_row[0], rgb_row[1]),
                "depth": (depth_row[0], depth_row[1]),
                "groundtruth": (
                    pose_row[0],
                    float(pose_row[1]),
                    float(pose_row[2]),
                    float(pose_row[3]),
                    float(pose_row[4]),
                    float(pose_row[5]),
                    float(pose_row[6]),
                    float(pose_row[7]),
                ),
            }
        )
    return associations


def pose_to_matrix(values: Sequence[float], *, opengl: bool = True) -> list[list[float]]:
    if len(values) != 7:
        raise ValueError("pose must contain tx ty tz qx qy qz qw")
    tx, ty, tz, qx, qy, qz, qw = [float(value) for value in values]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("quaternion norm is zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rotation = [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
    if opengl:
        for row in rotation:
            row[1] *= -1.0
            row[2] *= -1.0
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], tx],
        [rotation[1][0], rotation[1][1], rotation[1][2], ty],
        [rotation[2][0], rotation[2][1], rotation[2][2], tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _download(url: str, output_path: Path) -> None:
    print(f"downloading {url} -> {output_path}", flush=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_if_needed(archive_path: Path, raw_root: Path) -> Path:
    candidates = [path for path in raw_root.iterdir() if path.is_dir() and (path / "rgb.txt").exists()]
    if candidates:
        return sorted(candidates)[0]
    print(f"extracting {archive_path}", flush=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(raw_root, filter="data")
    candidates = [path for path in raw_root.iterdir() if path.is_dir() and (path / "rgb.txt").exists()]
    if not candidates:
        raise FileNotFoundError(f"No extracted TUM sequence with rgb.txt found under {raw_root}")
    return sorted(candidates)[0]


def _read_timestamp_file(path: Path, *, min_fields: int = 2) -> list[tuple]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < min_fields:
            continue
        rows.append((float(parts[0]), *parts[1:]))
    return rows


def _nearest_row(timestamp: float, rows: Sequence[tuple], *, tolerance: float) -> tuple | None:
    timestamps = [row[0] for row in rows]
    index = bisect.bisect_left(timestamps, timestamp)
    candidates = []
    if index < len(rows):
        candidates.append(rows[index])
    if index > 0:
        candidates.append(rows[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda row: abs(row[0] - timestamp))
    if abs(nearest[0] - timestamp) > tolerance:
        return None
    return nearest


def _link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    if target.exists() or target.is_symlink():
        return
    if copy:
        shutil.copytree(source, target)
    else:
        os.symlink(source, target, target_is_directory=True)


if __name__ == "__main__":
    raise SystemExit(main())
