#!/usr/bin/env python3
"""Download and convert an ICL-NUIM living-room trajectory to Nerfstudio format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import TypeVar

from prepare_tum_rgbd import pose_to_matrix


DATASET_PAGE = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"
BASE_URL = "http://www.doc.ic.ac.uk/~ahanda"
POSE_BASE_URL = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC"
TRAJECTORIES = {
    f"lr_kt{index}": {
        "clean_url": f"{BASE_URL}/living_room_traj{index}_frei_png.tar.gz",
        "noisy_url": f"{BASE_URL}/living_room_traj{index}n_frei_png.tar.gz",
        "pose_url": f"{POSE_BASE_URL}/livingRoom{index}.gt.freiburg",
    }
    for index in range(4)
}

INTRINSICS = {
    "w": 640,
    "h": 480,
    "fl_x": 481.20,
    "fl_y": 480.00,
    "cx": 319.50,
    "cy": 239.50,
}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=sorted(TRAJECTORIES), default="lr_kt0")
    parser.add_argument("--variant", choices=("clean", "noisy"), default="clean")
    parser.add_argument("--url", default=None, help="Override RGB-D archive URL.")
    parser.add_argument("--pose-url", default=None, help="Override Freiburg pose URL.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--copy-assets", action="store_true")
    parser.add_argument(
        "--no-opengl-conversion",
        action="store_true",
        help="Keep Freiburg camera axes instead of flipping y/z for Nerfstudio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive.")

    metadata = TRAJECTORIES[args.trajectory]
    archive_url = args.url or str(metadata[f"{args.variant}_url"])
    pose_url = args.pose_url or str(metadata["pose_url"])
    raw_root = Path(args.raw_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = raw_root / Path(archive_url).name
    pose_path = raw_root / Path(pose_url).name
    if not archive_path.exists():
        _download(archive_url, archive_path)
    if not pose_path.exists():
        _download(pose_url, pose_path)

    sequence_dir = _extract_sequence(archive_path, raw_root)
    image_pairs = indexed_image_pairs(sequence_dir)
    poses = load_indexed_poses(pose_path)
    aligned = [(index, rgb, depth, poses[index]) for index, rgb, depth in image_pairs if index in poses]
    aligned = aligned[:: args.frame_stride]
    if args.max_frames is not None:
        aligned = uniform_subsample(aligned, args.max_frames)
    if len(aligned) < 8:
        raise SystemExit(f"Only {len(aligned)} indexed RGB-D pose frames found.")

    rgb_dir = _find_asset_dir(sequence_dir, "rgb")
    depth_dir = _find_asset_dir(sequence_dir, "depth")
    _link_or_copy(rgb_dir, output_dir / "rgb", copy=args.copy_assets)
    _link_or_copy(depth_dir, output_dir / "depth", copy=args.copy_assets)

    frames = []
    for frame_index, rgb_path, depth_path, pose in aligned:
        frames.append(
            {
                "file_path": f"rgb/{rgb_path.name}",
                "depth_file_path": f"depth/{depth_path.name}",
                "transform_matrix": pose_to_matrix(
                    pose,
                    opengl=not args.no_opengl_conversion,
                ),
                "frame_index": frame_index,
            }
        )

    scene_name = args.scene_name or f"icl_nuim_{args.trajectory}_{args.variant}"
    payload = {
        "camera_model": "OPENCV",
        **INTRINSICS,
        "depth_unit_scale_factor": 1.0 / 5000.0,
        "frames": frames,
        "metadata": {
            "dataset": "icl_nuim",
            "trajectory": args.trajectory,
            "variant": args.variant,
            "scene_name": scene_name,
            "dataset_page": DATASET_PAGE,
            "source_url": archive_url,
            "pose_url": pose_url,
            "license": "CC BY 3.0",
            "raw_sequence_dir": str(sequence_dir),
            "available_rgbd_pairs": len(image_pairs),
            "available_poses": len(poses),
            "associated_frame_count": len(frames),
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "pose_convention": (
                "freiburg_camera_to_world_with_yz_flip"
                if not args.no_opengl_conversion
                else "freiburg_camera_to_world_raw"
            ),
        },
    }
    transforms_path = output_dir / "transforms.json"
    transforms_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {
        "dataset": "icl_nuim",
        "trajectory": args.trajectory,
        "variant": args.variant,
        "source_url": archive_url,
        "pose_url": pose_url,
        "raw_sequence_dir": str(sequence_dir),
        "output_dir": str(output_dir),
        "available_rgbd_pairs": len(image_pairs),
        "available_poses": len(poses),
        "associated_frame_count": len(frames),
        "first_frame_index": frames[0]["frame_index"],
        "last_frame_index": frames[-1]["frame_index"],
    }
    (output_dir / "association_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {transforms_path} with {len(frames)} indexed RGB-D frames")
    return 0


def indexed_image_pairs(sequence_dir: Path) -> list[tuple[int, Path, Path]]:
    """Return RGB/depth files paired by their integer filename stems."""

    rgb_dir = _find_asset_dir(sequence_dir, "rgb")
    depth_dir = _find_asset_dir(sequence_dir, "depth")
    rgb = {_integer_stem(path): path for path in rgb_dir.glob("*.png")}
    depth = {_integer_stem(path): path for path in depth_dir.glob("*.png")}
    common = sorted(index for index in rgb.keys() & depth.keys() if index is not None)
    return [(int(index), rgb[index], depth[index]) for index in common]


def load_indexed_poses(path: Path) -> dict[int, tuple[float, ...]]:
    """Load ``frame tx ty tz qx qy qz qw`` rows keyed by frame index."""

    poses: dict[int, tuple[float, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        frame_index = int(float(fields[0]))
        poses[frame_index] = tuple(float(value) for value in fields[1:8])
    return poses


def uniform_subsample(values: list[T], max_count: int) -> list[T]:
    """Select at most ``max_count`` values while retaining both trajectory endpoints."""

    if len(values) <= max_count:
        return list(values)
    if max_count == 1:
        return [values[0]]
    denominator = max_count - 1
    last_index = len(values) - 1
    indices = [(step * last_index) // denominator for step in range(max_count)]
    return [values[index] for index in indices]


def _integer_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def _find_asset_dir(root: Path, name: str) -> Path:
    candidates = sorted(path for path in root.rglob(name) if path.is_dir())
    for candidate in candidates:
        if any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(f"No {name}/ directory containing PNGs found under {root}")


def _download(url: str, output_path: Path) -> None:
    print(f"downloading {url} -> {output_path}", flush=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_sequence(archive_path: Path, raw_root: Path) -> Path:
    marker = raw_root / f".{archive_path.name}.extracted"
    if not marker.exists():
        print(f"extracting {archive_path}", flush=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(raw_root, filter="data")
        marker.touch()
    _find_asset_dir(raw_root, "rgb")
    _find_asset_dir(raw_root, "depth")
    return raw_root


def _link_or_copy(source: Path, destination: Path, *, copy: bool) -> None:
    if destination.is_symlink() or destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if copy:
        shutil.copytree(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)


if __name__ == "__main__":
    raise SystemExit(main())
