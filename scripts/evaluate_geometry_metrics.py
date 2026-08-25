#!/usr/bin/env python3
"""Evaluate held-out world-space surface geometry from rendered and RGB-D depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # Lightweight unit-test environments may omit numeric extras.
    np = None  # type: ignore[assignment]

from evaluate_depth_metrics import (
    _accumulation,
    _load_object_json,
    _predicted_depth,
    _target_depth,
    _test_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-source", choices=("auto", "batch", "file"), default="auto")
    parser.add_argument("--cache-images", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--min-accumulation", type=float, default=0.0)
    parser.add_argument("--max-pixels-per-frame", type=int, default=20000)
    parser.add_argument("--max-points", type=int, default=500000)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.05, 0.10))
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    if np is None:
        raise SystemExit("NumPy is required; install the numeric or Modal dependencies.")
    load_config = Path(args.load_config)
    data_dir = Path(args.data)
    output_path = Path(args.output)
    payload = _load_object_json(data_dir / "transforms.json")
    depth_unit_scale = float(payload.get("depth_unit_scale_factor", 0.001))
    test_frames = _test_frames(payload)
    if args.max_frames is not None:
        test_frames = test_frames[: args.max_frames]
    if not test_frames:
        raise SystemExit("No held-out frames found.")

    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    def update_config(config: Any) -> Any:
        config.pipeline.datamanager.dataparser.data = data_dir
        if hasattr(config.pipeline.datamanager, "data"):
            config.pipeline.datamanager.data = None
        if hasattr(config.pipeline.datamanager, "cache_images"):
            config.pipeline.datamanager.cache_images = args.cache_images
        if hasattr(config.pipeline.datamanager.dataparser, "depth_unit_scale_factor"):
            config.pipeline.datamanager.dataparser.depth_unit_scale_factor = depth_unit_scale
        return config

    config, pipeline, checkpoint_path, checkpoint_step = eval_setup(
        load_config,
        test_mode="test",
        update_config_callback=update_config,
    )
    dataloader = pipeline.datamanager.fixed_indices_eval_dataloader
    if len(dataloader) != len(test_frames):
        raise SystemExit(
            f"Dataset has {len(test_frames)} test frames, but eval dataloader has {len(dataloader)}."
        )

    predicted_clouds: list[np.ndarray] = []
    reference_clouds: list[np.ndarray] = []
    frame_reports: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, (camera, batch) in enumerate(dataloader):
            frame = test_frames[index]
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            predicted, output_depth_key = _predicted_depth(outputs)
            target, target_source = _target_depth(
                batch,
                frame=frame,
                data_dir=data_dir,
                depth_unit_scale=depth_unit_scale,
                target_source=args.target_source,
                device=predicted.device,
                shape=predicted.shape,
            )
            accumulation = _accumulation(outputs, predicted.shape)
            prediction_valid = (
                predicted.isfinite()
                & (predicted >= args.min_depth)
                & (predicted <= args.max_depth)
            )
            if accumulation is not None and args.min_accumulation > 0:
                prediction_valid &= accumulation.isfinite() & (accumulation >= args.min_accumulation)
            target_valid = (
                target.isfinite()
                & (target >= args.min_depth)
                & (target <= args.max_depth)
            )
            prediction_pixels = _sample_flat_indices(prediction_valid, args.max_pixels_per_frame)
            target_pixels = _sample_flat_indices(target_valid, args.max_pixels_per_frame)
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
            predicted_points = backproject_depth(
                predicted.detach().cpu().numpy(),
                prediction_pixels.detach().cpu().numpy(),
                c2w,
                intrinsics_for_shape(payload, predicted.shape),
            )
            reference_points = backproject_depth(
                target.detach().cpu().numpy(),
                target_pixels.detach().cpu().numpy(),
                c2w,
                intrinsics_for_shape(payload, target.shape),
            )
            predicted_clouds.append(predicted_points)
            reference_clouds.append(reference_points)
            frame_reports.append(
                {
                    "file_path": frame["file_path"],
                    "target_source": target_source,
                    "output_depth_key": output_depth_key,
                    "predicted_points": len(predicted_points),
                    "reference_points": len(reference_points),
                }
            )
            print(
                f"{index + 1:03d}/{len(test_frames):03d} {frame['file_path']} "
                f"predicted={len(predicted_points)} reference={len(reference_points)}",
                flush=True,
            )

    predicted_cloud = prepare_cloud(predicted_clouds, args.max_points, args.voxel_size)
    reference_cloud = prepare_cloud(reference_clouds, args.max_points, args.voxel_size)
    summary = geometry_metrics(predicted_cloud, reference_cloud, args.thresholds)
    report = {
        "metadata": {
            "load_config": str(load_config),
            "data": str(data_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "method_name": getattr(config, "method_name", None),
            "test_frame_count": len(test_frames),
            "depth_unit_scale_factor": depth_unit_scale,
            "target_source": args.target_source,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "min_accumulation": args.min_accumulation,
            "max_pixels_per_frame": args.max_pixels_per_frame,
            "max_points": args.max_points,
            "voxel_size_m": args.voxel_size,
            "thresholds_m": list(args.thresholds),
            "coordinate_convention": "Nerfstudio OpenGL camera-to-world; depth backprojected along -z",
        },
        "summary": summary,
        "frames": frame_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} with {len(predicted_cloud)} predicted and {len(reference_cloud)} reference points")
    return 0


def intrinsics_for_shape(payload: dict[str, Any], shape: Sequence[int]) -> dict[str, float]:
    height, width = int(shape[0]), int(shape[1])
    source_width = float(payload.get("w", width))
    source_height = float(payload.get("h", height))
    scale_x = width / source_width
    scale_y = height / source_height
    return {
        "fx": float(payload["fl_x"]) * scale_x,
        "fy": float(payload.get("fl_y", payload["fl_x"])) * scale_y,
        "cx": float(payload["cx"]) * scale_x,
        "cy": float(payload["cy"]) * scale_y,
    }


def backproject_depth(
    depth: np.ndarray,
    flat_indices: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: dict[str, float],
) -> np.ndarray:
    """Backproject selected image pixels using Nerfstudio's OpenGL camera convention."""

    if flat_indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    height, width = depth.shape
    rows, columns = np.unravel_index(flat_indices.astype(np.int64), (height, width))
    z_depth = depth[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - intrinsics["cx"]) * z_depth / intrinsics["fx"]
    y = -(rows.astype(np.float64) - intrinsics["cy"]) * z_depth / intrinsics["fy"]
    camera_points = np.stack((x, y, -z_depth, np.ones_like(z_depth)), axis=1)
    return (camera_to_world @ camera_points.T).T[:, :3]


def prepare_cloud(clouds: Sequence[np.ndarray], max_points: int, voxel_size: float) -> np.ndarray:
    nonempty = [cloud for cloud in clouds if len(cloud)]
    if not nonempty:
        raise SystemExit("No valid surface points were generated.")
    cloud = np.concatenate(nonempty, axis=0)
    if len(cloud) > max_points:
        indices = np.linspace(0, len(cloud) - 1, max_points, dtype=np.int64)
        cloud = cloud[indices]
    if voxel_size > 0:
        voxel_keys = np.floor(cloud / voxel_size).astype(np.int64)
        _, indices = np.unique(voxel_keys, axis=0, return_index=True)
        cloud = cloud[np.sort(indices)]
    return cloud


def geometry_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, float | int]:
    """Compute symmetric surface distances and thresholded precision/recall/F-score."""

    from scipy.spatial import cKDTree

    if len(predicted) == 0 or len(reference) == 0:
        raise ValueError("Both point clouds must be non-empty.")
    pred_to_ref = cKDTree(reference).query(predicted, workers=-1)[0]
    ref_to_pred = cKDTree(predicted).query(reference, workers=-1)[0]
    output: dict[str, float | int] = {
        "predicted_point_count": len(predicted),
        "reference_point_count": len(reference),
        "accuracy_mean_m": float(np.mean(pred_to_ref)),
        "accuracy_median_m": float(np.median(pred_to_ref)),
        "completeness_mean_m": float(np.mean(ref_to_pred)),
        "completeness_median_m": float(np.median(ref_to_pred)),
        "chamfer_l1_mean_m": float((np.mean(pred_to_ref) + np.mean(ref_to_pred)) / 2.0),
    }
    for threshold in thresholds:
        precision = float(np.mean(pred_to_ref <= threshold))
        recall = float(np.mean(ref_to_pred <= threshold))
        fscore = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        suffix = f"{int(round(threshold * 100)):02d}cm"
        output[f"precision_{suffix}"] = precision
        output[f"recall_{suffix}"] = recall
        output[f"fscore_{suffix}"] = fscore
    return output


def _sample_flat_indices(valid: Any, max_pixels: int) -> Any:
    indices = valid.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    if indices.numel() > max_pixels:
        positions = np.linspace(0, indices.numel() - 1, max_pixels, dtype=np.int64)
        indices = indices[indices.new_tensor(positions)]
    return indices


def _validate_args(args: argparse.Namespace) -> None:
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise SystemExit("Depth bounds must be positive and increasing.")
    if args.max_pixels_per_frame <= 0 or args.max_points <= 0:
        raise SystemExit("Point sampling limits must be positive.")
    if args.voxel_size < 0:
        raise SystemExit("--voxel-size cannot be negative.")
    if not args.thresholds or any(threshold <= 0 for threshold in args.thresholds):
        raise SystemExit("Geometry thresholds must be positive.")


if __name__ == "__main__":
    raise SystemExit(main())
