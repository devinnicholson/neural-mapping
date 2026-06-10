#!/usr/bin/env python3
"""Evaluate rendered depth against depth images in a Nerfstudio dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


DEPTH_KEYS = ("depth", "expected_depth", "median_depth")
BATCH_DEPTH_KEYS = ("depth_image", "depth", "depths")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-config", required=True, help="Trained Nerfstudio config.yml.")
    parser.add_argument("--data", required=True, help="Nerfstudio dataset directory used for evaluation.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--target-source",
        choices=("auto", "batch", "file"),
        default="auto",
        help="Use dataloader depth when available, load depth_file_path directly, or auto-detect.",
    )
    parser.add_argument(
        "--cache-images",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Nerfstudio image cache device for evaluation.",
    )
    parser.add_argument("--min-depth", type=float, default=0.05, help="Minimum valid target depth in meters.")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Maximum valid target depth in meters.")
    parser.add_argument(
        "--min-accumulation",
        type=float,
        default=0.0,
        help="Optional rendered accumulation threshold for valid predicted depth pixels.",
    )
    parser.add_argument(
        "--max-pixels-per-frame",
        type=int,
        default=200000,
        help="Deterministically sample at most this many valid pixels per frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on held-out frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_pixels_per_frame <= 0:
        raise SystemExit("--max-pixels-per-frame must be positive.")
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise SystemExit("--min-depth must be positive and lower than --max-depth.")
    if args.min_accumulation < 0:
        raise SystemExit("--min-accumulation must be non-negative.")

    load_config = Path(args.load_config)
    data_dir = Path(args.data)
    output_path = Path(args.output)
    payload = _load_object_json(data_dir / "transforms.json")
    depth_unit_scale = float(payload.get("depth_unit_scale_factor", 0.001))
    test_frames = _test_frames(payload)
    if args.max_frames is not None:
        test_frames = test_frames[: args.max_frames]
    if not test_frames:
        raise SystemExit(f"No test frames found in {data_dir / 'transforms.json'}")

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
            f"Dataset has {len(test_frames)} test frames, but eval dataloader has "
            f"{len(dataloader)} frames."
        )

    raw_accumulator = DepthAccumulator()
    aligned_accumulator = DepthAccumulator()
    frames = []
    target_source_counts: dict[str, int] = {}
    output_depth_keys: dict[str, int] = {}

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
            valid = _valid_mask(
                predicted,
                target,
                accumulation=accumulation,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                min_accumulation=args.min_accumulation,
            )
            indices = _sample_indices(valid, args.max_pixels_per_frame)
            if indices.numel() == 0:
                frame_report = {
                    "file_path": frame["file_path"],
                    "depth_file_path": frame.get("depth_file_path"),
                    "target_source": target_source,
                    "output_depth_key": output_depth_key,
                    "valid_pixels": 0,
                    "valid_fraction": 0.0,
                }
                frames.append(frame_report)
                continue

            predicted_values = predicted.reshape(-1)[indices]
            target_values = target.reshape(-1)[indices]
            raw = depth_metrics(predicted_values, target_values)
            aligned_prediction, median_scale = median_align(predicted_values, target_values)
            aligned = depth_metrics(aligned_prediction, target_values)
            raw_accumulator.update(predicted_values, target_values)
            aligned_accumulator.update(aligned_prediction, target_values)
            target_source_counts[target_source] = target_source_counts.get(target_source, 0) + 1
            output_depth_keys[output_depth_key] = output_depth_keys.get(output_depth_key, 0) + 1
            valid_pixels = int(indices.numel())
            total_pixels = int(valid.numel())
            frames.append(
                {
                    "file_path": frame["file_path"],
                    "depth_file_path": frame.get("depth_file_path"),
                    "target_source": target_source,
                    "output_depth_key": output_depth_key,
                    "valid_pixels": valid_pixels,
                    "valid_fraction": valid_pixels / total_pixels if total_pixels else 0.0,
                    "median_alignment_scale": median_scale,
                    "raw": raw,
                    "median_aligned": aligned,
                }
            )
            print(
                f"{index + 1:03d}/{len(test_frames):03d} {frame['file_path']} "
                f"pixels={valid_pixels} raw_abs_rel={raw['abs_rel']:.4f} "
                f"aligned_abs_rel={aligned['abs_rel']:.4f}",
                flush=True,
            )

    if raw_accumulator.count == 0:
        raise SystemExit("No valid depth pixels were evaluated.")

    output = {
        "metadata": {
            "load_config": str(load_config),
            "data": str(data_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "method_name": getattr(config, "method_name", None),
            "depth_unit_scale_factor": depth_unit_scale,
            "target_source": args.target_source,
            "target_source_counts": target_source_counts,
            "output_depth_keys": output_depth_keys,
            "test_frame_count": len(test_frames),
            "evaluated_frame_count": sum(1 for frame in frames if frame.get("valid_pixels", 0) > 0),
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "min_accumulation": args.min_accumulation,
            "max_pixels_per_frame": args.max_pixels_per_frame,
            "cache_images": args.cache_images,
        },
        "summary": {
            "raw": raw_accumulator.summary(),
            "median_aligned": aligned_accumulator.summary(),
        },
        "frames": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} with {raw_accumulator.count} sampled depth pixels")
    return 0


@dataclass
class DepthAccumulator:
    count: int = 0
    sum_abs_rel: float = 0.0
    sum_sq_rel: float = 0.0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    delta1: int = 0
    delta2: int = 0
    delta3: int = 0
    frame_metrics: list[dict[str, float]] = field(default_factory=list)

    def update(self, prediction: Any, target: Any) -> None:
        metrics = depth_metrics(prediction, target)
        count = int(metrics["count"])
        self.count += count
        self.sum_abs_rel += float(metrics["abs_rel"]) * count
        self.sum_sq_rel += float(metrics["sq_rel"]) * count
        self.sum_abs += float(metrics["mae"]) * count
        self.sum_sq += float(metrics["rmse"]) ** 2 * count
        self.delta1 += int(round(float(metrics["delta1"]) * count))
        self.delta2 += int(round(float(metrics["delta2"]) * count))
        self.delta3 += int(round(float(metrics["delta3"]) * count))
        self.frame_metrics.append(metrics)

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0}
        return {
            "count": self.count,
            "abs_rel": self.sum_abs_rel / self.count,
            "sq_rel": self.sum_sq_rel / self.count,
            "mae": self.sum_abs / self.count,
            "rmse": math.sqrt(self.sum_sq / self.count),
            "delta1": self.delta1 / self.count,
            "delta2": self.delta2 / self.count,
            "delta3": self.delta3 / self.count,
            "frame_count": len(self.frame_metrics),
            "mean_frame_abs_rel": _mean(metric["abs_rel"] for metric in self.frame_metrics),
            "mean_frame_rmse": _mean(metric["rmse"] for metric in self.frame_metrics),
        }


def depth_metrics(prediction: Any, target: Any) -> dict[str, float]:
    import torch

    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    valid = prediction.isfinite() & target.isfinite() & (prediction > 0) & (target > 0)
    prediction = prediction[valid]
    target = target[valid]
    if prediction.numel() == 0:
        return _empty_metrics()
    abs_error = torch.abs(prediction - target)
    sq_error = (prediction - target) ** 2
    ratio = torch.maximum(prediction / target, target / prediction)
    return {
        "count": float(prediction.numel()),
        "abs_rel": float((abs_error / target).mean().detach().cpu().item()),
        "sq_rel": float((sq_error / target).mean().detach().cpu().item()),
        "mae": float(abs_error.mean().detach().cpu().item()),
        "rmse": float(torch.sqrt(sq_error.mean()).detach().cpu().item()),
        "delta1": float((ratio < 1.25).float().mean().detach().cpu().item()),
        "delta2": float((ratio < 1.25**2).float().mean().detach().cpu().item()),
        "delta3": float((ratio < 1.25**3).float().mean().detach().cpu().item()),
    }


def median_align(prediction: Any, target: Any) -> tuple[Any, float]:
    import torch

    prediction = prediction.detach().float()
    target = target.detach().float()
    valid = prediction.isfinite() & target.isfinite() & (prediction > 0) & (target > 0)
    if int(valid.sum().detach().cpu().item()) == 0:
        return prediction, math.nan
    pred_median = torch.median(prediction[valid])
    target_median = torch.median(target[valid])
    if float(pred_median.detach().cpu().item()) <= 0:
        return prediction, math.nan
    scale = target_median / pred_median
    return prediction * scale, float(scale.detach().cpu().item())


def _predicted_depth(outputs: dict[str, Any]) -> tuple[Any, str]:
    for key in DEPTH_KEYS:
        if key not in outputs:
            continue
        depth = outputs[key].detach().float()
        if depth.shape[-1:] == (1,):
            depth = depth.squeeze(-1)
        return depth, key
    raise SystemExit(f"Model outputs are missing a depth-like key. Tried: {DEPTH_KEYS}")


def _accumulation(outputs: dict[str, Any], shape: Sequence[int]) -> Any | None:
    if "accumulation" not in outputs:
        return None
    accumulation = outputs["accumulation"].detach().float()
    if accumulation.shape[-1:] == (1,):
        accumulation = accumulation.squeeze(-1)
    if tuple(accumulation.shape) != tuple(shape):
        return None
    return accumulation


def _target_depth(
    batch: dict[str, Any],
    *,
    frame: dict[str, Any],
    data_dir: Path,
    depth_unit_scale: float,
    target_source: str,
    device: Any,
    shape: Sequence[int],
) -> tuple[Any, str]:
    if target_source in {"auto", "batch"}:
        depth = _batch_depth(batch, depth_unit_scale=depth_unit_scale)
        if depth is not None:
            return _resize_to_shape(depth.to(device).detach().float(), shape), "batch"
        if target_source == "batch":
            raise SystemExit("Batch depth target requested, but eval batch has no depth tensor.")
    if target_source not in {"auto", "file"}:
        raise ValueError(f"Unknown target source: {target_source}")
    return _file_depth(frame, data_dir, depth_unit_scale, device=device, shape=shape), "file"


def _batch_depth(batch: dict[str, Any], depth_unit_scale: float) -> Any | None:
    for key in BATCH_DEPTH_KEYS:
        value = batch.get(key)
        if value is None:
            continue
        if hasattr(value, "detach"):
            if value.shape[-1:] == (1,):
                value = value.squeeze(-1)
            value = value.detach().float()
            finite_positive = value[value.isfinite() & (value > 0)]
            if finite_positive.numel() > 0:
                median_depth = float(finite_positive.median().detach().cpu().item())
                if median_depth > 100.0:
                    value = value * depth_unit_scale
            return value
    return None


def _file_depth(
    frame: dict[str, Any],
    data_dir: Path,
    depth_unit_scale: float,
    *,
    device: Any,
    shape: Sequence[int],
) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    depth_path_value = frame.get("depth_file_path")
    if not isinstance(depth_path_value, str):
        raise SystemExit(f"Frame is missing depth_file_path: {frame.get('file_path')}")
    depth_path = data_dir / _normalize_path(depth_path_value)
    if not depth_path.exists():
        raise SystemExit(f"Missing depth image: {depth_path}")
    image = Image.open(depth_path)
    depth_array = np.asarray(image, dtype=np.float32)
    depth = torch.as_tensor(depth_array, dtype=torch.float32, device=device) * depth_unit_scale
    return _resize_to_shape(depth, shape)


def _resize_to_shape(depth: Any, shape: Sequence[int]) -> Any:
    import torch.nn.functional as functional

    if tuple(depth.shape) == tuple(shape):
        return depth
    if depth.ndim != 2 or len(shape) != 2:
        raise SystemExit(f"Cannot resize depth with shape {tuple(depth.shape)} to {tuple(shape)}")
    resized = functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=tuple(shape),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0)


def _valid_mask(
    prediction: Any,
    target: Any,
    *,
    accumulation: Any | None,
    min_depth: float,
    max_depth: float,
    min_accumulation: float,
) -> Any:
    valid = (
        prediction.isfinite()
        & target.isfinite()
        & (prediction > 0)
        & (target >= min_depth)
        & (target <= max_depth)
    )
    if accumulation is not None and min_accumulation > 0:
        valid = valid & accumulation.isfinite() & (accumulation >= min_accumulation)
    return valid


def _sample_indices(valid: Any, max_pixels: int) -> Any:
    indices = valid.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    if indices.numel() > max_pixels:
        step = max(1, indices.numel() // max_pixels)
        indices = indices[::step][:max_pixels]
    return indices


def _test_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit("transforms.json must contain list-valued frames.")
    by_path = {}
    for frame in frames:
        if isinstance(frame, dict) and isinstance(frame.get("file_path"), str):
            by_path[_normalize_path(str(frame["file_path"]))] = frame
    test_paths = payload.get("test_filenames") or payload.get("eval_filenames")
    if not isinstance(test_paths, list):
        raise SystemExit("transforms.json must contain list-valued test_filenames or eval_filenames.")
    selected = []
    missing = []
    for path in test_paths:
        frame = by_path.get(_normalize_path(str(path)))
        if frame is None:
            missing.append(str(path))
        else:
            selected.append(frame)
    if missing:
        raise SystemExit(f"test_filenames references frames missing from transforms.json: {missing[:10]}")
    return selected


def _load_object_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON must contain an object: {path}")
    return payload


def _normalize_path(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def _empty_metrics() -> dict[str, float]:
    return {
        "count": 0.0,
        "abs_rel": math.nan,
        "sq_rel": math.nan,
        "mae": math.nan,
        "rmse": math.nan,
        "delta1": math.nan,
        "delta2": math.nan,
        "delta3": math.nan,
    }


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return math.nan
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
