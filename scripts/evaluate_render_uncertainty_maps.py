#!/usr/bin/env python3
"""Evaluate renderer-derived uncertainty maps against RGB or depth error maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uncertainty_3dgs.metrics import evaluate_uncertainty


SIGNALS = (
    "transmittance",
    "local-mean-transmittance",
    "local-std-transmittance",
    "accumulation-gradient",
    "depth-gradient",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-config", required=True, help="Seed run config.yml.")
    parser.add_argument("--candidate-data", required=True, help="Candidate eval dataset directory.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--error-metric",
        choices=("rgb-l1", "rgb-l2", "depth-l1", "depth-abs-rel", "depth-aligned-abs-rel"),
        default="rgb-l1",
        help="Per-pixel error target. Higher values mean worse pixels.",
    )
    parser.add_argument(
        "--bad-error-quantile",
        type=float,
        default=0.8,
        help="Per-report/per-frame error quantile used as the bad-pixel threshold.",
    )
    parser.add_argument(
        "--max-pixels-per-frame",
        type=int,
        default=50000,
        help="Deterministically sample at most this many valid pixels per frame.",
    )
    parser.add_argument(
        "--max-global-pixels",
        type=int,
        default=500000,
        help="Deterministically cap the global report metrics to this many sampled pixels.",
    )
    parser.add_argument(
        "--signals",
        choices=SIGNALS,
        nargs="+",
        default=("transmittance",),
        help="Renderer-derived uncertainty maps to evaluate.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=15,
        help="Odd local window size for patch transmittance statistics.",
    )
    parser.add_argument(
        "--cache-images",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Nerfstudio image cache device for candidate evaluation.",
    )
    parser.add_argument("--min-depth", type=float, default=0.05, help="Minimum valid target depth in meters.")
    parser.add_argument("--max-depth", type=float, default=10.0, help="Maximum valid target depth in meters.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_quantile(args.bad_error_quantile)
    if args.max_pixels_per_frame <= 0:
        raise SystemExit("--max-pixels-per-frame must be positive.")
    if args.max_global_pixels <= 0:
        raise SystemExit("--max-global-pixels must be positive.")
    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise SystemExit("--patch-size must be a positive odd integer.")
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise SystemExit("--min-depth must be positive and lower than --max-depth.")

    load_config = Path(args.load_config)
    candidate_data = Path(args.candidate_data)
    output_path = Path(args.output)
    manifest = _load_object_json(candidate_data / "candidate_manifest.json")
    candidates = _string_list(manifest.get("candidates"), "candidates")
    transform_payload = _load_object_json(candidate_data / "transforms.json")
    candidate_frames = _candidate_frames(transform_payload, candidates)
    depth_unit_scale = float(transform_payload.get("depth_unit_scale_factor", 0.001))

    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    def update_config(config: Any) -> Any:
        config.pipeline.datamanager.dataparser.data = candidate_data
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

    data_loader = pipeline.datamanager.fixed_indices_eval_dataloader
    if len(data_loader) != len(candidates):
        raise SystemExit(
            f"Candidate manifest has {len(candidates)} frames, but eval dataloader has "
            f"{len(data_loader)} images."
        )

    frames = []
    all_uncertainty: dict[str, list[float]] = {signal: [] for signal in args.signals}
    all_error: list[float] = []
    with torch.no_grad():
        for index, (camera, batch) in enumerate(data_loader):
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            uncertainty_maps, error = _sample_uncertainty_and_error(
                outputs,
                batch,
                frame=candidate_frames[index],
                data_dir=candidate_data,
                depth_unit_scale=depth_unit_scale,
                error_metric=args.error_metric,
                signals=args.signals,
                patch_size=args.patch_size,
                max_pixels=args.max_pixels_per_frame,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            if not error:
                continue
            bad_threshold = _quantile(error, args.bad_error_quantile)
            signal_summaries = {
                signal: evaluate_uncertainty(
                    uncertainty,
                    error,
                    bad_threshold=bad_threshold,
                )
                for signal, uncertainty in uncertainty_maps.items()
            }
            frame_uncertainty_fields = {
                signal: _uncertainty_summary_fields(uncertainty)
                for signal, uncertainty in uncertainty_maps.items()
            }
            frames.append(
                {
                    "file_path": candidates[index],
                    "sampled_pixels": len(error),
                    "mean_uncertainty": {
                        signal: fields["mean_uncertainty"]
                        for signal, fields in frame_uncertainty_fields.items()
                    },
                    "top_decile_mean_uncertainty": {
                        signal: fields["top_decile_mean_uncertainty"]
                        for signal, fields in frame_uncertainty_fields.items()
                    },
                    "p90_uncertainty": {
                        signal: fields["p90_uncertainty"]
                        for signal, fields in frame_uncertainty_fields.items()
                    },
                    "mean_error": _mean(error),
                    "bad_threshold": bad_threshold,
                    "signals": signal_summaries,
                }
            )
            for signal, uncertainty in uncertainty_maps.items():
                all_uncertainty[signal].extend(uncertainty)
            all_error.extend(error)
            print(
                f"{index + 1:03d}/{len(candidates):03d} "
                f"{candidates[index]} pixels={len(error)}",
                flush=True,
            )

    if not all_error:
        raise SystemExit("No valid pixels were evaluated.")
    all_uncertainty, all_error = _downsample_global_samples(
        all_uncertainty,
        all_error,
        max_pixels=args.max_global_pixels,
    )

    global_bad_threshold = _quantile(all_error, args.bad_error_quantile)
    output = {
        "metadata": {
            "load_config": str(load_config),
            "candidate_data": str(candidate_data),
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "method_name": getattr(config, "method_name", None),
            "error_metric": args.error_metric,
            "uncertainty_signals": list(args.signals),
            "bad_error_quantile": args.bad_error_quantile,
            "bad_threshold": global_bad_threshold,
            "candidate_count": len(frames),
            "sampled_pixels": len(all_error),
            "raw_sampled_pixels_before_global_cap": sum(frame["sampled_pixels"] for frame in frames),
            "max_pixels_per_frame": args.max_pixels_per_frame,
            "max_global_pixels": args.max_global_pixels,
            "patch_size": args.patch_size,
            "cache_images": args.cache_images,
            "depth_unit_scale_factor": depth_unit_scale,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        },
        "signals": {
            signal: evaluate_uncertainty(
                uncertainty,
                all_error,
                bad_threshold=global_bad_threshold,
            )
            for signal, uncertainty in all_uncertainty.items()
        },
        "frames": frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} with {len(frames)} frames and {len(all_error)} sampled pixels")
    return 0


def _sample_uncertainty_and_error(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    frame: dict[str, Any],
    data_dir: Path,
    depth_unit_scale: float,
    error_metric: str,
    signals: Sequence[str],
    patch_size: int,
    max_pixels: int,
    min_depth: float,
    max_depth: float,
) -> tuple[dict[str, list[float]], list[float]]:
    if "accumulation" not in outputs:
        raise SystemExit("Model outputs are missing required key 'accumulation'.")

    accumulation = outputs["accumulation"].detach().float()
    error, valid_error = _error_map(
        outputs,
        batch,
        frame=frame,
        data_dir=data_dir,
        depth_unit_scale=depth_unit_scale,
        error_metric=error_metric,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    transmittance = 1.0 - accumulation
    if transmittance.shape[-1:] == (1,):
        transmittance = transmittance.squeeze(-1)
    if transmittance.shape != error.shape:
        raise SystemExit(f"Accumulation and error shapes differ: {transmittance.shape} vs {error.shape}")

    signal_maps = _signal_maps(
        outputs,
        transmittance=transmittance,
        signals=signals,
        patch_size=patch_size,
    )
    flat_error = error.reshape(-1)
    flat_signals = {signal: values.reshape(-1) for signal, values in signal_maps.items()}
    valid = flat_error.isfinite() & valid_error.reshape(-1)
    for values in flat_signals.values():
        valid = valid & values.isfinite()
    indices = valid.nonzero(as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        return {}, []
    if indices.numel() > max_pixels:
        step = max(1, indices.numel() // max_pixels)
        indices = indices[::step][:max_pixels]

    return (
        {
            signal: values[indices].detach().cpu().tolist()
            for signal, values in flat_signals.items()
        },
        flat_error[indices].detach().cpu().tolist(),
    )


def _error_map(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    frame: dict[str, Any],
    data_dir: Path,
    depth_unit_scale: float,
    error_metric: str,
    min_depth: float,
    max_depth: float,
) -> tuple[Any, Any]:
    import torch

    if error_metric in {"rgb-l1", "rgb-l2"}:
        if "rgb" not in outputs:
            raise SystemExit("Model outputs are missing required key 'rgb'.")
        if "image" not in batch:
            raise SystemExit("Eval batch is missing required key 'image'.")
        prediction = outputs["rgb"].detach().float()
        target = batch["image"].to(prediction.device).detach().float()
        prediction = _normalize_rgb_tensor(prediction[..., :3])
        target = _normalize_rgb_tensor(target[..., :3])
        if prediction.shape != target.shape:
            raise SystemExit(f"RGB prediction and target shapes differ: {prediction.shape} vs {target.shape}")
        if error_metric == "rgb-l1":
            error = (prediction - target).abs().mean(dim=-1)
        else:
            error = ((prediction - target) ** 2).mean(dim=-1).sqrt()
        return error, torch.ones_like(error, dtype=torch.bool)

    prediction = _depth_map(outputs)
    target = _file_depth(
        frame,
        data_dir,
        depth_unit_scale,
        device=prediction.device,
        shape=prediction.shape,
    )
    valid = (
        prediction.isfinite()
        & target.isfinite()
        & (prediction > 0)
        & (target >= min_depth)
        & (target <= max_depth)
    )
    if error_metric == "depth-l1":
        return (prediction - target).abs(), valid
    if error_metric == "depth-abs-rel":
        return (prediction - target).abs() / torch.clamp(target, min=min_depth), valid
    if error_metric == "depth-aligned-abs-rel":
        prediction = _median_align(prediction, target, valid)
        return (prediction - target).abs() / torch.clamp(target, min=min_depth), valid
    raise ValueError(f"Unknown error metric: {error_metric}")


def _signal_maps(
    outputs: dict[str, Any],
    *,
    transmittance: Any,
    signals: Sequence[str],
    patch_size: int,
) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    for signal in signals:
        if signal == "transmittance":
            maps[signal] = transmittance
        elif signal == "local-mean-transmittance":
            maps[signal] = _local_mean(transmittance, patch_size)
        elif signal == "local-std-transmittance":
            maps[signal] = _local_std(transmittance, patch_size)
        elif signal == "accumulation-gradient":
            maps[signal] = _gradient_magnitude(1.0 - transmittance)
        elif signal == "depth-gradient":
            depth = _depth_map(outputs)
            maps[signal] = _gradient_magnitude(depth)
        else:
            raise ValueError(f"Unknown signal: {signal}")
    return maps


def _local_mean(values: Any, patch_size: int) -> Any:
    import torch.nn.functional as functional

    image = values.unsqueeze(0).unsqueeze(0)
    pooled = functional.avg_pool2d(
        image,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
        count_include_pad=False,
    )
    return pooled.squeeze(0).squeeze(0)


def _local_std(values: Any, patch_size: int) -> Any:
    import torch

    mean = _local_mean(values, patch_size)
    mean_square = _local_mean(values * values, patch_size)
    variance = torch.clamp(mean_square - mean * mean, min=0.0)
    return torch.sqrt(variance)


def _gradient_magnitude(values: Any) -> Any:
    import torch

    vertical = torch.zeros_like(values)
    horizontal = torch.zeros_like(values)
    vertical[1:, :] = torch.abs(values[1:, :] - values[:-1, :])
    horizontal[:, 1:] = torch.abs(values[:, 1:] - values[:, :-1])
    return torch.sqrt(vertical * vertical + horizontal * horizontal)


def _depth_map(outputs: dict[str, Any]) -> Any:
    for key in ("depth", "expected_depth", "median_depth"):
        if key not in outputs:
            continue
        depth = outputs[key].detach().float()
        if depth.shape[-1:] == (1,):
            depth = depth.squeeze(-1)
        return depth
    raise SystemExit("Model outputs are missing a depth-like key required by depth-gradient.")


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
    depth_path = data_dir / _normalize_frame_path(depth_path_value)
    if not depth_path.exists():
        raise SystemExit(f"Missing depth image: {depth_path}")
    image = Image.open(depth_path)
    depth = torch.as_tensor(np.asarray(image).copy(), dtype=torch.float32, device=device) * depth_unit_scale
    return _resize_to_shape(depth, shape)


def _resize_to_shape(values: Any, shape: Sequence[int]) -> Any:
    import torch.nn.functional as functional

    if tuple(values.shape) == tuple(shape):
        return values
    if values.ndim != 2 or len(shape) != 2:
        raise SystemExit(f"Cannot resize tensor with shape {tuple(values.shape)} to {tuple(shape)}")
    resized = functional.interpolate(
        values.unsqueeze(0).unsqueeze(0),
        size=tuple(shape),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0)


def _median_align(prediction: Any, target: Any, valid: Any) -> Any:
    import torch

    if int(valid.sum().detach().cpu().item()) == 0:
        return prediction
    pred_median = torch.median(prediction[valid])
    target_median = torch.median(target[valid])
    if float(pred_median.detach().cpu().item()) <= 0:
        return prediction
    return prediction * (target_median / pred_median)


def _normalize_rgb_tensor(value: Any) -> Any:
    max_value = float(value.max().detach().cpu().item())
    if max_value > 2.0:
        return value / 255.0
    return value


def _load_object_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON must contain an object: {path}")
    return payload


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise SystemExit(f"{name} must be a list.")
    return [str(item) for item in value]


def _candidate_frames(payload: dict[str, Any], candidates: Sequence[str]) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit("transforms.json must contain list-valued frames.")
    by_path: dict[str, dict[str, Any]] = {}
    for frame in frames:
        if isinstance(frame, dict) and isinstance(frame.get("file_path"), str):
            by_path[_normalize_frame_path(str(frame["file_path"]))] = frame
    selected = []
    missing = []
    for candidate in candidates:
        frame = by_path.get(_normalize_frame_path(candidate))
        if frame is None:
            missing.append(candidate)
        else:
            selected.append(frame)
    if missing:
        raise SystemExit(f"Candidate manifest references frames missing from transforms.json: {missing[:10]}")
    return selected


def _normalize_frame_path(path: str) -> str:
    return path[2:] if path.startswith("./") else path


def _downsample_global_samples(
    all_uncertainty: dict[str, list[float]],
    all_error: list[float],
    *,
    max_pixels: int,
) -> tuple[dict[str, list[float]], list[float]]:
    if len(all_error) <= max_pixels:
        return all_uncertainty, all_error
    step = max(1, len(all_error) // max_pixels)
    indices = range(0, len(all_error), step)
    selected = list(indices)[:max_pixels]
    return (
        {
            signal: [values[index] for index in selected]
            for signal, values in all_uncertainty.items()
        },
        [all_error[index] for index in selected],
    )


def _validate_quantile(quantile: float) -> None:
    if not 0.0 <= quantile <= 1.0:
        raise SystemExit("quantile must be between 0 and 1.")


def _quantile(values: Sequence[float], quantile: float) -> float:
    _validate_quantile(quantile)
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _uncertainty_summary_fields(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_uncertainty": math.nan,
            "p90_uncertainty": math.nan,
            "top_decile_mean_uncertainty": math.nan,
        }
    mean = _mean(values)
    p90 = _quantile(values, 0.9)
    top_decile = [float(value) for value in values if float(value) >= p90]
    return {
        "mean_uncertainty": mean,
        "p90_uncertainty": p90,
        "top_decile_mean_uncertainty": _mean(top_decile),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return sum(float(value) for value in values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
