#!/usr/bin/env python3
"""Evaluate renderer-derived uncertainty maps against RGB error maps."""

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
        choices=("rgb-l1", "rgb-l2"),
        default="rgb-l1",
        help="Per-pixel RGB error target. Higher values mean worse pixels.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_quantile(args.bad_error_quantile)
    if args.max_pixels_per_frame <= 0:
        raise SystemExit("--max-pixels-per-frame must be positive.")
    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise SystemExit("--patch-size must be a positive odd integer.")

    load_config = Path(args.load_config)
    candidate_data = Path(args.candidate_data)
    output_path = Path(args.output)
    manifest = _load_object_json(candidate_data / "candidate_manifest.json")
    candidates = _string_list(manifest.get("candidates"), "candidates")

    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    def update_config(config: Any) -> Any:
        config.pipeline.datamanager.dataparser.data = candidate_data
        if hasattr(config.pipeline.datamanager, "data"):
            config.pipeline.datamanager.data = None
        if hasattr(config.pipeline.datamanager, "cache_images"):
            config.pipeline.datamanager.cache_images = args.cache_images
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
                error_metric=args.error_metric,
                signals=args.signals,
                patch_size=args.patch_size,
                max_pixels=args.max_pixels_per_frame,
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
            frames.append(
                {
                    "file_path": candidates[index],
                    "sampled_pixels": len(error),
                    "mean_uncertainty": {
                        signal: _mean(uncertainty)
                        for signal, uncertainty in uncertainty_maps.items()
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
            "max_pixels_per_frame": args.max_pixels_per_frame,
            "patch_size": args.patch_size,
            "cache_images": args.cache_images,
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
    error_metric: str,
    signals: Sequence[str],
    patch_size: int,
    max_pixels: int,
) -> tuple[dict[str, list[float]], list[float]]:
    if "rgb" not in outputs:
        raise SystemExit("Model outputs are missing required key 'rgb'.")
    if "accumulation" not in outputs:
        raise SystemExit("Model outputs are missing required key 'accumulation'.")
    if "image" not in batch:
        raise SystemExit("Eval batch is missing required key 'image'.")

    prediction = outputs["rgb"].detach().float()
    target = batch["image"].to(prediction.device).detach().float()
    accumulation = outputs["accumulation"].detach().float()

    prediction = _normalize_rgb_tensor(prediction[..., :3])
    target = _normalize_rgb_tensor(target[..., :3])
    if prediction.shape != target.shape:
        raise SystemExit(f"RGB prediction and target shapes differ: {prediction.shape} vs {target.shape}")

    if error_metric == "rgb-l1":
        error = (prediction - target).abs().mean(dim=-1)
    elif error_metric == "rgb-l2":
        error = ((prediction - target) ** 2).mean(dim=-1).sqrt()
    else:
        raise ValueError(f"Unknown error metric: {error_metric}")

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
    valid = flat_error.isfinite()
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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return sum(float(value) for value in values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
