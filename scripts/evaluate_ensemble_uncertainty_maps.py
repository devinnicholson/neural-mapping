#!/usr/bin/env python3
"""Evaluate ensemble RGB disagreement against RGB error maps."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-config",
        nargs="+",
        required=True,
        help="Two or more seed run config.yml files.",
    )
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
        "--cache-images",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Nerfstudio image cache device for candidate evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_quantile(args.bad_error_quantile)
    if len(args.load_config) < 2:
        raise SystemExit("--load-config requires at least two configs for an ensemble.")
    if args.max_pixels_per_frame <= 0:
        raise SystemExit("--max-pixels-per-frame must be positive.")

    load_configs = [Path(path) for path in args.load_config]
    candidate_data = Path(args.candidate_data)
    output_path = Path(args.output)
    manifest = _load_object_json(candidate_data / "candidate_manifest.json")
    candidates = _string_list(manifest.get("candidates"), "candidates")

    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    pipelines = []
    checkpoints = []
    checkpoint_steps = []
    method_names = []
    for load_config in load_configs:
        config, pipeline, checkpoint_path, checkpoint_step = eval_setup(
            load_config,
            test_mode="test",
            update_config_callback=_candidate_update_callback(candidate_data, args.cache_images),
        )
        pipelines.append(pipeline)
        checkpoints.append(str(checkpoint_path))
        checkpoint_steps.append(checkpoint_step)
        method_names.append(getattr(config, "method_name", None))

    data_loader = pipelines[0].datamanager.fixed_indices_eval_dataloader
    if len(data_loader) != len(candidates):
        raise SystemExit(
            f"Candidate manifest has {len(candidates)} frames, but eval dataloader has "
            f"{len(data_loader)} images."
        )

    frames = []
    all_uncertainty: list[float] = []
    all_error: list[float] = []
    with torch.no_grad():
        for index, (camera, batch) in enumerate(data_loader):
            predictions = []
            for pipeline in pipelines:
                outputs = pipeline.model.get_outputs_for_camera(camera=camera)
                predictions.append(_normalize_rgb_tensor(outputs["rgb"].detach().float()[..., :3]))

            stacked = torch.stack(predictions, dim=0)
            ensemble_mean = stacked.mean(dim=0)
            uncertainty = stacked.var(dim=0, unbiased=False).mean(dim=-1)

            target = batch["image"].to(ensemble_mean.device).detach().float()
            target = _normalize_rgb_tensor(target[..., :3])
            if ensemble_mean.shape != target.shape:
                raise SystemExit(
                    f"RGB prediction and target shapes differ: {ensemble_mean.shape} vs {target.shape}"
                )
            error = _rgb_error(ensemble_mean, target, args.error_metric)
            uncertainty_values, error_values = _sample_map_and_error(
                uncertainty,
                error,
                max_pixels=args.max_pixels_per_frame,
            )
            if not error_values:
                continue

            bad_threshold = _quantile(error_values, args.bad_error_quantile)
            summary = evaluate_uncertainty(
                uncertainty_values,
                error_values,
                bad_threshold=bad_threshold,
            )
            frames.append(
                {
                    "file_path": candidates[index],
                    "sampled_pixels": len(error_values),
                    "mean_uncertainty": _mean(uncertainty_values),
                    "mean_error": _mean(error_values),
                    "bad_threshold": bad_threshold,
                    "signals": {"ensemble-rgb-variance": summary},
                }
            )
            all_uncertainty.extend(uncertainty_values)
            all_error.extend(error_values)
            print(
                f"{index + 1:03d}/{len(candidates):03d} "
                f"{candidates[index]} pixels={len(error_values)}",
                flush=True,
            )

    if not all_error:
        raise SystemExit("No valid pixels were evaluated.")

    global_bad_threshold = _quantile(all_error, args.bad_error_quantile)
    output = {
        "metadata": {
            "load_configs": [str(path) for path in load_configs],
            "candidate_data": str(candidate_data),
            "checkpoints": checkpoints,
            "checkpoint_steps": checkpoint_steps,
            "method_names": method_names,
            "error_metric": args.error_metric,
            "uncertainty_signal": "ensemble-rgb-variance",
            "bad_error_quantile": args.bad_error_quantile,
            "bad_threshold": global_bad_threshold,
            "candidate_count": len(frames),
            "sampled_pixels": len(all_error),
            "max_pixels_per_frame": args.max_pixels_per_frame,
            "cache_images": args.cache_images,
        },
        "signals": {
            "ensemble-rgb-variance": evaluate_uncertainty(
                all_uncertainty,
                all_error,
                bad_threshold=global_bad_threshold,
            )
        },
        "frames": frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} with {len(frames)} frames and {len(all_error)} sampled pixels")
    return 0


def _candidate_update_callback(candidate_data: Path, cache_images: str) -> Any:
    def update_config(config: Any) -> Any:
        config.pipeline.datamanager.dataparser.data = candidate_data
        if hasattr(config.pipeline.datamanager, "data"):
            config.pipeline.datamanager.data = None
        if hasattr(config.pipeline.datamanager, "cache_images"):
            config.pipeline.datamanager.cache_images = cache_images
        return config

    return update_config


def _rgb_error(prediction: Any, target: Any, error_metric: str) -> Any:
    if error_metric == "rgb-l1":
        return (prediction - target).abs().mean(dim=-1)
    if error_metric == "rgb-l2":
        return ((prediction - target) ** 2).mean(dim=-1).sqrt()
    raise ValueError(f"Unknown error metric: {error_metric}")


def _sample_map_and_error(
    uncertainty: Any,
    error: Any,
    *,
    max_pixels: int,
) -> tuple[list[float], list[float]]:
    if uncertainty.shape != error.shape:
        raise SystemExit(f"Uncertainty and error shapes differ: {uncertainty.shape} vs {error.shape}")

    flat_uncertainty = uncertainty.reshape(-1)
    flat_error = error.reshape(-1)
    valid = flat_uncertainty.isfinite() & flat_error.isfinite()
    indices = valid.nonzero(as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        return [], []
    if indices.numel() > max_pixels:
        step = max(1, indices.numel() // max_pixels)
        indices = indices[::step][:max_pixels]

    return (
        flat_uncertainty[indices].detach().cpu().tolist(),
        flat_error[indices].detach().cpu().tolist(),
    )


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
