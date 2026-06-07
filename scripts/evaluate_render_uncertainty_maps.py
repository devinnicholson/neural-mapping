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
    all_uncertainty: list[float] = []
    all_error: list[float] = []
    with torch.no_grad():
        for index, (camera, batch) in enumerate(data_loader):
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            uncertainty, error = _sample_transmittance_and_error(
                outputs,
                batch,
                error_metric=args.error_metric,
                max_pixels=args.max_pixels_per_frame,
            )
            if not uncertainty:
                continue
            bad_threshold = _quantile(error, args.bad_error_quantile)
            summary = evaluate_uncertainty(
                uncertainty,
                error,
                bad_threshold=bad_threshold,
            )
            frames.append(
                {
                    "file_path": candidates[index],
                    "sampled_pixels": len(error),
                    "mean_uncertainty": _mean(uncertainty),
                    "mean_error": _mean(error),
                    "bad_threshold": bad_threshold,
                    "signals": {"transmittance": summary},
                }
            )
            all_uncertainty.extend(uncertainty)
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
            "uncertainty_signal": "transmittance",
            "bad_error_quantile": args.bad_error_quantile,
            "bad_threshold": global_bad_threshold,
            "candidate_count": len(frames),
            "sampled_pixels": len(all_error),
            "max_pixels_per_frame": args.max_pixels_per_frame,
            "cache_images": args.cache_images,
        },
        "signals": {
            "transmittance": evaluate_uncertainty(
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


def _sample_transmittance_and_error(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    error_metric: str,
    max_pixels: int,
) -> tuple[list[float], list[float]]:
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

    flat_uncertainty = transmittance.reshape(-1)
    flat_error = error.reshape(-1)
    valid = (
        flat_uncertainty.isfinite()
        & flat_error.isfinite()
        & (flat_uncertainty >= 0.0)
        & (flat_uncertainty <= 1.0)
    )
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
