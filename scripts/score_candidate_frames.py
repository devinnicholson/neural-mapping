#!/usr/bin/env python3
"""Score candidate frames with a trained Nerfstudio checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-config", required=True, help="Seed run config.yml.")
    parser.add_argument("--candidate-data", required=True, help="Candidate eval dataset directory.")
    parser.add_argument("--output", required=True, help="Output JSON score file.")
    parser.add_argument(
        "--score-metric",
        choices=("lpips", "negative-psnr", "one-minus-ssim"),
        default="lpips",
        help="Metric converted to a higher-is-worse active selection score.",
    )
    parser.add_argument(
        "--cache-images",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Nerfstudio image cache device for candidate scoring.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_config = Path(args.load_config)
    candidate_data = Path(args.candidate_data)
    output = Path(args.output)
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

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, (camera, batch) in enumerate(data_loader):
            start = time()
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            metrics_dict, _ = pipeline.model.get_image_metrics_and_images(outputs, batch)
            row = _metrics_to_row(metrics_dict)
            height = _scalar(camera.height)
            width = _scalar(camera.width)
            num_rays = height * width
            row["num_rays_per_sec"] = float(num_rays / (time() - start))
            row["fps"] = float(row["num_rays_per_sec"] / num_rays)
            row["file_path"] = candidates[index]
            row["score_metric"] = args.score_metric
            row["score"] = _active_score(row, args.score_metric)
            rows.append(row)
            print(
                f"{index + 1:03d}/{len(candidates):03d} "
                f"{row['file_path']} score={row['score']:.6f}",
                flush=True,
            )

    rows.sort(key=lambda row: (-float(row["score"]), str(row["file_path"])))
    payload = {
        "metadata": {
            "load_config": str(load_config),
            "candidate_data": str(candidate_data),
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "score_metric": args.score_metric,
            "selection_direction": "higher_is_worse",
            "candidate_count": len(rows),
            "cache_images": args.cache_images,
            "method_name": getattr(config, "method_name", None),
        },
        "scores": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} with {len(rows)} candidate scores")
    return 0


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


def _metrics_to_row(metrics: dict[str, Any]) -> dict[str, float]:
    row: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "num_rays":
            continue
        row[key] = _scalar(value)
    return row


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _active_score(row: dict[str, Any], score_metric: str) -> float:
    if score_metric == "lpips":
        return float(row["lpips"])
    if score_metric == "negative-psnr":
        return -float(row["psnr"])
    if score_metric == "one-minus-ssim":
        return 1.0 - float(row["ssim"])
    raise ValueError(f"Unknown score metric: {score_metric}")


if __name__ == "__main__":
    raise SystemExit(main())
