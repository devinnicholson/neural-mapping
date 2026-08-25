#!/usr/bin/env python3
"""Build and audit the frozen multi-trajectory ICL-NUIM benchmark record."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


TRAJECTORIES = tuple(f"kt{index}" for index in range(4))
SEEDS = (42, 43)
METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "raw_depth_abs_rel",
    "raw_depth_rmse",
    "raw_depth_delta1",
    "median_aligned_depth_abs_rel",
    "median_aligned_depth_rmse",
    "median_aligned_depth_delta1",
    "geometry_accuracy_mean_m",
    "geometry_completeness_mean_m",
    "geometry_chamfer_l1_mean_m",
    "geometry_precision_05cm",
    "geometry_recall_05cm",
    "geometry_fscore_05cm",
    "geometry_precision_10cm",
    "geometry_recall_10cm",
    "geometry_fscore_10cm",
)
LOWER_IS_BETTER = {
    "lpips",
    "raw_depth_abs_rel",
    "raw_depth_rmse",
    "median_aligned_depth_abs_rel",
    "median_aligned_depth_rmse",
    "geometry_accuracy_mean_m",
    "geometry_completeness_mean_m",
    "geometry_chamfer_l1_mean_m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Directory containing downloaded runs/ and reports/ trees.",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _required_number(payload: dict[str, Any], key: str, path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Missing finite {key!r} in {path}")
    return float(value)


def load_run_metrics(root: Path, scene: str, budget: int = 50) -> tuple[dict[str, float], list[Path]]:
    metric_dir = root / "runs" / scene / "splatfacto" / f"budget_{budget:03d}" / "metrics"
    paths = {
        "rgb": metric_dir / "ns_eval.json",
        "depth": metric_dir / "depth_eval.json",
        "geometry": metric_dir / "geometry_eval.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete metrics for {scene}: {missing}")

    rgb = _read(paths["rgb"])
    rgb_results = rgb.get("results")
    if not isinstance(rgb_results, dict):
        raise ValueError(f"Missing results in {paths['rgb']}")
    depth = _read(paths["depth"])
    depth_summary = depth.get("summary")
    if not isinstance(depth_summary, dict):
        raise ValueError(f"Missing summary in {paths['depth']}")
    raw = depth_summary.get("raw")
    aligned = depth_summary.get("median_aligned")
    if not isinstance(raw, dict) or not isinstance(aligned, dict):
        raise ValueError(f"Missing raw/median_aligned depth summary in {paths['depth']}")
    geometry = _read(paths["geometry"])
    geometry_summary = geometry.get("summary")
    if not isinstance(geometry_summary, dict):
        raise ValueError(f"Missing summary in {paths['geometry']}")

    row = {
        "psnr": _required_number(rgb_results, "psnr", paths["rgb"]),
        "ssim": _required_number(rgb_results, "ssim", paths["rgb"]),
        "lpips": _required_number(rgb_results, "lpips", paths["rgb"]),
        "raw_depth_abs_rel": _required_number(raw, "abs_rel", paths["depth"]),
        "raw_depth_rmse": _required_number(raw, "rmse", paths["depth"]),
        "raw_depth_delta1": _required_number(raw, "delta1", paths["depth"]),
        "median_aligned_depth_abs_rel": _required_number(aligned, "abs_rel", paths["depth"]),
        "median_aligned_depth_rmse": _required_number(aligned, "rmse", paths["depth"]),
        "median_aligned_depth_delta1": _required_number(aligned, "delta1", paths["depth"]),
    }
    for key in METRICS:
        if key.startswith("geometry_"):
            row[key] = _required_number(geometry_summary, key.removeprefix("geometry_"), paths["geometry"])
    return row, list(paths.values())


def _delta(active: dict[str, float], random: dict[str, float]) -> dict[str, float]:
    return {metric: active[metric] - random[metric] for metric in METRICS}


def _favorable(metric: str, delta: float) -> bool:
    return delta < 0 if metric in LOWER_IS_BETTER else delta > 0


def _cluster_bootstrap_interval(trajectory_values: dict[str, list[float]]) -> list[float]:
    names = sorted(trajectory_values)
    values = []
    for sample in itertools.product(names, repeat=len(names)):
        sampled = [value for name in sample for value in trajectory_values[name]]
        values.append(statistics.mean(sampled))
    values.sort()
    return [
        values[int(0.025 * (len(values) - 1))],
        values[int(0.975 * (len(values) - 1))],
    ]


def summarize_pairs(pairs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory_summary: dict[str, Any] = {}
    for trajectory in TRAJECTORIES:
        selected = [pair for pair in pairs if pair["trajectory"] == trajectory]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Expected two pairs for {trajectory}, found {len(selected)}")
        trajectory_summary[trajectory] = {
            metric: statistics.mean(pair["active_minus_random"][metric] for pair in selected)
            for metric in METRICS
        }

    summary: dict[str, Any] = {"pair_count": len(pairs), "trajectory_count": len(TRAJECTORIES)}
    for metric in METRICS:
        deltas = [pair["active_minus_random"][metric] for pair in pairs]
        clustered = {
            trajectory: [
                pair["active_minus_random"][metric]
                for pair in pairs
                if pair["trajectory"] == trajectory
            ]
            for trajectory in TRAJECTORIES
        }
        summary[metric] = {
            "mean": statistics.mean(deltas),
            "population_std": statistics.pstdev(deltas),
            "favorable_pairs": sum(_favorable(metric, value) for value in deltas),
            "favorable_trajectories": sum(
                _favorable(metric, trajectory_summary[name][metric]) for name in TRAJECTORIES
            ),
            "trajectory_cluster_bootstrap_95_interval": _cluster_bootstrap_interval(clustered),
        }
    return summary, trajectory_summary


def _mean_method(controls: list[dict[str, Any]], method: str, metric: str) -> float:
    values = [row["metrics"][metric] for row in controls if row["method"] == method]
    if len(values) != len(TRAJECTORIES):
        raise ValueError(f"Expected four {method} control rows, found {len(values)}")
    return statistics.mean(values)


def _audit_split_manifest(path: Path, train_count: int) -> dict[str, Any]:
    payload = _read(path)
    expected = {"train": train_count, "val": 10, "test": 20}
    sets: dict[str, set[str]] = {}
    for name, count in expected.items():
        values = payload.get(name)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Missing string-list {name!r} in {path}")
        if len(values) != count or len(set(values)) != count:
            raise ValueError(f"Expected {count} unique {name} frames in {path}, found {len(values)}")
        sets[name] = set(values)
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError(f"Train/validation/test leakage in {path}")
    if payload.get("budget") != train_count:
        raise ValueError(f"Budget mismatch in {path}: {payload.get('budget')!r}")
    transforms = payload.get("transforms_frames")
    if not isinstance(transforms, list) or not sets["train"] | sets["val"] | sets["test"] <= set(transforms):
        raise ValueError(f"Manifest references frames absent from transforms_frames in {path}")
    return {"path": str(path), "train": sets["train"], "val": sets["val"], "test": sets["test"]}


def audit_split_family(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    split_root = root / "splits"
    summaries: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    for trajectory in TRAJECTORIES:
        base_path = split_root / f"icl_lr_{trajectory}_clean_v1_budget_025_split_manifest.json"
        base = _audit_split_manifest(base_path, 25)
        source_paths.append(base_path)
        methods = {
            "random": f"icl_lr_{trajectory}_clean_v1_budget_050_split_manifest.json",
            "pose": f"icl_lr_{trajectory}_pose_v1_budget_050_split_manifest.json",
            "trajectory": f"icl_lr_{trajectory}_trajectory_v1_budget_050_split_manifest.json",
            "active_seed_42": f"icl_lr_{trajectory}_s42_active_v2_budget_050_split_manifest.json",
            "active_seed_43": f"icl_lr_{trajectory}_s43_active_v2_budget_050_split_manifest.json",
        }
        method_summaries: dict[str, Any] = {}
        for method, filename in methods.items():
            path = split_root / filename
            manifest = _audit_split_manifest(path, 50)
            source_paths.append(path)
            if manifest["val"] != base["val"] or manifest["test"] != base["test"]:
                raise ValueError(f"Mismatched holdout for {trajectory} {method}")
            if not base["train"] <= manifest["train"]:
                raise ValueError(f"{trajectory} {method} does not expand the frozen 25-frame seed")
            method_summaries[method] = {
                "manifest": str(path),
                "train_count": len(manifest["train"]),
                "base_frames_retained": len(base["train"] & manifest["train"]),
                "validation_matches_base": True,
                "test_matches_base": True,
                "partitions_disjoint": True,
            }
        summaries.append(
            {
                "trajectory": trajectory,
                "base_manifest": str(base_path),
                "base_train_count": len(base["train"]),
                "validation_count": len(base["val"]),
                "test_count": len(base["test"]),
                "methods": method_summaries,
            }
        )
    return summaries, source_paths


def decisions(
    summary: dict[str, Any],
    trajectory_summary: dict[str, Any],
    controls: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, bool]:
    h1 = summary["psnr"]["mean"] > 0 and summary["geometry_fscore_05cm"]["mean"] > 0
    h2 = (
        summary["psnr"]["favorable_pairs"] >= 6
        and summary["geometry_fscore_05cm"]["favorable_pairs"] >= 6
        and summary["psnr"]["favorable_trajectories"] >= 3
        and summary["geometry_fscore_05cm"]["favorable_trajectories"] >= 3
    )
    active_psnr = _mean_method(controls, "active", "psnr")
    active_fscore = _mean_method(controls, "active", "geometry_fscore_05cm")
    h3 = all(
        active_psnr > _mean_method(controls, method, "psnr")
        and active_fscore > _mean_method(controls, method, "geometry_fscore_05cm")
        for method in ("random", "trajectory", "pose")
    )
    h4 = (
        statistics.mean(row["depth_gradient_spearman"] for row in diagnostics) > 0
        and sum(row["depth_gradient_auroc"] > 0.5 for row in diagnostics) >= 6
    )
    lpips_mean = summary["lpips"]["mean"]
    lpips_trajectory_max = max(trajectory_summary[name]["lpips"] for name in TRAJECTORIES)
    perceptual_safety = lpips_mean <= 0.01 and lpips_trajectory_max <= 0.02
    return {
        "h1_primary": h1,
        "h2_replication": h2,
        "h3_control_superiority": h3,
        "h4_uncertainty": h4,
        "perceptual_safety_gate": perceptual_safety,
        "overall_support": h1 and h2 and h4 and perceptual_safety,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_record(
    artifact_root: Path,
    protocol_path: Path,
    run_manifest_path: Path | None = None,
) -> dict[str, Any]:
    protocol = _read(protocol_path)
    split_audit, split_paths = audit_split_family(artifact_root)
    pairs: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    source_paths: list[Path] = list(split_paths)

    for trajectory in TRAJECTORIES:
        for seed in SEEDS:
            random_scene = f"icl_lr_{trajectory}_s{seed}_random_v2"
            active_scene = f"icl_lr_{trajectory}_s{seed}_active_v2"
            random_metrics, random_paths = load_run_metrics(artifact_root, random_scene)
            active_metrics, active_paths = load_run_metrics(artifact_root, active_scene)
            source_paths.extend(random_paths + active_paths)
            pairs.append(
                {
                    "trajectory": trajectory,
                    "training_seed": seed,
                    "random_scene": random_scene,
                    "active_scene": active_scene,
                    "random_budget_50": random_metrics,
                    "active_budget_50": active_metrics,
                    "active_minus_random": _delta(active_metrics, random_metrics),
                }
            )

        for method in ("random", "trajectory", "pose", "active"):
            scene = f"icl_lr_{trajectory}_s42_{method}_v2"
            metrics, paths = load_run_metrics(artifact_root, scene)
            source_paths.extend(paths)
            controls.append({"trajectory": trajectory, "method": method, "scene": scene, "metrics": metrics})

    diagnostics: list[dict[str, Any]] = []
    for trajectory in TRAJECTORIES:
        for seed in SEEDS:
            report = (
                artifact_root
                / "reports"
                / "render_uncertainty_maps"
                / f"icl_lr_{trajectory}_s{seed}_uncertainty_v2_budget_025_depth-aligned-abs-rel.json"
            )
            payload = _read(report)
            signal = payload.get("signals", {}).get("depth-gradient")
            if not isinstance(signal, dict):
                raise ValueError(f"Missing depth-gradient signal in {report}")
            sparsification = signal.get("sparsification")
            if not isinstance(sparsification, dict):
                raise ValueError(f"Missing sparsification summary in {report}")
            metadata = payload.get("metadata", {})
            diagnostics.append(
                {
                    "trajectory": trajectory,
                    "training_seed": seed,
                    "candidate_frames": int(metadata.get("candidate_count", 0)),
                    "depth_gradient_spearman": _required_number(signal, "spearman", report),
                    "depth_gradient_auroc": _required_number(signal, "auroc", report),
                    "depth_gradient_auprc": _required_number(signal, "auprc", report),
                    "depth_gradient_risk_coverage_auc": _required_number(
                        signal, "risk_coverage_auc", report
                    ),
                    "depth_gradient_ause": _required_number(sparsification, "ause", report),
                }
            )
            source_paths.append(report)

    summary, trajectory_summary = summarize_pairs(pairs)
    gate_results = decisions(summary, trajectory_summary, controls, diagnostics)
    unique_paths = sorted(set(source_paths))
    run_manifest = _read(run_manifest_path) if run_manifest_path else None
    if run_manifest_path:
        unique_paths.append(run_manifest_path)
    return {
        "schema_version": 1,
        "status": "completed_preregistered_multitrajectory_study",
        "protocol": str(protocol_path),
        "protocol_status": protocol.get("status"),
        "policy": protocol.get("acquisition_methods", {}).get("frozen_active"),
        "split_integrity": split_audit,
        "paired_comparisons": pairs,
        "active_random_summary": summary,
        "trajectory_summary": trajectory_summary,
        "seed_42_controls": controls,
        "uncertainty_diagnostics": diagnostics,
        "decision": gate_results,
        "modal_runs": run_manifest,
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path)} for path in sorted(set(unique_paths))
        ],
    }


def audit_record(record: dict[str, Any], tolerance: float = 1e-9) -> None:
    pairs = record.get("paired_comparisons")
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise ValueError("Record must contain eight matched active-random pairs.")
    for pair in pairs:
        for metric in METRICS:
            expected = pair["active_budget_50"][metric] - pair["random_budget_50"][metric]
            actual = pair["active_minus_random"][metric]
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"Delta mismatch for {pair['trajectory']} seed {pair['training_seed']} {metric}")
    summary, trajectory_summary = summarize_pairs(pairs)
    stored = record["active_random_summary"]
    for metric in METRICS:
        for field in ("mean", "population_std"):
            if not math.isclose(
                stored[metric][field], summary[metric][field], rel_tol=0.0, abs_tol=tolerance
            ):
                raise ValueError(f"Summary mismatch for {metric}.{field}")
    recomputed = decisions(summary, trajectory_summary, record["seed_42_controls"], record["uncertainty_diagnostics"])
    if record["decision"] != recomputed:
        raise ValueError(f"Decision mismatch: stored={record['decision']}, recomputed={recomputed}")


def main() -> int:
    args = parse_args()
    record = build_record(args.artifact_root, args.protocol, args.run_manifest)
    audit_record(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "pairs": len(record["paired_comparisons"]),
                "decision": record["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
