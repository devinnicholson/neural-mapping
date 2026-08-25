#!/usr/bin/env python3
"""Recompute paired summaries and decision gates in a replication record."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "raw_depth_abs_rel",
    "raw_depth_delta1",
    "median_aligned_depth_abs_rel",
    "median_aligned_depth_delta1",
)
LOWER_IS_BETTER = {"lpips", "raw_depth_abs_rel", "median_aligned_depth_abs_rel"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def audit_record(record: dict[str, Any], *, tolerance: float = 1e-9) -> dict[str, Any]:
    pairs = record["interleaved_pairs"]
    if len(pairs) != 3:
        raise ValueError(f"Expected three interleaved pairs, found {len(pairs)}")

    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for pair in pairs:
        for metric in METRICS:
            expected = float(pair["active_budget_50"][metric]) - float(
                pair["random_budget_50"][metric]
            )
            stored = float(pair["active_minus_random"][metric])
            _assert_close(f"{pair['scene']}.{metric}", stored, expected, tolerance)
            values[metric].append(expected)

    recomputed: dict[str, Any] = {}
    for metric, deltas in values.items():
        favorable = sum(delta < 0 for delta in deltas) if metric in LOWER_IS_BETTER else sum(
            delta > 0 for delta in deltas
        )
        bootstrap = sorted(
            sum(sample) / len(sample)
            for sample in itertools.product(deltas, repeat=len(deltas))
        )
        interval = [
            bootstrap[int(0.025 * (len(bootstrap) - 1))],
            bootstrap[int(0.975 * (len(bootstrap) - 1))],
        ]
        recomputed[metric] = {
            "mean": statistics.mean(deltas),
            "population_std": statistics.pstdev(deltas),
            "favorable_pairs": favorable,
            "paired_bootstrap_95_interval": interval,
        }
        stored = record["interleaved_summary"][metric]
        _assert_close(f"summary.{metric}.mean", stored["mean"], recomputed[metric]["mean"], tolerance)
        _assert_close(
            f"summary.{metric}.population_std",
            stored["population_std"],
            recomputed[metric]["population_std"],
            tolerance,
        )
        if int(stored["favorable_pairs"]) != favorable:
            raise ValueError(
                f"summary.{metric}.favorable_pairs: stored={stored['favorable_pairs']}, "
                f"recomputed={favorable}"
            )
        for index, value in enumerate(interval):
            _assert_close(
                f"summary.{metric}.paired_bootstrap_95_interval[{index}]",
                stored["paired_bootstrap_95_interval"][index],
                value,
                tolerance,
            )

    blocked = record["blocked_pair"]
    blocked_delta: dict[str, float] = {}
    for metric in METRICS:
        expected = float(blocked["active_budget_50"][metric]) - float(
            blocked["random_budget_50"][metric]
        )
        stored = float(blocked["active_minus_random"][metric])
        _assert_close(f"blocked.{metric}", stored, expected, tolerance)
        blocked_delta[metric] = expected

    interleaved_support = (
        recomputed["psnr"]["mean"] > 0
        and recomputed["raw_depth_abs_rel"]["mean"] < 0
        and recomputed["psnr"]["favorable_pairs"] >= 2
        and recomputed["raw_depth_abs_rel"]["favorable_pairs"] >= 2
    )
    blocked_support = blocked_delta["psnr"] > 0 and blocked_delta["raw_depth_abs_rel"] < 0
    lpips_safety_gate = all(delta <= 0.01 for delta in values["lpips"]) and blocked_delta[
        "lpips"
    ] <= 0.01
    robust_support = interleaved_support and blocked_support and lpips_safety_gate
    decisions = {
        "interleaved_support": interleaved_support,
        "blocked_support": blocked_support,
        "lpips_safety_gate": lpips_safety_gate,
        "robust_support": robust_support,
    }
    for key, value in decisions.items():
        if bool(record["decision"][key]) is not value:
            raise ValueError(
                f"decision.{key}: stored={record['decision'][key]!r}, recomputed={value!r}"
            )

    return {
        "status": "ok",
        "record_status": record["status"],
        "interleaved_summary": recomputed,
        "blocked_active_minus_random": blocked_delta,
        "decision": decisions,
    }


def _assert_close(label: str, stored: Any, recomputed: float, tolerance: float) -> None:
    stored_number = float(stored)
    if not math.isclose(stored_number, recomputed, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label}: stored={stored_number}, recomputed={recomputed}")


def main() -> int:
    args = parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    print(json.dumps(audit_record(record, tolerance=args.tolerance), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
