#!/usr/bin/env python3
"""Summarize active-vs-baseline metric deltas from Modal metric rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


METRIC_NAMES = ("psnr", "ssim", "lpips", "fps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Metric rows JSON file, Modal metrics command output, or '-' for stdin. "
            "Rows should contain scene, budget, psnr, ssim, lpips, and fps."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="GROUP:SEED:BASELINE_SCENE:ACTIVE_SCENE",
        help="Metric pair to compare. May be repeated.",
    )
    parser.add_argument(
        "--pairs-file",
        default=None,
        help=(
            "Optional JSON file containing a 'pairs' list with group, seed, "
            "baseline_scene, and active_scene fields. May also contain 'budget'."
        ),
    )
    parser.add_argument(
        "--budget",
        default=None,
        help="Budget to compare, normalized to three digits. Defaults to pairs-file budget or 050.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for stdout. JSON is always written when --output is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs_payload = _read_pairs_file(args.pairs_file) if args.pairs_file else {}
    budget = _normalize_budget(args.budget or pairs_payload.get("budget", "050"))
    pair_specs = [_parse_pair_spec(spec) for spec in args.pair]
    pair_specs.extend(_parse_pairs_payload(pairs_payload))
    if not pair_specs:
        raise SystemExit(
            "At least one --pair GROUP:SEED:BASELINE_SCENE:ACTIVE_SCENE "
            "or --pairs-file entry is required."
        )

    rows = load_metric_rows(_read_input(args.input))
    summary = summarize_pairs(rows, pair_specs, budget=budget)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output_path}", file=sys.stderr)

    if args.format == "markdown":
        print(_format_markdown(summary))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def load_metric_rows(text: str) -> list[dict[str, Any]]:
    """Load metric rows from JSON or noisy Modal CLI output."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        return [_normalize_metric_row(row) for row in payload if _looks_like_metric_row(row)]
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return [
                _normalize_metric_row(row)
                for row in payload["rows"]
                if _looks_like_metric_row(row)
            ]
        if _looks_like_metric_row(payload):
            return [_normalize_metric_row(payload)]

    rows = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if _looks_like_metric_row(value):
            rows.append(_normalize_metric_row(value))
        index = start + end
    return rows


def summarize_pairs(
    rows: list[dict[str, Any]],
    pair_specs: list[dict[str, str]],
    *,
    budget: str,
) -> dict[str, Any]:
    by_scene_budget = {
        (row["scene"], row["budget"]): row for row in rows if row.get("budget") is not None
    }
    pair_rows = []
    for spec in pair_specs:
        baseline = _require_row(by_scene_budget, spec["baseline_scene"], budget)
        active = _require_row(by_scene_budget, spec["active_scene"], budget)
        deltas = {
            metric: _finite_or_none(active.get(metric)) - _finite_or_none(baseline.get(metric))
            for metric in METRIC_NAMES
        }
        pair_rows.append(
            {
                "group": spec["group"],
                "seed": spec["seed"],
                "budget": budget,
                "baseline_scene": spec["baseline_scene"],
                "active_scene": spec["active_scene"],
                "baseline": _metric_subset(baseline),
                "active": _metric_subset(active),
                "delta": deltas,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        groups.setdefault(row["group"], []).append(row)

    group_rows = []
    for group_name in sorted(groups):
        members = groups[group_name]
        group_rows.append(
            {
                "group": group_name,
                "budget": budget,
                "count": len(members),
                "seeds": [row["seed"] for row in members],
                "mean_baseline": {
                    metric: _mean(row["baseline"][metric] for row in members)
                    for metric in METRIC_NAMES
                },
                "mean_active": {
                    metric: _mean(row["active"][metric] for row in members)
                    for metric in METRIC_NAMES
                },
                "mean_delta": {
                    metric: _mean(row["delta"][metric] for row in members)
                    for metric in METRIC_NAMES
                },
            }
        )

    return {
        "budget": budget,
        "metric_directions": {
            "psnr": "higher is better",
            "ssim": "higher is better",
            "lpips": "lower is better",
            "fps": "higher is better",
        },
        "pairs": pair_rows,
        "groups": group_rows,
    }


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _read_pairs_file(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"pairs": payload}
    if not isinstance(payload, dict):
        raise SystemExit(f"Pairs file must contain an object or list, got {type(payload).__name__}.")
    return payload


def _parse_pair_spec(spec: str) -> dict[str, str]:
    parts = spec.split(":")
    if len(parts) != 4 or any(not part for part in parts):
        raise SystemExit(
            "Invalid --pair value. Expected GROUP:SEED:BASELINE_SCENE:ACTIVE_SCENE, "
            f"got {spec!r}."
        )
    group, seed, baseline_scene, active_scene = parts
    return {
        "group": group,
        "seed": seed,
        "baseline_scene": baseline_scene,
        "active_scene": active_scene,
    }


def _parse_pairs_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_pairs = payload.get("pairs", [])
    if not isinstance(raw_pairs, list):
        raise SystemExit("Pairs file field 'pairs' must be a list.")
    pairs = []
    for index, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise SystemExit(f"Pairs file entry {index} must be an object.")
        try:
            group = str(item["group"])
            seed = str(item["seed"])
            baseline_scene = str(item["baseline_scene"])
            active_scene = str(item["active_scene"])
        except KeyError as exc:
            raise SystemExit(f"Pairs file entry {index} is missing field {exc.args[0]!r}.") from exc
        pairs.append(
            {
                "group": group,
                "seed": seed,
                "baseline_scene": baseline_scene,
                "active_scene": active_scene,
            }
        )
    return pairs


def _normalize_metric_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError("metric row must be an object")
    normalized = dict(row)
    normalized["scene"] = str(normalized["scene"])
    normalized["budget"] = _normalize_budget(normalized["budget"])
    for metric in METRIC_NAMES:
        normalized[metric] = _finite_or_none(normalized.get(metric))
    return normalized


def _looks_like_metric_row(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("scene"), str)
        and "budget" in value
        and all(metric in value for metric in METRIC_NAMES)
    )


def _normalize_budget(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("budget_"):
        text = text.removeprefix("budget_")
    return f"{int(text):03d}"


def _require_row(rows: dict[tuple[str, str], dict[str, Any]], scene: str, budget: str) -> dict[str, Any]:
    try:
        return rows[(scene, budget)]
    except KeyError as exc:
        raise SystemExit(f"Missing metrics row for scene={scene!r}, budget={budget!r}.") from exc


def _metric_subset(row: dict[str, Any]) -> dict[str, float]:
    return {metric: _finite_or_none(row.get(metric)) for metric in METRIC_NAMES}


def _finite_or_none(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected finite metric value, got {value!r}.")
    return number


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return math.nan
    return sum(items) / len(items)


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| Group | Seeds | Delta PSNR | Delta SSIM | Delta LPIPS | Delta FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["groups"]:
        delta = row["mean_delta"]
        lines.append(
            "| {group} | {count} | {psnr:.3f} | {ssim:.3f} | {lpips:.3f} | {fps:.3f} |".format(
                group=row["group"],
                count=row["count"],
                psnr=delta["psnr"],
                ssim=delta["ssim"],
                lpips=delta["lpips"],
                fps=delta["fps"],
            )
        )
    lines.extend(
        [
            "",
            "| Group | Seed | Baseline | Active | Delta PSNR | Delta SSIM | Delta LPIPS |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["pairs"]:
        delta = row["delta"]
        lines.append(
            "| {group} | {seed} | `{baseline}` | `{active}` | {psnr:.3f} | {ssim:.3f} | {lpips:.3f} |".format(
                group=row["group"],
                seed=row["seed"],
                baseline=row["baseline_scene"],
                active=row["active_scene"],
                psnr=delta["psnr"],
                ssim=delta["ssim"],
                lpips=delta["lpips"],
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
