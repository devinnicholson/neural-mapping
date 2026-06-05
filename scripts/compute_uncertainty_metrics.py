#!/usr/bin/env python3
"""Compute lightweight uncertainty/error alignment metrics from JSON arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uncertainty_3dgs.metrics import evaluate_uncertainty, load_metric_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="JSON file with 'uncertainty' and 'error' arrays, plus optional 'mask'.",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=None,
        help="Error threshold for AUROC/AUPRC bad-sample detection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_metric_json(args.input)
    if "uncertainty" not in payload:
        raise SystemExit("input JSON is missing required key: uncertainty")
    if "error" not in payload:
        raise SystemExit("input JSON is missing required key: error")

    summary = evaluate_uncertainty(
        payload["uncertainty"],
        payload["error"],
        bad_threshold=args.bad_threshold,
        mask=payload.get("mask"),
    )

    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
