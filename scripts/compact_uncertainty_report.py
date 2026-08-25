#!/usr/bin/env python3
"""Create a reviewable summary of a pixel-heavy uncertainty report.

The raw report remains the authoritative Modal artifact.  The compact report
records its SHA-256 digest and retains every scalar diagnostic while removing
per-pixel/per-frame samples and sparsification curves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_scalars(payload: dict[str, Any]) -> dict[str, int | float | bool | str | None]:
    scalars: dict[str, int | float | bool | str | None] = {}
    for key, value in payload.items():
        if value is None or isinstance(value, (bool, str, int)):
            scalars[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            scalars[key] = value
    return scalars


def compact_report(source: Path) -> dict[str, Any]:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("signals"), dict):
        raise ValueError(f"Malformed uncertainty report: {source}")

    signals: dict[str, Any] = {}
    for name, signal in raw["signals"].items():
        if not isinstance(signal, dict):
            raise ValueError(f"Malformed signal {name!r} in {source}")
        compact = _finite_scalars(signal)
        sparsification = signal.get("sparsification")
        if isinstance(sparsification, dict):
            compact["sparsification"] = _finite_scalars(sparsification)
        bins = signal.get("uncertainty_bins")
        if isinstance(bins, list):
            compact["uncertainty_bins"] = bins
        signals[name] = compact

    return {
        "schema_version": 1,
        "status": "compact_summary",
        "source": {
            "filename": source.name,
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
            "omitted_fields": ["frames", "signals.*.sparsification.curve"],
        },
        "metadata": raw.get("metadata", {}),
        "signals": signals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = compact_report(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "source": payload["source"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
