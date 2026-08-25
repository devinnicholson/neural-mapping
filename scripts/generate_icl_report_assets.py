#!/usr/bin/env python3
"""Generate paper tables and a study card from the audited ICL-NUIM record."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from build_icl_benchmark_record import METRICS, audit_record


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _decision_label(value: bool) -> str:
    return "Pass" if value else "Fail"


def _method_means(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = record["seed_42_controls"]
    return {
        method: {
            metric: statistics.mean(
                row["metrics"][metric] for row in rows if row["method"] == method
            )
            for metric in METRICS
        }
        for method in ("random", "trajectory", "pose", "active")
    }


def generate_main_table(record: dict[str, Any]) -> str:
    rows = []
    for trajectory, values in sorted(record["trajectory_summary"].items()):
        rows.append(
            f"{trajectory.upper()} & {_signed(values['psnr'])} & {_signed(values['ssim'])} & "
            f"{_signed(values['lpips'])} & {_signed(values['raw_depth_abs_rel'])} & "
            f"{_signed(values['geometry_fscore_05cm'])} \\\\"
        )
    summary = record["active_random_summary"]
    rows.append("\\midrule")
    rows.append(
        "Mean & "
        + " & ".join(
            _signed(summary[metric]["mean"])
            for metric in ("psnr", "ssim", "lpips", "raw_depth_abs_rel", "geometry_fscore_05cm")
        )
        + " \\\\"
    )
    return "\n".join(
        [
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Trajectory & $\\Delta$PSNR $\\uparrow$ & $\\Delta$SSIM $\\uparrow$ & $\\Delta$LPIPS $\\downarrow$ & $\\Delta$AbsRel $\\downarrow$ & $\\Delta F_{5}$ $\\uparrow$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )


def generate_control_table(record: dict[str, Any]) -> str:
    means = _method_means(record)
    labels = {"random": "Random", "trajectory": "Temporal coverage", "pose": "Pose coverage", "active": "Depth-gradient + pose"}
    rows = []
    for method in ("random", "trajectory", "pose", "active"):
        values = means[method]
        rows.append(
            f"{labels[method]} & {values['psnr']:.3f} & {values['ssim']:.3f} & "
            f"{values['lpips']:.3f} & {values['raw_depth_abs_rel']:.3f} & "
            f"{values['geometry_fscore_05cm']:.3f} \\\\"
        )
    return "\n".join(
        [
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Method & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & AbsRel $\\downarrow$ & $F_{5}$ $\\uparrow$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )


def generate_diagnostic_table(record: dict[str, Any]) -> str:
    diagnostics = record["uncertainty_diagnostics"]
    rows = [
        f"{row['trajectory'].upper()} & {row['training_seed']} & {row['candidate_frames']} & "
        f"{row['depth_gradient_spearman']:.3f} & {row['depth_gradient_auroc']:.3f} & "
        f"{row['depth_gradient_auprc']:.3f} & {row['depth_gradient_ause']:.3f} \\\\"
        for row in diagnostics
    ]
    rows.extend(
        [
            "\\midrule",
            "Mean & -- & -- & "
            + " & ".join(
                f"{statistics.mean(row[key] for row in diagnostics):.3f}"
                for key in (
                    "depth_gradient_spearman",
                    "depth_gradient_auroc",
                    "depth_gradient_auprc",
                    "depth_gradient_ause",
                )
            )
            + " \\\\"
        ]
    )
    return "\n".join(
        [
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Trajectory & Seed & Views & $\\rho$ & AUROC & AUPRC & AUSE $\\downarrow$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )


def generate_gate_table(record: dict[str, Any]) -> str:
    decision = record["decision"]
    labels = (
        ("h1_primary", "H1: positive mean PSNR and $F_5$"),
        ("h2_replication", "H2: replication sign counts"),
        ("h3_control_superiority", "H3: coverage-control superiority"),
        ("h4_uncertainty", "H4: error-ranking diagnostic"),
        ("perceptual_safety_gate", "LPIPS safety"),
        ("overall_support", "Overall support"),
    )
    rows = [f"{label} & {_decision_label(decision[key])} \\\\" for key, label in labels]
    return "\n".join(
        ["\\begin{tabular}{lr}", "\\toprule", "Rule & Outcome \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
    )


def generate_pair_table(record: dict[str, Any]) -> str:
    rows = []
    for pair in record["paired_comparisons"]:
        values = pair["active_minus_random"]
        rows.append(
            f"{pair['trajectory'].upper()} & {pair['training_seed']} & {_signed(values['psnr'])} & "
            f"{_signed(values['ssim'])} & {_signed(values['lpips'])} & "
            f"{_signed(values['raw_depth_abs_rel'])} & {_signed(values['geometry_fscore_05cm'])} \\\\"
        )
    return "\n".join(
        [
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Trajectory & Seed & $\\Delta$PSNR & $\\Delta$SSIM & $\\Delta$LPIPS & $\\Delta$AbsRel & $\\Delta F_5$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )


def generate_secondary_table(record: dict[str, Any]) -> str:
    labels = {
        "psnr": "PSNR $\\uparrow$",
        "ssim": "SSIM $\\uparrow$",
        "lpips": "LPIPS $\\downarrow$",
        "raw_depth_abs_rel": "Raw AbsRel $\\downarrow$",
        "raw_depth_rmse": "Raw RMSE $\\downarrow$",
        "raw_depth_delta1": "Raw $\\delta_1$ $\\uparrow$",
        "median_aligned_depth_abs_rel": "Aligned AbsRel $\\downarrow$",
        "median_aligned_depth_rmse": "Aligned RMSE $\\downarrow$",
        "median_aligned_depth_delta1": "Aligned $\\delta_1$ $\\uparrow$",
        "geometry_accuracy_mean_m": "Accuracy $\\downarrow$",
        "geometry_completeness_mean_m": "Completeness $\\downarrow$",
        "geometry_chamfer_l1_mean_m": "Chamfer-L1 $\\downarrow$",
        "geometry_precision_05cm": "Precision@5 $\\uparrow$",
        "geometry_recall_05cm": "Recall@5 $\\uparrow$",
        "geometry_fscore_05cm": "$F_5$ $\\uparrow$",
        "geometry_precision_10cm": "Precision@10 $\\uparrow$",
        "geometry_recall_10cm": "Recall@10 $\\uparrow$",
        "geometry_fscore_10cm": "$F_{10}$ $\\uparrow$",
    }
    rows = []
    for metric in METRICS:
        result = record["active_random_summary"][metric]
        interval = result["trajectory_cluster_bootstrap_95_interval"]
        rows.append(
            f"{labels[metric]} & {_signed(result['mean'], 4)} & {result['population_std']:.4f} & "
            f"{result['favorable_pairs']}/8 & {_signed(interval[0], 4)} & {_signed(interval[1], 4)} \\\\"
        )
    return "\n".join(
        [
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Metric & Mean $\\Delta$ & SD & Fav. & CI low & CI high \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )


def write_csvs(record: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "active_random_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory", "training_seed", *METRICS])
        writer.writeheader()
        for pair in record["paired_comparisons"]:
            writer.writerow({"trajectory": pair["trajectory"], "training_seed": pair["training_seed"], **pair["active_minus_random"]})
    with (output_dir / "seed42_controls.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory", "method", *METRICS])
        writer.writeheader()
        for row in record["seed_42_controls"]:
            writer.writerow({"trajectory": row["trajectory"], "method": row["method"], **row["metrics"]})
    keys = [
        "trajectory", "training_seed", "candidate_frames", "depth_gradient_spearman",
        "depth_gradient_auroc", "depth_gradient_auprc", "depth_gradient_risk_coverage_auc",
        "depth_gradient_ause",
    ]
    with (output_dir / "uncertainty_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(record["uncertainty_diagnostics"])


def generate_study_card(record: dict[str, Any]) -> str:
    summary = record["active_random_summary"]
    diagnostics = record["uncertainty_diagnostics"]
    decisions = record["decision"]
    lines = [
        "# ICL-NUIM multi-trajectory confirmatory study",
        "",
        "This study was executed under the frozen protocol in `experiments/protocols/icl_nuim_multitrajectory_v1.json`. All eight matched active–random pairs and all four seed-42 coverage-control sets completed.",
        "",
        "## Headline result",
        "",
        f"The depth-gradient/pose policy improved held-out appearance over random on average (PSNR {_signed(summary['psnr']['mean'])} dB; LPIPS {_signed(summary['lpips']['mean'])}) but did not improve the co-primary 5 cm surface F-score ({_signed(summary['geometry_fscore_05cm']['mean'], 4)}). The prespecified overall-support gate therefore **{_decision_label(decisions['overall_support']).lower()}ed**.",
        "",
        "| Outcome | Mean active − random | Favorable pairs | Favorable trajectories | Trajectory-cluster 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("PSNR", "psnr"), ("LPIPS", "lpips"), ("Raw depth AbsRel", "raw_depth_abs_rel"), ("5 cm surface F-score", "geometry_fscore_05cm")):
        row = summary[key]
        interval = row["trajectory_cluster_bootstrap_95_interval"]
        lines.append(f"| {label} | {_signed(row['mean'], 4)} | {row['favorable_pairs']}/8 | {row['favorable_trajectories']}/4 | [{_signed(interval[0], 4)}, {_signed(interval[1], 4)}] |")
    lines.extend(["", "## Error-ranking diagnostic", "", f"The target-free depth-gradient proxy had positive rank correlation in 8/8 seed models (mean Spearman {statistics.mean(row['depth_gradient_spearman'] for row in diagnostics):.3f}) and AUROC above 0.5 in 8/8 (mean {statistics.mean(row['depth_gradient_auroc'] for row in diagnostics):.3f}). This supports error ranking, not calibrated uncertainty or downstream acquisition benefit.", "", "## Frozen gate outcomes", ""])
    for key in ("h1_primary", "h2_replication", "h3_control_superiority", "h4_uncertainty", "perceptual_safety_gate", "overall_support"):
        lines.append(f"- `{key}`: **{_decision_label(decisions[key])}**")
    lines.extend(["", "## Recompute", "", "```bash", "python scripts/build_icl_benchmark_record.py \\", "  --artifact-root experiments/artifacts/icl_nuim_multitrajectory_v1 \\", "  --protocol experiments/protocols/icl_nuim_multitrajectory_v1.json \\", "  --run-manifest experiments/run_manifests/icl_nuim_multitrajectory_v1.json \\", "  --output experiments/records/icl_nuim_multitrajectory_v1.json", "python scripts/generate_icl_report_assets.py", "```", "", "The study record hashes every compact metric, diagnostic, split manifest, and run-manifest input. The compact diagnostic files also retain SHA-256 digests of their pixel-heavy source reports."])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path("experiments/records/icl_nuim_multitrajectory_v1.json"))
    parser.add_argument("--paper-tables", type=Path, default=Path("paper/tables"))
    parser.add_argument("--csv-dir", type=Path, default=Path("experiments/tables/icl_nuim_multitrajectory_v1"))
    parser.add_argument("--study-card", type=Path, default=Path("docs/icl_nuim_multitrajectory_v1.md"))
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    audit_record(record)
    _write(args.paper_tables / "icl_main_results.tex", generate_main_table(record))
    _write(args.paper_tables / "icl_control_results.tex", generate_control_table(record))
    _write(args.paper_tables / "icl_diagnostics.tex", generate_diagnostic_table(record))
    _write(args.paper_tables / "icl_gates.tex", generate_gate_table(record))
    _write(args.paper_tables / "icl_pair_results.tex", generate_pair_table(record))
    _write(args.paper_tables / "icl_secondary_results.tex", generate_secondary_table(record))
    write_csvs(record, args.csv_dir)
    _write(args.study_card, generate_study_card(record))
    print(json.dumps({"status": "ok", "record": str(args.record), "decision": record["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
