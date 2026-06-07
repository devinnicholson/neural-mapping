"""Modal entrypoints for the uncertainty-aware 3DGS workflow.

Run examples:

    modal run modal_app.py --action env
    modal run modal_app.py --action prepare
    modal run modal_app.py --action smoke --budget 25 --iterations 3000
    modal run modal_app.py --action eval --scene-name poster_modal_smoke --budget 25

The Modal workflow mirrors the SLURM smoke path:

    download -> filter -> split -> materialize -> train Splatfacto -> eval
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


APP_NAME = "uncertainty-3dgs-neural-mapping"
REMOTE_PROJECT = Path("/workspace/neural-mapping")
DATA_ROOT = REMOTE_PROJECT / "data"
OUTPUT_ROOT = REMOTE_PROJECT / "outputs"

DEFAULT_GPU = os.environ.get("MODAL_GPU", "L4")
TRAIN_TIMEOUT_SECONDS = int(os.environ.get("MODAL_TRAIN_TIMEOUT_SECONDS", "7200"))
EVAL_TIMEOUT_SECONDS = int(os.environ.get("MODAL_EVAL_TIMEOUT_SECONDS", "3600"))


data_volume = modal.Volume.from_name("u3dgs-data", create_if_missing=True)
outputs_volume = modal.Volume.from_name("u3dgs-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("u3dgs-hf-cache", create_if_missing=True)

volumes = {
    str(DATA_ROOT): data_volume,
    str(OUTPUT_ROOT): outputs_volume,
    "/root/.cache/huggingface": hf_cache_volume,
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "4"})
    .apt_install("git", "curl", "ffmpeg", "build-essential", "clang", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip wheel 'setuptools<82' "
        "'numpy<2.0.0,>=1.26.0' ninja"
    )
    .run_commands(
        "python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 "
        "torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "python -m pip install nerfstudio==1.1.5 gsplat==1.4.0 "
        "'huggingface_hub[hf_xet]' 'numpy<2.0.0,>=1.26.0' 'setuptools<82'"
    )
    .run_commands(
        "python -c 'from gsplat.cuda._backend import _C; "
        'print("gsplat CUDA extension ready", type(_C).__name__)\''
    )
    .add_local_dir("scripts", remote_path=str(REMOTE_PROJECT / "scripts"), copy=True)
    .add_local_dir("src", remote_path=str(REMOTE_PROJECT / "src"), copy=True)
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_PROJECT / "pyproject.toml"), copy=True)
)

app = modal.App(APP_NAME)


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> None:
    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = str(REMOTE_PROJECT / "src")
    merged_env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    merged_env.setdefault("MAX_JOBS", "4")
    if env:
        merged_env.update(env)
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=REMOTE_PROJECT,
        env=merged_env,
        input=input_text,
        text=input_text is not None,
        check=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _budget_dir_name(budget: int) -> str:
    return f"budget_{budget:03d}"


def _compact_uncertainty_signals(signals: object) -> dict[str, dict[str, Any]]:
    if not isinstance(signals, dict):
        return {}
    compact: dict[str, dict[str, Any]] = {}
    keys = (
        "count",
        "mean_error",
        "spearman",
        "auroc",
        "auprc",
        "bad_threshold",
        "bad_fraction",
        "risk_coverage_auc",
    )
    for signal_name, summary in signals.items():
        if not isinstance(summary, dict):
            continue
        row = {key: summary.get(key) for key in keys if key in summary}
        sparsification = summary.get("sparsification")
        if isinstance(sparsification, dict):
            row["ause"] = sparsification.get("ause")
        bins = summary.get("uncertainty_bins")
        if isinstance(bins, list):
            row["uncertainty_bins"] = _compact_uncertainty_bins(bins)
        compact[str(signal_name)] = row
    return compact


def _compact_uncertainty_bins(bins: object) -> list[dict[str, Any]]:
    if not isinstance(bins, list):
        return []
    keys = (
        "bin",
        "count",
        "lower_quantile",
        "upper_quantile",
        "mean_uncertainty",
        "mean_error",
        "bad_fraction",
    )
    compact = []
    for item in bins:
        if isinstance(item, dict):
            compact.append({key: item.get(key) for key in keys if key in item})
    return compact


def _resolve_output_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return OUTPUT_ROOT / candidate


def _split_fields(value: str) -> list[str]:
    return [field for field in value.replace(",", " ").split() if field]


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=900)
def env_check() -> dict[str, Any]:
    """Verify the Modal GPU image can import torch, Nerfstudio, and gsplat."""

    _run(["nvidia-smi"])
    _run(["which", "nvcc"])
    _run(["nvcc", "--version"])
    script = """
import json
import torch
import nerfstudio
import gsplat
from gsplat.cuda._backend import _C as gsplat_backend

print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nerfstudio": "ok",
    "gsplat": "ok",
    "gsplat_path": getattr(gsplat, "__file__", None),
    "gsplat_backend": type(gsplat_backend).__name__,
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
"""
    _run(["python", "-c", script])
    return {"status": "ok", "gpu": DEFAULT_GPU}


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=1800)
def prepare_poster_sample(
    capture_name: str = "poster",
    scene_name: str = "poster_available",
    budgets: list[int] | None = None,
    val_count: int = 10,
    test_count: int = 10,
    seed: int = 20260529,
    selection_method: str = "random",
) -> dict[str, Any]:
    """Download, filter, split, and materialize the Nerfstudio poster sample."""

    budgets = budgets or [25, 50]
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "nerfstudio").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "splits").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "nerfstudio_splits").mkdir(parents=True, exist_ok=True)

    print("Downloading Hugging Face Nerfstudio sample mirror", flush=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="nerfstudioteam/datasets",
        repo_type="dataset",
        local_dir=str(DATA_ROOT / "nerfstudio"),
        allow_patterns=[f"{capture_name}/**"],
    )

    source_dir = DATA_ROOT / "nerfstudio" / capture_name
    filtered_dir = DATA_ROOT / "nerfstudio" / scene_name
    split_json = DATA_ROOT / "splits" / f"{scene_name}.json"
    materialized_root = DATA_ROOT / "nerfstudio_splits" / scene_name

    _run(
        [
            "python",
            "scripts/filter_nerfstudio_transforms.py",
            "--input-dir",
            str(source_dir),
            "--output-dir",
            str(filtered_dir),
            "--min-frames",
            "8",
        ]
    )
    _run(
        [
            "python",
            "scripts/generate_splits.py",
            "--frames",
            str(filtered_dir / "transforms.json"),
            "--budgets",
            *[str(budget) for budget in budgets],
            "--val-count",
            str(val_count),
            "--test-count",
            str(test_count),
            "--scene",
            scene_name,
            "--seed",
            str(seed),
            "--selection-method",
            selection_method,
            "--output",
            str(split_json),
        ]
    )
    _run(
        [
            "python",
            "scripts/materialize_nerfstudio_split.py",
            "--source-dir",
            str(filtered_dir),
            "--split-json",
            str(split_json),
            "--output-root",
            str(materialized_root),
            "--budgets",
            *[str(budget) for budget in budgets],
        ]
    )

    data_volume.commit()
    hf_cache_volume.commit()
    return {
        "status": "ok",
        "scene_name": scene_name,
        "selection_method": selection_method,
        "split_json": str(split_json),
        "materialized_root": str(materialized_root),
        "summary": _read_json(materialized_root / "materialization_summary.json"),
    }


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=900)
def prepare_active_split(
    source_scene_name: str = "poster_available",
    base_split_scene_name: str = "poster_available",
    active_scene_name: str = "poster_available_active_pose",
    base_budget: int = 25,
    target_budget: int = 50,
    strategy: str = "pose-novelty",
    scores_path: str | None = None,
    score_key: str = "score",
    score_weight: float = 0.65,
) -> dict[str, Any]:
    """Create and materialize an active-expansion split from a seed split."""

    source_dir = DATA_ROOT / "nerfstudio" / source_scene_name
    base_split_json = DATA_ROOT / "splits" / f"{base_split_scene_name}.json"
    active_split_json = DATA_ROOT / "splits" / f"{active_scene_name}.json"
    materialized_root = DATA_ROOT / "nerfstudio_splits" / active_scene_name

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source scene directory: {source_dir}")
    if not base_split_json.exists():
        raise FileNotFoundError(f"Missing base split JSON: {base_split_json}")
    if strategy in {"score-desc", "score-pose-hybrid"} and scores_path is None:
        scores_path = str(DATA_ROOT / "scores" / f"{active_scene_name}.json")

    command = [
        "python",
        "scripts/generate_active_split.py",
        "--frames",
        str(source_dir / "transforms.json"),
        "--base-split",
        str(base_split_json),
        "--base-budget",
        str(base_budget),
        "--target-budget",
        str(target_budget),
        "--scene",
        active_scene_name,
        "--strategy",
        strategy,
        "--output",
        str(active_split_json),
    ]
    if scores_path is not None:
        command.extend(["--scores", scores_path, "--score-key", score_key])
    if strategy == "score-pose-hybrid":
        command.extend(["--score-weight", str(score_weight)])
    _run(command)
    _run(
        [
            "python",
            "scripts/materialize_nerfstudio_split.py",
            "--source-dir",
            str(source_dir),
            "--split-json",
            str(active_split_json),
            "--output-root",
            str(materialized_root),
            "--budgets",
            str(base_budget),
            str(target_budget),
        ]
    )

    data_volume.commit()
    return {
        "status": "ok",
        "source_scene_name": source_scene_name,
        "base_split_scene_name": base_split_scene_name,
        "active_scene_name": active_scene_name,
        "strategy": strategy,
        "scores_path": scores_path,
        "score_weight": score_weight,
        "split_json": str(active_split_json),
        "materialized_root": str(materialized_root),
        "summary": _read_json(materialized_root / "materialization_summary.json"),
    }


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=EVAL_TIMEOUT_SECONDS)
def score_candidate_frames(
    source_scene_name: str = "poster_available",
    base_split_scene_name: str = "poster_available",
    score_scene_name: str = "poster_available_active_error",
    seed_scene_name: str = "poster_modal_b25_10k",
    base_budget: int = 25,
    score_metric: str = "lpips",
) -> dict[str, Any]:
    """Score active-selection candidate frames using a seed checkpoint."""

    source_dir = DATA_ROOT / "nerfstudio" / source_scene_name
    base_split_json = DATA_ROOT / "splits" / f"{base_split_scene_name}.json"
    candidate_dir = DATA_ROOT / "candidate_eval" / score_scene_name
    scores_path = DATA_ROOT / "scores" / f"{score_scene_name}.json"
    seed_run_root = OUTPUT_ROOT / "runs" / seed_scene_name / "splatfacto" / _budget_dir_name(base_budget)

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source scene directory: {source_dir}")
    if not base_split_json.exists():
        raise FileNotFoundError(f"Missing base split JSON: {base_split_json}")
    configs = sorted((seed_run_root / "train").glob("**/config.yml"))
    if not configs:
        raise FileNotFoundError(f"No seed config.yml found under {seed_run_root / 'train'}")
    load_config = configs[-1]

    _run(
        [
            "python",
            "scripts/materialize_candidate_eval.py",
            "--source-dir",
            str(source_dir),
            "--base-split",
            str(base_split_json),
            "--base-budget",
            str(base_budget),
            "--output-dir",
            str(candidate_dir),
        ]
    )
    _run(
        [
            "python",
            "scripts/score_candidate_frames.py",
            "--load-config",
            str(load_config),
            "--candidate-data",
            str(candidate_dir),
            "--score-metric",
            score_metric,
            "--output",
            str(scores_path),
        ],
        env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        input_text="y\n",
    )

    data_volume.commit()
    scores = _read_json(scores_path)
    top_scores = scores.get("scores", [])[:10]
    return {
        "status": "ok",
        "source_scene_name": source_scene_name,
        "base_split_scene_name": base_split_scene_name,
        "score_scene_name": score_scene_name,
        "seed_scene_name": seed_scene_name,
        "base_budget": base_budget,
        "score_metric": score_metric,
        "candidate_dir": str(candidate_dir),
        "scores_path": str(scores_path),
        "top_scores": top_scores,
    }


@app.function(image=image, volumes=volumes, timeout=300)
def evaluate_frame_uncertainty_baseline(
    source_scene_name: str = "poster_available",
    base_split_scene_name: str = "poster_available",
    score_scene_name: str = "poster_available_active_error",
    base_budget: int = 25,
    error_metric: str = "lpips",
    bad_quantile: float = 0.8,
    score_signal_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate simple frame-level uncertainty baselines against scored errors."""

    score_signal_fields = score_signal_fields or []
    frames_path = DATA_ROOT / "nerfstudio" / source_scene_name / "transforms.json"
    split_json = DATA_ROOT / "splits" / f"{base_split_scene_name}.json"
    scores_path = DATA_ROOT / "scores" / f"{score_scene_name}.json"
    output_path = (
        OUTPUT_ROOT
        / "reports"
        / "frame_uncertainty"
        / f"{score_scene_name}_budget_{base_budget:03d}_{error_metric}.json"
    )

    command = [
        "python",
        "scripts/evaluate_frame_uncertainty.py",
        "--frames",
        str(frames_path),
        "--split-json",
        str(split_json),
        "--budget",
        str(base_budget),
        "--scores",
        str(scores_path),
        "--error-metric",
        error_metric,
        "--bad-quantile",
        str(bad_quantile),
        "--output",
        str(output_path),
    ]
    if score_signal_fields:
        command.extend(["--score-signal-fields", *score_signal_fields])

    _run(command)

    outputs_volume.commit()
    report = _read_json(output_path)
    return {
        "status": "ok",
        "source_scene_name": source_scene_name,
        "base_split_scene_name": base_split_scene_name,
        "score_scene_name": score_scene_name,
        "base_budget": base_budget,
        "error_metric": error_metric,
        "bad_quantile": bad_quantile,
        "score_signal_fields": score_signal_fields,
        "report_path": str(output_path),
        "signals": _compact_uncertainty_signals(report.get("signals")),
    }


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=EVAL_TIMEOUT_SECONDS)
def evaluate_render_uncertainty_maps(
    source_scene_name: str = "poster_available",
    base_split_scene_name: str = "poster_available",
    report_scene_name: str = "poster_available_render_maps",
    seed_scene_name: str = "poster_modal_b25_10k",
    base_budget: int = 25,
    error_metric: str = "rgb-l1",
    bad_error_quantile: float = 0.8,
    max_pixels_per_frame: int = 50000,
    render_map_signals: list[str] | None = None,
    patch_size: int = 15,
) -> dict[str, Any]:
    """Evaluate per-pixel renderer confidence maps against RGB error maps."""

    render_map_signals = render_map_signals or ["transmittance"]
    source_dir = DATA_ROOT / "nerfstudio" / source_scene_name
    base_split_json = DATA_ROOT / "splits" / f"{base_split_scene_name}.json"
    candidate_dir = DATA_ROOT / "candidate_eval" / report_scene_name
    seed_run_root = OUTPUT_ROOT / "runs" / seed_scene_name / "splatfacto" / _budget_dir_name(base_budget)
    output_path = (
        OUTPUT_ROOT
        / "reports"
        / "render_uncertainty_maps"
        / f"{report_scene_name}_budget_{base_budget:03d}_{error_metric}.json"
    )

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source scene directory: {source_dir}")
    if not base_split_json.exists():
        raise FileNotFoundError(f"Missing base split JSON: {base_split_json}")
    configs = sorted((seed_run_root / "train").glob("**/config.yml"))
    if not configs:
        raise FileNotFoundError(f"No seed config.yml found under {seed_run_root / 'train'}")
    load_config = configs[-1]

    _run(
        [
            "python",
            "scripts/materialize_candidate_eval.py",
            "--source-dir",
            str(source_dir),
            "--base-split",
            str(base_split_json),
            "--base-budget",
            str(base_budget),
            "--output-dir",
            str(candidate_dir),
        ]
    )
    _run(
        [
            "python",
            "scripts/evaluate_render_uncertainty_maps.py",
            "--load-config",
            str(load_config),
            "--candidate-data",
            str(candidate_dir),
            "--error-metric",
            error_metric,
            "--bad-error-quantile",
            str(bad_error_quantile),
            "--max-pixels-per-frame",
            str(max_pixels_per_frame),
            "--signals",
            *render_map_signals,
            "--patch-size",
            str(patch_size),
            "--output",
            str(output_path),
        ],
        env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        input_text="y\n",
    )

    data_volume.commit()
    outputs_volume.commit()
    report = _read_json(output_path)
    return {
        "status": "ok",
        "source_scene_name": source_scene_name,
        "base_split_scene_name": base_split_scene_name,
        "report_scene_name": report_scene_name,
        "seed_scene_name": seed_scene_name,
        "base_budget": base_budget,
        "error_metric": error_metric,
        "bad_error_quantile": bad_error_quantile,
        "max_pixels_per_frame": max_pixels_per_frame,
        "render_map_signals": render_map_signals,
        "patch_size": patch_size,
        "candidate_dir": str(candidate_dir),
        "report_path": str(output_path),
        "signals": _compact_uncertainty_signals(report.get("signals")),
    }


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=EVAL_TIMEOUT_SECONDS)
def evaluate_ensemble_uncertainty_maps(
    source_scene_name: str = "poster_available",
    base_split_scene_name: str = "poster_available",
    report_scene_name: str = "poster_available_ensemble_maps",
    ensemble_scene_names: list[str] | None = None,
    base_budget: int = 25,
    error_metric: str = "rgb-l1",
    bad_error_quantile: float = 0.8,
    max_pixels_per_frame: int = 50000,
) -> dict[str, Any]:
    """Evaluate pixel-level RGB variance across seed models against RGB error."""

    ensemble_scene_names = ensemble_scene_names or []
    if len(ensemble_scene_names) < 2:
        raise ValueError("ensemble_scene_names must contain at least two scene names.")

    source_dir = DATA_ROOT / "nerfstudio" / source_scene_name
    base_split_json = DATA_ROOT / "splits" / f"{base_split_scene_name}.json"
    candidate_dir = DATA_ROOT / "candidate_eval" / report_scene_name
    output_path = (
        OUTPUT_ROOT
        / "reports"
        / "ensemble_uncertainty_maps"
        / f"{report_scene_name}_budget_{base_budget:03d}_{error_metric}.json"
    )

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source scene directory: {source_dir}")
    if not base_split_json.exists():
        raise FileNotFoundError(f"Missing base split JSON: {base_split_json}")

    load_configs = []
    for scene_name in ensemble_scene_names:
        run_root = OUTPUT_ROOT / "runs" / scene_name / "splatfacto" / _budget_dir_name(base_budget)
        configs = sorted((run_root / "train").glob("**/config.yml"))
        if not configs:
            raise FileNotFoundError(f"No config.yml found under {run_root / 'train'}")
        load_configs.append(str(configs[-1]))

    _run(
        [
            "python",
            "scripts/materialize_candidate_eval.py",
            "--source-dir",
            str(source_dir),
            "--base-split",
            str(base_split_json),
            "--base-budget",
            str(base_budget),
            "--output-dir",
            str(candidate_dir),
        ]
    )
    _run(
        [
            "python",
            "scripts/evaluate_ensemble_uncertainty_maps.py",
            "--load-config",
            *load_configs,
            "--candidate-data",
            str(candidate_dir),
            "--error-metric",
            error_metric,
            "--bad-error-quantile",
            str(bad_error_quantile),
            "--max-pixels-per-frame",
            str(max_pixels_per_frame),
            "--output",
            str(output_path),
        ],
        env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        input_text="y\n",
    )

    data_volume.commit()
    outputs_volume.commit()
    report = _read_json(output_path)
    return {
        "status": "ok",
        "source_scene_name": source_scene_name,
        "base_split_scene_name": base_split_scene_name,
        "report_scene_name": report_scene_name,
        "ensemble_scene_names": ensemble_scene_names,
        "base_budget": base_budget,
        "error_metric": error_metric,
        "bad_error_quantile": bad_error_quantile,
        "max_pixels_per_frame": max_pixels_per_frame,
        "candidate_dir": str(candidate_dir),
        "report_path": str(output_path),
        "signals": _compact_uncertainty_signals(report.get("signals")),
    }


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=TRAIN_TIMEOUT_SECONDS)
def train_splatfacto(
    ns_data_dir: str,
    scene_name: str,
    budget: int,
    max_num_iterations: int = 3000,
    downscale_factor: int = 1,
    vis: str = "tensorboard",
) -> dict[str, Any]:
    """Train Splatfacto on a Nerfstudio-compatible dataset directory."""

    run_root = OUTPUT_ROOT / "runs" / scene_name / "splatfacto" / _budget_dir_name(budget)
    run_root.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "ns-train",
            "splatfacto",
            "--output-dir",
            str(run_root / "train"),
            "--max-num-iterations",
            str(max_num_iterations),
            "--vis",
            vis,
            "--viewer.quit-on-train-completion",
            "True",
            "nerfstudio-data",
            "--data",
            ns_data_dir,
            "--downscale-factor",
            str(downscale_factor),
        ],
        input_text="y\n",
    )

    data_volume.commit()
    outputs_volume.commit()
    configs = sorted((run_root / "train").glob("**/config.yml"))
    return {
        "status": "ok",
        "scene_name": scene_name,
        "budget": budget,
        "iterations": max_num_iterations,
        "run_root": str(run_root),
        "latest_config": str(configs[-1]) if configs else None,
    }


@app.function(image=image, gpu=DEFAULT_GPU, volumes=volumes, timeout=EVAL_TIMEOUT_SECONDS)
def eval_latest_run(
    scene_name: str,
    budget: int,
    render_outputs: bool = True,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate the latest Splatfacto run for a scene/budget pair."""

    run_root = OUTPUT_ROOT / "runs" / scene_name / "splatfacto" / _budget_dir_name(budget)
    (run_root / "metrics").mkdir(parents=True, exist_ok=True)
    (run_root / "renders").mkdir(parents=True, exist_ok=True)

    if config_path is None:
        configs = sorted((run_root / "train").glob("**/config.yml"))
        if not configs:
            raise FileNotFoundError(f"No config.yml found under {run_root / 'train'}")
        config_path = str(configs[-1])

    output_path = run_root / "metrics" / "ns_eval.json"
    command = [
        "ns-eval",
        "--load-config",
        config_path,
        "--output-path",
        str(output_path),
    ]
    if render_outputs:
        command.extend(["--render-output-path", str(run_root / "renders" / "eval")])

    _run(command, env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    outputs_volume.commit()
    metrics = _read_json(output_path)
    return {"status": "ok", "metrics_path": str(output_path), "metrics": metrics}


@app.function(image=image, volumes=volumes, timeout=300)
def summarize_report(report_path: str) -> dict[str, Any]:
    """Read a report JSON from the Modal output volume and compact key metrics."""

    path = _resolve_output_path(report_path)
    report = _read_json(path)
    return {
        "status": "ok",
        "report_path": str(path),
        "metadata": report.get("metadata"),
        "signals": _compact_uncertainty_signals(report.get("signals")),
    }


@app.function(image=image, volumes=volumes, timeout=300)
def collect_metric_rows() -> list[dict[str, Any]]:
    """Collect compact metric rows from Modal output volume."""

    rows = []
    for path in sorted(OUTPUT_ROOT.glob("runs/*/splatfacto/budget_*/metrics/ns_eval.json")):
        metrics = _read_json(path)
        parts = path.parts
        scene = parts[parts.index("runs") + 1]
        budget = parts[parts.index("splatfacto") + 1].replace("budget_", "")
        result = metrics["results"]
        rows.append(
            {
                "scene": scene,
                "budget": budget,
                "checkpoint": metrics.get("checkpoint"),
                "psnr": result.get("psnr"),
                "ssim": result.get("ssim"),
                "lpips": result.get("lpips"),
                "fps": result.get("fps"),
                "metrics_path": str(path),
            }
        )
    return rows


@app.function(image=image, volumes=volumes, timeout=300)
def summarize_splits(split_scene_names: list[str], budget: int = 50, base_budget: int = 25) -> list[dict[str, Any]]:
    """Collect compact train/val/test summaries from Modal split JSON files."""

    rows = []
    for scene_name in split_scene_names:
        path = DATA_ROOT / "splits" / f"{scene_name}.json"
        payload = _read_json(path)
        splits = payload.get("splits")
        if not isinstance(splits, dict):
            raise ValueError(f"Split JSON is missing object-valued 'splits': {path}")
        entry = splits.get(str(budget))
        if not isinstance(entry, dict):
            raise ValueError(f"Split JSON {path} is missing budget {budget}.")
        base_entry = splits.get(str(base_budget))
        train = _json_list(entry, "train")
        val = _json_list(entry, "val")
        test = _json_list(entry, "test")
        base_train = _json_list(base_entry, "train") if isinstance(base_entry, dict) else []
        base_train_set = set(base_train)
        added_train = [frame for frame in train if frame not in base_train_set]
        rows.append(
            {
                "scene": scene_name,
                "path": str(path),
                "budget": budget,
                "base_budget": base_budget,
                "train_count": len(train),
                "val_count": len(val),
                "test_count": len(test),
                "added_train_count": len(added_train),
                "train": train,
                "added_train": added_train,
                "val": val,
                "test": test,
            }
        )
    return rows


def _json_list(payload: dict[str, Any], key: str) -> list[str]:
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Expected list-valued {key!r}.")
    return [str(value) for value in values]


@app.local_entrypoint()
def main(
    action: str = "smoke",
    budget: int = 25,
    base_budget: int = 25,
    target_budget: int = 50,
    iterations: int = 3000,
    scene_name: str = "poster_modal_smoke",
    data_scene_name: str = "poster_available",
    capture_name: str = "poster",
    source_data_scene_name: str = "poster_available",
    base_split_scene_name: str = "poster_available",
    selection_method: str = "random",
    split_seed: int = 20260529,
    downscale_factor: int = 1,
    active_strategy: str = "pose-novelty",
    score_path: str = "",
    score_key: str = "score",
    score_weight: float = 0.65,
    score_metric: str = "lpips",
    bad_quantile: float = 0.8,
    score_signal_fields: str = "",
    max_pixels_per_frame: int = 50000,
    render_map_signals: str = "transmittance",
    patch_size: int = 15,
    ensemble_scene_names: str = "",
    report_path: str = "",
    render_outputs: bool = True,
) -> None:
    """Run Modal workflow stages from the local CLI."""

    if action == "env":
        print(env_check.remote())
    elif action == "prepare":
        print(
            prepare_poster_sample.remote(
                capture_name=capture_name,
                scene_name=data_scene_name,
                selection_method=selection_method,
                seed=split_seed,
            )
        )
    elif action == "prepare-active":
        print(
            prepare_active_split.remote(
                source_scene_name=source_data_scene_name,
                base_split_scene_name=base_split_scene_name,
                active_scene_name=data_scene_name,
                base_budget=base_budget,
                target_budget=target_budget,
                strategy=active_strategy,
                scores_path=score_path or None,
                score_key=score_key,
                score_weight=score_weight,
            )
        )
    elif action == "score-candidates":
        print(
            score_candidate_frames.remote(
                source_scene_name=source_data_scene_name,
                base_split_scene_name=base_split_scene_name,
                score_scene_name=data_scene_name,
                seed_scene_name=scene_name,
                base_budget=budget,
                score_metric=score_metric,
            )
        )
    elif action == "frame-uncertainty":
        print(
            evaluate_frame_uncertainty_baseline.remote(
                source_scene_name=source_data_scene_name,
                base_split_scene_name=base_split_scene_name,
                score_scene_name=data_scene_name,
                base_budget=budget,
                error_metric=score_metric,
                bad_quantile=bad_quantile,
                score_signal_fields=_split_fields(score_signal_fields),
            )
        )
    elif action == "render-uncertainty-maps":
        print(
            evaluate_render_uncertainty_maps.remote(
                source_scene_name=source_data_scene_name,
                base_split_scene_name=base_split_scene_name,
                report_scene_name=data_scene_name,
                seed_scene_name=scene_name,
                base_budget=budget,
                error_metric=score_metric,
                bad_error_quantile=bad_quantile,
                max_pixels_per_frame=max_pixels_per_frame,
                render_map_signals=_split_fields(render_map_signals),
                patch_size=patch_size,
            )
        )
    elif action == "ensemble-uncertainty-maps":
        print(
            evaluate_ensemble_uncertainty_maps.remote(
                source_scene_name=source_data_scene_name,
                base_split_scene_name=base_split_scene_name,
                report_scene_name=data_scene_name,
                ensemble_scene_names=_split_fields(ensemble_scene_names),
                base_budget=budget,
                error_metric=score_metric,
                bad_error_quantile=bad_quantile,
                max_pixels_per_frame=max_pixels_per_frame,
            )
        )
    elif action == "train":
        ns_data_dir = str(DATA_ROOT / "nerfstudio_splits" / data_scene_name / _budget_dir_name(budget))
        print(
            train_splatfacto.remote(
                ns_data_dir=ns_data_dir,
                scene_name=scene_name,
                budget=budget,
                max_num_iterations=iterations,
                downscale_factor=downscale_factor,
            )
        )
    elif action == "eval":
        print(eval_latest_run.remote(scene_name=scene_name, budget=budget, render_outputs=render_outputs))
    elif action == "metrics":
        for row in collect_metric_rows.remote():
            print(json.dumps(row, indent=2))
    elif action == "split-summary":
        split_scene_names = _split_fields(data_scene_name)
        if not split_scene_names:
            raise ValueError("--data-scene-name must contain at least one split scene name.")
        for row in summarize_splits.remote(split_scene_names, budget=budget, base_budget=base_budget):
            print(json.dumps(row, indent=2))
    elif action == "report-summary":
        if not report_path:
            raise ValueError("--report-path is required for report-summary.")
        print(json.dumps(summarize_report.remote(report_path), indent=2))
    elif action == "smoke":
        print(env_check.remote())
        print(
            prepare_poster_sample.remote(
                capture_name=capture_name,
                scene_name=data_scene_name,
                selection_method=selection_method,
                seed=split_seed,
            )
        )
        ns_data_dir = str(DATA_ROOT / "nerfstudio_splits" / data_scene_name / _budget_dir_name(budget))
        print(
            train_splatfacto.remote(
                ns_data_dir=ns_data_dir,
                scene_name=scene_name,
                budget=budget,
                max_num_iterations=iterations,
                downscale_factor=downscale_factor,
            )
        )
        print(eval_latest_run.remote(scene_name=scene_name, budget=budget, render_outputs=render_outputs))
    else:
        raise ValueError(
            f"Unknown action {action!r}. "
            "Use env, prepare, score-candidates, frame-uncertainty, "
            "render-uncertainty-maps, ensemble-uncertainty-maps, "
            "prepare-active, train, eval, metrics, split-summary, report-summary, or smoke."
        )
