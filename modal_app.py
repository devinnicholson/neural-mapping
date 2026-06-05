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
    .apt_install("git", "curl", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
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
    .add_local_dir("scripts", remote_path=str(REMOTE_PROJECT / "scripts"), copy=True)
    .add_local_dir("src", remote_path=str(REMOTE_PROJECT / "src"), copy=True)
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_PROJECT / "pyproject.toml"), copy=True)
)

app = modal.App(APP_NAME)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = str(REMOTE_PROJECT / "src")
    if env:
        merged_env.update(env)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE_PROJECT, env=merged_env, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _budget_dir_name(budget: int) -> str:
    return f"budget_{budget:03d}"


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
from gsplat.cuda import _wrapper as gsplat_wrapper

print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nerfstudio": "ok",
    "gsplat": "ok",
    "gsplat_cuda_extension": gsplat_wrapper._C is not None,
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if gsplat_wrapper._C is None:
    raise SystemExit("gsplat CUDA extension is unavailable")
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
        "split_json": str(split_json),
        "materialized_root": str(materialized_root),
        "summary": _read_json(materialized_root / "materialization_summary.json"),
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
        ]
    )

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


@app.local_entrypoint()
def main(
    action: str = "smoke",
    budget: int = 25,
    iterations: int = 3000,
    scene_name: str = "poster_modal_smoke",
    render_outputs: bool = True,
) -> None:
    """Run Modal workflow stages from the local CLI."""

    if action == "env":
        print(env_check.remote())
    elif action == "prepare":
        print(prepare_poster_sample.remote())
    elif action == "train":
        ns_data_dir = str(DATA_ROOT / "nerfstudio_splits" / "poster_available" / _budget_dir_name(budget))
        print(
            train_splatfacto.remote(
                ns_data_dir=ns_data_dir,
                scene_name=scene_name,
                budget=budget,
                max_num_iterations=iterations,
            )
        )
    elif action == "eval":
        print(eval_latest_run.remote(scene_name=scene_name, budget=budget, render_outputs=render_outputs))
    elif action == "metrics":
        for row in collect_metric_rows.remote():
            print(json.dumps(row, indent=2))
    elif action == "smoke":
        print(env_check.remote())
        print(prepare_poster_sample.remote())
        ns_data_dir = str(DATA_ROOT / "nerfstudio_splits" / "poster_available" / _budget_dir_name(budget))
        print(
            train_splatfacto.remote(
                ns_data_dir=ns_data_dir,
                scene_name=scene_name,
                budget=budget,
                max_num_iterations=iterations,
            )
        )
        print(eval_latest_run.remote(scene_name=scene_name, budget=budget, render_outputs=render_outputs))
    else:
        raise ValueError(f"Unknown action {action!r}. Use env, prepare, train, eval, metrics, or smoke.")
