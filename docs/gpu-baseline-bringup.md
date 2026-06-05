# GPU Baseline Bring-Up

This project needs a Linux machine with an NVIDIA CUDA GPU for the Nerfstudio/Splatfacto baseline. The local lightweight utilities can run on macOS, but Splatfacto/gsplat training should happen on a CUDA box.

## Goal

Bring up the first full baseline:

1. Install Nerfstudio + Splatfacto/gsplat on Linux CUDA.
2. Run a tiny Nerfstudio smoke test.
3. Prepare one Replica or placeholder scene.
4. Train Splatfacto on a small frame budget.
5. Render held-out RGB/depth outputs.
6. Feed exported uncertainty/error arrays into this repo's metric utilities.

## Machine Requirements

Minimum:

- Linux.
- NVIDIA GPU with CUDA support.
- 12 GB VRAM for small scenes.
- Conda or Docker.

Preferred:

- Ubuntu 22.04.
- NVIDIA GPU with 24 GB VRAM.
- Docker with NVIDIA Container Toolkit, or a clean conda environment.

This is not a good fit for Apple Silicon because the primary stack depends on CUDA-oriented packages.

## Recommended Path: Nerfstudio Docker

Use Docker first if the GPU host supports it. It avoids many CUDA and tiny-cuda-nn build mismatches.

On the GPU machine:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Then follow the current Nerfstudio Docker instructions from the official installation guide.

Mount this project into the container with absolute paths, for example:

```bash
docker run --gpus all -it --rm \
  -v /ABS/PATH/uncertainty-3dgs-neural-mapping:/workspace/project \
  -v /ABS/PATH/datasets:/workspace/datasets \
  -v /ABS/PATH/outputs:/workspace/outputs \
  --workdir /workspace/project \
  nerfstudio/nerfstudio:latest
```

Inside the container:

```bash
ns-train --help
ns-render --help
ns-eval --help
```

## Fallback Path: Conda Install

Use this when Docker is unavailable.

```bash
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with a CUDA build compatible with the GPU host, then install Nerfstudio dependencies. Check the official Nerfstudio installation guide before pinning exact CUDA/PyTorch versions.

Typical sequence:

```bash
pip install ninja
pip install "setuptools<82" "numpy<2.0.0,>=1.26.0"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install nerfstudio
pip install gsplat
ns-install-cli
```

If the cluster driver reports CUDA 12.8, avoid unpinned `pip install torch` defaults that may pull wheels requiring a newer driver.

If `tiny-cuda-nn` fails, use the official Nerfstudio troubleshooting guidance for the host GPU architecture.

## First Baseline Smoke Test

Use Nerfstudio's quickstart `poster` sample before touching Replica:

```bash
ns-download-data nerfstudio --capture-name=poster
ns-train splatfacto --help
ns-train splatfacto --data data/nerfstudio/poster
```

If the Google Drive-backed downloader fails, use the Hugging Face mirror:

```bash
pip install "huggingface_hub[cli,hf_xet]"
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nerfstudioteam/datasets",
    repo_type="dataset",
    local_dir="data/nerfstudio",
    allow_patterns=["poster/**"],
)
PY
```

Expected result:

- `splatfacto` is available as a method.
- CUDA kernels compile on first use.
- A small training run can start without import/build errors.

## First Real Project Run

Target a single scene first:

- Dataset: Replica if available, otherwise a tiny COLMAP/Nerfstudio-compatible scene.
- Training budget: 25 frames.
- Holdouts: fixed validation/calibration and test split.
- Method: Splatfacto.

Generate split manifests with the lightweight local utility:

```bash
python scripts/generate_splits.py \
  --frames /ABS/PATH/frame_manifest.txt \
  --budgets 25 50 100 200 \
  --val-count 20 \
  --test-count 50 \
  --scene replica_room0 \
  --seed 20260529 \
  --output data/splits/replica_room0.json
```

Use the split manifest to create Nerfstudio train/eval inputs. The first implementation can be manual: copy or symlink selected training images into a budget-specific dataset directory.

Recommended output layout:

```text
outputs/runs/
  replica_room0/
    splatfacto/
      budget_025/
        train/
        render_test/
        metrics/
      budget_050/
      budget_100/
      budget_200/
```

## What To Export Back Into This Harness

For every held-out test frame, save:

- rendered RGB
- rendered depth
- ground-truth RGB
- ground-truth depth
- RGB error map
- depth error map
- uncertainty maps
- valid-pixel mask

The lightweight metric CLI expects JSON arrays shaped like:

```json
{
  "uncertainty": [[0.05, 0.12], [0.30, 0.91]],
  "error": [[0.02, 0.08], [0.25, 0.88]],
  "mask": [[true, true], [true, true]]
}
```

Run:

```bash
python scripts/compute_uncertainty_metrics.py \
  --input outputs/runs/replica_room0/splatfacto/budget_025/metrics/depth_uncertainty_input.json \
  --bad-threshold 0.05 \
  --output outputs/reports/replica_room0_budget_025_depth_uncertainty.json
```

## Immediate Done Criteria

The first heavy-stack milestone is complete when:

- `nvidia-smi` works on the GPU host.
- `ns-train splatfacto --help` works.
- One tiny Splatfacto run starts successfully.
- One scene has a fixed split manifest.
- One held-out render/export can be evaluated by `compute_uncertainty_metrics.py`.

## References

- Nerfstudio installation: https://docs.nerf.studio/quickstart/installation.html
- Splatfacto docs: https://docs.nerf.studio/nerfology/methods/splat.html
- gsplat docs: https://docs.gsplat.studio/main/
