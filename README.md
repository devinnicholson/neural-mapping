# Uncertainty-Aware 3D Gaussian Neural Mapping

Research project roadmap for calibrated uncertainty in 3D Gaussian scene maps.

Core question:

> Can uncertainty maps from a 3D Gaussian scene representation predict where novel-view RGB, depth, or geometry will fail under partial indoor observations?

Primary roadmap: [ROADMAP.md](ROADMAP.md)

## Current Build Target

This repository is being built as a lightweight research harness around a heavier Nerfstudio/Splatfacto workflow.

Initial local tooling focuses on:

- dataset split manifests for Replica, ScanNet++, ScanNet, and TUM-style frame pools
- uncertainty/error metric computation
- experiment configuration templates
- reproducible protocol documentation

Heavy training dependencies such as Nerfstudio, PyTorch, CUDA, and gsplat should stay outside the lightweight utilities until the baseline pipeline is ready.

## Repository Layout

- `configs/`: machine-readable experiment templates.
- `docs/`: implementation plan, experiment protocol, and literature map.
- `scripts/`: lightweight command-line utilities.
- `src/uncertainty_3dgs/`: dependency-light split and metric helpers.
- `tests/`: standard-library unit tests.
- `data/`: local datasets, processed manifests, and split files.
- `outputs/`: generated metrics, reports, and run artifacts.

## Lightweight Commands

Run tests:

```bash
make test
```

Generate a deterministic split manifest:

```bash
python scripts/generate_splits.py \
  --frames examples/frames.txt \
  --budgets 4 6 \
  --val-count 2 \
  --test-count 2 \
  --scene example \
  --seed 7 \
  --selection-method random \
  --output data/splits/example_split.json
```

Use `--selection-method farthest-index` for a lightweight trajectory coverage
baseline that keeps the same validation/test holdouts and selects training
frames with farthest-first coverage over input-frame order.
Use `--selection-method farthest-pose` with a Nerfstudio `transforms.json` to
select training frames by farthest-first coverage over camera-center positions.

Generate an active-expansion split from a locked seed split:

```bash
python scripts/generate_active_split.py \
  --frames data/nerfstudio/poster_available/transforms.json \
  --base-split data/splits/poster_available.json \
  --base-budget 25 \
  --target-budget 50 \
  --scene poster_available_active_pose \
  --strategy pose-novelty \
  --output data/splits/poster_available_active_pose.json
```

`--strategy pose-novelty` keeps the seed train/val/test split fixed and adds
frames that are farthest from the current seed set in camera-center space.
`--strategy score-desc --scores scores.json` can be used later for real
model-error or uncertainty scores.

Compute uncertainty/error alignment metrics:

```bash
python scripts/compute_uncertainty_metrics.py \
  --input examples/metric_input.json \
  --bad-threshold 0.5 \
  --output outputs/reports/example_metrics.json
```

Run both example commands:

```bash
make smoke
```

Prepare a filtered Nerfstudio dataset view when `transforms.json` references missing image files:

```bash
python scripts/filter_nerfstudio_transforms.py \
  --input-dir data/nerfstudio/poster \
  --output-dir data/nerfstudio/poster_available
```

Materialize budgeted Nerfstudio dataset directories from a split manifest:

```bash
python scripts/materialize_nerfstudio_split.py \
  --source-dir data/nerfstudio/poster_available \
  --split-json data/splits/poster_available.json \
  --output-root data/nerfstudio_splits/poster_available \
  --budgets 25 50
```

## Heavy Stack Boundary

The first heavy integration should be a separate Nerfstudio/Splatfacto environment on Linux with NVIDIA CUDA. This repo should initially consume exported frame manifests, rendered outputs, depth maps, error maps, and uncertainty maps rather than requiring Nerfstudio at import time.

Heavy-stack runbook: [docs/gpu-baseline-bringup.md](docs/gpu-baseline-bringup.md)

SLURM cluster runbook: [docs/cluster-slurm.md](docs/cluster-slurm.md)

Modal workflow: [docs/modal.md](docs/modal.md)
