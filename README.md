# Uncertainty-Aware 3D Gaussian Neural Mapping

Research project roadmap for calibrated uncertainty in 3D Gaussian scene maps.

Core question:

> Can uncertainty maps from a 3D Gaussian scene representation predict where novel-view RGB, depth, or geometry will fail under partial indoor observations?

Primary roadmap: [ROADMAP.md](ROADMAP.md)

## Current Status

The current strongest result is an offline active-view selection signal for
Nerfstudio/Splatfacto scenes:

- Train budget-25 seed models.
- Score candidate views with ensemble RGB disagreement.
- Aggregate each frame by top-decile uncertainty.
- Expand to budget 50 with a `score-pose-hybrid` selector that mixes
  uncertainty tail risk and camera-pose diversity.

As of 2026-06-08 UTC, this rule improved held-out budget-50 quality over
same-seed random selection on repeated `dozer`, `redwoods2`, and `library`
Nerfstudio sample splits. The strongest clean replication is still `dozer` plus
`redwoods2`, averaging about +0.739 PSNR, +0.023 SSIM, and -0.015 LPIPS across
eight ensemble-tail seeds. On `library` v1-v3 it averaged about +0.496 PSNR,
+0.015 SSIM, and -0.003 LPIPS; LPIPS improved on v2/v3 but regressed on v1.
See [docs/results.md](docs/results.md) for the full tables, Modal run IDs, and
caveats.

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

Prepare a candidate-eval dataset for model-error scoring:

```bash
python scripts/materialize_candidate_eval.py \
  --source-dir data/nerfstudio/poster_available \
  --base-split data/splits/poster_available.json \
  --base-budget 25 \
  --output-dir data/candidate_eval/poster_available_active_error
```

Score those candidates with a trained Nerfstudio checkpoint:

```bash
python scripts/score_candidate_frames.py \
  --load-config outputs/runs/poster_modal_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/.../config.yml \
  --candidate-data data/candidate_eval/poster_available_active_error \
  --score-metric lpips \
  --output data/scores/poster_available_active_error.json
```

Evaluate whether simple frame-level uncertainty baselines predict those
candidate errors:

```bash
python scripts/evaluate_frame_uncertainty.py \
  --frames data/nerfstudio/poster_available/transforms.json \
  --split-json data/splits/poster_available.json \
  --budget 25 \
  --scores data/scores/poster_available_active_error.json \
  --error-metric lpips \
  --score-signal-fields mean_transmittance low_accumulation_fraction \
  --bad-quantile 0.8 \
  --output outputs/reports/poster_available_frame_uncertainty.json
```

This compares nearest-training-camera distance, temporal distance, and a
uniform control against held-out frame errors using the same
uncertainty-alignment metrics as pixel-level maps. When the score file was
generated by `score_candidate_frames.py`, renderer-derived fields such as
`mean_transmittance` and `low_accumulation_fraction` can be evaluated as
additional frame-level uncertainty proxies.

Compute uncertainty/error alignment metrics:

```bash
python scripts/compute_uncertainty_metrics.py \
  --input examples/metric_input.json \
  --bad-threshold 0.5 \
  --reliability-bins 10 \
  --output outputs/reports/example_metrics.json
```

The summary includes equal-count `uncertainty_bins` for reliability-style plots:
each bin is sorted by increasing uncertainty and reports observed mean error,
plus bad-sample fraction when `--bad-threshold` is set.

On a Nerfstudio/Splatfacto runtime, evaluate per-pixel renderer confidence maps
directly against RGB error maps:

```bash
python scripts/evaluate_render_uncertainty_maps.py \
  --load-config outputs/runs/poster_modal_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/.../config.yml \
  --candidate-data data/candidate_eval/poster_available_active_error \
  --error-metric rgb-l1 \
  --signals transmittance local-mean-transmittance local-std-transmittance accumulation-gradient depth-gradient \
  --patch-size 15 \
  --bad-error-quantile 0.8 \
  --max-pixels-per-frame 50000 \
  --output outputs/reports/poster_available_render_uncertainty_maps.json
```

Evaluate pixel-level ensemble RGB disagreement when multiple independently
trained seed models are available:

```bash
python scripts/evaluate_ensemble_uncertainty_maps.py \
  --load-config outputs/runs/seed_a/splatfacto/budget_025/train/unnamed/splatfacto/.../config.yml \
                outputs/runs/seed_b/splatfacto/budget_025/train/unnamed/splatfacto/.../config.yml \
  --candidate-data data/candidate_eval/poster_available_active_error \
  --error-metric rgb-l1 \
  --bad-error-quantile 0.8 \
  --max-pixels-per-frame 50000 \
  --output outputs/reports/poster_available_ensemble_uncertainty_maps.json
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

This writes explicit `train_filenames`, `val_filenames`, and `test_filenames`
fields so Nerfstudio uses the intended split instead of falling back to its
dataparser fraction split.

## Heavy Stack Boundary

The first heavy integration should be a separate Nerfstudio/Splatfacto environment on Linux with NVIDIA CUDA. This repo should initially consume exported frame manifests, rendered outputs, depth maps, error maps, and uncertainty maps rather than requiring Nerfstudio at import time.

Heavy-stack runbook: [docs/gpu-baseline-bringup.md](docs/gpu-baseline-bringup.md)

SLURM cluster runbook: [docs/cluster-slurm.md](docs/cluster-slurm.md)

Modal workflow: [docs/modal.md](docs/modal.md)
