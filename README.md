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

As of 2026-06-09 UTC, this rule improved held-out budget-50 quality over
same-seed random selection on repeated `dozer`, `redwoods2`, `library`, and
`bww_entrance` Nerfstudio-style splits. The strongest fresh replication is
`bww_entrance` v1-v4, averaging about +2.016 PSNR, +0.031 SSIM, and -0.021
LPIPS. Across `dozer`, `redwoods2`, and `bww_entrance`, the same rule averages
about +1.165 PSNR, +0.026 SSIM, and -0.017 LPIPS across twelve ensemble-tail
seeds. On `library` v1-v3 it averaged about +0.496 PSNR, +0.015 SSIM, and an
LPIPS change of -0.003. On the harder `kitchen` scene, the same rule averaged
about +0.687 PSNR, +0.002 SSIM, and -0.028 LPIPS across v1-v4, but one seed
regressed sharply, so this remains a mixed hard-scene result rather than a
clean replication. See [docs/results.md](docs/results.md) for the full tables,
Modal run IDs, and caveats.

As of 2026-06-11 UTC, the first depth-bearing active loop also runs on
TUM RGB-D `freiburg1_desk`. Across three locked RGB-D split seeds, random
budget-50 training improves over budget 25 on both RGB and held-out depth, and
transmittance-tail active expansion improves over the same-seed random
budget-50 baseline. On v3, the random baseline moved from 15.432 PSNR /
0.576 SSIM / 0.420 LPIPS at budget 25 to 17.676 PSNR / 0.664 SSIM /
0.343 LPIPS at budget 50; transmittance-tail active selection then improved the
budget-50 result to 18.362 PSNR / 0.684 SSIM / 0.319 LPIPS. The same v3 active
run reduced raw depth AbsRel from 0.358 to 0.347 and median-aligned AbsRel
from 0.351 to 0.315 versus random budget 50. This is now a three-seed RGB-D
pilot on `freiburg1_desk`.

The same depth-bearing loop now has a second-sequence check on TUM RGB-D
`freiburg1_room`. On room v1, transmittance-tail active expansion improved
random budget 50 from 17.343 PSNR / 0.635 SSIM / 0.344 LPIPS to 17.844 PSNR /
0.643 SSIM / 0.345 LPIPS and reduced raw depth RMSE from 0.957m to 0.919m. On
room v2, local-mean-transmittance active selection traded RGB quality for depth:
PSNR moved from 16.886 to 16.621, but raw AbsRel improved from 0.507 to 0.490
and raw RMSE improved from 1.046m to 1.003m versus random budget 50. On room
v3, accumulation-gradient active selection was a small all-around win over
random budget 50: 17.571 to 17.590 PSNR, 0.647 to 0.650 SSIM, raw AbsRel 0.461
to 0.458, and median-aligned AbsRel 0.410 to 0.390. Across three room seeds,
adaptive active selection improves depth consistently, while RGB is positive on
two of three seeds and mixed on average. A fixed transmittance-only room control
was less robust: it improved room v2 depth versus random budget 50, but
regressed room v3 RGB and depth. A fixed accumulation-gradient room control was
also mixed: it was strong on room v1 and modestly useful on room v3, but
regressed room v2. A room v2/v3 rank ensemble of transmittance and
local-mean-transmittance was also only diagnostic: v2 improved depth versus
random budget 50 but still regressed RGB, while v3 regressed both RGB and depth
versus random. `freiburg1_room` currently needs signal-specific selection
rather than a one-size-fits-all RGB-D rule.

As of 2026-06-22 UTC, a third TUM RGB-D sequence, `freiburg1_xyz`, has a
matched random/depth-gradient/transmittance check. Random budget 50 improved
over random budget 25 from 18.747 to 19.432 PSNR and from 0.697 to 0.725 SSIM.
Depth-gradient active expansion then improved random budget 50 to 20.047 PSNR /
0.742 SSIM / 0.239 LPIPS, while transmittance-tail active expansion was the
stronger control at 20.189 PSNR / 0.745 SSIM / 0.232 LPIPS. Transmittance also
reduced raw depth AbsRel from 1.726 to 1.604 and median-aligned AbsRel from
0.171 to 0.157 versus random budget 50.

The current static dashboard is available at [docs/dashboard.html](docs/dashboard.html).

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
model-error or uncertainty scores. For render-uncertainty reports, `--score-key`
also accepts comma-separated nested fields; each field is rank-normalized over
the candidate pool and averaged into a composite score before selection.

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

Summarize active-vs-random metric deltas from saved Modal metric rows:

```bash
python scripts/summarize_active_metrics.py \
  --input outputs/modal_metrics.log \
  --pairs-file configs/active_metric_pairs.json \
  --format markdown
```

The input may be a JSON array of metric rows or the noisy output from
`modal run modal_app.py --action metrics`. The checked-in pair manifest covers
the current dozer, redwoods2, library, kitchen, and bww_entrance
active-selection comparisons.

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
