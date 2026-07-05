# Run Manifest

Date: 2026-07-05

This manifest ties the current headline claims to the commands and artifacts
needed to audit or reproduce them. Large artifacts remain in Modal volumes or
local experiment storage and are intentionally not tracked in Git.

## Verification Commands

Collect the Modal metric index:

```bash
modal run modal_app.py --action metrics
```

Summarize the active-vs-random sample-scene pairs:

```bash
python scripts/summarize_active_metrics.py \
  --input outputs/modal_metrics.log \
  --pairs-file configs/active_metric_pairs.json \
  --format markdown
```

Run the lightweight repo tests:

```bash
make test
```

## Headline Claims

| Claim | Source | Metrics | Audit artifacts |
|---|---|---|---|
| Sample-scene active selection improves budget-50 held-out RGB quality across repeated scenes. | `docs/results.md`, `docs/dashboard.html` | Dozer v1-v4: +0.779 PSNR, +0.020 SSIM, -0.018 LPIPS. Redwoods2 v1-v4: +0.699 PSNR, +0.027 SSIM, -0.012 LPIPS. BWW v1-v4: +2.016 PSNR, +0.031 SSIM, -0.021 LPIPS. Library v1-v3: +0.496 PSNR, +0.015 SSIM, -0.003 LPIPS. | Pair definitions: `configs/active_metric_pairs.json`. Metrics index command: `modal run modal_app.py --action metrics`. Summary command: `scripts/summarize_active_metrics.py`. |
| TUM RGB-D `freiburg1_xyz` v1-v6 shows depth-gradient as the strongest fixed xyz RGB-D policy at budget 50. | `docs/results.md`, `docs/dashboard.html` | +0.264 PSNR, +0.007 SSIM, -0.003 LPIPS, -0.005 aligned AbsRel, +0.008 aligned delta1 vs same-seed random b50. | Modal metrics under `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v{1..6}_*/splatfacto/budget_050/metrics/`. Depth reports under each run's `metrics/depth_eval.json`. |
| TUM RGB-D `freiburg1_xyz` v9 budget sweep supports compact subset selection, with b100 as the cleanest active result and b150 as a saturation negative control. | `docs/results.md`, `docs/dashboard.html`, `docs/blog-assets/tum-fr1-xyz-v9-budget-sweep.csv` | b100: +0.127 PSNR, +0.008 SSIM, -0.004 LPIPS, -0.008 aligned AbsRel, +0.031 aligned delta1. b150: -0.779 PSNR, -0.019 SSIM, +0.028 LPIPS, +0.065 aligned AbsRel, -0.057 aligned delta1. | Modal metrics listed below. Blog plot: `docs/blog-assets/tum-fr1-xyz-v9-budget-sweep.svg`. |

## TUM FR1 XYZ v9 Reproduction

Protocol:

- Source sequence: TUM RGB-D `freiburg1_xyz`.
- Prepared scene: `tum_fr1_xyz_v9`.
- Source frames: 180 RGB-D frames sampled with `frame_stride=3`.
- Split seed: `20260623`.
- Validation/test split: 10 validation frames, 20 held-out test frames.
- Random budgets: 25, 50, 75, 100, 125, 150.
- Active policy: start from random b25, then expand with
  `score-pose-hybrid`, `score_weight=0.65`, and score key
  `top_decile_mean_uncertainty.depth-gradient`.
- Training: Nerfstudio `splatfacto`, 7,000 iterations, downscale factor 1.

Prepare the base split:

```bash
modal run modal_app.py \
  --action prepare-tum \
  --tum-sequence freiburg1_xyz \
  --data-scene-name tum_fr1_xyz_v9 \
  --budgets "25 50 75 100 125 150" \
  --split-seed 20260623 \
  --val-count 10 \
  --test-count 20 \
  --max-frames 180 \
  --frame-stride 3
```

Train and evaluate a random budget run:

```bash
modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr1_xyz_v9 \
  --scene-name tum_fr1_xyz_v9_b100_7k \
  --budget 100 \
  --iterations 7000 \
  --downscale-factor 1

modal run modal_app.py \
  --action eval \
  --scene-name tum_fr1_xyz_v9_b100_7k \
  --budget 100

modal run modal_app.py \
  --action depth-eval \
  --data-scene-name tum_fr1_xyz_v9 \
  --scene-name tum_fr1_xyz_v9_b100_7k \
  --budget 100 \
  --depth-cache-images cpu
```

Generate the seed-model depth-error report used for active expansion:

```bash
modal run modal_app.py \
  --action render-uncertainty-maps \
  --source-data-scene-name tum_fr1_xyz_v9 \
  --base-split-scene-name tum_fr1_xyz_v9 \
  --data-scene-name tum_fr1_xyz_v9_depth_error_maps \
  --scene-name tum_fr1_xyz_v9_b25_7k \
  --budget 25 \
  --score-metric depth-aligned-abs-rel \
  --bad-quantile 0.8 \
  --max-pixels-per-frame 50000 \
  --render-map-signals transmittance,local-mean-transmittance,local-std-transmittance,accumulation-gradient,depth-gradient \
  --patch-size 15
```

Materialize an active budget and train/evaluate it:

```bash
modal run modal_app.py \
  --action prepare-active \
  --source-data-scene-name tum_fr1_xyz_v9 \
  --base-split-scene-name tum_fr1_xyz_v9 \
  --data-scene-name tum_fr1_xyz_v9_active_depth_grad_hybrid_b100 \
  --base-budget 25 \
  --target-budget 100 \
  --active-strategy score-pose-hybrid \
  --score-path /workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr1_xyz_v9_depth_error_maps_budget_025_depth-aligned-abs-rel.json \
  --score-key top_decile_mean_uncertainty.depth-gradient \
  --score-weight 0.65

modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr1_xyz_v9_active_depth_grad_hybrid_b100 \
  --scene-name tum_fr1_xyz_v9_active_depth_grad_hybrid_b100_7k \
  --budget 100 \
  --iterations 7000 \
  --downscale-factor 1

modal run modal_app.py \
  --action eval \
  --scene-name tum_fr1_xyz_v9_active_depth_grad_hybrid_b100_7k \
  --budget 100

modal run modal_app.py \
  --action depth-eval \
  --data-scene-name tum_fr1_xyz_v9_active_depth_grad_hybrid_b100 \
  --scene-name tum_fr1_xyz_v9_active_depth_grad_hybrid_b100_7k \
  --budget 100 \
  --depth-cache-images cpu
```

Repeat the train/eval/depth-eval pattern for b50, b75, b125, and b150.

## TUM FR1 XYZ v9 Artifact Index

| Budget | Selection | Scene | RGB metrics | Depth metrics | Checkpoint |
|---:|---|---|---|---|---|
| 25 | Random | `tum_fr1_xyz_v9_b25_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b25_7k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b25_7k/splatfacto/budget_025/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b25_7k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-26_004323/nerfstudio_models/step-000006999.ckpt` |
| 50 | Random | `tum_fr1_xyz_v9_b50_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b50_7k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b50_7k/splatfacto/budget_050/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b50_7k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-26_004326/nerfstudio_models/step-000006999.ckpt` |
| 50 | Active depth-gradient hybrid | `tum_fr1_xyz_v9_active_depth_grad_hybrid_b50_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b50_7k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b50_7k/splatfacto/budget_050/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b50_7k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-26_011343/nerfstudio_models/step-000006999.ckpt` |
| 75 | Random | `tum_fr1_xyz_v9_b75_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b75_7k/splatfacto/budget_075/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b75_7k/splatfacto/budget_075/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b75_7k/splatfacto/budget_075/train/unnamed/splatfacto/2026-06-26_012108/nerfstudio_models/step-000006999.ckpt` |
| 75 | Active depth-gradient hybrid | `tum_fr1_xyz_v9_active_depth_grad_hybrid_b75_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b75_7k/splatfacto/budget_075/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b75_7k/splatfacto/budget_075/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b75_7k/splatfacto/budget_075/train/unnamed/splatfacto/2026-06-26_012107/nerfstudio_models/step-000006999.ckpt` |
| 100 | Random | `tum_fr1_xyz_v9_b100_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b100_7k/splatfacto/budget_100/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b100_7k/splatfacto/budget_100/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b100_7k/splatfacto/budget_100/train/unnamed/splatfacto/2026-06-26_012648/nerfstudio_models/step-000006999.ckpt` |
| 100 | Active depth-gradient hybrid | `tum_fr1_xyz_v9_active_depth_grad_hybrid_b100_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b100_7k/splatfacto/budget_100/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b100_7k/splatfacto/budget_100/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b100_7k/splatfacto/budget_100/train/unnamed/splatfacto/2026-06-26_012649/nerfstudio_models/step-000006999.ckpt` |
| 125 | Random | `tum_fr1_xyz_v9_b125_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b125_7k/splatfacto/budget_125/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b125_7k/splatfacto/budget_125/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b125_7k/splatfacto/budget_125/train/unnamed/splatfacto/2026-06-26_013302/nerfstudio_models/step-000006999.ckpt` |
| 125 | Active depth-gradient hybrid | `tum_fr1_xyz_v9_active_depth_grad_hybrid_b125_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b125_7k/splatfacto/budget_125/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b125_7k/splatfacto/budget_125/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b125_7k/splatfacto/budget_125/train/unnamed/splatfacto/2026-06-26_013303/nerfstudio_models/step-000006999.ckpt` |
| 150 | Random | `tum_fr1_xyz_v9_b150_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b150_7k/splatfacto/budget_150/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b150_7k/splatfacto/budget_150/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_b150_7k/splatfacto/budget_150/train/unnamed/splatfacto/2026-06-26_013826/nerfstudio_models/step-000006999.ckpt` |
| 150 | Active depth-gradient hybrid | `tum_fr1_xyz_v9_active_depth_grad_hybrid_b150_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b150_7k/splatfacto/budget_150/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b150_7k/splatfacto/budget_150/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v9_active_depth_grad_hybrid_b150_7k/splatfacto/budget_150/train/unnamed/splatfacto/2026-06-26_013823/nerfstudio_models/step-000006999.ckpt` |
