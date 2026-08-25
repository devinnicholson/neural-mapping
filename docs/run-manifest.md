# Run Manifest

Date: 2026-08-25

This manifest ties the current headline claims to the commands and artifacts
needed to audit or reproduce them. Large datasets, checkpoints, and pixel-heavy
reports remain in Modal volumes; compact metrics, split memberships, source
hashes, protocols, and claim-bearing records are tracked in Git.

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

Recompute the completed ICL-NUIM multi-trajectory record and every generated
paper/CSV table:

```bash
python scripts/build_icl_benchmark_record.py \
  --artifact-root experiments/artifacts/icl_nuim_multitrajectory_v1 \
  --protocol experiments/protocols/icl_nuim_multitrajectory_v1.json \
  --run-manifest experiments/run_manifests/icl_nuim_multitrajectory_v1.json \
  --output experiments/records/icl_nuim_multitrajectory_v1.json
python scripts/generate_icl_report_assets.py
```

Recompute every stored FR2 paired delta, aggregate, bootstrap interval, and
decision gate:

```bash
python scripts/verify_replication_record.py \
  experiments/records/tum_fr2_desk_replication_v1.json
```

## Headline Claims

| Claim | Source | Metrics | Audit artifacts |
|---|---|---|---|
| Cross-trajectory ICL-NUIM confirms error ranking but rejects the geometry-improving acquisition claim. | `docs/icl_nuim_multitrajectory_v1.md`, `paper/main.tex` | Active−random: +0.637 PSNR, −0.023 LPIPS, −0.0047 raw AbsRel, −0.0016 surface F-score at 5 cm. Mean diagnostic AUROC 0.716. Overall gate: fail. | Frozen/amended protocol, 72 metric files, eight compact diagnostics with raw-source hashes, 24 split manifests, run manifest, and recomputable record under `experiments/`. |
| Sample-scene active selection improves budget-50 held-out RGB quality across repeated scenes. | `docs/results.md`, `docs/dashboard.html` | Dozer v1-v4: +0.779 PSNR, +0.020 SSIM, -0.018 LPIPS. Redwoods2 v1-v4: +0.699 PSNR, +0.027 SSIM, -0.012 LPIPS. BWW v1-v4: +2.016 PSNR, +0.031 SSIM, -0.021 LPIPS. Library v1-v3: +0.496 PSNR, +0.015 SSIM, -0.003 LPIPS. | Pair definitions: `configs/active_metric_pairs.json`. Metrics index command: `modal run modal_app.py --action metrics`. Summary command: `scripts/summarize_active_metrics.py`. |
| TUM RGB-D `freiburg1_xyz` v1-v6 shows depth-gradient as the strongest fixed xyz RGB-D policy at budget 50. | `docs/results.md`, `docs/dashboard.html` | +0.264 PSNR, +0.007 SSIM, -0.003 LPIPS, -0.005 aligned AbsRel, +0.008 aligned delta1 vs same-seed random b50. | Modal metrics under `/workspace/neural-mapping/outputs/runs/tum_fr1_xyz_v{1..6}_*/splatfacto/budget_050/metrics/`. Depth reports under each run's `metrics/depth_eval.json`. |
| TUM RGB-D `freiburg1_xyz` v9 budget sweep supports compact subset selection, with b100 as the cleanest active result and b150 as a saturation negative control. | `docs/results.md`, `docs/dashboard.html`, `docs/blog-assets/tum-fr1-xyz-v9-budget-sweep.csv` | b100: +0.127 PSNR, +0.008 SSIM, -0.004 LPIPS, -0.008 aligned AbsRel, +0.031 aligned delta1. b150: -0.779 PSNR, -0.019 SSIM, +0.028 LPIPS, +0.065 aligned AbsRel, -0.057 aligned delta1. | Modal metrics listed below. Blog plot: `docs/blog-assets/tum-fr1-xyz-v9-budget-sweep.svg`. |
| TUM RGB-D `freiburg1_desk` v4 compact validation transfers the xyz depth-gradient policy to a second scene. | `docs/results.md`, `docs/dashboard.html`, `docs/blog-assets/tum-fr1-desk-v4-compact-validation.csv` | b50: +0.994 PSNR, +0.047 SSIM, -0.041 LPIPS, -0.033 aligned AbsRel. b100: +0.708 PSNR, +0.025 SSIM, -0.024 LPIPS, -0.024 aligned AbsRel. | Modal metrics listed below. Blog plot: `docs/blog-assets/tum-fr1-desk-v4-compact-validation.svg`. |
| TUM RGB-D `freiburg1_room` v4 compact stress test is mixed and does not transfer cleanly. | `docs/results.md`, `docs/dashboard.html`, `docs/blog-assets/tum-fr1-room-v4-compact-stress.csv` | b50: +0.035 PSNR, -0.010 SSIM, +0.003 LPIPS, -0.032 aligned AbsRel. b100: -0.400 PSNR, -0.017 SSIM, +0.019 LPIPS, +0.031 aligned AbsRel. | Modal metrics listed below. Blog plot: `docs/blog-assets/tum-fr1-room-v4-compact-stress.svg`. |
| TUM RGB-D `freiburg2_desk` passes the preregistered within-sequence robustness gate. | `docs/results.md`, `experiments/protocols/tum_fr2_desk_replication_v1.json`, `experiments/records/tum_fr2_desk_replication_v1.json` | Three interleaved pairs average +0.441 PSNR, +0.0222 SSIM, -0.0125 LPIPS, and -0.0108 raw AbsRel. A guarded temporal-block pair gives +0.297 PSNR and -0.120 raw AbsRel. | Protocol commit, exact paired metrics, split/training seeds, holdout spans, uncertainty diagnostics, and every Modal run ID are in the machine-readable record. Aligned AbsRel regresses on all three interleaved pairs. |

## TUM FR2 Desk Preregistered Robustness Reproduction

Protocol:

- Source sequence: TUM RGB-D `freiburg2_desk`.
- Interleaved prepared scenes: `tum_fr2_desk_frozen_v1` through `v3`.
- Blocked prepared scene: `tum_fr2_desk_blocked_v1` with five-frame guard
  bands around contiguous validation and test blocks.
- Source frames: 180 RGB-D frames sampled with `frame_stride=3`.
- Split seeds: `20260824`-`20260827`; 10 validation and 20 test frames each.
- Random budgets: 25 and 50.
- Active policy: expand the random b25 seed to b50 with
  `score-pose-hybrid`, `score_weight=0.35`, and
  `top_decile_mean_uncertainty.depth-gradient`.
- Training: Nerfstudio `splatfacto`, matched pair seeds 42-45, 7,000
  iterations, downscale factor 1, NVIDIA L4.
- Preregistered protocol commit:
  `1b81c56cef90f440784b4fc3cff45c6eb5f6ab73`.
- Implementation commit:
  `2d0a12090ab41c92c9e3db6eb356bfd0f56ff366`.

The commands below show the v1 execution pattern. Substitute the scene, split
seed, training seed, and holdout parameters exactly as listed in the protocol
record for v2, v3, and the temporal-block pair.

```bash
modal run modal_app.py \
  --action prepare-tum \
  --tum-sequence freiburg2_desk \
  --data-scene-name tum_fr2_desk_frozen_v1 \
  --budgets "25 50" \
  --split-seed 20260824 \
  --val-count 10 \
  --test-count 20 \
  --max-frames 180 \
  --frame-stride 3

modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr2_desk_frozen_v1 \
  --scene-name tum_fr2_desk_frozen_v1_random_b25_7k \
  --budget 25 \
  --iterations 7000 \
  --downscale-factor 1

modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr2_desk_frozen_v1 \
  --scene-name tum_fr2_desk_frozen_v1_random_b50_7k \
  --budget 50 \
  --iterations 7000 \
  --downscale-factor 1

modal run modal_app.py \
  --action render-uncertainty-maps \
  --source-data-scene-name tum_fr2_desk_frozen_v1 \
  --base-split-scene-name tum_fr2_desk_frozen_v1 \
  --data-scene-name tum_fr2_desk_frozen_v1_depth_error_maps \
  --scene-name tum_fr2_desk_frozen_v1_random_b25_7k \
  --budget 25 \
  --score-metric depth-aligned-abs-rel \
  --bad-quantile 0.8 \
  --max-pixels-per-frame 50000 \
  --render-map-signals transmittance,local-mean-transmittance,local-std-transmittance,accumulation-gradient,depth-gradient \
  --patch-size 15

modal run modal_app.py \
  --action prepare-active \
  --source-data-scene-name tum_fr2_desk_frozen_v1 \
  --base-split-scene-name tum_fr2_desk_frozen_v1 \
  --data-scene-name tum_fr2_desk_frozen_v1_active_depthgrad_w035_b50 \
  --base-budget 25 \
  --target-budget 50 \
  --active-strategy score-pose-hybrid \
  --score-path /workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr2_desk_frozen_v1_depth_error_maps_budget_025_depth-aligned-abs-rel.json \
  --score-key top_decile_mean_uncertainty.depth-gradient \
  --score-weight 0.35

modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr2_desk_frozen_v1_active_depthgrad_w035_b50 \
  --scene-name tum_fr2_desk_frozen_v1_active_depthgrad_w035_b50_7k \
  --budget 50 \
  --iterations 7000 \
  --downscale-factor 1
```

Run `eval` and `depth-eval` for each scene/budget pair above. The completed
record at `experiments/records/tum_fr2_desk_replication_v1.json` stores every
confirmatory value and Modal application ID; the earlier v1 pilot record is
retained for provenance.

Artifact index:

| Stage | Artifact or run |
|---|---|
| Dataset preparation | Modal `ap-60QQoYZ9fxXFLUirwuwFX9`; `/workspace/neural-mapping/data/splits/tum_fr2_desk_frozen_v1.json` |
| Random b25 training/evaluation | `ap-xGSCuJRFEH219JK5trZhmS`, `ap-thunXveuXm8zsBXjJGMVbd`, `ap-0XIFgAF3KMaI7IPy5glbHx` |
| Random b50 training/evaluation | `ap-1w15vog8y0Kob9qFAYLxkA`, `ap-7Pgt0eGYoR0BrcHQqqSA0L`, `ap-qUR8zHZ17gMgaeEfaMCo0a` |
| Uncertainty report | Modal `ap-4GsSdaiorCLz7ARgPHrNIj`; `/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr2_desk_frozen_v1_depth_error_maps_budget_025_depth-aligned-abs-rel.json` |
| Active split | Modal `ap-cUWQLNamTKTL0O8pgZ5Dtc`; `/workspace/neural-mapping/data/splits/tum_fr2_desk_frozen_v1_active_depthgrad_w035_b50.json` |
| Active b50 training/evaluation | `ap-C63cc0pQ0VEs9ONMoSRGkZ`, `ap-pkDtS4fdKczhmeGdN2YGfF`, `ap-FDXWLA7BQZ7O9qvzQI2EO9` |
| Interleaved v2 confirmatory pair | Random b50: `ap-nkaDvRoiGkJRLhRTHtZCZ7`, `ap-WkXLagk7tzg9gF08ekWtLZ`, `ap-L4Z4ZzLOLaZC8JUlpwR5by`; uncertainty/active: `ap-UH6sXGT047mwiWwM1HfNmh`, `ap-2Lyw4ccjqCd2VeADdVD2QG`, `ap-WzmmMJEPZ0DV0GgAwP9zXZ`, `ap-AUUQO59uBMKkXpGfIWNS2l` |
| Interleaved v3 confirmatory pair | Random b50: `ap-d39UYLhSP3ZMEnqezRi2He`, `ap-nxhwqfpI6voWJMAE20kszF`, `ap-HrBaDh0wcB6CNiJDbCk574`; uncertainty/active: `ap-bN1JP6g9nBhNuwj1hOJRiw`, `ap-hjdmi374rHl2I4IW4GT51W`, `ap-ZBFSPbIixJcolx2zh80wnp`, `ap-OpgDuwzWzqh123wdj3ogmW` |
| Temporal-block confirmatory pair | Random b50: `ap-EYzVLhs5sJi1FaLQJ8f5HY`, `ap-Ro367r4VrvCh5C7UGh5DJd`, `ap-l3UKvCrBKCuCJ3p7jXMxIW`; uncertainty/active: `ap-l3QgFeGt7CRGwBrFjHJ1lE`, `ap-87M43ISRDQLbYDtIAUQRXy`, `ap-S3Ko3irBKmlPTzMzbF12vs`, `ap-xppEqtbWMjKQnndlrCaaBH` |

## TUM FR1 Desk v4 Compact Reproduction

Protocol:

- Source sequence: TUM RGB-D `freiburg1_desk`.
- Prepared scene: `tum_fr1_desk_v4_compact`.
- Source frames: 180 RGB-D frames sampled with `frame_stride=3`.
- Split seed: `20260627`.
- Validation/test split: 10 validation frames, 20 held-out test frames.
- Random budgets: 25, 50, 100.
- Active policy: start from random b25, then expand with
  `score-pose-hybrid`, `score_weight=0.65`, and score key
  `top_decile_mean_uncertainty.depth-gradient`.
- Training: Nerfstudio `splatfacto`, 7,000 iterations, downscale factor 1.

Prepare the base split:

```bash
modal run modal_app.py \
  --action prepare-tum \
  --tum-sequence freiburg1_desk \
  --data-scene-name tum_fr1_desk_v4_compact \
  --budgets "25 50 100" \
  --split-seed 20260627 \
  --val-count 10 \
  --test-count 20 \
  --max-frames 180 \
  --frame-stride 3
```

Train/evaluate random budgets:

```bash
modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr1_desk_v4_compact \
  --scene-name tum_fr1_desk_v4_compact_b100_7k \
  --budget 100 \
  --iterations 7000 \
  --downscale-factor 1

modal run modal_app.py \
  --action eval \
  --scene-name tum_fr1_desk_v4_compact_b100_7k \
  --budget 100

modal run modal_app.py \
  --action depth-eval \
  --data-scene-name tum_fr1_desk_v4_compact \
  --scene-name tum_fr1_desk_v4_compact_b100_7k \
  --budget 100 \
  --depth-cache-images cpu
```

Repeat the same pattern for random b25 and b50.

Generate the seed-model depth-error report:

```bash
modal run modal_app.py \
  --action render-uncertainty-maps \
  --source-data-scene-name tum_fr1_desk_v4_compact \
  --base-split-scene-name tum_fr1_desk_v4_compact \
  --data-scene-name tum_fr1_desk_v4_compact_depth_error_maps \
  --scene-name tum_fr1_desk_v4_compact_b25_7k \
  --budget 25 \
  --score-metric depth-aligned-abs-rel \
  --bad-quantile 0.8 \
  --max-pixels-per-frame 50000 \
  --render-map-signals transmittance,local-mean-transmittance,local-std-transmittance,accumulation-gradient,depth-gradient \
  --patch-size 15
```

Materialize and run active budgets:

```bash
modal run modal_app.py \
  --action prepare-active \
  --source-data-scene-name tum_fr1_desk_v4_compact \
  --base-split-scene-name tum_fr1_desk_v4_compact \
  --data-scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100 \
  --base-budget 25 \
  --target-budget 100 \
  --active-strategy score-pose-hybrid \
  --score-path /workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr1_desk_v4_compact_depth_error_maps_budget_025_depth-aligned-abs-rel.json \
  --score-key top_decile_mean_uncertainty.depth-gradient \
  --score-weight 0.65

modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100 \
  --scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100_7k \
  --budget 100 \
  --iterations 7000 \
  --downscale-factor 1

modal run modal_app.py \
  --action eval \
  --scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100_7k \
  --budget 100

modal run modal_app.py \
  --action depth-eval \
  --data-scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100 \
  --scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100_7k \
  --budget 100 \
  --depth-cache-images cpu
```

Repeat the active pattern for b50.

## TUM FR1 Desk v4 Artifact Index

| Budget | Selection | Scene | RGB metrics | Depth metrics | Checkpoint |
|---:|---|---|---|---|---|
| 25 | Random | `tum_fr1_desk_v4_compact_b25_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b25_7k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b25_7k/splatfacto/budget_025/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b25_7k/splatfacto/budget_025/train/unnamed/splatfacto/2026-07-05_212620/nerfstudio_models/step-000006999.ckpt` |
| 50 | Random | `tum_fr1_desk_v4_compact_b50_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b50_7k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b50_7k/splatfacto/budget_050/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b50_7k/splatfacto/budget_050/train/unnamed/splatfacto/2026-07-05_213440/nerfstudio_models/step-000006999.ckpt` |
| 50 | Active depth-gradient hybrid | `tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b50_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b50_7k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b50_7k/splatfacto/budget_050/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b50_7k/splatfacto/budget_050/train/unnamed/splatfacto/2026-07-05_220400/nerfstudio_models/step-000006999.ckpt` |
| 100 | Random | `tum_fr1_desk_v4_compact_b100_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b100_7k/splatfacto/budget_100/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b100_7k/splatfacto/budget_100/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_b100_7k/splatfacto/budget_100/train/unnamed/splatfacto/2026-07-05_214111/nerfstudio_models/step-000006999.ckpt` |
| 100 | Active depth-gradient hybrid | `tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100_7k/splatfacto/budget_100/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100_7k/splatfacto/budget_100/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b100_7k/splatfacto/budget_100/train/unnamed/splatfacto/2026-07-05_220948/nerfstudio_models/step-000006999.ckpt` |

Depth-error report:

`/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr1_desk_v4_compact_depth_error_maps_budget_025_depth-aligned-abs-rel.json`

## TUM FR1 Room v4 Compact Reproduction

Protocol:

- Source sequence: TUM RGB-D `freiburg1_room`.
- Prepared scene: `tum_fr1_room_v4_compact`.
- Source frames: 180 RGB-D frames sampled with `frame_stride=3`.
- Split seed: `20260705`.
- Validation/test split: 10 validation frames, 20 held-out test frames.
- Random budgets: 25, 50, 100.
- Active policy: start from random b25, then expand with
  `score-pose-hybrid`, `score_weight=0.65`, and score key
  `top_decile_mean_uncertainty.transmittance`. The local-mean control swaps
  the score key to `top_decile_mean_uncertainty.local-mean-transmittance`.
- Training: Nerfstudio `splatfacto`, 7,000 iterations, downscale factor 1.

Prepare the base split:

```bash
modal run modal_app.py \
  --action prepare-tum \
  --tum-sequence freiburg1_room \
  --data-scene-name tum_fr1_room_v4_compact \
  --budgets "25 50 100" \
  --split-seed 20260705 \
  --val-count 10 \
  --test-count 20 \
  --max-frames 180 \
  --frame-stride 3
```

Generate the seed-model depth-error report:

```bash
modal run modal_app.py \
  --action render-uncertainty-maps \
  --source-data-scene-name tum_fr1_room_v4_compact \
  --base-split-scene-name tum_fr1_room_v4_compact \
  --data-scene-name tum_fr1_room_v4_compact_depth_error_maps \
  --scene-name tum_fr1_room_v4_compact_b25_7k \
  --budget 25 \
  --score-metric depth-aligned-abs-rel \
  --bad-quantile 0.8 \
  --max-pixels-per-frame 50000 \
  --render-map-signals transmittance,local-mean-transmittance,local-std-transmittance,accumulation-gradient,depth-gradient \
  --patch-size 15
```

Materialize and run an active budget:

```bash
modal run modal_app.py \
  --action prepare-active \
  --source-data-scene-name tum_fr1_room_v4_compact \
  --base-split-scene-name tum_fr1_room_v4_compact \
  --data-scene-name tum_fr1_room_v4_compact_active_trans_hybrid_b100 \
  --base-budget 25 \
  --target-budget 100 \
  --active-strategy score-pose-hybrid \
  --score-path /workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr1_room_v4_compact_depth_error_maps_budget_025_depth-aligned-abs-rel.json \
  --score-key top_decile_mean_uncertainty.transmittance \
  --score-weight 0.65

modal run modal_app.py \
  --action train \
  --data-scene-name tum_fr1_room_v4_compact_active_trans_hybrid_b100 \
  --scene-name tum_fr1_room_v4_compact_active_trans_hybrid_b100_7k \
  --budget 100 \
  --iterations 7000 \
  --downscale-factor 1

modal run modal_app.py \
  --action eval \
  --scene-name tum_fr1_room_v4_compact_active_trans_hybrid_b100_7k \
  --budget 100

modal run modal_app.py \
  --action depth-eval \
  --data-scene-name tum_fr1_room_v4_compact_active_trans_hybrid_b100 \
  --scene-name tum_fr1_room_v4_compact_active_trans_hybrid_b100_7k \
  --budget 100 \
  --depth-cache-images cpu
```

Repeat the train/eval/depth-eval pattern for random b25/b50/b100 and active
b50.

## TUM FR1 Room v4 Artifact Index

| Budget | Selection | Scene | RGB metrics | Depth metrics | Checkpoint |
|---:|---|---|---|---|---|
| 25 | Random | `tum_fr1_room_v4_compact_b25_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b25_7k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b25_7k/splatfacto/budget_025/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b25_7k/splatfacto/budget_025/train/unnamed/splatfacto/2026-07-05_231719/nerfstudio_models/step-000006999.ckpt` |
| 50 | Random | `tum_fr1_room_v4_compact_b50_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b50_7k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b50_7k/splatfacto/budget_050/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b50_7k/splatfacto/budget_050/train/unnamed/splatfacto/2026-07-05_231707/nerfstudio_models/step-000006999.ckpt` |
| 50 | Active transmittance hybrid | `tum_fr1_room_v4_compact_active_trans_hybrid_b50_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_trans_hybrid_b50_7k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_trans_hybrid_b50_7k/splatfacto/budget_050/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_trans_hybrid_b50_7k/splatfacto/budget_050/train/unnamed/splatfacto/2026-07-05_234635/nerfstudio_models/step-000006999.ckpt` |
| 50 | Active local-mean hybrid | `tum_fr1_room_v4_compact_active_lmean_hybrid_b50_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_lmean_hybrid_b50_7k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_lmean_hybrid_b50_7k/splatfacto/budget_050/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_lmean_hybrid_b50_7k/splatfacto/budget_050/train/unnamed/splatfacto/2026-07-06_001545/nerfstudio_models/step-000006999.ckpt` |
| 100 | Random | `tum_fr1_room_v4_compact_b100_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b100_7k/splatfacto/budget_100/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b100_7k/splatfacto/budget_100/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_b100_7k/splatfacto/budget_100/train/unnamed/splatfacto/2026-07-05_231712/nerfstudio_models/step-000006999.ckpt` |
| 100 | Active transmittance hybrid | `tum_fr1_room_v4_compact_active_trans_hybrid_b100_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_trans_hybrid_b100_7k/splatfacto/budget_100/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_trans_hybrid_b100_7k/splatfacto/budget_100/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_trans_hybrid_b100_7k/splatfacto/budget_100/train/unnamed/splatfacto/2026-07-05_234636/nerfstudio_models/step-000006999.ckpt` |
| 100 | Active local-mean hybrid | `tum_fr1_room_v4_compact_active_lmean_hybrid_b100_7k` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_lmean_hybrid_b100_7k/splatfacto/budget_100/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_lmean_hybrid_b100_7k/splatfacto/budget_100/metrics/depth_eval.json` | `/workspace/neural-mapping/outputs/runs/tum_fr1_room_v4_compact_active_lmean_hybrid_b100_7k/splatfacto/budget_100/train/unnamed/splatfacto/2026-07-06_001546/nerfstudio_models/step-000006999.ckpt` |

Depth-error report:

`/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr1_room_v4_compact_depth_error_maps_budget_025_depth-aligned-abs-rel.json`

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
