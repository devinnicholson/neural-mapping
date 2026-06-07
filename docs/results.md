# Results

This page records lightweight experiment results that are small enough to track in Git.
Checkpoints, renders, and full Modal output volumes are intentionally not tracked.

## Modal Poster Baselines

Date: 2026-06-05 Pacific / 2026-06-06 UTC

Dataset:

- Nerfstudio `poster` sample downloaded from the Hugging Face mirror.
- Filtered scene: `poster_available`.
- Split seed: `20260529`.
- Split manifest: 10 held-out test frames and 10 validation frames.
- Validation frames: 10 frames included in the materialized training `transforms.json` for this smoke baseline.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- GPU: Modal L4.

Note: these rows were produced before `materialize_nerfstudio_split.py`
emitted Nerfstudio-native `train_filenames`/`val_filenames`/`test_filenames`.
The legacy `eval_filenames` field was ignored by Nerfstudio, so treat these as
smoke results over the materialized frame subsets, not final held-out-test
claims. Future runs should use the corrected materializer and rerun the
baseline table.

| Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random | `poster_modal_b25_10k` | 25 | 10,000 | 31.197 | 0.952 | 0.167 | 0.669 |
| Random | `poster_modal_b50_10k` | 50 | 10,000 | 30.588 | 0.947 | 0.197 | 0.839 |
| Farthest index | `poster_modal_farthest_b25_10k` | 25 | 10,000 | 22.842 | 0.909 | 0.272 | 0.622 |
| Farthest index | `poster_modal_farthest_b50_10k` | 50 | 10,000 | 29.247 | 0.930 | 0.229 | 0.590 |
| Farthest pose | `poster_modal_pose_b25_10k` | 25 | 10,000 | 22.835 | 0.901 | 0.290 | 0.552 |
| Farthest pose | `poster_modal_pose_b50_10k` | 50 | 10,000 | 27.571 | 0.929 | 0.243 | 0.791 |
| Active pose novelty | `poster_modal_active_pose_b50_10k` | 50 | 10,000 | 27.335 | 0.926 | 0.242 | 0.764 |
| Active model error | `poster_modal_active_error_b50_10k` | 50 | 10,000 | 31.357 | 0.946 | 0.204 | 0.607 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `poster_modal_b25_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-06_044723/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-06_051459/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_farthest_b25_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_farthest_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_farthest_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-06_055801/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_farthest_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_farthest_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_farthest_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-06_171033/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_pose_b25_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_pose_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_pose_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-06_173356/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_pose_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_pose_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_pose_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-06_175953/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_pose_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_pose_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_pose_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-06_203558/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_error_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-06_225705/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- This is a smoke baseline, not a final claim.
- The random 50-frame run did not improve over the random 25-frame run on this single small split.
- `farthest-index` underperformed random at both budgets, especially at 25 frames.
- `farthest-pose` improved from 25 to 50 frames, but still underperformed random at both budgets; camera-center spread alone is not sufficient for this scene.
- Active pose novelty kept the random 25-frame seed set fixed and added 25 pose-novel candidates, but it still underperformed random 50 and farthest-index 50.
- Active model error kept the random 25-frame seed set fixed and added the 25 highest-LPIPS candidate frames from the seed model. It produced the strongest 50-frame PSNR in this smoke table, but LPIPS remained worse than the random 25/50 runs.
- The next baseline should rerun random, pose, and model-error active selection with the corrected explicit Nerfstudio split fields before making a final comparison.

## Corrected Modal Poster V2 Baselines

Date: 2026-06-06 Pacific / 2026-06-07 UTC

Dataset:

- Source scene: `poster_available_v2`, filtered from the Nerfstudio `poster` sample.
- Split seed: `20260529`.
- Materialized datasets include explicit Nerfstudio-native `train_filenames`, `val_filenames`, and `test_filenames`.
- Each budget uses the same 10 held-out test frames and 10 validation frames.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- GPU: Modal L4.

| Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random v2 | `poster_modal_v2_b25_10k` | 25 | 10,000 | 27.810 | 0.913 | 0.245 | 0.756 |
| Random v2 | `poster_modal_v2_b50_10k` | 50 | 10,000 | 29.323 | 0.930 | 0.225 | 0.690 |
| Active model error v2 | `poster_modal_active_error_v2_b50_10k` | 50 | 10,000 | 30.390 | 0.945 | 0.199 | 0.806 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `poster_modal_v2_b25_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_v2_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_v2_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-06_233548/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_v2_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_v2_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_v2_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-06_234721/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_error_v2_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_v2_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_v2_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_000035/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- These are the first corrected held-out-test rows for the Modal poster workflow.
- Random 50 improved over random 25 by +1.513 PSNR, +0.017 SSIM, and -0.021 LPIPS.
- Active model error v2 kept the corrected random 25-frame seed set fixed, scored the remaining candidates with the seed model, and added the 25 highest-LPIPS frames.
- Active model error v2 improved over corrected random 50 by +1.067 PSNR, +0.015 SSIM, and -0.026 LPIPS.
- The corrected result supports the active-error selection hypothesis on this small poster scene; the next step is to repeat on another scene or another seed before treating it as robust.

## Corrected Modal Poster V3 Seed Repeat

Date: 2026-06-06 Pacific / 2026-06-07 UTC

Dataset:

- Source scene: `poster_available_v3`, filtered from the Nerfstudio `poster` sample.
- Split seed: `20260607`.
- Materialized datasets include explicit Nerfstudio-native `train_filenames`, `val_filenames`, and `test_filenames`.
- Each budget uses the same 10 held-out test frames and 10 validation frames.
- Active-error scene: `poster_available_active_error_v3`.
- Active-error candidate scores came from the 25-frame random seed model and selected the 25 highest-LPIPS remaining candidate frames.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- GPU: Modal L4.

| Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random v3 | `poster_modal_v3_b25_10k` | 25 | 10,000 | 28.153 | 0.931 | 0.216 | 0.901 |
| Random v3 | `poster_modal_v3_b50_10k` | 50 | 10,000 | 29.788 | 0.945 | 0.201 | 0.825 |
| Active model error v3 | `poster_modal_active_error_v3_b50_10k` | 50 | 10,000 | 30.061 | 0.951 | 0.187 | 0.912 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `poster_modal_v3_b25_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_v3_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_v3_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-07_004005/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_v3_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_v3_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_v3_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_005156/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_error_v3_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_v3_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_v3_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_010444/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- Random 50 improved over random 25 by +1.635 PSNR, +0.015 SSIM, and -0.015 LPIPS.
- Active model error v3 improved over corrected random 50 by +0.273 PSNR, +0.006 SSIM, and -0.014 LPIPS.
- Compared with v2, the active-error advantage is smaller in PSNR but still positive across PSNR, SSIM, and LPIPS.
- Across these two corrected seeds, active-error 50 beats random 50 on the held-out test split by an average of about +0.670 PSNR, +0.010 SSIM, and -0.020 LPIPS.
- The result is still a small-scene signal, not a robust claim. The v4 seed repeat below tests whether the active-error advantage is stable.

## Corrected Modal Poster V4 Seed Repeat

Date: 2026-06-06 Pacific / 2026-06-07 UTC

Dataset:

- Source scene: `poster_available_v4`, filtered from the Nerfstudio `poster` sample.
- Split seed: `20260608`.
- Materialized datasets include explicit Nerfstudio-native `train_filenames`, `val_filenames`, and `test_filenames`.
- Each budget uses the same 10 held-out test frames and 10 validation frames.
- Active-error scene: `poster_available_active_error_v4`.
- Active-error candidate scores came from the 25-frame random seed model and selected the 25 highest-LPIPS remaining candidate frames.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- GPU: Modal L4.

| Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random v4 | `poster_modal_v4_b25_10k` | 25 | 10,000 | 24.249 | 0.895 | 0.289 | 0.777 |
| Random v4 | `poster_modal_v4_b50_10k` | 50 | 10,000 | 28.461 | 0.936 | 0.215 | 0.826 |
| Active model error v4 | `poster_modal_active_error_v4_b50_10k` | 50 | 10,000 | 26.615 | 0.919 | 0.253 | 0.889 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `poster_modal_v4_b25_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_v4_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_v4_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-07_030616/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_v4_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_v4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_v4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_030602/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_error_v4_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_v4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_error_v4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_031925/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- Random 50 improved over random 25 by +4.212 PSNR, +0.041 SSIM, and -0.074 LPIPS.
- Active model error v4 kept the corrected random 25-frame seed set fixed, scored the remaining candidates with the seed model, and added the 25 highest-LPIPS frames.
- Active model error v4 underperformed corrected random 50 by -1.846 PSNR, -0.017 SSIM, and +0.038 LPIPS.
- The active-error strategy is not consistently better across seeds on this scene. Across v2, v3, and v4, active-error 50 versus random 50 averages about -0.169 PSNR, +0.001 SSIM, and -0.001 LPIPS.
- The current evidence says active-error selection can help, but it is high variance on this small poster sample. The next step should be either another scene or a less myopic selection rule that mixes high-error frames with pose/coverage diversity.

## Corrected Modal Poster V4 Hybrid Selector Sweep

Date: 2026-06-06 Pacific / 2026-06-07 UTC

Dataset:

- Source scene: `poster_available_v4`, filtered from the Nerfstudio `poster` sample.
- Split seed: `20260608`.
- Base split scene: `poster_available_v4`.
- Hybrid active scenes: `poster_available_active_hybrid_v4`, `poster_available_active_hybrid_w050_v4`, `poster_available_active_hybrid_w035_v4`.
- Hybrid strategy: `score-pose-hybrid`.
- Score source: `/workspace/neural-mapping/data/scores/poster_available_active_error_v4.json`.
- Score source model: `poster_modal_v4_b25_10k`.
- Score weights tested: `0.65`, `0.50`, `0.35`; pose-novelty weight is `1 - score_weight`.
- Candidate score: held-out seed-model LPIPS.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- GPU: Modal L4.

| Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random v4 | `poster_modal_v4_b50_10k` | 50 | 10,000 | 28.461 | 0.936 | 0.215 | 0.826 |
| Active model error v4 | `poster_modal_active_error_v4_b50_10k` | 50 | 10,000 | 26.615 | 0.919 | 0.253 | 0.889 |
| Active score-pose hybrid v4, `score_weight=0.65` | `poster_modal_active_hybrid_v4_b50_10k` | 50 | 10,000 | 28.051 | 0.932 | 0.223 | 0.830 |
| Active score-pose hybrid v4, `score_weight=0.50` | `poster_modal_active_hybrid_w050_v4_b50_10k` | 50 | 10,000 | 28.186 | 0.933 | 0.221 | 0.651 |
| Active score-pose hybrid v4, `score_weight=0.35` | `poster_modal_active_hybrid_w035_v4_b50_10k` | 50 | 10,000 | 28.645 | 0.935 | 0.221 | 0.753 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `poster_modal_active_hybrid_v4_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_v4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_v4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_034145/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_hybrid_w050_v4_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w050_v4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w050_v4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_041639/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_hybrid_w035_v4_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_041636/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- The first hybrid setting, `score_weight=0.65`, improved over pure active-error v4 by +1.437 PSNR, +0.013 SSIM, and -0.030 LPIPS, but still underperformed corrected random v4 50.
- Reducing score weight improved v4 PSNR monotonically in this small sweep: `0.65` -> 28.051, `0.50` -> 28.186, `0.35` -> 28.645.
- The best sweep row, `score_weight=0.35`, beat corrected random v4 50 by +0.184 PSNR, but still trailed by -0.001 SSIM and +0.006 LPIPS.
- Mixing model error with stronger pose diversity reduced the v4 failure mode. The `score_weight=0.35` setting was repeated on v2/v3 below.

## Corrected Modal Poster Hybrid Seed Repeat

Date: 2026-06-06 Pacific / 2026-06-07 UTC

Dataset:

- Source scenes: `poster_available_v2`, `poster_available_v3`, and `poster_available_v4`.
- Base split scenes: same as the source scene for each seed.
- Hybrid strategy: `score-pose-hybrid`.
- Score weight: `0.35`; pose-novelty weight is `0.65`.
- Candidate score: held-out seed-model LPIPS from each seed's 25-frame random model.
- Each active set kept the corrected 25-frame random seed set fixed and added 25 hybrid-selected candidate frames.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- GPU: Modal L4.

| Seed | Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v2 | Random | `poster_modal_v2_b50_10k` | 50 | 10,000 | 29.323 | 0.930 | 0.225 | 0.690 |
| v2 | Active model error | `poster_modal_active_error_v2_b50_10k` | 50 | 10,000 | 30.390 | 0.945 | 0.199 | 0.806 |
| v2 | Active score-pose hybrid, `score_weight=0.35` | `poster_modal_active_hybrid_w035_v2_b50_10k` | 50 | 10,000 | 30.667 | 0.948 | 0.196 | 0.877 |
| v3 | Random | `poster_modal_v3_b50_10k` | 50 | 10,000 | 29.788 | 0.945 | 0.201 | 0.825 |
| v3 | Active model error | `poster_modal_active_error_v3_b50_10k` | 50 | 10,000 | 30.061 | 0.951 | 0.187 | 0.912 |
| v3 | Active score-pose hybrid, `score_weight=0.35` | `poster_modal_active_hybrid_w035_v3_b50_10k` | 50 | 10,000 | 31.668 | 0.957 | 0.171 | 0.668 |
| v4 | Random | `poster_modal_v4_b50_10k` | 50 | 10,000 | 28.461 | 0.936 | 0.215 | 0.826 |
| v4 | Active model error | `poster_modal_active_error_v4_b50_10k` | 50 | 10,000 | 26.615 | 0.919 | 0.253 | 0.889 |
| v4 | Active score-pose hybrid, `score_weight=0.35` | `poster_modal_active_hybrid_w035_v4_b50_10k` | 50 | 10,000 | 28.645 | 0.935 | 0.221 | 0.753 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `poster_modal_active_hybrid_w035_v2_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v2_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v2_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_043555/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_hybrid_w035_v3_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v3_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v3_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_043554/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_active_hybrid_w035_v4_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_active_hybrid_w035_v4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_041636/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- The `score_weight=0.35` hybrid beat random 50 on PSNR in all three corrected seeds.
- On v2 it beat random 50 by +1.344 PSNR, +0.017 SSIM, and -0.028 LPIPS; it also edged active-error by +0.277 PSNR, +0.002 SSIM, and -0.003 LPIPS.
- On v3 it beat random 50 by +1.880 PSNR, +0.011 SSIM, and -0.030 LPIPS; it also beat active-error by +1.607 PSNR, +0.006 SSIM, and -0.016 LPIPS.
- On v4 it beat random 50 by +0.184 PSNR but trailed by -0.001 SSIM and +0.006 LPIPS; it still repaired the pure active-error failure by +2.031 PSNR, +0.016 SSIM, and -0.032 LPIPS.
- Across v2/v3/v4, hybrid `score_weight=0.35` versus random 50 averages about +1.136 PSNR, +0.009 SSIM, and -0.017 LPIPS.
- This is the strongest current selector for the corrected poster experiments. The next useful step is to validate it on a second scene, then try a small local sweep around `score_weight=0.25` to `0.45` if it still holds up.

## Modal Dozer V1 Second-Scene Check

Date: 2026-06-06 Pacific / 2026-06-07 UTC

Dataset:

- Nerfstudio `dozer` sample downloaded from the Hugging Face mirror.
- Source scene: `dozer_available_v1`, filtered to 100 usable frames from 359 transform entries.
- Split seed: `20260609`.
- Materialized datasets include explicit Nerfstudio-native `train_filenames`, `val_filenames`, and `test_filenames`.
- Each budget uses the same 10 held-out test frames and 10 validation frames.
- Hybrid strategy: `score-pose-hybrid`.
- Score source: `/workspace/neural-mapping/data/scores/dozer_available_active_error_v1.json`.
- Score source model: `dozer_modal_v1_d4_fixed_b25_10k`.
- Score weights tested: `0.25`, `0.35`, `0.45`; pose-novelty weight is `1 - score_weight`.
- Candidate score: held-out seed-model LPIPS.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- Downscale factor: 4.
- GPU: Modal L4.

| Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random dozer v1 d4 | `dozer_modal_v1_d4_fixed_b25_10k` | 25 | 10,000 | 22.094 | 0.735 | 0.186 | 3.620 |
| Random dozer v1 d4 | `dozer_modal_v1_d4_fixed_b50_10k` | 50 | 10,000 | 23.625 | 0.780 | 0.158 | 4.595 |
| Active score-pose hybrid dozer v1 d4, `score_weight=0.25` | `dozer_modal_active_hybrid_w025_v1_d4_b50_10k` | 50 | 10,000 | 23.834 | 0.785 | 0.151 | 4.630 |
| Active score-pose hybrid dozer v1 d4, `score_weight=0.35` | `dozer_modal_active_hybrid_w035_v1_d4_b50_10k` | 50 | 10,000 | 23.844 | 0.785 | 0.150 | 4.624 |
| Active score-pose hybrid dozer v1 d4, `score_weight=0.45` | `dozer_modal_active_hybrid_w045_v1_d4_b50_10k` | 50 | 10,000 | 23.879 | 0.784 | 0.152 | 4.310 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `dozer_modal_v1_d4_fixed_b25_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v1_d4_fixed_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v1_d4_fixed_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-07_050422/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_v1_d4_fixed_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v1_d4_fixed_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v1_d4_fixed_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_050421/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_active_hybrid_w025_v1_d4_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w025_v1_d4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w025_v1_d4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_052604/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_active_hybrid_w035_v1_d4_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w035_v1_d4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w035_v1_d4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_051337/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_active_hybrid_w045_v1_d4_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w045_v1_d4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w045_v1_d4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_052607/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- Random 50 improved over random 25 by +1.531 PSNR, +0.045 SSIM, and -0.028 LPIPS.
- All three dozer hybrid weights beat dozer random 50 on PSNR, SSIM, and LPIPS.
- The best PSNR row was `score_weight=0.45`, beating random 50 by +0.254 PSNR, +0.004 SSIM, and -0.006 LPIPS.
- The best SSIM and LPIPS row was `score_weight=0.35`, beating random 50 by +0.219 PSNR, +0.006 SSIM, and -0.008 LPIPS.
- This is a modest but positive second-scene result for the hybrid selector; it does not prove robustness, but it reduces the chance that the poster result is purely scene-specific.
- Dozer required `downscale_factor=4` and non-interactive COLMAP point conversion because the mirrored dataset was processed by an older Nerfstudio pipeline without `sparse_pc.ply`.

## Modal Dozer Hybrid Seed Repeat

Date: 2026-06-06 Pacific / 2026-06-07 UTC

Dataset:

- Source capture: Nerfstudio `dozer`.
- Filtered seed scenes: `dozer_available_v1`, `dozer_available_v2`, `dozer_available_v3`.
- Split seeds: `20260609`, `20260610`, `20260611`.
- Each seed uses 100 filtered frames, 10 held-out test frames, and 10 validation frames.
- Hybrid strategy: `score-pose-hybrid`.
- Hybrid score weight: `0.35`; pose-novelty weight is `0.65`.
- Candidate score: held-out seed-model LPIPS from each seed's budget-25 checkpoint.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- Downscale factor: 4.
- GPU: Modal L4.

| Seed | Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v1 | Random dozer d4 | `dozer_modal_v1_d4_fixed_b25_10k` | 25 | 10,000 | 22.094 | 0.735 | 0.186 | 3.620 |
| v1 | Random dozer d4 | `dozer_modal_v1_d4_fixed_b50_10k` | 50 | 10,000 | 23.625 | 0.780 | 0.158 | 4.595 |
| v1 | Active score-pose hybrid, `score_weight=0.35` | `dozer_modal_active_hybrid_w035_v1_d4_b50_10k` | 50 | 10,000 | 23.844 | 0.785 | 0.150 | 4.624 |
| v2 | Random dozer d4 | `dozer_modal_v2_d4_b25_10k` | 25 | 10,000 | 20.903 | 0.724 | 0.202 | 4.780 |
| v2 | Random dozer d4 | `dozer_modal_v2_d4_b50_10k` | 50 | 10,000 | 22.367 | 0.773 | 0.169 | 4.699 |
| v2 | Active score-pose hybrid, `score_weight=0.35` | `dozer_modal_active_hybrid_w035_v2_d4_b50_10k` | 50 | 10,000 | 23.311 | 0.789 | 0.153 | 4.848 |
| v3 | Random dozer d4 | `dozer_modal_v3_d4_b25_10k` | 25 | 10,000 | 20.850 | 0.692 | 0.230 | 4.649 |
| v3 | Random dozer d4 | `dozer_modal_v3_d4_b50_10k` | 50 | 10,000 | 23.036 | 0.750 | 0.192 | 4.444 |
| v3 | Active score-pose hybrid, `score_weight=0.35` | `dozer_modal_active_hybrid_w035_v3_d4_b50_10k` | 50 | 10,000 | 24.714 | 0.796 | 0.152 | 4.589 |

New metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `dozer_modal_v2_d4_b25_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v2_d4_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v2_d4_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-07_053914/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_v2_d4_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v2_d4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v2_d4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_053916/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_active_hybrid_w035_v2_d4_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w035_v2_d4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w035_v2_d4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_054702/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_v3_d4_b25_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v3_d4_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v3_d4_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-07_053915/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_v3_d4_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v3_d4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_v3_d4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_053920/nerfstudio_models/step-000009999.ckpt` |
| `dozer_modal_active_hybrid_w035_v3_d4_b50_10k` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w035_v3_d4_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/dozer_modal_active_hybrid_w035_v3_d4_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-07_054720/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- The hybrid `score_weight=0.35` selector beat random 50 on PSNR, SSIM, and LPIPS in all three dozer seeds.
- On v1 it beat random 50 by +0.219 PSNR, +0.006 SSIM, and -0.008 LPIPS.
- On v2 it beat random 50 by +0.944 PSNR, +0.016 SSIM, and -0.016 LPIPS.
- On v3 it beat random 50 by +1.678 PSNR, +0.046 SSIM, and -0.040 LPIPS.
- Across dozer v1/v2/v3, hybrid `score_weight=0.35` versus random 50 averages +0.947 PSNR, +0.023 SSIM, and -0.021 LPIPS.
- This is the strongest current evidence that hybrid error-plus-pose selection is not just a poster-scene artifact.

## Modal Dozer Frame-Level Coverage Failure Prediction

Date: 2026-06-07

Question:

- Do simple camera-coverage uncertainty signals predict which candidate frames
  the budget-25 seed model renders poorly?

Setup:

- Source scenes: `dozer_available_v1`, `dozer_available_v2`, `dozer_available_v3`.
- Seed model for each scene: the corresponding random budget-25 Splatfacto run.
- Candidate error target: candidate-frame LPIPS from the seed model.
- Bad-frame threshold: seed-local 80th percentile of candidate LPIPS.
- Signals:
  - `nearest-train-distance`: camera-center distance to the nearest seed training frame.
  - `temporal-index-distance`: nearest seed training frame by trajectory index.
  - `uniform`: constant-score control.

| Seed | Signal | Spearman | AUROC | AUPRC | AUSE |
|---|---|---:|---:|---:|---:|
| v1 | nearest-train-distance | 0.178 | 0.729 | 0.648 | 0.047 |
| v1 | temporal-index-distance | 0.010 | 0.528 | 0.226 | 0.073 |
| v1 | uniform | n/a | 0.500 | 0.218 | 0.119 |
| v2 | nearest-train-distance | -0.293 | 0.329 | 0.172 | 0.085 |
| v2 | temporal-index-distance | -0.103 | 0.366 | 0.198 | 0.109 |
| v2 | uniform | n/a | 0.500 | 0.218 | 0.138 |
| v3 | nearest-train-distance | 0.074 | 0.391 | 0.195 | 0.050 |
| v3 | temporal-index-distance | 0.018 | 0.384 | 0.190 | 0.081 |
| v3 | uniform | n/a | 0.500 | 0.218 | 0.109 |
| mean | nearest-train-distance | -0.014 | 0.483 | 0.338 | 0.061 |
| mean | temporal-index-distance | -0.025 | 0.426 | 0.205 | 0.088 |

Report artifact paths in Modal:

| Seed | Report path |
|---|---|
| v1 | `/workspace/neural-mapping/outputs/reports/frame_uncertainty/dozer_available_active_error_v1_budget_025_lpips.json` |
| v2 | `/workspace/neural-mapping/outputs/reports/frame_uncertainty/dozer_available_active_error_v2_budget_025_lpips.json` |
| v3 | `/workspace/neural-mapping/outputs/reports/frame_uncertainty/dozer_available_active_error_v3_budget_025_lpips.json` |

Interpretation:

- Nearest training-camera distance predicted seed-model LPIPS failures on v1, but it was anti-correlated or weak on v2/v3.
- Temporal index distance was weaker than nearest camera distance and mostly near the uniform control.
- The mean nearest-distance AUROC was below random because v2/v3 failed despite v1 looking strong.
- This is an important negative baseline: simple camera distance is not a robust failure predictor on the dozer candidate pools, even though pose diversity helped active frame selection.
- The next uncertainty signal should be render-derived rather than purely geometric, for example opacity/transmittance confidence, residual maps, or ensemble disagreement.

## Modal Dozer Renderer Proxy Failure Prediction

Date: 2026-06-07

Question:

- Do Splatfacto renderer-output summaries predict which candidate frames the
  budget-25 seed model renders poorly?

Setup:

- Source scenes: `dozer_available_v1`, `dozer_available_v2`, `dozer_available_v3`.
- Seed model for each scene: the corresponding random budget-25 Splatfacto run.
- Candidate error target: candidate-frame LPIPS from the seed model.
- Bad-frame threshold: seed-local 80th percentile of candidate LPIPS.
- Candidate scorer additionally exported renderer-derived fields from
  `outputs["accumulation"]` and `outputs["depth"]`.
- Signals:
  - `mean_transmittance`: `1 - mean(accumulation)`.
  - `low_accumulation_fraction`: fraction of rendered pixels with accumulation below 0.5.
  - `std_accumulation`: standard deviation of rendered accumulation.
  - `std_depth`: standard deviation of rendered depth.

| Seed | Signal | Spearman | AUROC | AUPRC | AUSE |
|---|---|---:|---:|---:|---:|
| v1 | mean_transmittance | -0.005 | 0.624 | 0.397 | 0.049 |
| v1 | low_accumulation_fraction | -0.094 | 0.515 | 0.357 | 0.058 |
| v1 | std_accumulation | 0.007 | 0.630 | 0.398 | 0.046 |
| v1 | std_depth | -0.219 | 0.438 | 0.319 | 0.086 |
| v2 | mean_transmittance | -0.482 | 0.134 | 0.146 | 0.113 |
| v2 | low_accumulation_fraction | -0.028 | 0.395 | 0.204 | 0.102 |
| v2 | std_accumulation | -0.225 | 0.271 | 0.167 | 0.095 |
| v2 | std_depth | -0.169 | 0.205 | 0.159 | 0.085 |
| v3 | mean_transmittance | 0.240 | 0.473 | 0.218 | 0.043 |
| v3 | low_accumulation_fraction | 0.213 | 0.527 | 0.221 | 0.056 |
| v3 | std_accumulation | 0.303 | 0.548 | 0.245 | 0.040 |
| v3 | std_depth | 0.138 | 0.393 | 0.237 | 0.062 |
| mean | mean_transmittance | -0.083 | 0.410 | 0.254 | 0.068 |
| mean | low_accumulation_fraction | 0.030 | 0.479 | 0.260 | 0.072 |
| mean | std_accumulation | 0.028 | 0.483 | 0.270 | 0.060 |
| mean | std_depth | -0.083 | 0.346 | 0.238 | 0.078 |

Report artifact paths in Modal:

| Seed | Score path | Report path |
|---|---|---|
| v1 | `/workspace/neural-mapping/data/scores/dozer_available_render_proxy_v1.json` | `/workspace/neural-mapping/outputs/reports/frame_uncertainty/dozer_available_render_proxy_v1_budget_025_lpips.json` |
| v2 | `/workspace/neural-mapping/data/scores/dozer_available_render_proxy_v2.json` | `/workspace/neural-mapping/outputs/reports/frame_uncertainty/dozer_available_render_proxy_v2_budget_025_lpips.json` |
| v3 | `/workspace/neural-mapping/data/scores/dozer_available_render_proxy_v3.json` | `/workspace/neural-mapping/outputs/reports/frame_uncertainty/dozer_available_render_proxy_v3_budget_025_lpips.json` |

Modal run URLs:

- v1 scoring: `ap-NbkFTgddX79IimjpLw1IPs`; v1 evaluation: `ap-dyWwxPvLJzbUqvcTipSExc`.
- v2 scoring: `ap-ey2uSHmGvbFTVG3GB860Cr`; v2 evaluation: `ap-BQcr1myjgncUmxWFl0TNUF`.
- v3 scoring: `ap-DMiZQjx6yVDkYrYp2u62Jw`; v3 evaluation: `ap-gf1VyRK8hx1aQ85iyUPSXa`.

Interpretation:

- The renderer proxy fields were present and measurable, so the extraction path works.
- These simple accumulation/depth summaries are not robust frame-level failure predictors on dozer.
- `std_accumulation` was the best average proxy, but mean AUROC was only 0.483, essentially random and close to the geometric nearest-distance baseline.
- The next signal should use richer render information than global frame summaries, for example patch-level accumulation/residual maps, per-pixel error alignment, or ensemble disagreement.

## Modal Dozer Pixel-Level Transmittance Failure Prediction

Date: 2026-06-07

Question:

- Does per-pixel transmittance (`1 - accumulation`) align with per-pixel RGB
  rendering error on candidate views?

Setup:

- Source scenes: `dozer_available_v1`, `dozer_available_v2`, `dozer_available_v3`.
- Seed model for each scene: the corresponding random budget-25 Splatfacto run.
- Candidate pool: 55 non-seed candidate frames per split.
- Error target: per-pixel RGB L1 error after normalizing rendered and target RGB to `0..1`.
- Pixel sampling: deterministic sample of 10,000 valid pixels per frame, 550,000 pixels per seed.
- Bad-pixel threshold: report-local 80th percentile RGB L1 error.

| Seed | Signal | Count | Mean Error | Spearman | AUROC | AUPRC | AUSE |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | transmittance | 550,000 | 0.125 | -0.013 | 0.481 | 0.205 | 0.085 |
| v2 | transmittance | 550,000 | 0.150 | -0.108 | 0.439 | 0.179 | 0.105 |
| v3 | transmittance | 550,000 | 0.119 | -0.033 | 0.479 | 0.201 | 0.082 |
| mean | transmittance | 550,000 | 0.131 | -0.051 | 0.466 | 0.195 | 0.090 |

Report artifact paths in Modal:

| Seed | Report path |
|---|---|
| v1 | `/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/dozer_available_render_maps_v1_budget_025_rgb-l1.json` |
| v2 | `/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/dozer_available_render_maps_v2_budget_025_rgb-l1.json` |
| v3 | `/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/dozer_available_render_maps_v3_budget_025_rgb-l1.json` |

Modal run URLs:

- v1 corrected run: `ap-rBry06nCbzXWfCPE35s2UP`.
- v2 run: `ap-Fs4eIGJPyjV1jQoOV9pd4s`.
- v3 run: `ap-PGl5A1iAuwVKBtaCKdsyns`.
- Superseded pre-normalization smoke: `ap-hJCUdTi4auo6QC8YYq19Wo`.

Interpretation:

- The pixel-level extraction path works and catches raw renderer confidence maps without saving image files.
- Raw transmittance is a negative result across all three dozer seeds: mean Spearman is -0.051 and mean bad-pixel AUROC is 0.466.
- The next pixel-level signal should be less naive than raw transmittance, for example local residual propagation, patch-level statistics, or ensemble RGB disagreement.

## Modal Dozer Expanded Pixel-Level Renderer Signals

Date: 2026-06-07

Question:

- Do simple local renderer-map statistics or gradients improve over raw
  transmittance for pixel-level RGB failure prediction?

Setup:

- Source scenes: `dozer_available_v1`, `dozer_available_v2`, `dozer_available_v3`.
- Seed model for each scene: the corresponding random budget-25 Splatfacto run.
- Error target: per-pixel RGB L1 error, normalized to `0..1`.
- Pixel sampling: deterministic sample of 10,000 valid pixels per frame, 550,000 pixels per seed.
- Patch size: 15 pixels for local transmittance mean/std.
- Bad-pixel threshold: report-local 80th percentile RGB L1 error.

| Seed | Signal | Spearman | AUROC | AUPRC | AUSE |
|---|---|---:|---:|---:|---:|
| v1 | transmittance | -0.013 | 0.481 | 0.205 | 0.085 |
| v1 | local-mean-transmittance | -0.006 | 0.483 | 0.207 | 0.086 |
| v1 | local-std-transmittance | 0.014 | 0.491 | 0.208 | 0.083 |
| v1 | accumulation-gradient | 0.019 | 0.492 | 0.204 | 0.082 |
| v1 | depth-gradient | -0.039 | 0.488 | 0.199 | 0.085 |
| v2 | transmittance | -0.108 | 0.439 | 0.179 | 0.105 |
| v2 | local-mean-transmittance | -0.143 | 0.422 | 0.178 | 0.111 |
| v2 | local-std-transmittance | -0.126 | 0.429 | 0.180 | 0.109 |
| v2 | accumulation-gradient | -0.089 | 0.443 | 0.180 | 0.104 |
| v2 | depth-gradient | -0.185 | 0.424 | 0.168 | 0.110 |
| v3 | transmittance | -0.033 | 0.479 | 0.201 | 0.082 |
| v3 | local-mean-transmittance | -0.037 | 0.478 | 0.204 | 0.084 |
| v3 | local-std-transmittance | -0.019 | 0.486 | 0.206 | 0.081 |
| v3 | accumulation-gradient | -0.004 | 0.490 | 0.203 | 0.080 |
| v3 | depth-gradient | -0.079 | 0.467 | 0.188 | 0.085 |
| mean | transmittance | -0.051 | 0.466 | 0.195 | 0.090 |
| mean | local-mean-transmittance | -0.062 | 0.461 | 0.196 | 0.093 |
| mean | local-std-transmittance | -0.044 | 0.468 | 0.198 | 0.091 |
| mean | accumulation-gradient | -0.024 | 0.475 | 0.196 | 0.089 |
| mean | depth-gradient | -0.101 | 0.460 | 0.185 | 0.093 |

Report artifact paths in Modal:

| Seed | Report path |
|---|---|
| v1 | `/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/dozer_available_render_maps_patch_v1_budget_025_rgb-l1.json` |
| v2 | `/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/dozer_available_render_maps_patch_v2_budget_025_rgb-l1.json` |
| v3 | `/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/dozer_available_render_maps_patch_v3_budget_025_rgb-l1.json` |

Modal run URLs:

- v1: `ap-36ScrnOWYXR1plb5omR4JU`.
- v2: `ap-hMUDbcwvOFkOvpcJYjgh0l`.
- v3: `ap-zqUBGxNgZjG1u5drMaM9kG`.

Interpretation:

- Local transmittance statistics and simple renderer-map gradients did not fix the raw-transmittance failure.
- `accumulation-gradient` was the best average signal, but mean AUROC was only 0.475, still below random.
- The next credible uncertainty baseline should be model-disagreement based, for example rendering the same candidate views from multiple independently trained seed models and scoring per-pixel RGB variance.

## Modal Dozer Ensemble Pixel-Level Disagreement Smoke

Date: 2026-06-07

Question:

- Does per-pixel RGB variance across independently trained seed models align
  with per-pixel RGB rendering error on candidate views?

Setup:

- Candidate source scene: `dozer_available_v1`.
- Candidate pool: 55 non-seed candidate frames from the budget-25 split.
- Ensemble seed models:
  - `dozer_modal_v1_d4_fixed_b25_10k`
  - `dozer_modal_v2_d4_b25_10k`
  - `dozer_modal_v3_d4_b25_10k`
- Error target: per-pixel RGB L1 error between the ensemble mean render and target RGB, normalized to `0..1`.
- Uncertainty signal: per-pixel mean RGB variance across the three rendered predictions.
- Pixel sampling: deterministic sample of 5,000 valid pixels per frame, 275,000 total sampled pixels.
- Bad-pixel threshold: report-local 80th percentile RGB L1 error.

| Candidate Scene | Signal | Count | Mean Error | Spearman | AUROC | AUPRC | AUSE |
|---|---|---:|---:|---:|---:|---:|---:|
| `dozer_available_v1` | ensemble-rgb-variance | 275,000 | 0.100 | 0.597 | 0.733 | 0.412 | 0.023 |

Report artifact path in Modal:

| Report path |
|---|
| `/workspace/neural-mapping/outputs/reports/ensemble_uncertainty_maps/dozer_available_ensemble_maps_v1_budget_025_rgb-l1.json` |

Modal run URL:

- `ap-eHnUaQrGL5Ff4jttG1JWwD`

Interpretation:

- This is the first uncertainty signal in the dozer sequence that clearly tracks pixel-level render failure.
- Ensemble RGB variance substantially outperformed single-model renderer proxies: AUROC improved from roughly random (`0.475` best expanded single-model mean) to `0.733`.
- AUSE dropped to `0.023`, meaning sorting pixels by ensemble disagreement removes high-error pixels much closer to the oracle risk-coverage curve than the renderer-map baselines.
- This is still one candidate pool and one three-model ensemble. The next check should repeat the same ensemble-disagreement evaluation on v2/v3 candidate pools or use it to drive active frame selection, then train/evaluate the selected 50-frame set.
