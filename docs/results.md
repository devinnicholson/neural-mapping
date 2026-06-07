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
