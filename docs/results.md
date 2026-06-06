# Results

This page records lightweight experiment results that are small enough to track in Git.
Checkpoints, renders, and full Modal output volumes are intentionally not tracked.

## Modal Poster Baselines

Date: 2026-06-05 Pacific / 2026-06-06 UTC

Dataset:

- Nerfstudio `poster` sample downloaded from the Hugging Face mirror.
- Filtered scene: `poster_available`.
- Split seed: `20260529`.
- Test frames: 10 held-out frames via `eval_filenames`.
- Validation frames: 10 frames included in the materialized training `transforms.json` for this smoke baseline.
- Method: Nerfstudio `splatfacto`.
- Training length: 10,000 iterations.
- GPU: Modal L4.

| Selection | Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| Random | `poster_modal_b25_10k` | 25 | 10,000 | 31.197 | 0.952 | 0.167 | 0.669 |
| Random | `poster_modal_b50_10k` | 50 | 10,000 | 30.588 | 0.947 | 0.197 | 0.839 |
| Farthest index | `poster_modal_farthest_b25_10k` | 25 | 10,000 | 22.842 | 0.909 | 0.272 | 0.622 |
| Farthest index | `poster_modal_farthest_b50_10k` | 50 | 10,000 | 29.247 | 0.930 | 0.229 | 0.590 |
| Farthest pose | `poster_modal_pose_b25_10k` | 25 | 10,000 | 22.835 | 0.901 | 0.290 | 0.552 |
| Farthest pose | `poster_modal_pose_b50_10k` | 50 | 10,000 | 27.571 | 0.929 | 0.243 | 0.791 |
| Active pose novelty | `poster_modal_active_pose_b50_10k` | 50 | 10,000 | 27.335 | 0.926 | 0.242 | 0.764 |

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

Interpretation:

- This is a smoke baseline, not a final claim.
- The random 50-frame run did not improve over the random 25-frame run on this single small split.
- `farthest-index` underperformed random at both budgets, especially at 25 frames.
- `farthest-pose` improved from 25 to 50 frames, but still underperformed random at both budgets; camera-center spread alone is not sufficient for this scene.
- Active pose novelty kept the random 25-frame seed set fixed and added 25 pose-novel candidates, but it still underperformed random 50 and farthest-index 50.
- The next baseline should score candidates using view content, validation error, visibility, or uncertainty instead of camera-center distance alone.
