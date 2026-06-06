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

| Scene | Budget | Iterations | PSNR | SSIM | LPIPS | FPS |
|---|---:|---:|---:|---:|---:|---:|
| `poster_modal_b25_10k` | 25 | 10,000 | 31.197 | 0.952 | 0.167 | 0.669 |
| `poster_modal_b50_10k` | 50 | 10,000 | 30.588 | 0.947 | 0.197 | 0.839 |

Metric artifact paths in Modal:

| Scene | Metrics path | Checkpoint |
|---|---|---|
| `poster_modal_b25_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_b25_10k/splatfacto/budget_025/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_b25_10k/splatfacto/budget_025/train/unnamed/splatfacto/2026-06-06_044723/nerfstudio_models/step-000009999.ckpt` |
| `poster_modal_b50_10k` | `/workspace/neural-mapping/outputs/runs/poster_modal_b50_10k/splatfacto/budget_050/metrics/ns_eval.json` | `/workspace/neural-mapping/outputs/runs/poster_modal_b50_10k/splatfacto/budget_050/train/unnamed/splatfacto/2026-06-06_051459/nerfstudio_models/step-000009999.ckpt` |

Interpretation:

- This is a smoke baseline, not a final claim.
- The 50-frame run did not improve over the 25-frame run on this single small split.
- The next comparison should add repeated seeds or move directly to an active/coverage selection baseline under the same train/eval policy.
