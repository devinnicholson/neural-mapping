# RGB-D Validation Roadmap

Purpose: track the next depth-bearing validation after the TUM FR1 xyz v9
budget sweep.

## Completed: TUM FR1 Desk v4 Compact

Status: completed on 2026-07-05 UTC.

Run `freiburg1_desk` as `tum_fr1_desk_v4_compact` with budgets 25, 50, and
100. The held-out split was fixed, and random b50/b100 were compared against
active depth-gradient hybrid b50/b100.

Result:

| Budget | Delta PSNR | Delta LPIPS | Delta raw AbsRel | Delta aligned AbsRel | Delta aligned delta1 | Read |
|---:|---:|---:|---:|---:|---:|---|
| 50 | +0.994 | -0.041 | -0.021 | -0.033 | +0.045 | RGB positive and depth positive |
| 100 | +0.708 | -0.024 | +0.005 | -0.024 | +0.021 | RGB positive and aligned-depth positive |

Success gate:

- Primary: passed for active b100 on PSNR, LPIPS, aligned AbsRel, and aligned
  delta1.
- Secondary: passed for active b50 on RGB and aligned depth.
- Caveat: b100 raw AbsRel moved from 0.285 to 0.290, so the precise claim is
  aligned-geometry improvement, not all-depth-metrics improvement.

Tracked artifacts:

- Results: `docs/results.md`.
- Manifest: `docs/run-manifest.md`.
- Blog CSV: `docs/blog-assets/tum-fr1-desk-v4-compact-validation.csv`.
- Blog plot: `docs/blog-assets/tum-fr1-desk-v4-compact-validation.svg`.

## Next Recommended Experiment

Run a compact `freiburg1_room` validation as `tum_fr1_room_v4_compact` with the
same budgets 25, 50, and 100. Keep the held-out split fixed and compare random
b50/b100 against active depth-gradient hybrid b50/b100.

Why this scene:

- Room is the hardest TUM RGB-D sequence already explored in this project.
- Earlier room evidence was depth-positive but RGB-mixed, so it is the right
  stress test after the desk v4 transfer success.
- Budgets 50 and 100 test the same useful compact range as xyz v9 and desk v4
  without spending on another b150 saturation control.

Success gate:

- Primary: active b100 beats random b100 on PSNR, LPIPS, aligned AbsRel, and
  aligned delta1.
- Secondary: active b50 is RGB-positive and does not materially regress aligned
  AbsRel.
- Diagnostic read: if depth-gradient loses, use the room uncertainty report to
  choose between transmittance, local-mean-transmittance, and
  accumulation-gradient before changing the active split code.

## Commands

Prepare the data and random splits:

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

Train and evaluate random baselines:

```bash
for budget in 25 50 100; do
  modal run modal_app.py \
    --action train \
    --data-scene-name tum_fr1_room_v4_compact \
    --scene-name tum_fr1_room_v4_compact_b${budget}_7k \
    --budget ${budget} \
    --iterations 7000 \
    --downscale-factor 1

  modal run modal_app.py \
    --action eval \
    --scene-name tum_fr1_room_v4_compact_b${budget}_7k \
    --budget ${budget}

  modal run modal_app.py \
    --action depth-eval \
    --data-scene-name tum_fr1_room_v4_compact \
    --scene-name tum_fr1_room_v4_compact_b${budget}_7k \
    --budget ${budget} \
    --depth-cache-images cpu
done
```

Generate the budget-25 depth-error uncertainty report:

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

Materialize active b50 and b100:

```bash
for budget in 50 100; do
  modal run modal_app.py \
    --action prepare-active \
    --source-data-scene-name tum_fr1_room_v4_compact \
    --base-split-scene-name tum_fr1_room_v4_compact \
    --data-scene-name tum_fr1_room_v4_compact_active_depth_grad_hybrid_b${budget} \
    --base-budget 25 \
    --target-budget ${budget} \
    --active-strategy score-pose-hybrid \
    --score-path /workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr1_room_v4_compact_depth_error_maps_budget_025_depth-aligned-abs-rel.json \
    --score-key top_decile_mean_uncertainty.depth-gradient \
    --score-weight 0.65
done
```

Train and evaluate active b50 and b100:

```bash
for budget in 50 100; do
  modal run modal_app.py \
    --action train \
    --data-scene-name tum_fr1_room_v4_compact_active_depth_grad_hybrid_b${budget} \
    --scene-name tum_fr1_room_v4_compact_active_depth_grad_hybrid_b${budget}_7k \
    --budget ${budget} \
    --iterations 7000 \
    --downscale-factor 1

  modal run modal_app.py \
    --action eval \
    --scene-name tum_fr1_room_v4_compact_active_depth_grad_hybrid_b${budget}_7k \
    --budget ${budget}

  modal run modal_app.py \
    --action depth-eval \
    --data-scene-name tum_fr1_room_v4_compact_active_depth_grad_hybrid_b${budget} \
    --scene-name tum_fr1_room_v4_compact_active_depth_grad_hybrid_b${budget}_7k \
    --budget ${budget} \
    --depth-cache-images cpu
done
```

## Result Logging

After the run finishes:

```bash
modal run modal_app.py --action metrics
```

Then add the b50 and b100 RGB/depth numbers to `docs/results.md`, append the
new Modal paths to `docs/run-manifest.md`, and update the dashboard if the
result changes the headline story.
