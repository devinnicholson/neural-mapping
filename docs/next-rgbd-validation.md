# Next RGB-D Validation Experiment

Purpose: test whether the compact-budget behavior from TUM FR1 xyz v9 transfers
to a second RGB-D scene without spending a full saturation sweep.

## Recommended Experiment

Run `freiburg1_desk` as `tum_fr1_desk_v4_compact` with budgets 25, 50, and
100. Keep the held-out split fixed and compare random b50/b100 against active
depth-gradient hybrid b50/b100.

Why this scene:

- Desk already produced stable transmittance-positive evidence across v1-v3.
- It is less chaotic than room, so it is a better second-scene check for the
  v9 compact-budget claim before returning to harder room sequences.
- Budgets 50 and 100 test the useful range without wasting compute on another
  b150 saturation control.

Success gate:

- Primary: active b100 beats random b100 on PSNR, LPIPS, aligned AbsRel, and
  aligned delta1.
- Secondary: active b50 is RGB-positive and does not materially regress aligned
  AbsRel.
- Negative-control read: if active b100 loses, inspect the uncertainty report
  before changing the selector. A loss is evidence about signal transfer, not
  an infrastructure failure.

## Commands

Prepare the data and random splits:

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

Train and evaluate random baselines:

```bash
for budget in 25 50 100; do
  modal run modal_app.py \
    --action train \
    --data-scene-name tum_fr1_desk_v4_compact \
    --scene-name tum_fr1_desk_v4_compact_b${budget}_7k \
    --budget ${budget} \
    --iterations 7000 \
    --downscale-factor 1

  modal run modal_app.py \
    --action eval \
    --scene-name tum_fr1_desk_v4_compact_b${budget}_7k \
    --budget ${budget}

  modal run modal_app.py \
    --action depth-eval \
    --data-scene-name tum_fr1_desk_v4_compact \
    --scene-name tum_fr1_desk_v4_compact_b${budget}_7k \
    --budget ${budget} \
    --depth-cache-images cpu
done
```

Generate the budget-25 depth-error uncertainty report:

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

Materialize active b50 and b100:

```bash
for budget in 50 100; do
  modal run modal_app.py \
    --action prepare-active \
    --source-data-scene-name tum_fr1_desk_v4_compact \
    --base-split-scene-name tum_fr1_desk_v4_compact \
    --data-scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b${budget} \
    --base-budget 25 \
    --target-budget ${budget} \
    --active-strategy score-pose-hybrid \
    --score-path /workspace/neural-mapping/outputs/reports/render_uncertainty_maps/tum_fr1_desk_v4_compact_depth_error_maps_budget_025_depth-aligned-abs-rel.json \
    --score-key top_decile_mean_uncertainty.depth-gradient \
    --score-weight 0.65
done
```

Train and evaluate active b50 and b100:

```bash
for budget in 50 100; do
  modal run modal_app.py \
    --action train \
    --data-scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b${budget} \
    --scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b${budget}_7k \
    --budget ${budget} \
    --iterations 7000 \
    --downscale-factor 1

  modal run modal_app.py \
    --action eval \
    --scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b${budget}_7k \
    --budget ${budget}

  modal run modal_app.py \
    --action depth-eval \
    --data-scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b${budget} \
    --scene-name tum_fr1_desk_v4_compact_active_depth_grad_hybrid_b${budget}_7k \
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
new Modal paths to `docs/run-manifest.md`, and update the dashboard only if the
result changes the headline story.
