# Experiment Protocol

This protocol defines how to run experiments so uncertainty scores can be compared against actual error without leakage. The default benchmark is offline posed mapping, not online SLAM.

## Experimental Invariants

- Use Replica first for all pipeline bring-up and controlled claims.
- Use Nerfstudio Splatfacto with gsplat as the main method when possible.
- Use depth-Nerfacto only as a documented fallback or baseline.
- Keep validation/calibration and test frames separate.
- Lock scene splits before inspecting test uncertainty-error results.
- Report uncertainty against RGB, depth, and geometry error when each target is available.
- Compare every learned or derived uncertainty signal against trivial baselines.

## Datasets

### Replica

Role:

- Main synthetic indoor benchmark.
- First dataset for debugging, fixed splits, stress tests, and final controlled results.

Scene counts:

- Bring-up: 1 scene.
- Controlled study: 4 scenes.
- Final synthetic results: 6-8 scenes if compute allows.

Required assets:

- RGB frames.
- Camera intrinsics and poses.
- Depth frames.
- Mesh or geometry target if available.

### ScanNet++

Role:

- Real-world validation after the Replica protocol is stable.

Scene counts:

- Target: 3-6 scenes.
- Minimum: 3 scenes.

Fallback:

- Use ScanNet if ScanNet++ setup, access, or preprocessing blocks progress.
- Use TUM RGB-D only for a small trajectory sanity check.

## Splits

For each scene:

1. Sort frames by trajectory order.
2. Build a fixed frame pool after filtering missing RGB, pose, or depth.
3. Reserve held-out test frames before selecting any training budget.
4. Reserve validation/calibration frames from non-test frames.
5. Select training budgets from the remaining candidate pool.

Training budgets:

- 25 frames.
- 50 frames.
- 100 frames.
- 200 frames.

Required split metadata:

- Scene id.
- Dataset version or source path.
- Frame ids for train, validation, and test.
- Random seed.
- Selection method.
- Resolution and crop policy.
- Whether depth and mesh targets are available.

Calibration rule:

- Validation frames may tune score normalization, uncertainty thresholds, reliability bins, and temperature or scale parameters.
- Test frames may only be used for final metric reporting.

## Training And Rendering

Default method:

- Nerfstudio Splatfacto with gsplat.

Baselines:

- Nerfacto for radiance-field comparison.
- depth-Nerfacto for depth-aware fallback comparison.
- Optional SplaTAM, VarSplat, or CG-SLAM only after the core benchmark runs.

For every scene and budget:

1. Train from the fixed training split.
2. Save the exact Nerfstudio config and checkpoint.
3. Render validation and test RGB.
4. Render validation and test depth where supported.
5. Export opacity, alpha accumulation, transmittance, or any available renderer confidence fields.
6. Save camera metadata with every render.

Run naming:

```text
{dataset}/{scene}/{method}/budget_{frames}/seed_{seed}/{split}
```

Example:

```text
replica/office0/splatfacto/budget_050/seed_000/test
```

## Error Targets

RGB error:

- Pixel L1 or L2 error in linearized or consistently normalized RGB.
- PSNR, SSIM, and LPIPS as aggregate rendering metrics.

Depth error:

- Absolute depth error.
- RMSE.
- AbsRel.
- Valid-pixel masked metrics.
- Separate reporting for all valid pixels and near-surface/depth-edge regions if masks are reliable.

Geometry error:

- Chamfer distance.
- Accuracy.
- Completeness.
- F-score at 5 cm and 10 cm.
- Normal consistency only when normals are reliable.

Bad-pixel or bad-patch labels:

- Prefer fixed percentile thresholds within each target, such as top 10 percent error on validation.
- Also report absolute thresholds when they are meaningful for depth or geometry.
- Use patch aggregation to reduce pixel noise, especially for depth and geometry.

## Uncertainty Signals

Required baselines:

- Uniform uncertainty.
- Distance to nearest training camera.
- Number of visible training views or coverage count.
- Image-space depth-edge heuristic when depth is available.

Primary signals:

- Alpha or transmittance confidence from Splatfacto/gsplat.
- Training-view photometric residual.
- Depth disagreement with observed or nearby-view projected depth.
- Lightweight ensemble disagreement across seeds or frame subsets.

Optional signal:

- Per-splat variance or learned covariance extension after post-hoc signals are stable.

Normalization:

- Convert every signal so larger means more uncertain.
- Fit any min/max, z-score, isotonic, temperature, or bin calibration on validation only.
- Apply the frozen transform to test frames.

## Metrics

Rendering quality:

- PSNR.
- SSIM.
- LPIPS.

Uncertainty-error alignment:

- Spearman correlation between uncertainty and RGB/depth/geometry error.
- AUROC for bad-pixel or bad-patch detection.
- AUPRC for bad-pixel or bad-patch detection.
- Risk-coverage curves.
- Sparsification curves.
- AUSE against oracle error ranking.
- Reliability diagrams.
- Calibration error for regression-style uncertainty.
- Negative log likelihood only for uncertainty methods that produce a predictive distribution.

Aggregation:

- Report per-scene metrics.
- Report mean and standard error across scenes.
- Keep Replica and real-data results in separate tables.
- Separate pixel-level and patch-level results.

## Stress Tests

Run on Replica first:

- Sparse views: 25, 50, 100, and 200 frames.
- Extrapolation: hold out a room side, trajectory segment, or spatial region.
- Pose noise: 1-5 cm translation and 1-5 degrees rotation.
- Depth corruption: dropout, Gaussian noise, and missing-depth masks.
- Appearance corruption: blur, exposure shifts, and low light.
- Surface categories when labels are available: textureless walls, reflective surfaces, glass, clutter, and depth edges.

Repeat only the most informative stress tests on ScanNet++ or fallback real data.

## Optional Keyframe Selection Protocol

Use this only after the uncertainty benchmark is stable.

Setup:

1. Start from a fixed small seed training set.
2. Keep a fixed candidate pool and test set.
3. Score each candidate using uncertainty reduction, expected information gain, or a calibrated proxy.
4. Add one frame or a fixed-size batch per round.
5. Retrain or update using the same compute policy for all methods.
6. Evaluate after each budget.

Baselines:

- Random frame selection with multiple seeds.
- Uniform temporal sampling.
- Farthest-camera-distance sampling.
- Motion-threshold keyframe selection.
- Coverage-only selection.

Outputs:

- Quality versus number of frames.
- Uncertainty-error alignment versus number of frames.
- Area under the frame-efficiency curve.
- Failure cases where uncertainty selects redundant or unhelpful views.

## Reporting Rules

- State whether each result uses Splatfacto, depth-Nerfacto, or another method.
- Do not mix validation and test metrics in a single headline table.
- Report failed uncertainty signals rather than omitting them silently.
- Include qualitative maps only beside quantitative uncertainty-error metrics.
- Treat keyframe selection as optional and report it as negative if it does not improve over random or uniform sampling.
