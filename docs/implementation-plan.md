# Implementation Plan

This plan translates the roadmap into implementation work packages. It keeps the first contribution narrow: calibrated uncertainty-error evaluation for posed indoor 3D Gaussian maps. Keyframe selection is optional and starts only after failure prediction is reproducible.

## Scope

Primary stack:

- Nerfstudio from source.
- Splatfacto as the main 3D Gaussian method.
- gsplat as the rasterization backend.
- Replica as the first and main controlled dataset.
- ScanNet++ as real-world validation, with ScanNet or TUM RGB-D as fallbacks.

Primary claim:

- Uncertainty maps from a 3D Gaussian scene representation can predict held-out RGB, depth, and geometry error better than trivial baselines.

Out of scope until the core result works:

- Full online SLAM.
- Loop closure, global planning, or robotics control.
- Broad novelty claims around uncertainty-aware 3DGS SLAM.
- Keyframe selection beyond a controlled offline candidate-pool experiment.

## Work Packages

### 1. Environment And Baseline Bring-Up

Deliverables:

- Nerfstudio environment with Splatfacto training on a toy/example scene.
- Documented CUDA, Python, Nerfstudio, and gsplat versions.
- One smoke-test command sequence that trains, renders, and evaluates.

Exit criteria:

- Splatfacto completes a train/render/eval cycle without local code changes.
- The output directory contains config, checkpoint, rendered RGB, and standard metrics.

Fallback:

- If Splatfacto or gsplat blocks progress by the end of the first baseline cycle, switch the benchmark pipeline to depth-Nerfacto while keeping Splatfacto as a later target.

### 2. Replica Data Pipeline

Deliverables:

- One Replica scene converted into a Nerfstudio-compatible layout.
- Fixed frame pool sorted by trajectory order.
- Reproducible train, validation/calibration, and test splits.
- Training budgets: 25, 50, 100, and 200 frames.

Exit criteria:

- One Replica scene runs end to end at every frame budget.
- Held-out RGB and depth are rendered for the same fixed test frames.

Implementation notes:

- Treat validation as the only split allowed for score normalization, threshold selection, or calibration.
- Keep test frames untouched until final metric computation.
- Save split manifests with scene name, frame ids, seed, resolution, and camera/depth availability.

### 3. Error Map Generation

Deliverables:

- Per-test-frame RGB error maps.
- Per-test-frame depth error maps with valid-depth masks.
- Optional geometry targets from rendered depth fusion or dataset meshes.
- Visualization grids for RGB, rendered RGB, depth, rendered depth, error, and masks.

Exit criteria:

- For every held-out frame, RGB and depth errors use the same camera, resolution, crop, and validity mask.
- Aggregate metrics match the per-frame error maps.

### 4. Uncertainty Signals

Implement signals in increasing complexity:

1. Distance to nearest training camera.
2. View coverage count.
3. Alpha or transmittance confidence from Splatfacto/gsplat rendering.
4. Training-view photometric residual projected or aggregated into candidate views.
5. Depth disagreement between rendered depth and observed or nearby-view projected depth.
6. Lightweight ensemble disagreement if compute allows.
7. Per-splat variance only after the post-hoc signals are stable.

Exit criteria:

- At least three signals produce aligned uncertainty maps for every held-out frame.
- At least one signal is not simply a camera-distance or coverage proxy.

### 5. Calibration And Failure Prediction

Deliverables:

- Pixel-level and patch-level uncertainty-error tables.
- Spearman correlation with RGB and depth error.
- AUROC and AUPRC for bad-pixel or bad-patch detection.
- Risk-coverage and sparsification curves.
- AUSE where an oracle ranking is available.
- Reliability diagrams for calibrated uncertainty bins.
- Negative log likelihood only for signals that define a proper predictive distribution.

Exit criteria:

- Validation frames are used to fit score transforms, binning, thresholds, or calibration parameters.
- Test metrics are reported once per locked split.
- Every uncertainty result is compared against uniform, camera-distance, and coverage baselines.

### 6. Multi-Scene Study And Stress Tests

Replica progression:

- Bring-up: 1 scene.
- Controlled study: 4 scenes.
- Final synthetic benchmark: 6-8 scenes if compute allows.

Stress tests:

- Sparse views at 25, 50, 100, and 200 frames.
- Extrapolation by holding out a spatial region or side of a room.
- Pose perturbations at 1-5 cm and 1-5 degrees.
- Depth dropout, Gaussian depth noise, and missing-depth regions.
- Appearance perturbations such as blur, exposure shifts, and low light.

Exit criteria:

- Results identify when uncertainty tracks error and when it fails.
- Claims are tied to uncertainty-error metrics, not only PSNR/SSIM/LPIPS.

### 7. Real-World Validation

Preferred path:

- Prepare 3-6 ScanNet++ scenes after the Replica pipeline is stable.
- Reuse the same split, rendering, uncertainty, and evaluation code.

Fallback path:

- Use ScanNet if ScanNet++ setup is too heavy.
- Use TUM RGB-D only for small trajectory checks or as an emergency real-data fallback.

Exit criteria:

- At least three real scenes are evaluated, or the fallback decision is documented with the exact blocker.

### 8. Optional Keyframe Selection

Start only after uncertainty-error evaluation is complete.

Protocol:

- Begin from a small initial training set.
- Maintain a fixed candidate frame pool.
- Score candidates using uncertainty reduction or expected information gain.
- Add one frame or a small batch per round.
- Retrain or update under the same compute budget as baselines.

Baselines:

- Random selection.
- Uniform temporal sampling.
- Farthest-camera-distance sampling.
- Motion-threshold keyframe selection.
- Coverage-only selection.

Exit criteria:

- Plot RGB, depth, geometry, and uncertainty metrics against number of selected frames.
- Report a negative result if uncertainty-guided selection does not beat random or uniform sampling.

## Decision Gates

| Gate | Time | Decision |
|---|---|---|
| Splatfacto viability | End of baseline bring-up | Switch to depth-Nerfacto if Splatfacto cannot train/render reliably |
| Uncertainty extraction | After first three signals | Use post-hoc signals only if renderer internals are too expensive to modify |
| Replica stability | After 4 scenes | Drop keyframe selection if uncertainty metrics are noisy or inconclusive |
| Real-data setup | Before final experiments | Fall back from ScanNet++ to ScanNet or TUM RGB-D if setup blocks progress |
| Keyframe extension | After first full curve | Keep only if it improves quality per frame or yields a clear negative finding |

## Definition Of Done

Minimum serious project:

- Splatfacto or depth-Nerfacto trains on Replica.
- Fixed splits and frame budgets are committed.
- Held-out RGB and depth errors are computed.
- At least three uncertainty methods are compared quantitatively.
- Calibration and failure-detection metrics are reported.

Strong project:

- Multi-scene Replica evaluation.
- Real-world ScanNet++ or ScanNet validation.
- Stress tests for sparse views, pose noise, and missing depth.
- Optional keyframe selection with random, uniform, motion, and coverage baselines.
