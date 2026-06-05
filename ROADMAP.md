# Roadmap: Calibrated Uncertainty for 3D Gaussian Neural Maps

Created: 2026-05-29

## Project Thesis

This project investigates whether uncertainty estimates from 3D Gaussian scene representations can predict where neural maps fail under partial indoor observations.

The core research question:

> Can uncertainty maps from a 3D Gaussian scene representation predict where novel-view RGB, depth, or geometry will fail?

The extension question:

> Can calibrated uncertainty select better next frames or keyframes than random, uniform, or simple motion-based sampling?

The goal is not to build a full robot SLAM stack from scratch. The goal is a rigorous, reproducible research project with a measurable claim.

## Target Contribution

The strongest final contribution is:

> A calibrated failure-prediction benchmark and method for 3D Gaussian indoor maps, with an uncertainty-guided keyframe-selection extension.

This should demonstrate:

- 3D Gaussian neural mapping competence.
- Serious uncertainty and calibration evaluation.
- RGB, depth, and geometry failure prediction.
- Controlled synthetic results and real-world validation.
- A clear distinction between uncertainty signals that look plausible and uncertainty signals that actually predict error.

## Non-Goals

Avoid these until the core project works:

- Building full real-time robotic exploration.
- Implementing SLAM, loop closure, and planning from scratch.
- Claiming novelty as "uncertainty-aware 3DGS SLAM" broadly.
- Reporting only pretty renderings or PSNR/SSIM/LPIPS.
- Treating heatmaps as meaningful without calibration or error correlation.

## Recommended Stack

Primary implementation stack:

- Nerfstudio from source.
- Splatfacto as the main 3D Gaussian method.
- gsplat as the rasterization backend.
- Nerfacto and depth-Nerfacto as baselines.

Why this stack:

- Nerfstudio provides training, rendering, eval, viewer, export, and modular method/dataparser support.
- Splatfacto is a practical 3D Gaussian baseline inside Nerfstudio.
- gsplat exposes depth rendering, alpha/transmittance behavior, feature rendering, and efficient differentiable rasterization.
- This avoids starting from brittle full-SLAM code.

Stretch/comparison systems:

- SplaTAM for actual RGB-D 3DGS SLAM comparison.
- VarSplat as a recent uncertainty-aware 3DGS SLAM reference.
- CG-SLAM as an uncertainty-aware Gaussian SLAM reference.

Fallback stack:

- Nerfstudio + depth-Nerfacto + ray/depth uncertainty.
- Use this if Splatfacto/gsplat integration becomes the bottleneck.

Hardware expectation:

- Minimum: NVIDIA CUDA GPU with 12 GB VRAM for small serious experiments.
- Preferred: 24 GB VRAM for larger scenes, ensembles, and multi-baseline experiments.

## Datasets

Primary datasets:

1. Replica
   - Use for controlled synthetic indoor experiments.
   - Strengths: clean geometry, depth, photorealistic rooms, semantics, controlled trajectories.
   - Role: main debugging and controlled uncertainty/error evaluation.

2. ScanNet++
   - Use for real-world validation if setup time allows.
   - Strengths: high-fidelity real indoor scenes, laser scans, DSLR imagery, iPhone RGB-D streams.
   - Role: show the method survives realistic data.

Fallback datasets:

- ScanNet for real RGB-D validation if ScanNet++ is too heavy.
- TUM RGB-D for small SLAM-style trajectory checks.
- ICL-NUIM as an older synthetic RGB-D fallback.
- Bonn RGB-D Dynamic as a dynamic-scene stress test, not a primary benchmark.

Avoid as primary datasets:

- LLFF and Mip-NeRF 360, because they are strong neural rendering datasets but weak for robotics-style depth and geometry claims.
- Tanks and Temples, because it is useful for reconstruction but less aligned with indoor RGB-D neural mapping.

## Evaluation Metrics

Rendering quality:

- PSNR.
- SSIM.
- LPIPS.

Depth quality:

- L1 depth error.
- RMSE.
- AbsRel.
- Valid-pixel masked metrics.

Geometry quality:

- Chamfer distance.
- Accuracy.
- Completeness.
- F-score at 5 cm and 10 cm.
- Normal consistency if reliable normals are available.

Uncertainty quality:

- Spearman correlation between uncertainty and actual error.
- AUROC/AUPRC for bad-pixel or bad-patch detection.
- Risk-coverage curves.
- Sparsification curves and AUSE.
- Reliability diagrams.
- Calibration error for regression-style uncertainty.
- Negative log likelihood if the method outputs a proper predictive distribution.

Tracking quality, only if running a live/semi-live SLAM system:

- ATE RMSE.
- RPE.

## Uncertainty Signals To Compare

Start simple and measurable:

1. View coverage
   - Count how many training cameras observe each point/ray/region.
   - Strong baseline because missing observations are a major source of failure.

2. Alpha/transmittance confidence
   - Use accumulated opacity or ray termination behavior as a confidence proxy.
   - Low opacity or diffuse termination can indicate uncertain geometry.

3. Photometric residual
   - Use training-view residuals projected into candidate views or local regions.
   - Can catch inconsistent appearance, reflections, and dynamic artifacts.

4. Depth disagreement
   - Compare rendered depth against input depth or nearby-view projected depth.
   - Useful for geometry failure prediction.

5. Lightweight ensembles
   - Train several smaller models or vary initialization/subsets.
   - Use disagreement in RGB/depth/rendered features.

6. Per-splat variance
   - Longer-term method inspired by uncertainty-aware 3DGS work.
   - Useful if there is time to modify splat parameters and rendering outputs.

Baselines to beat:

- Uniform uncertainty.
- Distance from nearest training camera.
- Number of visible training views.
- Image-space depth edge heuristic.
- Training photometric residual alone.

## Experimental Protocol

For each scene:

1. Build a fixed frame pool from the camera trajectory.
2. Select training frame budgets: 25, 50, 100, 200.
3. Split frames into train, validation/calibration, and test.
4. Train Splatfacto on the selected frames.
5. Render held-out RGB and depth.
6. Generate uncertainty maps.
7. Compute actual RGB/depth error maps.
8. Evaluate uncertainty-error alignment.
9. Repeat under stress tests.
10. Compare against baselines.

Recommended initial scene count:

- Phase 1: 1 Replica scene for pipeline bring-up.
- Phase 2: 4 Replica scenes for controlled experiments.
- Phase 3: 6-8 Replica scenes for final synthetic results.
- Phase 4: 3-6 ScanNet++ or ScanNet scenes for real validation.

## Stress Tests

Run these first on Replica, then repeat the most informative ones on real data:

- Sparse-view budgets: 25, 50, 100, 200 frames.
- Extrapolation: hold out a side or region of a room.
- Pose noise: perturb training poses by 1-5 cm and 1-5 degrees.
- Depth corruption: dropout, Gaussian noise, missing-depth regions.
- Appearance corruption: blur, exposure shifts, low light.
- Dynamic occlusion: injected moving masks or Bonn RGB-D Dynamic.
- Surface categories: textureless walls, depth edges, reflective surfaces, glass, clutter.

## Active Keyframe Selection Extension

Only start this after uncertainty evaluation works.

Setup:

1. Begin with a small initial training set.
2. Maintain a candidate pool of future frames.
3. Score candidate frames by predicted uncertainty reduction or expected information gain.
4. Add the top-scoring frame.
5. Retrain or incrementally update.
6. Compare quality per frame budget.

Baselines:

- Random frame selection.
- Uniform temporal sampling.
- Farthest-camera-distance sampling.
- Motion-threshold keyframe selection.
- Coverage-only selection.

Expected claim:

> Uncertainty-guided keyframe selection improves RGB/depth/geometry quality per added frame compared with random and uniform sampling.

## 12-Week Plan

### Week 1: Setup And Literature Lock

Tasks:

- Install Nerfstudio from source on Linux/CUDA.
- Run one Nerfstudio example successfully.
- Read and summarize the core papers.
- Create a notes file with the exact gap and project claim.

Key papers:

- 3D Gaussian Splatting.
- Nerfstudio.
- SplaTAM.
- CG-SLAM.
- VarSplat.
- Density-aware NeRF Ensembles.
- Bayes' Rays.
- Neural Visibility Field.

Exit criteria:

- Environment runs.
- Splatfacto trains on a toy/example dataset.
- One-page problem statement exists.

### Week 2: Baseline Scene Pipeline

Tasks:

- Download or prepare one Replica scene.
- Convert data into a Nerfstudio-compatible format.
- Train Splatfacto on a fixed frame subset.
- Render held-out views.
- Run PSNR/SSIM/LPIPS evaluation.

Exit criteria:

- One full train/render/eval cycle works.
- Results are saved reproducibly.

### Week 3: Depth And Error Maps

Tasks:

- Render depth for held-out views.
- Load ground-truth or dataset depth.
- Compute RGB error maps.
- Compute depth error maps.
- Save per-frame visualizations.

Exit criteria:

- For each held-out view, there is RGB, depth, RGB error, and depth error.

### Week 4: First Uncertainty Baselines

Tasks:

- Implement view coverage uncertainty.
- Implement alpha/transmittance confidence.
- Implement distance-to-training-camera baseline.
- Compare each uncertainty map against actual error.

Exit criteria:

- Spearman correlation and AUROC are computed for at least three uncertainty signals.
- First table of uncertainty vs error exists.

### Week 5: Better Uncertainty Signals

Tasks:

- Add photometric residual uncertainty.
- Add depth disagreement uncertainty.
- Add patch-level uncertainty aggregation.
- Normalize/calibrate uncertainty scores on validation frames.

Exit criteria:

- At least five uncertainty signals are compared on one Replica scene.
- Calibration split is separate from test split.

### Week 6: Multi-Scene Replica Study

Tasks:

- Run 4 Replica scenes.
- Run frame budgets: 25, 50, 100, 200.
- Aggregate metrics across scenes.
- Identify which uncertainty signals are robust.

Exit criteria:

- Multi-scene table exists.
- Early plots show uncertainty-error alignment or failure.

### Week 7: Calibration And Reliability

Tasks:

- Add reliability diagrams.
- Add risk-coverage curves.
- Add sparsification curves/AUSE.
- Evaluate pixel-level and patch-level failure detection.

Exit criteria:

- Calibration section is backed by quantitative plots.
- There is a clear winner or failure analysis across uncertainty methods.

### Week 8: Stress Tests

Tasks:

- Run sparse-view stress tests.
- Run pose-noise stress tests.
- Run missing-depth or appearance-corruption stress tests.
- Evaluate surface-category failure if labels are available.

Exit criteria:

- Results show when uncertainty works and when it fails.
- The project has more than standard rendering metrics.

### Week 9: Real-World Validation

Tasks:

- Prepare ScanNet++ or ScanNet scenes.
- Run the best uncertainty methods from Replica.
- Compare synthetic vs real behavior.
- Document setup friction and data caveats.

Exit criteria:

- At least 3 real scenes evaluated, or a clear documented fallback to ScanNet/TUM.

### Week 10: Active Keyframe Selection

Tasks:

- Define candidate-frame selection experiment.
- Implement random, uniform, and coverage baselines.
- Implement uncertainty-guided selection.
- Compare reconstruction quality per frame budget.

Exit criteria:

- One scene shows a full keyframe-selection curve.

### Week 11: Final Experiments And Ablations

Tasks:

- Run final selected experiments.
- Add ablations for uncertainty components.
- Remove weak or unsupported claims.
- Generate final plots and qualitative figures.

Exit criteria:

- Final results tables are stable.
- Claims are tied directly to metrics.

### Week 12: Report, Demo, And Cleanup

Tasks:

- Write a 6-8 page paper-style report.
- Prepare a short demo video or viewer export.
- Clean scripts and configs.
- Write reproducibility instructions.

Exit criteria:

- Final report exists.
- Repository can reproduce main tables.
- Demo artifacts are present.

## Decision Gates

Gate 1, end of Week 2:

- If Splatfacto cannot run reliably, switch to depth-Nerfacto.

Gate 2, end of Week 4:

- If uncertainty maps cannot be extracted cleanly from the stack, use post-hoc uncertainty signals only.

Gate 3, end of Week 6:

- If results are too noisy on Replica, narrow to failure prediction and drop keyframe selection.

Gate 4, end of Week 9:

- If ScanNet++ setup is too heavy, validate on ScanNet or TUM RGB-D instead.

Gate 5, end of Week 10:

- If keyframe selection does not improve quality, report it honestly as a negative result and focus on calibrated failure prediction.

## Final Deliverables

Required:

- Reproducible training/evaluation pipeline.
- Fixed dataset splits.
- Quantitative uncertainty-error evaluation.
- RGB, depth, and uncertainty visualizations.
- Stress-test results.
- Paper-style final report.

Strong optional deliverables:

- Uncertainty-guided keyframe selection experiment.
- Web viewer or demo video.
- Comparison with SplaTAM or VarSplat outputs.
- Clean benchmark scripts others could reuse.

## Success Criteria

Minimum serious project:

- Splatfacto or depth-Nerfacto trained on Replica.
- Held-out RGB/depth error computed.
- At least three uncertainty methods compared quantitatively.
- Calibration and failure detection metrics reported.
- Clear failure analysis.

Strong project:

- Multi-scene Replica evaluation.
- Real-world ScanNet++ or ScanNet validation.
- Stress tests with pose noise and sparse views.
- Uncertainty-guided keyframe selection beats random/uniform baselines.

Outstanding project:

- Calibrated uncertainty predicts depth and geometry failure, not just RGB error.
- Method generalizes from Replica to real indoor data.
- Keyframe selection improves reconstruction quality per frame budget.
- Report is close to workshop-paper quality.

## Risk Matrix

| Risk | Severity | Mitigation |
|---|---:|---|
| Full SLAM code is brittle | High | Start with offline posed RGB-D/RGB mapping |
| Novelty is crowded | High | Frame around calibrated failure prediction and benchmark quality |
| Dataset setup takes too long | Medium | Start with Replica, fall back from ScanNet++ to ScanNet/TUM |
| Uncertainty does not beat simple baselines | Medium | Treat this as a real negative finding if evaluation is rigorous |
| Compute gets expensive | Medium | Use small scenes, reduced resolution, limited ensembles |
| Keyframe selection balloons scope | Medium | Make it optional after core uncertainty results |
| Results are noisy | High | Lock splits and metrics early, aggregate over scenes |

## Reference List

- 3D Gaussian Splatting: https://arxiv.org/abs/2308.04079
- Nerfstudio: https://arxiv.org/abs/2302.04264
- Splatfacto docs: https://docs.nerf.studio/nerfology/methods/splat.html
- gsplat docs: https://docs.gsplat.studio/main/
- SplaTAM: https://arxiv.org/abs/2312.02126
- CG-SLAM: https://arxiv.org/abs/2403.16095
- VarSplat: https://arxiv.org/abs/2603.09673
- Density-aware NeRF Ensembles: https://arxiv.org/abs/2209.08718
- Bayes' Rays: https://arxiv.org/abs/2309.03185
- Neural Visibility Field: https://arxiv.org/abs/2406.06948
- Replica: https://arxiv.org/abs/1906.05797
- ScanNet: https://arxiv.org/abs/1702.04405
- ScanNet++: https://arxiv.org/abs/2308.11417

