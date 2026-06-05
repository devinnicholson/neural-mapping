# Literature Map

This map organizes the reading around the project claim: calibrated uncertainty in 3D Gaussian indoor maps should predict where held-out RGB, depth, or geometry will fail. The goal is to extract implementation constraints and evaluation expectations, not to build a broad survey.

## Reading Order

1. 3D Gaussian Splatting, Nerfstudio, Splatfacto, and gsplat.
2. Replica, ScanNet++, and ScanNet dataset papers.
3. SplaTAM, CG-SLAM, and VarSplat for Gaussian mapping and SLAM context.
4. Density-aware NeRF Ensembles, Bayes' Rays, and Neural Visibility Field for uncertainty signals.
5. Calibration and risk-ranking evaluation references for reliability, sparsification, and failure detection.

## Core Threads

| Thread | References | What To Extract | Role In This Project |
|---|---|---|---|
| 3D Gaussian representation | 3D Gaussian Splatting | Splat parameters, alpha compositing, visibility, rendering failure modes | Base representation for Splatfacto experiments |
| Practical training stack | Nerfstudio, Splatfacto docs, gsplat docs | Dataparsers, training commands, renderer outputs, depth rendering support | Avoids building a renderer or SLAM system from scratch |
| Synthetic indoor benchmark | Replica | Clean RGB-D, known geometry, controlled trajectories | First dataset and main controlled benchmark |
| Real indoor validation | ScanNet++, ScanNet | Real sensors, reconstruction artifacts, pose/depth caveats | Tests whether Replica conclusions survive realistic data |
| 3DGS mapping/SLAM | SplaTAM, CG-SLAM, VarSplat | How Gaussian maps behave in RGB-D mapping, tracking, and uncertainty-aware variants | Context and optional comparison, not the initial implementation target |
| Neural rendering uncertainty | Density-aware NeRF Ensembles, Bayes' Rays, Neural Visibility Field | View coverage, ensemble disagreement, visibility, density and ray-based uncertainty | Source of post-hoc uncertainty signals and baselines |
| Calibration and ranking | Reliability diagrams, risk-coverage, sparsification, AUSE, regression calibration | How to test whether uncertainty predicts error rather than only looking plausible | Core evaluation standard |

## Paper Notes And Project Actions

### 3D Gaussian Splatting

Reference:

- 3D Gaussian Splatting for Real-Time Radiance Field Rendering: https://arxiv.org/abs/2308.04079

Relevant ideas:

- Scene represented by anisotropic Gaussians with position, covariance, opacity, color, and spherical harmonics.
- Rendering depends on visibility ordering, alpha compositing, opacity accumulation, and view-dependent appearance.

Project actions:

- Use opacity accumulation, alpha, or transmittance as a confidence proxy.
- Evaluate failures near disocclusions, thin structure, reflective surfaces, and sparse coverage.
- Avoid claiming uncertainty is inherent unless the method outputs a calibrated distribution.

### Nerfstudio, Splatfacto, And gsplat

References:

- Nerfstudio: https://arxiv.org/abs/2302.04264
- Splatfacto docs: https://docs.nerf.studio/nerfology/methods/splat.html
- gsplat docs: https://docs.gsplat.studio/main/

Relevant ideas:

- Nerfstudio provides training, rendering, dataparser, eval, viewer, and export workflows.
- Splatfacto is the practical 3DGS path inside Nerfstudio.
- gsplat exposes efficient rasterization and renderer-level signals that may be useful for confidence.

Project actions:

- Treat Nerfstudio outputs as the canonical train/render/eval artifacts.
- Prefer renderer-exported alpha, depth, and accumulation fields over reimplementing rasterization.
- Keep depth-Nerfacto available as a fallback if Splatfacto depth or internal confidence extraction blocks progress.

### Replica

Reference:

- The Replica Dataset: https://arxiv.org/abs/1906.05797

Relevant ideas:

- High-quality synthetic indoor scenes with clean geometry and depth.
- Good for controlled sparse-view, pose-noise, depth-corruption, and extrapolation experiments.

Project actions:

- Start every new metric or uncertainty signal on one Replica scene.
- Use Replica for the main 25, 50, 100, and 200 frame budget study.
- Use Replica mesh/depth as the first geometry and depth targets.

### ScanNet++ And ScanNet

References:

- ScanNet++: https://arxiv.org/abs/2308.11417
- ScanNet: https://arxiv.org/abs/1702.04405

Relevant ideas:

- ScanNet++ provides high-fidelity real indoor data, but setup is heavier than Replica.
- ScanNet is older and easier to use as a real RGB-D fallback.

Project actions:

- Move to ScanNet++ only after Replica metrics and split logic are stable.
- Document data access, preprocessing friction, and target availability.
- Fall back to ScanNet or TUM RGB-D if ScanNet++ blocks final validation.

### SplaTAM

Reference:

- SplaTAM: https://arxiv.org/abs/2312.02126

Relevant ideas:

- Demonstrates RGB-D 3D Gaussian mapping and tracking.
- Useful for understanding online mapping constraints and Gaussian map update behavior.

Project actions:

- Use as a comparison point for what a full SLAM stack would require.
- Do not make SplaTAM a dependency for the first benchmark.
- Consider optional qualitative or metric comparison only after the offline Splatfacto protocol is complete.

### CG-SLAM

Reference:

- CG-SLAM: https://arxiv.org/abs/2403.16095

Relevant ideas:

- Relevant uncertainty-aware or confidence-aware Gaussian SLAM context.
- Helps frame why uncertainty should be evaluated against actual map failures.

Project actions:

- Extract uncertainty definitions, map update rules, and reported metrics.
- Compare their SLAM-oriented metrics with this project's offline failure-prediction metrics.
- Avoid claiming direct superiority unless the same data and protocol are run.

### VarSplat

Reference:

- VarSplat: https://arxiv.org/abs/2603.09673

Relevant ideas:

- Directly relevant recent uncertainty-aware 3DGS SLAM reference.
- Likely the strongest positioning risk for broad novelty claims.

Project actions:

- Use it to narrow the project claim toward calibrated failure prediction and benchmark quality.
- Compare uncertainty outputs conceptually against post-hoc coverage, transmittance, residual, and ensemble signals.
- Treat full method reproduction as optional.

### Density-Aware NeRF Ensembles

Reference:

- Density-aware NeRF Ensembles: https://arxiv.org/abs/2209.08718

Relevant ideas:

- Ensemble disagreement can estimate epistemic uncertainty.
- Density or occupancy information can prevent overconfidence in empty or under-observed regions.

Project actions:

- Implement lightweight ensemble disagreement only after single-model signals are stable.
- Compare ensemble gains against its extra compute cost.
- Use density/visibility ideas to interpret alpha and coverage baselines.

### Bayes' Rays

Reference:

- Bayes' Rays: https://arxiv.org/abs/2309.03185

Relevant ideas:

- Ray-level uncertainty can be estimated post hoc for neural rendering.
- Useful distinction between appearance uncertainty and geometry uncertainty.

Project actions:

- Use as motivation for ray-based uncertainty evaluation.
- Check whether uncertainty aligns better with RGB error, depth error, or both.
- Keep post-hoc uncertainty attractive because it avoids modifying the core renderer early.

### Neural Visibility Field

Reference:

- Neural Visibility Field: https://arxiv.org/abs/2406.06948

Relevant ideas:

- Visibility and observation support are strong cues for where novel-view predictions fail.
- View coverage can be a hard baseline to beat.

Project actions:

- Implement view coverage and nearest-camera-distance baselines early.
- Treat failure to beat coverage as a meaningful result, not a failed experiment.
- Use visibility concepts for optional keyframe scoring.

## Claim Boundaries

Supported claim if experiments succeed:

- Calibrated uncertainty signals for Splatfacto-style indoor Gaussian maps predict held-out RGB, depth, or geometry errors better than simple baselines under fixed splits.

Unsupported claim without extra work:

- The project is a complete uncertainty-aware SLAM system.
- The method is real-time.
- The method improves tracking.
- The uncertainty is Bayesian or probabilistic unless a proper predictive distribution and NLL evaluation are implemented.

## Evaluation Expectations From The Literature

Must include:

- Standard rendering metrics: PSNR, SSIM, LPIPS.
- Depth metrics: L1, RMSE, AbsRel, valid-mask reporting.
- Geometry metrics: Chamfer, accuracy, completeness, F-score.
- Uncertainty metrics: Spearman, AUROC, AUPRC, risk-coverage, sparsification, AUSE, reliability diagrams.

Must avoid:

- Showing uncertainty heatmaps without error correlation.
- Reporting only average rendering quality.
- Tuning thresholds on the test split.
- Comparing against complex uncertainty methods while omitting distance and coverage baselines.
