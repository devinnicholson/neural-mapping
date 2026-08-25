# Uncertainty-Aware 3D Gaussian Neural Mapping

Research code for uncertainty-guided frame selection in sparse-view 3D Gaussian
scene reconstruction.

> **Status:** active research prototype. The repository contains a working
> lightweight evaluation harness, a preregistered RGB-D robustness study, and
> repeated pilot results. The current evidence supports a narrow within-sequence
> claim; cross-dataset evaluation and external reproduction are still required
> before this should be interpreted as a submission-ready result.

## Research Question

Given an initial set of posed RGB or RGB-D observations and a pool of candidate
camera views, can model-derived uncertainty identify which additional frames
will most improve a 3D Gaussian scene representation?

The current hypothesis is deliberately narrower:

> At compact frame budgets, upper-tail model uncertainty is a useful acquisition
> signal when it is constrained by camera-pose diversity. Uncertainty-only
> selection is less robust because it can concentrate on redundant or
> systematically difficult views.

This repository studies **offline, pool-based frame selection**. It does not
claim to implement online SLAM, autonomous exploration, real-time next-best-view
planning, or a universally calibrated probabilistic model.

## Method Overview

The current acquisition loop is:

1. Construct a deterministic seed split with fixed validation and test frames.
2. Train one or more Splatfacto seed models on the initial frame budget.
3. Render the remaining candidate poses and compute per-pixel uncertainty.
4. Aggregate each candidate by upper-tail risk, currently the mean uncertainty
   in its highest-uncertainty pixel decile.
5. Greedily select additional frames using uncertainty and pose novelty.
6. Retrain under the same optimization budget and evaluate on the untouched
   test set.

For candidate view $v$, selected frame set $S$, normalized tail-risk score
$\widetilde{U}(v)$, and normalized distance to the nearest selected camera
$\widetilde{D}(v,S)$, the hybrid acquisition score is

$$
A(v \mid S) = \lambda\widetilde{U}(v) + (1-\lambda)\widetilde{D}(v,S).
$$

The experiments compare this rule with random selection, trajectory coverage,
pose coverage, uncertainty-only selection, and renderer-derived uncertainty
signals. Final claims will use one policy frozen on development scenes and
evaluated unchanged on held-out scenes.

## Current Evidence

The results below are pilot evidence. They establish that the pipeline works and
motivate the frozen benchmark; they do not yet establish a general acquisition
policy.

| Study | Current result | Interpretation |
|---|---|---|
| Nerfstudio sample scenes | Across twelve `dozer`, `redwoods2`, and `bww_entrance` ensemble-tail seeds, the hybrid selector improved the same-seed random budget-50 baseline by approximately **+1.165 PSNR**, **+0.026 SSIM**, and **-0.017 LPIPS** on average. | Strongest repeated RGB pilot. |
| `library` and `kitchen` | `library` improved on average; `kitchen` contained a substantial regression on one seed despite positive average PSNR and LPIPS. | Evidence of transfer, but not uniform robustness. |
| TUM RGB-D `freiburg1_xyz` v1-v6 | Depth-gradient selection improved the random budget-50 baseline by approximately **+0.264 PSNR**, **+0.007 SSIM**, **-0.003 LPIPS**, and **-0.005 median-aligned AbsRel** on average. | Best current fixed-policy RGB-D result. |
| TUM RGB-D budget sweep | The hybrid policy helped at intermediate budgets on `freiburg1_xyz` v9, with the cleanest RGB/depth result at budget 100, but regressed at the saturated budget of 150. | Supports compact subset selection rather than full-trajectory ordering. |
| RGB-D transfer checks | `freiburg1_desk` v4 transferred positively at budgets 50 and 100; `freiburg1_room` v4 was mixed at budget 50 and negative at budget 100. | The current policy does not transfer reliably to every scene regime. |
| TUM RGB-D `freiburg2_desk` preregistered robustness study | Across three locked interleaved splits, active b50 improves matched random b50 by **+0.441 PSNR**, **+0.0222 SSIM**, **-0.0125 LPIPS**, and **-0.0108 raw AbsRel** on average. A separate temporal-block holdout improves by **+0.297 PSNR**, **-0.0437 LPIPS**, and **-0.120 raw AbsRel**. | Clears the prespecified interleaved and blocked gates. Median-aligned AbsRel regresses on all three interleaved splits, so the supported depth claim is limited to raw metric behavior. |

Full per-scene tables, negative results, run identifiers, and interpretation are
maintained in [docs/results.md](docs/results.md). The corresponding run commands
and artifact locations are recorded in
[docs/run-manifest.md](docs/run-manifest.md). A compact visual summary is
available in [docs/dashboard.html](docs/dashboard.html). New experiments also
store compact machine-readable records under [experiments/records](experiments/records).

## Claim Boundary

The evidence currently supports the following statement:

> Ensemble disagreement and renderer-derived signals can identify useful
> candidate views, and a tail-risk/pose-diversity hybrid can improve reconstruction
> at compact frame budgets on repeated scene splits. On TUM RGB-D
> `freiburg2_desk`, one policy fixed before the confirmatory runs improves RGB
> and raw depth outcomes across independent splits and a guarded temporal-block
> holdout.

It does **not** yet support these stronger statements:

- one fixed selector generalizes across synthetic and real RGB-D datasets;
- the uncertainty estimates are probabilistically calibrated;
- the method improves complete-scene geometry or tracking;
- the improvement remains after matching ensemble and baseline compute;
- the method is state of the art against established active-view systems.

These boundaries are part of the evaluation protocol rather than post-hoc
disclaimers. Failed signals, regressions, and saturation effects remain in the
reported results.

## Evaluation Protocol

The benchmark separates model development from final evaluation:

- Validation data may tune normalization, thresholds, calibration, and the
  acquisition weight.
- Test frames are reserved before model or policy inspection and are used only
  for final reporting.
- Development scenes determine the final signal and selector.
- Final scenes evaluate that frozen policy without scene-specific adaptation.
- All comparisons use the same train/test split, optimization schedule, image
  resolution, and evaluation implementation.

Primary outcome families are:

- **Rendering:** PSNR, SSIM, and LPIPS.
- **Depth:** RMSE, AbsRel, and valid-pixel threshold accuracy.
- **Geometry:** accuracy, completeness, Chamfer distance, and F-score when a
  reference surface is available.
- **Failure prediction:** Spearman correlation, AUROC, AUPRC, risk-coverage,
  sparsification, AUSE, and reliability analysis.
- **Efficiency:** selected-frame budget, GPU-hours, wall time, peak memory, and
  scoring overhead.

The complete protocol is in
[docs/experiment-protocol.md](docs/experiment-protocol.md).

## Roadmap to a Submission-Grade Result

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 1. Protocol lock | Fixed development/test scene partition, information boundary, primary metrics, selector, and hyperparameters. | No final-test observation can change the acquisition rule. |
| 2. Artifact provenance | Committed split manifests and machine-readable run records containing code, data, environment, seed, hardware, and artifact identifiers. | Every reported number traces to a reproducible run record. |
| 3. Controlled benchmark | Multi-scene Replica evaluation followed by a frozen real RGB-D transfer study. | At least 6 controlled scenes, 3 real scenes, and repeated seeds for the primary comparison. |
| 4. Statistical and compute audit | Paired confidence intervals, scene-level effects, ablations, and compute-matched baselines. | The main conclusion survives uncertainty estimates and fair resource accounting. |
| 5. Research release | Generated figures and tables, paper, limitations, public artifacts, and clean-environment reproduction. | An independent user can reproduce at least one principal result from documented inputs. |

The detailed implementation sequence remains in
[ROADMAP.md](ROADMAP.md) and [docs/implementation-plan.md](docs/implementation-plan.md).

## Repository Structure

```text
configs/                  experiment, dataset, uncertainty, and stress-test definitions
data/                     local raw/processed data and split manifests
docs/                     protocol, results, runbooks, literature, and project decisions
examples/                 small dependency-light inputs for smoke tests
outputs/                  generated reports and run artifacts
scripts/                  command-line experiment and evaluation utilities
src/uncertainty_3dgs/     reusable split and uncertainty-metric implementations
tests/                    dependency-light unit and command-line tests
modal_app.py              GPU workflow for Nerfstudio/Splatfacto experiments
```

Heavy training dependencies remain isolated from the lightweight utilities so
split generation, metric validation, and result summarization can run without a
CUDA environment.

## Quick Start

The lightweight package requires Python 3.10 or newer.

```bash
python -m pip install -e ".[dev,numeric]"
make test
make smoke
```

`make test` runs the dependency-light test suite. `make smoke` generates a small
deterministic split and computes uncertainty/error alignment metrics from the
checked-in examples.

Generate a deterministic split manifest:

```bash
python scripts/generate_splits.py \
  --frames examples/frames.txt \
  --budgets 4 6 \
  --val-count 2 \
  --test-count 2 \
  --scene example \
  --seed 7 \
  --selection-method random \
  --output data/splits/example_split.json
```

Compute uncertainty/error metrics:

```bash
python scripts/compute_uncertainty_metrics.py \
  --input examples/metric_input.json \
  --bad-threshold 0.5 \
  --reliability-bins 10 \
  --output outputs/reports/example_metrics.json
```

Summarize active-versus-random results from saved metric rows:

```bash
python scripts/summarize_active_metrics.py \
  --input outputs/modal_metrics.log \
  --pairs-file configs/active_metric_pairs.json \
  --format markdown
```

## GPU Workflows

Training and rendering use Nerfstudio Splatfacto with gsplat on Linux and NVIDIA
CUDA. They are intentionally kept outside the base package.

- [GPU baseline bring-up](docs/gpu-baseline-bringup.md)
- [Modal workflow](docs/modal.md)
- [SLURM workflow](docs/cluster-slurm.md)
- [RGB-D validation runbook](docs/next-rgbd-validation.md)

The full workflow is:

```text
acquire data -> validate frames and poses -> freeze split -> materialize dataset
-> train seed model(s) -> render candidate uncertainty -> select frames
-> retrain matched baselines -> evaluate untouched test views -> aggregate results
```

## Reproducibility Status

The repository currently provides deterministic split generation, lightweight
metric implementations, unit tests, configuration templates, runbooks, and a
manifest linking headline claims to stored GPU runs.

The following items remain required for a complete external reproduction:

- a frozen dependency lock and GPU image digest;
- committed split manifests for every headline experiment;
- public resolved run configurations and large evaluation artifacts;
- checksummed model, render, and evaluation artifacts;
- one command that regenerates every reported table and figure;
- a clean-machine reproduction report.

Until those artifacts are released, the checked-in results should be treated as
documented internal experiments rather than independently reproduced findings.

## Documentation

- [Literature map](docs/literature-map.md)
- [Experiment protocol](docs/experiment-protocol.md)
- [Results and negative findings](docs/results.md)
- [Run manifest](docs/run-manifest.md)
- [Implementation plan](docs/implementation-plan.md)
- [Project roadmap](ROADMAP.md)
