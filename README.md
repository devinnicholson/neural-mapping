# Error Prediction Is Not Acquisition

[![CI](https://github.com/devinnicholson/neural-mapping/actions/workflows/ci.yml/badge.svg)](https://github.com/devinnicholson/neural-mapping/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A controlled study of frame acquisition for sparse-view 3D Gaussian neural
mapping. The repository contains the full path from deterministic RGB-D splits
and GPU training to metric surface evaluation, frozen decision rules, compact
artifacts, and a workshop-style paper.

The central result is deliberately reported as mixed: a target-free rendered
depth-gradient proxy identifies difficult candidate views, and its hybrid
selector improves held-out appearance over random on average, but it does not
improve metric surface reconstruction or beat direct coverage controls.

## Confirmatory result

The completed study expands a shared 25-frame seed to 50 frames on each of four
ICL-NUIM living-room trajectories. Active and random methods are paired by
trajectory, optimization seed, held-out frames, training schedule, and NVIDIA
L4 hardware.

| Outcome | Mean active − random | Favorable pairs | Favorable trajectories | Trajectory-cluster 95% interval |
|---|---:|---:|---:|---:|
| PSNR ↑ | +0.6372 dB | 5/8 | 3/4 | [-0.0231, +1.4764] |
| LPIPS ↓ | -0.0231 | 6/8 | 2/4 | [-0.0546, +0.0084] |
| Raw depth AbsRel ↓ | -0.0047 | 5/8 | 2/4 | [-0.0653, +0.0531] |
| Surface F-score at 5 cm ↑ | -0.0016 | 5/8 | 2/4 | [-0.0459, +0.0368] |

The depth-gradient diagnostic passes independently: Spearman correlation is
positive in 8/8 seed models and AUROC exceeds 0.5 in 8/8 (mean AUROC 0.716).
The preregistered primary, replication, coverage-control superiority, and
overall-support gates fail. The LPIPS safety and error-ranking gates pass.

This supports one narrow conclusion: candidate-error discrimination is not a
sufficient surrogate for geometry-improving acquisition. It does not establish
state-of-the-art next-best-view planning, calibrated epistemic uncertainty,
online SLAM, or transfer to physical scenes.

Read the [study card](docs/icl_nuim_multitrajectory_v1.md), the
[frozen protocol](experiments/protocols/icl_nuim_multitrajectory_v1.json), the
[audited record](experiments/records/icl_nuim_multitrajectory_v1.json), or the
[paper source](paper/main.tex).

## Research question

Given a trained seed map, known candidate camera poses, and a fixed acquisition
budget, does renderer state identify observations whose addition improves both
novel-view appearance and world-space surface geometry beyond inexpensive
random and coverage rules?

For rendered expected depth $z_v$ at candidate view $v$, the frozen signal is
the forward finite-difference magnitude

$$
u_v(i,j)=\sqrt{|z_v(i,j)-z_v(i-1,j)|^2+|z_v(i,j)-z_v(i,j-1)|^2}.
$$

The view score $U(v)$ is the mean of its highest-scoring pixel decile. At each
greedy step, min–max-normalized signal and nearest selected-camera distance are
combined as

$$
A(v\mid S)=0.35\,\widetilde U(v)+0.65\,\widetilde D(v,S).
$$

Candidate RGB and reference depth never enter selection. They are used only
afterward to evaluate error ranking and reconstruction.

## Experimental design

| Component | Frozen specification |
|---|---|
| Dataset | Four clean ICL-NUIM living-room trajectories, uniformly sampled to 240 observations each |
| Acquisition | Shared 25-frame seed; 50-frame target budget |
| Holdout | 10 validation and contiguous 20-frame test blocks with five-sample guard bands where available |
| Primary comparison | Active vs matched random, seeds 42 and 43 on every trajectory |
| Controls | Random, acquisition-time farthest-first, and camera-center farthest-first; seed 42 |
| Training | Nerfstudio Splatfacto, 7,000 iterations, camera optimization off, NVIDIA L4 |
| Coordinate frame | Published metric poses; orientation, centering, and auto-scaling disabled |
| Co-primary outcomes | Held-out PSNR and fused-surface F-score at 5 cm |
| Secondary outcomes | SSIM, LPIPS, raw/aligned depth, accuracy, completeness, Chamfer-L1, precision/recall/F-score at 5 and 10 cm |
| Experimental unit | Trajectory-by-optimization-seed matched pair; pixels are not treated as independent replicates |

Every budget-50 split contains exactly 50 train, 10 validation, and 20 test
filenames. The record builder verifies partition disjointness, common holdouts,
retention of the 25-frame seed, source hashes, paired deltas, aggregate
statistics, and all decision gates.

## Reproduce the record

The CPU audit path does not require a GPU:

```bash
git clone https://github.com/devinnicholson/neural-mapping.git
cd neural-mapping
uv sync --all-extras --locked
uv run pytest -q

uv run python scripts/build_icl_benchmark_record.py \
  --artifact-root experiments/artifacts/icl_nuim_multitrajectory_v1 \
  --protocol experiments/protocols/icl_nuim_multitrajectory_v1.json \
  --run-manifest experiments/run_manifests/icl_nuim_multitrajectory_v1.json \
  --output experiments/records/icl_nuim_multitrajectory_v1.json

uv run python scripts/generate_icl_report_assets.py
git diff --exit-code -- experiments/records paper/tables experiments/tables docs/icl_nuim_multitrajectory_v1.md
```

The committed evidence package is about 1 MB and contains:

- 72 final RGB, depth, and geometry metric files;
- eight compact per-seed diagnostic reports, each linked by SHA-256 to its
  pixel-heavy source report;
- 24 exact split-membership manifests;
- the preregistration and dated pre-outcome amendment;
- Modal run provenance and deterministic generated tables.

Large datasets and model checkpoints remain outside Git.

## Run GPU experiments

Local tests, record verification, and table generation are CPU-only. Training
and rendered evaluation require a CUDA GPU. The reference environment is
defined in [modal_app.py](modal_app.py) and pins Python 3.12, CUDA 12.8.1,
PyTorch 2.11.0, Nerfstudio 1.1.5, gsplat 1.4.0, and SciPy 1.15.3.

```bash
modal setup
modal run modal_app.py --action env
modal run modal_app.py --action prepare-icl \
  --icl-trajectory lr_kt0 \
  --data-scene-name icl_lr_kt0_clean_v1
```

Matrix actions in `modal_app.py` prepare controls, train paired models, render
candidate diagnostics, materialize active splits, and evaluate RGB/depth/surface
outcomes. Use a new protocol and new scene names for any follow-up study; do not
overwrite or tune the completed confirmatory family.

## Repository map

```text
.
├── experiments/
│   ├── artifacts/       # compact source metrics, diagnostics, split manifests
│   ├── protocols/       # frozen hypotheses, methods, gates, amendments
│   ├── records/         # recomputable claim-bearing JSON records
│   ├── run_manifests/   # execution and infrastructure provenance
│   └── tables/          # generated machine-readable result tables
├── paper/               # workshop paper and generated LaTeX tables
├── scripts/             # splits, evaluation, auditing, artifact generation
├── src/uncertainty_3dgs/# CPU-testable selection and metric primitives
├── tests/               # math, split, metric, provenance, and record tests
├── modal_app.py         # pinned GPU orchestration
├── uv.lock              # resolved CPU/repository dependencies
└── ROADMAP.md           # claim-oriented next studies
```

## Paper

Generated numerical tables are derived from the audited JSON record; they are
not hand-maintained.

```bash
uv run python scripts/generate_icl_report_assets.py
make paper
```

The paper is framed around the image–geometry mismatch and reports the negative
primary outcome, stronger coverage baselines, every seed, descriptive clustered
intervals, limitations, and the pre-outcome correction.

## Standards for new claims

A new headline claim must include:

- a protocol committed before claim-bearing training;
- a public dataset or releasable fresh-scene data;
- locked train/validation/test membership with leakage checks;
- matched seeds, budgets, hardware class, and optimization schedules;
- random and domain-relevant coverage or literature baselines;
- appearance, depth, calibration/selection, and metric geometry outcomes;
- multi-scene or multi-trajectory replication with uncertainty intervals;
- complete negative results, runtime/resource provenance, and artifact hashes.

The next confirmatory family should test a decision-aware signal on distinct
physical environments and compare directly with an information-theoretic active
view method. It must use a new preregistration; the completed ICL-NUIM result is
immutable.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Code is released under the
[MIT License](LICENSE). ICL-NUIM data is distributed separately under CC BY
3.0; this repository commits only filenames, transforms-derived memberships,
and evaluation outputs, not the dataset images.
