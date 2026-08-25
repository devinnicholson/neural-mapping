# Research Roadmap

Updated: 2026-08-25

## Current evidence boundary

The first cross-trajectory confirmatory study is complete. A frozen
depth-gradient/pose selector predicts difficult candidate views and improves
average held-out appearance over matched random acquisition, but it does not
improve 5 cm surface F-score, meet the replication rule, or outperform direct
pose/temporal coverage controls. The result is immutable and should not be
retuned.

The next research question is therefore not “which weight makes the existing
table positive?” It is:

> Which target-free acquisition objective estimates reducible geometric value,
> beyond difficulty and camera coverage, and transfers to distinct physical
> environments?

## Claim-bearing milestones

### 1. Decision-aware acquisition baseline

- Implement one information-theoretic or optimal-design baseline from the
  active radiance-field/3DGS literature.
- Separate candidate difficulty, expected error reduction, visibility, and
  motion cost in the API and evaluation.
- Test the implementation against analytic toy scenes and finite-difference
  checks before GPU experiments.
- Freeze a new protocol; do not reuse ICL-NUIM as a tuning and test set.

Exit criterion: a CPU-testable scorer, a documented derivation, and a
reproduction of one published qualitative or quantitative behavior.

### 2. Multi-scene controlled benchmark

- Use at least four geometrically distinct synthetic indoor scenes with public
  depth and mesh ground truth.
- Use locked region/trajectory holdouts with guard bands and publish exact split
  membership.
- Compare random, temporal coverage, pose coverage, the completed depth-gradient
  heuristic, and the literature baseline at multiple budgets.
- Run at least three optimization seeds for claim-bearing comparisons.
- Evaluate RGB, raw metric depth, official-mesh geometry, uncertainty/error
  ranking, and resource cost.

Exit criterion: all preregistered runs complete, with scene-cluster intervals,
negative results, resource ledgers, and one-command record regeneration.

### 3. Real-scene replication

- Select a public RGB-D dataset with accurate registered geometry and licensing
  compatible with released split manifests.
- Lock at least three distinct physical environments before outcome inspection.
- Add pose-noise and missing-depth stress tests without changing the primary
  protocol.
- Report tracking assumptions explicitly; do not imply online SLAM if poses are
  supplied.

Exit criterion: the direction of both co-primary effects replicates on physical
scenes, or the paper is revised around the observed failure.

### 4. Representation breadth

- Reproduce the same acquisition protocol with an implicit radiance field and
  a point/Gaussian representation.
- Compare quality, memory, training time, selection latency, and sensitivity to
  pose/depth calibration.
- Derive which acquisition signals are representation-specific and which
  transfer.

Exit criterion: a controlled representation table with identical information
boundaries and holdouts.

### 5. Paper and release

- Replace reconstructed held-out surfaces with official-mesh evaluation where
  available.
- Include representative false positives and selection trajectories.
- Obtain an independent reproduction from a clean checkout and separate Modal
  namespace.
- Build the PDF in CI, archive the exact paper record, and mint a release DOI.

Exit criterion: workshop-ready PDF, anonymous artifact bundle, release tag,
archived environment, and independent reproduction report.

## Engineering requirements

Every new GPU runner must persist:

- protocol and code commit;
- dataset/split digest and information boundary;
- method, seed, budget, resolved configuration, and checkpoint step;
- GPU type, start/end timestamps, wall time, and retry lineage;
- RGB, depth, geometry, calibration/selection, latency, and memory outcomes;
- source-artifact hashes and a compact public record.

CPU CI must continue to verify split leakage, geometry math, metric directions,
record recomputation, generated tables, and malformed/missing artifact failure.

## Non-goals until the evidence supports them

- online exploration or real-time SLAM claims;
- “calibrated uncertainty” language for deterministic edge/coverage heuristics;
- state-of-the-art claims without direct contemporary baselines;
- choosing a method, metric, budget, or scene after inspecting confirmatory
  outcomes;
- treating pixels or frames from one trajectory as independent replicates.
