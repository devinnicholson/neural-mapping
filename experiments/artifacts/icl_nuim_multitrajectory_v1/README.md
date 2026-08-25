# Compact confirmatory artifacts

This directory is the public, claim-bearing input to
`experiments/records/icl_nuim_multitrajectory_v1.json`.

- `runs/` contains final `ns_eval.json`, `depth_eval.json`, and
  `geometry_eval.json` files for all 24 budget-50 models.
- `reports/render_uncertainty_maps/` contains scalar summaries for all eight
  seed-model diagnostics. Each summary records the SHA-256 digest and byte size
  of its pixel-heavy source report on the immutable Modal volume.
- `splits/` contains the four 25-frame seed manifests and all 20 budget-50
  method manifests. The record builder verifies counts, uniqueness,
  train/validation/test disjointness, common holdouts, and seed retention.

These files are intentionally sufficient to recompute every paper number and
decision gate without downloading checkpoints, rendered images, or the
licensed dataset. They are not sufficient to retrain models; use the frozen
protocol and GPU runner for that.
