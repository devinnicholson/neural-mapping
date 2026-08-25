# ICL-NUIM multi-trajectory confirmatory study

This study was executed under the frozen protocol in `experiments/protocols/icl_nuim_multitrajectory_v1.json`. All eight matched active–random pairs and all four seed-42 coverage-control sets completed.

## Headline result

The depth-gradient/pose policy improved held-out appearance over random on average (PSNR +0.637 dB; LPIPS -0.023) but did not improve the co-primary 5 cm surface F-score (-0.0016). The prespecified overall-support gate therefore **failed**.

| Outcome | Mean active − random | Favorable pairs | Favorable trajectories | Trajectory-cluster 95% interval |
|---|---:|---:|---:|---:|
| PSNR | +0.6372 | 5/8 | 3/4 | [-0.0231, +1.4764] |
| LPIPS | -0.0231 | 6/8 | 2/4 | [-0.0546, +0.0084] |
| Raw depth AbsRel | -0.0047 | 5/8 | 2/4 | [-0.0653, +0.0531] |
| 5 cm surface F-score | -0.0016 | 5/8 | 2/4 | [-0.0459, +0.0368] |

## Error-ranking diagnostic

The target-free depth-gradient proxy had positive rank correlation in 8/8 seed models (mean Spearman 0.141) and AUROC above 0.5 in 8/8 (mean 0.716). This supports error ranking, not calibrated uncertainty or downstream acquisition benefit.

## Frozen gate outcomes

- `h1_primary`: **Fail**
- `h2_replication`: **Fail**
- `h3_control_superiority`: **Fail**
- `h4_uncertainty`: **Pass**
- `perceptual_safety_gate`: **Pass**
- `overall_support`: **Fail**

## Recompute

```bash
python scripts/build_icl_benchmark_record.py \
  --artifact-root experiments/artifacts/icl_nuim_multitrajectory_v1 \
  --protocol experiments/protocols/icl_nuim_multitrajectory_v1.json \
  --run-manifest experiments/run_manifests/icl_nuim_multitrajectory_v1.json \
  --output experiments/records/icl_nuim_multitrajectory_v1.json
python scripts/generate_icl_report_assets.py
```

The study record hashes every compact metric, diagnostic, split manifest, and run-manifest input. The compact diagnostic files also retain SHA-256 digests of their pixel-heavy source reports.
