# Modal Workflow

This is the portable replacement for the SLURM smoke workflow. The cluster established the stack; Modal should now provide reproducible GPU jobs and persistent artifacts.

## Setup

Install and authenticate Modal locally:

```bash
python -m pip install ".[modal]"
modal setup
```

The app uses three Modal Volumes:

- `u3dgs-data` mounted at `/workspace/neural-mapping/data`
- `u3dgs-outputs` mounted at `/workspace/neural-mapping/outputs`
- `u3dgs-hf-cache` mounted at `/root/.cache/huggingface`

They are created automatically by `modal.Volume.from_name(..., create_if_missing=True)`.

The image is based on `nvidia/cuda:12.8.1-devel-ubuntu22.04` because `gsplat` needs the CUDA toolkit and `nvcc`, not only PyTorch CUDA wheels.
It precompiles the `gsplat` CUDA extension for L4 (`TORCH_CUDA_ARCH_LIST=8.9`) during image build, avoiding a long first training iteration.

## First Smoke

Run an environment check:

```bash
modal run modal_app.py --action env
```

This checks `nvidia-smi`, `nvcc`, PyTorch CUDA availability, imports `nerfstudio`/`gsplat`, and loads the `gsplat` CUDA backend before launching training.

Prepare the Nerfstudio `poster` sample:

```bash
modal run modal_app.py --action prepare
```

Run the full smoke path:

```bash
modal run modal_app.py --action smoke --budget 25 --iterations 3000 --scene-name poster_modal_smoke
```

This executes:

```text
env check -> Hugging Face download -> frame filtering -> split generation -> split materialization -> Splatfacto train -> ns-eval
```

## Separate Train/Eval Commands

Train a prepared split:

```bash
modal run modal_app.py --action train --budget 50 --iterations 10000 --scene-name poster_modal_50_10k
```

Evaluate a run:

```bash
modal run modal_app.py --action eval --budget 50 --scene-name poster_modal_50_10k
```

Print compact metrics:

```bash
modal run modal_app.py --action metrics
```

## Coverage Baseline

After the random 25/50-frame baselines pass, run a deterministic coverage
baseline. It keeps the same holdout policy but selects training frames with
farthest-first coverage over input-frame order:

```bash
modal run modal_app.py \
  --action prepare \
  --data-scene-name poster_available_farthest_index \
  --selection-method farthest-index

modal run modal_app.py \
  --action train \
  --data-scene-name poster_available_farthest_index \
  --budget 25 \
  --iterations 10000 \
  --scene-name poster_modal_farthest_b25_10k

modal run modal_app.py \
  --action eval \
  --budget 25 \
  --scene-name poster_modal_farthest_b25_10k

modal run modal_app.py \
  --action train \
  --data-scene-name poster_available_farthest_index \
  --budget 50 \
  --iterations 10000 \
  --scene-name poster_modal_farthest_b50_10k

modal run modal_app.py \
  --action eval \
  --budget 50 \
  --scene-name poster_modal_farthest_b50_10k

modal run modal_app.py --action metrics
```

This is still a baseline, not the uncertainty method. Its purpose is to test
whether simple coverage already beats random frame selection on the same
pipeline.

## Pose Coverage Baseline

If `farthest-index` underperforms, use camera positions from Nerfstudio
`transform_matrix` values instead of sequence order:

```bash
modal run modal_app.py \
  --action prepare \
  --data-scene-name poster_available_farthest_pose \
  --selection-method farthest-pose

modal run modal_app.py \
  --action train \
  --data-scene-name poster_available_farthest_pose \
  --budget 25 \
  --iterations 10000 \
  --scene-name poster_modal_pose_b25_10k

modal run modal_app.py \
  --action eval \
  --budget 25 \
  --scene-name poster_modal_pose_b25_10k

modal run modal_app.py \
  --action train \
  --data-scene-name poster_available_farthest_pose \
  --budget 50 \
  --iterations 10000 \
  --scene-name poster_modal_pose_b50_10k

modal run modal_app.py \
  --action eval \
  --budget 50 \
  --scene-name poster_modal_pose_b50_10k

modal run modal_app.py --action metrics
```

## Active Expansion Baseline

After random and coverage baselines, create a 50-frame active split by keeping
the random 25-frame seed set fixed and adding frames that are novel relative to
that seed set. This is a first active proxy; later runs can replace
`pose-novelty` with model-error or uncertainty scores.

```bash
modal run modal_app.py \
  --action prepare-active \
  --source-data-scene-name poster_available \
  --base-split-scene-name poster_available \
  --data-scene-name poster_available_active_pose \
  --base-budget 25 \
  --target-budget 50 \
  --active-strategy pose-novelty

modal run modal_app.py \
  --action train \
  --data-scene-name poster_available_active_pose \
  --budget 50 \
  --iterations 10000 \
  --scene-name poster_modal_active_pose_b50_10k

modal run modal_app.py \
  --action eval \
  --budget 50 \
  --scene-name poster_modal_active_pose_b50_10k

modal run modal_app.py --action metrics
```

## Model-Error Active Baseline

The first real active baseline scores candidate frames with the trained
25-frame seed model. It keeps the random 25-frame train/val/test split fixed,
uses the seed checkpoint to compute candidate LPIPS/PSNR/SSIM, then adds the
25 highest-error candidates to make a 50-frame split.

```bash
modal run modal_app.py \
  --action score-candidates \
  --source-data-scene-name poster_available \
  --base-split-scene-name poster_available \
  --data-scene-name poster_available_active_error \
  --scene-name poster_modal_b25_10k \
  --budget 25 \
  --score-metric lpips

modal run modal_app.py \
  --action prepare-active \
  --source-data-scene-name poster_available \
  --base-split-scene-name poster_available \
  --data-scene-name poster_available_active_error \
  --base-budget 25 \
  --target-budget 50 \
  --active-strategy score-desc

modal run modal_app.py \
  --action train \
  --data-scene-name poster_available_active_error \
  --budget 50 \
  --iterations 10000 \
  --scene-name poster_modal_active_error_b50_10k

modal run modal_app.py \
  --action eval \
  --budget 50 \
  --scene-name poster_modal_active_error_b50_10k

modal run modal_app.py --action metrics
```

After scoring candidates, evaluate frame-level failure-prediction baselines
against the candidate LPIPS errors. This compares nearest training-camera
distance, temporal index distance, a uniform control, and any requested numeric
fields from the candidate score rows. Recent score files include
renderer-derived accumulation summaries when Nerfstudio exposes them:

```bash
modal run modal_app.py \
  --action frame-uncertainty \
  --source-data-scene-name poster_available \
  --base-split-scene-name poster_available \
  --data-scene-name poster_available_active_error \
  --budget 25 \
  --score-metric lpips \
  --score-signal-fields mean_transmittance,low_accumulation_fraction \
  --bad-quantile 0.8
```

The report is written under
`/workspace/neural-mapping/outputs/reports/frame_uncertainty/` in the Modal
output volume.

For a direct pixel-level uncertainty/error test, render the candidate pool and
compare per-pixel transmittance (`1 - accumulation`) against per-pixel RGB
error. The command also evaluates local patch transmittance statistics and
simple renderer-map gradients in the same pass:

```bash
modal run modal_app.py \
  --action render-uncertainty-maps \
  --source-data-scene-name dozer_available_v1 \
  --base-split-scene-name dozer_available_v1 \
  --data-scene-name dozer_available_render_maps_v1 \
  --scene-name dozer_modal_v1_d4_fixed_b25_10k \
  --budget 25 \
  --score-metric rgb-l1 \
  --bad-quantile 0.8 \
  --max-pixels-per-frame 50000 \
  --render-map-signals transmittance,local-mean-transmittance,local-std-transmittance,accumulation-gradient,depth-gradient \
  --patch-size 15
```

The report is written under
`/workspace/neural-mapping/outputs/reports/render_uncertainty_maps/`.

To test model-disagreement uncertainty, render the same candidate views from
multiple independently trained seed models and score per-pixel RGB variance:

```bash
modal run modal_app.py \
  --action ensemble-uncertainty-maps \
  --source-data-scene-name dozer_available_v1 \
  --base-split-scene-name dozer_available_v1 \
  --data-scene-name dozer_available_ensemble_maps_v1 \
  --budget 25 \
  --score-metric rgb-l1 \
  --bad-quantile 0.8 \
  --max-pixels-per-frame 10000 \
  --ensemble-scene-names dozer_modal_v1_d4_fixed_b25_10k,dozer_modal_v2_d4_b25_10k,dozer_modal_v3_d4_b25_10k
```

The report is written under
`/workspace/neural-mapping/outputs/reports/ensemble_uncertainty_maps/`.

To use that report as an active-selection signal, pass it to `prepare-active`
and select the frame-level `mean_uncertainty` field:

```bash
modal run modal_app.py \
  --action prepare-active \
  --source-data-scene-name dozer_available_v1 \
  --base-split-scene-name dozer_available_v1 \
  --data-scene-name dozer_available_ensemble_hybrid_w035_v1 \
  --budget 25 \
  --target-budget 50 \
  --active-strategy score-pose-hybrid \
  --score-path /workspace/neural-mapping/outputs/reports/ensemble_uncertainty_maps/dozer_available_ensemble_maps_v1_budget_025_rgb-l1.json \
  --score-key mean_uncertainty \
  --score-weight 0.35
```

## GPU Choice

The default GPU is `L4`, matching the cluster smoke environment. Override it at image/function definition time by setting:

```bash
MODAL_GPU=L40S modal run modal_app.py --action smoke --budget 25 --iterations 3000
```

Modal currently supports GPU requests through `@app.function(gpu=...)`; common choices include `L4`, `A10G`, `L40S`, `A100`, and `H100`.

## Notes

- The image pins PyTorch to `2.11.0+cu128`, matching the CUDA 12.8 stack that worked on the cluster.
- Evaluation sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted locally generated Nerfstudio checkpoints.
- Keep checkpoints and rendered images in Modal Volumes, not GitHub.
- Use the `poster` sample only as a smoke test. The real project should move to Replica/ScanNet-style datasets after Modal smoke passes.

## References

- Modal Images: https://modal.com/docs/guide/images
- Modal Volumes: https://modal.com/docs/guide/volumes
- Modal GPU acceleration: https://modal.com/docs/guide/gpu
