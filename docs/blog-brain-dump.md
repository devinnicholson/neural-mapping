# Blog Brain Dump: Uncertainty-Aware Neural Mapping

Date: 2026-06-10

This is a working brain dump for a blog post. It is written for synthesis, not
as final prose. It avoids local usernames, hostnames, private paths, and raw
checkpoint locations.

## One-Line Version

We built a lightweight research harness around Nerfstudio/Splatfacto to test
whether uncertainty from partial 3D Gaussian scene reconstructions can tell us
which new camera views are worth adding. The current best rule uses ensemble
RGB disagreement plus camera-pose diversity to choose more training frames, and
it now improves held-out rendering quality across repeated scene seeds.

## Core Question

Can uncertainty maps from a 3D Gaussian scene representation predict where
novel-view RGB, depth, or geometry will fail under partial indoor observations?

The practical version:

- Start with a small number of observed frames.
- Train a 3D Gaussian scene model.
- Ask the model where it is uncertain or likely to fail.
- Use that uncertainty to choose the next frames to observe.
- Check whether the resulting model renders held-out views better than a
  random frame budget.

## Project Shape

The repo is intentionally lightweight. The Python utilities stay dependency
light, while heavy training dependencies such as Nerfstudio, PyTorch, CUDA, and
gsplat live in the runtime environment.

Main code/docs pieces:

- `scripts/generate_splits.py`: deterministic train/val/test split manifests.
- `scripts/materialize_nerfstudio_split.py`: creates Nerfstudio-ready split
  directories.
- `scripts/filter_nerfstudio_transforms.py`: filters broken transform files
  where image files are missing.
- `scripts/materialize_candidate_eval.py`: builds candidate-evaluation data.
- `scripts/score_candidate_frames.py`: scores candidates with a trained model.
- `scripts/generate_active_split.py`: expands a seed split using score, pose,
  or hybrid rules.
- `scripts/evaluate_frame_uncertainty.py`: frame-level uncertainty/error
  alignment.
- `scripts/evaluate_render_uncertainty_maps.py`: single-model renderer proxy
  maps.
- `scripts/evaluate_ensemble_uncertainty_maps.py`: ensemble RGB disagreement
  maps.
- `scripts/prepare_tum_rgbd.py`: TUM RGB-D download, association, and
  Nerfstudio transform generation.
- `scripts/evaluate_depth_metrics.py`: held-out rendered-depth evaluation
  against RGB-D depth images.
- `scripts/summarize_active_metrics.py`: active-vs-random metric summaries.
- `configs/active_metric_pairs.json`: manifest of paired random/active runs.
- `docs/results.md`: current evidence table and caveats.
- `docs/dashboard.html`: static results dashboard.

## Environment Bring-Up

Initial work happened on a GPU cluster with NVIDIA L4 GPUs and CUDA 12.8. We
validated:

- PyTorch CUDA build was available.
- CUDA was visible to Python.
- Nerfstudio imported cleanly.
- gsplat imported cleanly.
- `ns-train splatfacto --help` ran successfully.

Important environment lesson: installing the Nerfstudio/Splatfacto stack is not
the actual hard part. The hard part is making the dataset, split definitions,
Nerfstudio CLI arguments, and evaluation protocol line up exactly.

## Dataset Download Issue

The first Nerfstudio sample download path hit a Google Drive/gdown permission
or quota problem. The download failed with gdown unable to retrieve the public
file URL.

We worked around this by using a Hugging Face mirror. That successfully fetched
the dataset files.

Blog angle:

- The boring failure mode matters: "I had CUDA working, but the experiment
  still failed because the data source and transform file were inconsistent."

## Missing Frame Issue

After the dataset downloaded, the training run failed because the
`transforms.json` referenced image files that were not present locally. Example
failure shape:

- Nerfstudio tried to open an image listed in `transforms.json`.
- The referenced file did not exist.
- Training crashed during cached train/eval image loading.

We fixed this by filtering the transforms file down to frames that actually
exist on disk.

Concrete poster sample result:

- Source transforms referenced 226 frames.
- Only 100 usable frames were present.
- The filtered dataset became `poster_available`.

This was a key early lesson: before comparing active selection strategies, we
needed a reproducible dataset materialization pipeline.

## Early Poster Experiments

The first successful small training runs were on the filtered poster sample.

Early 3,000-iteration split runs:

- Budget 25 split: PSNR 25.047, SSIM 0.899, LPIPS 0.238.
- Budget 50 split: PSNR 24.805, SSIM 0.901, LPIPS 0.256.

The budget 50 result being worse than budget 25 was a useful warning. More
frames do not automatically make the result better if the split/materialization
or training protocol is not clean.

Later 10,000-iteration poster run:

- Budget 50 split at 10k iterations: PSNR 30.744, SSIM 0.946, LPIPS 0.203.

This showed that training length and corrected evaluation both mattered.

## Evaluation Gotcha: PyTorch Weights-Only Loading

Evaluation initially failed under newer PyTorch behavior because `torch.load`
defaults changed around `weights_only`. Nerfstudio checkpoints contained objects
that failed safe unpickling.

Workaround:

- Set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted checkpoints generated by
  our own run.

This let `ns-eval` load checkpoints and produce metrics.

Blog caveat:

- Do not present this as a general security recommendation.
- It is acceptable only because the checkpoints are self-generated/trusted.

## Moving From Cluster To Modal

After cluster access became limited, we moved the workflow to Modal.

Modal work included:

- Building a reusable `modal_app.py` workflow.
- Preparing datasets in Modal volumes.
- Running Splatfacto training on Modal L4 GPUs.
- Running eval and collecting `ns_eval.json` metrics.
- Adding a `metrics` action to collect compact rows.

There was a real gsplat/CUDA failure on the first Modal attempt:

- gsplat reported no CUDA toolkit found.
- Training crashed when gsplat's CUDA wrapper could not access the expected
  objects.

We fixed the Modal runtime so gsplat was enabled and Splatfacto training could
run successfully.

Blog angle:

- The cloud GPU is not enough. For 3D Gaussian training, the CUDA toolkit and
  extension build/runtime details matter.

## Split Corrections

Early poster smoke runs were not final because the materialized datasets used a
legacy split field that Nerfstudio ignored. The result was useful for smoke
testing but not strong enough for held-out claims.

We corrected the materializer to emit Nerfstudio-native split fields:

- `train_filenames`
- `val_filenames`
- `test_filenames`

This made later runs proper held-out-test comparisons.

## Baselines We Tried

We compared several frame-selection strategies:

- Random frame selection.
- Farthest-index coverage over frame order.
- Farthest-pose coverage over camera centers.
- Pose-novelty active expansion from a seed set.
- Active model-error selection using seed-model LPIPS.
- Pure uncertainty score selection.
- Hybrid score plus pose selection.
- Ensemble tail-risk plus pose diversity.

The key learning:

- Pure geometry/pose coverage is not enough.
- Pure score selection can be too myopic.
- Single-model renderer proxies were weak or misleading.
- Ensemble disagreement is useful, but it works best when mixed with pose
  diversity.

## Poster Results: Development Scene, Not Final Claim

Poster was the development scene. It helped discover the protocol, but the
final poster signal was based on seed-model LPIPS candidate error rather than
ensemble disagreement.

Corrected poster v2-v4 with `score-pose-hybrid`, `score_weight=0.35`:

- Mean active-vs-random budget-50 delta:
  - +1.136 PSNR
  - +0.009 SSIM
  - -0.017 LPIPS

Important caveat:

- Poster supports the acquisition shape, especially the need for a hybrid rule.
- It is not the cleanest evidence for ensemble uncertainty.

## The Rule That Started Working

Best-supported acquisition rule so far:

1. Start from a locked random budget-25 seed split.
2. Keep validation and held-out test frames fixed.
3. Train independent budget-25 Splatfacto seed models.
4. Render remaining candidate views from multiple seed models.
5. Compute per-pixel ensemble RGB variance.
6. Aggregate each candidate frame by `top_decile_mean_uncertainty`.
7. Expand from budget 25 to budget 50 with `score-pose-hybrid`.
8. Use `score_weight=0.35`, so uncertainty tail risk is mixed with camera-pose
   diversity.

Why top decile?

- Mean uncertainty over an entire image can dilute the important signal.
- The failure signal often lives in the tail: confusing geometry, uncertain
  object boundaries, holes, occlusions, or under-observed regions.
- Taking the mean of the highest-uncertainty pixel decile gives a frame-level
  score for "does this view contain a region the current model is likely to
  fail on?"

## Dozer Results

Dozer was where the ensemble-tail hybrid result first became convincing.

Mean active-vs-random budget-50 delta across v1-v4:

- +0.779 PSNR
- +0.020 SSIM
- -0.018 LPIPS

Important negative results:

- Single-model transmittance-style uncertainty maps were near random or worse.
- Pure ensemble-score selection underperformed random on the first dozer seed.

Interpretation:

- Ensemble disagreement was a useful failure signal.
- It needed pose diversity in the acquisition rule.

## Redwoods2 Results

Redwoods2 tested whether dozer was a one-scene accident.

Mean active-vs-random budget-50 delta across v1-v4:

- +0.699 PSNR
- +0.027 SSIM
- -0.012 LPIPS

Notes:

- No redwoods2 seed regressed on PSNR, SSIM, or LPIPS.
- v3 was the weak case: PSNR nearly tied, while SSIM and LPIPS still improved.

Interpretation:

- This upgraded the result from "dozer-specific" to a more scene-agnostic
  heuristic.

## Library Results

Library extended the experiment to another sample scene.

Mean active-vs-random budget-50 delta across v1-v3:

- +0.496 PSNR
- +0.015 SSIM
- -0.003 LPIPS

Caveat:

- PSNR and SSIM improved across seeds.
- LPIPS was weaker because one seed regressed.

Interpretation:

- Good support for transfer, but not as clean perceptually as dozer/redwoods2.

## Kitchen Results

Kitchen is the hard-scene stress test.

Mean active-vs-random budget-50 delta across v1-v4:

- +0.687 PSNR
- +0.002 SSIM
- -0.028 LPIPS

But:

- v1 regressed badly.
- A weight sweep toward stronger pose diversity did not recover the random
  baseline for v1.
- v2 and v3 were strong, and v4 was modestly positive.

Interpretation:

- The signal can rescue difficult splits.
- It is not uniformly robust yet.
- Kitchen should be presented separately from the clean repeats.

## BWW Entrance Results

BWW entrance is the strongest current replication.

Protocol:

- Four seeds: v1-v4.
- Budget-25 seed models.
- Expand to budget 50.
- Downscale factor 4.
- 10,000 training iterations.
- Score signal: ensemble RGB disagreement.
- Frame aggregation: `top_decile_mean_uncertainty`.
- Active rule: `score-pose-hybrid`, `score_weight=0.35`.

Mean active-vs-random budget-50 delta:

- +2.016 PSNR
- +0.031 SSIM
- -0.021 LPIPS

Per-seed deltas:

| Seed | Delta PSNR | Delta SSIM | Delta LPIPS |
|---|---:|---:|---:|
| v1 | +1.273 | +0.014 | -0.013 |
| v2 | +2.994 | +0.049 | -0.037 |
| v3 | +0.918 | +0.004 | -0.0004 |
| v4 | +2.880 | +0.057 | -0.035 |

Uncertainty signal quality:

| Seed | Spearman | AUROC | AUPRC |
|---|---:|---:|---:|
| v1 | 0.599 | 0.770 | 0.456 |
| v2 | 0.594 | 0.765 | 0.439 |
| v3 | 0.593 | 0.765 | 0.440 |
| v4 | 0.593 | 0.764 | 0.436 |

Interpretation:

- All four BWW seeds improved PSNR, SSIM, and LPIPS.
- The uncertainty signal was stable across seeds.
- This is the cleanest current support for the ensemble-tail-plus-pose rule.

## Current Aggregate Claim

Across dozer, redwoods2, and BWW entrance:

- 12 ensemble-tail seeds.
- No seed regressed on PSNR, SSIM, or LPIPS.
- Average delta:
  - +1.165 PSNR
  - +0.026 SSIM
  - -0.017 LPIPS

Across all five tracked scene groups, including library and mixed kitchen:

- 19 paired seeds.
- Weighted average delta:
  - about +0.959 PSNR
  - about +0.019 SSIM
  - about -0.017 LPIPS

Careful wording:

- "The current best rule improves held-out rendering quality across repeated
  Nerfstudio/Splatfacto sample-scene seeds."
- "The rule is not yet proven robust across harder indoor scenes."
- "Kitchen shows the failure mode: the average can improve while a seed
  regresses badly."

## What The Results Mean In Neural Rendering Terms

Splatfacto/3D Gaussian training builds a scene representation from a set of
camera images. If the camera set leaves some regions under-observed, the model
may still render plausible views, but it will fail under certain novel
viewpoints.

The useful finding is that independent seed models disagree most in places that
are actually error-prone. That disagreement can be turned into a practical
acquisition score:

- Render candidate frames from several seed models.
- Look for pixels where the models disagree.
- Summarize each candidate by the high-uncertainty tail.
- Prefer candidate frames that expose those uncertain regions.
- Keep enough pose diversity to avoid selecting redundant views.

In plain English:

The model is telling us where it does not understand the scene. If we choose
new views that see those weak regions, the next reconstruction improves more
than random sampling.

## First Depth-Bearing Bringup

After the sample-scene active-selection work, we started moving into RGB-D data
with TUM RGB-D `freiburg1_desk`.

What changed:

- Added a TUM RGB-D preparation script.
- Downloaded the TUM archive in Modal.
- Associated RGB frames, depth frames, and ground-truth poses.
- Converted the sequence into a Nerfstudio-style `transforms.json` with
  `depth_file_path` entries.
- Generated random budget-25 and budget-50 splits.
- Materialized those splits into Nerfstudio datasets.
- Trained Splatfacto for 7,000 iterations on each budget.
- Evaluated the held-out test split.
- Evaluated rendered depth against the held-out TUM depth maps.
- Evaluated renderer uncertainty signals against candidate-view depth error.
- Used the depth-error uncertainty report to build active budget-50 RGB-D
  splits from transmittance tail risk, and on v1/v3 also from depth-gradient
  tail risk.
- Trained and evaluated those active splits against the same held-out test set.
- Repeated the transmittance-tail active result on three TUM split seeds.

Initial RGB reconstruction results:

| Scene | Budget | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| `tum_fr1_desk_v1_b25_7k` | 25 | 15.705 | 0.578 | 0.405 |
| `tum_fr1_desk_v1_b50_7k` | 50 | 17.740 | 0.663 | 0.330 |
| `tum_fr1_desk_v1_active_depth_trans_b50_7k` | 50 | 18.016 | 0.677 | 0.323 |
| `tum_fr1_desk_v1_active_depth_grad_b50_7k` | 50 | 17.946 | 0.676 | 0.322 |
| `tum_fr1_desk_v2_b25_7k` | 25 | 14.926 | 0.565 | 0.413 |
| `tum_fr1_desk_v2_b50_7k` | 50 | 17.149 | 0.647 | 0.335 |
| `tum_fr1_desk_v2_active_depth_trans_b50_7k` | 50 | 17.914 | 0.675 | 0.317 |
| `tum_fr1_desk_v3_b25_7k` | 25 | 15.432 | 0.576 | 0.420 |
| `tum_fr1_desk_v3_b50_7k` | 50 | 17.676 | 0.664 | 0.343 |
| `tum_fr1_desk_v3_active_depth_trans_b50_7k` | 50 | 18.362 | 0.684 | 0.319 |
| `tum_fr1_desk_v3_active_depth_grad_b50_7k` | 50 | 18.169 | 0.687 | 0.318 |

Initial depth results:

| Scene | Budget | Raw AbsRel | Raw RMSE | Median-aligned AbsRel | Median-aligned delta1 |
|---|---:|---:|---:|---:|---:|
| `tum_fr1_desk_v1_b25_7k` | 25 | 0.572 | 0.982m | 0.643 | 0.381 |
| `tum_fr1_desk_v1_b50_7k` | 50 | 0.371 | 0.674m | 0.362 | 0.661 |
| `tum_fr1_desk_v1_active_depth_trans_b50_7k` | 50 | 0.359 | 0.643m | 0.330 | 0.683 |
| `tum_fr1_desk_v1_active_depth_grad_b50_7k` | 50 | 0.347 | 0.601m | 0.323 | 0.689 |
| `tum_fr1_desk_v2_b25_7k` | 25 | 0.520 | 0.934m | 0.598 | 0.417 |
| `tum_fr1_desk_v2_b50_7k` | 50 | 0.399 | 0.733m | 0.378 | 0.634 |
| `tum_fr1_desk_v2_active_depth_trans_b50_7k` | 50 | 0.357 | 0.629m | 0.314 | 0.707 |
| `tum_fr1_desk_v3_b25_7k` | 25 | 0.511 | 0.827m | 0.577 | 0.458 |
| `tum_fr1_desk_v3_b50_7k` | 50 | 0.358 | 0.644m | 0.351 | 0.656 |
| `tum_fr1_desk_v3_active_depth_trans_b50_7k` | 50 | 0.347 | 0.595m | 0.315 | 0.671 |
| `tum_fr1_desk_v3_active_depth_grad_b50_7k` | 50 | 0.366 | 0.606m | 0.341 | 0.654 |

Initial depth-error uncertainty alignment from the budget-25 model:

| Signal | Spearman | AUROC | AUPRC |
|---|---:|---:|---:|
| Transmittance | 0.203 | 0.710 | 0.472 |
| Local mean transmittance | 0.197 | 0.728 | 0.468 |
| Depth gradient | 0.183 | 0.753 | 0.467 |

V2 repeated the signal check on the same TUM sequence with a different split
seed. The uncertainty signal was weaker but still useful: transmittance reached
0.148 Spearman / 0.656 AUROC against per-pixel depth AbsRel, and its highest
uncertainty decile had mean depth AbsRel 1.324 versus a global mean of 0.552.

V3 repeated the same check with seed `20260612`. The signal was weaker by
Spearman, but the uncertainty tails still concentrated geometry failures:
transmittance reached 0.077 Spearman / 0.628 AUROC, and its highest decile had
mean depth AbsRel 1.015 versus a global mean of 0.465. Depth-gradient had the
best v3 AUROC/AUPRC, so we trained both v3 active variants.

Why this matters:

- It proves the workflow can now ingest a real RGB-D sequence, not only
  Nerfstudio sample RGB captures.
- The 50-frame run improved over 25 frames, which is the sanity check we wanted.
- Depth error also improved with more observed frames: raw AbsRel dropped from
  0.572 to 0.371 and raw RMSE dropped from 0.982m to 0.674m.
- Uncertainty is beginning to connect to geometry, not only RGB. On unobserved
  TUM candidate views, the highest-transmittance pixel decile had mean depth
  AbsRel 1.433 versus a global mean of 0.529, and its bad-pixel fraction was
  0.615 versus 0.200 globally.
- The absolute metrics are much lower than poster/BWW because TUM is a harder
  handheld indoor sequence and Splatfacto had no COLMAP point cloud, so it used
  random point initialization.
- The first active RGB-D loop is now closed. Starting from the same 25-frame
  seed set, we used `score-pose-hybrid` and `score_weight=0.35` to add 25
  depth-risk views. Transmittance tail risk gave the best RGB row, beating
  random budget 50 by +0.276 PSNR, +0.014 SSIM, and -0.007 LPIPS. Depth-gradient
  tail risk gave the best geometry row, beating random budget 50 by -0.024 raw
  AbsRel and -0.040 median-aligned AbsRel.
- The second TUM split seed made the RGB-D result more credible. On v2,
  transmittance tail risk beat random budget 50 by +0.765 PSNR, +0.029 SSIM,
  -0.018 LPIPS, -0.042 raw AbsRel, and -0.063 median-aligned AbsRel.
- The third TUM split seed repeated the pattern. On v3, transmittance tail risk
  beat random budget 50 by +0.686 PSNR, +0.020 SSIM, -0.024 LPIPS, -0.011 raw
  AbsRel, and -0.036 median-aligned AbsRel. The v3 depth-gradient active run
  was good for RGB but weaker on depth, so transmittance is the more stable
  current RGB-D selector.

## What Not To Overclaim

Do not claim:

- This is a complete active SLAM system.
- This is real-time.
- Active selection is robust on RGB-D scenes. The transmittance active variant
  improved three TUM split seeds, but robustness still needs another sequence
  or dataset.
- This is robust on all indoor scenes.
- Single-model uncertainty maps are enough.
- Pure uncertainty score selection is enough.
- More frames always improve quality.

Better claim:

- This is an offline active-view selection experiment.
- Ensemble disagreement is a useful failure signal.
- The hybrid of uncertainty tail risk and pose diversity is the strongest
  selection rule so far.
- Results are promising across repeated sample-scene seeds, with kitchen as a
  visible hard-case caveat.

## Engineering Things Worth Mentioning

The project is partly about research, partly about making the experiment
reproducible.

Useful engineering details:

- We separated lightweight split/metric code from heavy Nerfstudio runtime
  dependencies.
- We had to repair dataset manifests because transforms referenced missing
  image files.
- We added explicit Nerfstudio split fields to avoid accidental eval leakage or
  ignored test splits.
- We normalized budget naming because it appears as `25`, `025`, and
  `budget_025`.
- We parsed noisy Modal CLI logs into compact metric rows.
- We added a checked-in active-pair manifest so summaries are reproducible.
- We made a static dashboard so results can be inspected without a cloud
  runtime or local frontend stack.

## Dashboard Work

We added `docs/dashboard.html` as a static dashboard.

Current dashboard contents:

- Summary metrics across paired active-vs-random runs.
- Scene-group deltas.
- BWW seed-level deltas.
- Evidence matrix.
- BWW uncertainty signal metrics.
- Protocol strip.

Design direction:

- Clean, quiet, research-oriented.
- No frontend dependencies.
- No generated artifacts required to open the page.
- Copy was edited down to avoid AI-generated marketing language.

Future dashboard idea:

- Add a `scripts/build_dashboard_data.py` generator.
- Read `outputs/modal_metrics_latest.log`, `configs/active_metric_pairs.json`,
  and uncertainty report JSON.
- Emit a compact JSON file for the dashboard.
- Eventually graduate to a small Vite/React app only if needed.

## Possible Blog Titles

- "Teaching a 3D Gaussian Model Where to Look Next"
- "Using Uncertainty to Choose Better Views for Neural Rendering"
- "From Random Frames to Active View Selection in 3D Gaussian Mapping"
- "What My 3D Gaussian Model Gets Wrong, and How That Helps It Learn"
- "An Offline Active-Mapping Experiment with Nerfstudio and Splatfacto"

## Possible Blog Structure

1. Open with the question.
   - "If a neural renderer is unsure about part of a scene, can that
     uncertainty tell us where to move the camera next?"

2. Explain the setup.
   - 3D Gaussian/Splatfacto scene model.
   - Partial observations.
   - Fixed held-out views.
   - Frame budget 25 vs 50.

3. Explain the first failure.
   - Data was messy.
   - Google Drive download failed.
   - Transform files referenced missing frames.
   - Evaluation initially failed due checkpoint loading behavior.

4. Explain the baseline problem.
   - Random selection.
   - Pose-only/farthest selection.
   - Pure score selection.
   - Single-model uncertainty proxies.
   - None was reliably enough.

5. Explain the rule that worked.
   - Train multiple seed models.
   - Render candidates.
   - Compute ensemble RGB disagreement.
   - Score top-decile uncertainty.
   - Mix uncertainty score with pose diversity.

6. Show the results.
   - BWW entrance as headline.
   - Dozer and redwoods2 as clean repeats.
   - Library as supportive.
   - Kitchen as caveat.

7. Explain what it means.
   - Uncertainty can be turned into an acquisition signal.
   - The tail matters.
   - Diversity matters.

8. Explain what comes next.
   - Depth-bearing indoor data beyond the initial TUM bringup.
   - Geometry/depth uncertainty metrics.
   - Larger scenes.
   - Maybe robotic/Jetson-style deployment later.

## Images/Figures To Include

Potential figures:

- A diagram of the loop:
  - seed frames -> train models -> render candidates -> uncertainty map ->
    select frames -> retrain/evaluate.
- A simple table of BWW seed results.
- A bar chart of scene group deltas.
- Example uncertainty/error map if available.
- A screenshot of `docs/dashboard.html`.
- A small failure case section for kitchen.

## Jetson/Robotics Angle

We discussed using this kind of work on a Jetson Orin Nano Super Developer Kit.

Realistic framing:

- Training full Splatfacto models on the Jetson is probably not the best first
  use.
- A cooler near-term project is active data capture:
  - Jetson collects frames from a camera.
  - A remote GPU trains/evaluates a map.
  - The system recommends where to move next.
  - Jetson displays uncertainty or capture guidance.

Longer-term:

- Lightweight inference or visualization on-device.
- Cloud/desktop GPU for heavy training.
- Robot or handheld capture loop for active mapping.

## End Goal

The end goal is not just a nicer NeRF/3DGS demo.

The end goal is a mapping system that knows what it does not know:

- It can say which parts of a scene are unreliable.
- It can decide which next view would reduce uncertainty.
- It can improve a map efficiently under a limited observation budget.
- Eventually it can connect RGB, depth, and geometry uncertainty.

For the blog:

- The current project is an offline proof-of-concept for that larger active
  mapping loop.
- The main result is that ensemble disagreement is not just a pretty
  uncertainty map. It can drive frame acquisition and improve held-out
  rendering quality.

## Clean Summary Paragraph

This project started as a question about whether uncertainty in neural scene
representations can be useful, not just diagnostic. Using Nerfstudio/Splatfacto
as the rendering backbone, I built a lightweight experiment harness that locks
train/validation/test splits, trains small seed models, scores candidate camera
views, expands the training set, and evaluates held-out rendering quality. The
most reliable rule so far uses ensemble RGB disagreement, aggregated over each
candidate frame's highest-uncertainty pixels, then mixes that score with
camera-pose diversity. Across repeated dozer, redwoods2, and BWW entrance
seeds, this active selector improves random budget-50 training by about +1.165
PSNR, +0.026 SSIM, and -0.017 LPIPS, with BWW entrance showing the strongest
four-seed replication. Kitchen remains a hard-case caveat. The first TUM
RGB-D runs extend the pipeline to depth-bearing data and close an initial
active loop: transmittance-tail budget-50 splits improve over random budget 50
on three TUM split seeds. Depth-gradient was slightly stronger on v1 held-out
depth but weaker than transmittance on v3 geometry, so the stable RGB-D signal
right now is transmittance tail risk. The next step is to replicate that RGB-D
result on another TUM sequence or another RGB-D dataset.
