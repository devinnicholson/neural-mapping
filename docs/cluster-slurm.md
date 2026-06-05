# SLURM Cluster Runbook

The cluster example uses:

```bash
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu
```

The project SLURM scripts follow that pattern. Override `SPACK_SETUP` or `SPACK_ENV` only if the cluster path or environment name changes.

## 0. Copy Project To Cluster

From this machine or a Git remote, place the project on the cluster. Then run all commands from the project root:

```bash
cd /path/to/uncertainty-3dgs-neural-mapping
```

## 1. GPU Sanity Job

This checks `nvidia-smi`, `nvcc`, Python, optional PyTorch CUDA, and the lightweight project tests.

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD scripts/slurm/00_gpu_sanity.slurm
```

Check logs:

```bash
ls outputs/slurm
```

Pass criteria:

- `nvidia-smi` prints a GPU.
- `python -m unittest discover -s tests` passes.
- If PyTorch is already installed, `torch.cuda.is_available` should be `True`.

## 2. Nerfstudio Setup/Check Job

First run it without installing anything:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD scripts/slurm/01_nerfstudio_setup_check.slurm
```

If `ns-train` is missing and the cluster allows package installs, run:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,INSTALL_NERFSTUDIO=1 scripts/slurm/01_nerfstudio_setup_check.slurm
```

If PyTorch reports a CUDA driver mismatch, reinstall it from the CUDA 12.8 wheel index:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,INSTALL_NERFSTUDIO=1,FORCE_REINSTALL_TORCH=1,TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 scripts/slurm/01_nerfstudio_setup_check.slurm
```

The setup script also pins:

- `setuptools<82`, because recent PyTorch wheels may reject `setuptools 82+`.
- `numpy<2`, because some Nerfstudio ecosystem packages such as `nuscenes-devkit` still require NumPy 1.x.

This creates a virtualenv at:

```text
.venvs/nerfstudio
```

Pass criteria:

- `ns-train splatfacto --help` succeeds.
- `gsplat` imports.
- PyTorch sees CUDA.

## 3. First Splatfacto Run

Before training, download a small official Nerfstudio sample scene:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,CAPTURE_NAME=poster scripts/slurm/04_download_nerfstudio_sample.slurm
```

If Google Drive/gdown fails, force the Hugging Face mirror:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,CAPTURE_NAME=poster,DOWNLOAD_SOURCE=huggingface scripts/slurm/04_download_nerfstudio_sample.slurm
```

The expected dataset path is:

```text
data/nerfstudio/poster
```

This follows Nerfstudio's quickstart sample command:

```bash
ns-download-data nerfstudio --capture-name=poster
```

## 4. First Splatfacto Run

This requires a Nerfstudio-compatible dataset directory.

Set:

- `NS_DATA_DIR`: absolute path to the dataset.
- `SCENE_NAME`: short scene id.
- `BUDGET`: frame budget label, default `25`.
- `MAX_NUM_ITERATIONS`: default `30000`; use `1000-3000` for first smoke.

Example smoke run:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,NS_DATA_DIR=$PWD/data/nerfstudio/poster,SCENE_NAME=poster,BUDGET=25,MAX_NUM_ITERATIONS=3000,DOWNSCALE_FACTOR=1,VIS=tensorboard scripts/slurm/02_splatfacto_first_run.slurm
```

Full first run:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,NS_DATA_DIR=$PWD/data/nerfstudio/poster,SCENE_NAME=poster,BUDGET=25,MAX_NUM_ITERATIONS=30000,DOWNSCALE_FACTOR=1,VIS=tensorboard scripts/slurm/02_splatfacto_first_run.slurm
```

For the Hugging Face `poster` mirror, keep `DOWNSCALE_FACTOR=1`. Nerfstudio's automatic downscale selection may otherwise look for `images_2/`, which is not present in that mirror.

The training script passes this through the Nerfstudio dataparser subcommand:

```bash
ns-train splatfacto ... nerfstudio-data --data <dataset> --downscale-factor 1
```

If the Hugging Face mirror has a `transforms.json` entry for a missing image, prepare a filtered view:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,SOURCE_DATA_DIR=$PWD/data/nerfstudio/poster,FILTERED_DATA_DIR=$PWD/data/nerfstudio/poster_available scripts/slurm/05_prepare_nerfstudio_dataset.slurm
```

Then train against the filtered dataset:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,NS_DATA_DIR=$PWD/data/nerfstudio/poster_available,SCENE_NAME=poster_available,BUDGET=25,MAX_NUM_ITERATIONS=3000,DOWNSCALE_FACTOR=1,VIS=tensorboard scripts/slurm/02_splatfacto_first_run.slurm
```

To train against actual split budgets instead of the full filtered dataset, first materialize budgeted datasets:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,SCENE_NAME=poster_available,BUDGETS="25 50" scripts/slurm/07_materialize_split.slurm
```

Then train budget 25:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,NS_DATA_DIR=$PWD/data/nerfstudio_splits/poster_available/budget_025,SCENE_NAME=poster_available_split,BUDGET=25,MAX_NUM_ITERATIONS=3000,DOWNSCALE_FACTOR=1,VIS=tensorboard scripts/slurm/02_splatfacto_first_run.slurm
```

After training succeeds, evaluate the latest run:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,SCENE_NAME=poster_available,BUDGET=25,RENDER_OUTPUTS=1 scripts/slurm/06_eval_latest_run.slurm
```

The eval script sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` by default because PyTorch 2.6+ changed `torch.load` to use `weights_only=True`, while Nerfstudio checkpoints contain local training metadata. Only use this for checkpoints produced by this project or another trusted source.

This writes:

```text
outputs/runs/poster_available/splatfacto/budget_025/metrics/ns_eval.json
outputs/runs/poster_available/splatfacto/budget_025/renders/eval/
```

Outputs go to:

```text
outputs/runs/<scene>/splatfacto/budget_<budget>/train
```

## 5. Lightweight Metric Smoke Job

This does not train. It verifies the project split and metric CLIs on cluster Python.

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD scripts/slurm/03_metric_smoke.slurm
```

## Suggested Job Durations

- GPU sanity: 15-30 minutes.
- Nerfstudio install/check: 1-2 hours.
- First Splatfacto smoke: 1-2 hours with `MAX_NUM_ITERATIONS=3000`.
- First real 25-frame scene: 2-4 hours with `MAX_NUM_ITERATIONS=30000`.

## Notes

- Start with the sanity job. Do not burn a long GPU allocation until CUDA and `ns-train splatfacto --help` work.
- The first Splatfacto job trains only. Rendering held-out test views needs a prepared camera path/test split export, which should be added after the baseline training directory structure is known.
- Keep package installation in `.venvs/nerfstudio` so the Spack environment remains untouched.
