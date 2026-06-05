#!/bin/bash

#SBATCH --job-name=u3dgs-splatfacto-25
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SPACK_SETUP="${SPACK_SETUP:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV="${SPACK_ENV:-CS-2050-gpu}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venvs/nerfstudio}"

SCENE_NAME="${SCENE_NAME:-replica_room0}"
BUDGET="${BUDGET:-25}"
NS_DATA_DIR="${NS_DATA_DIR:-}"
MAX_NUM_ITERATIONS="${MAX_NUM_ITERATIONS:-30000}"
DOWNSCALE_FACTOR="${DOWNSCALE_FACTOR:-1}"
VIS="${VIS:-tensorboard}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/outputs/runs/$SCENE_NAME/splatfacto/budget_$(printf '%03d' "$BUDGET")}"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm "$RUN_ROOT"

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"
. "$VENV_DIR/bin/activate"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Project: $PROJECT_DIR"
echo "Scene: $SCENE_NAME"
echo "Budget: $BUDGET"
echo "Downscale factor: $DOWNSCALE_FACTOR"
echo "Visualization backend: $VIS"
echo "Run root: $RUN_ROOT"
nvidia-smi

if [ -z "$NS_DATA_DIR" ]; then
  echo "NS_DATA_DIR is required."
  echo "Set it to a Nerfstudio-compatible dataset directory, for example:"
  echo "  sbatch --export=ALL,PROJECT_DIR=$PROJECT_DIR,NS_DATA_DIR=/path/to/ns_dataset scripts/slurm/02_splatfacto_first_run.slurm"
  exit 2
fi

if [ ! -d "$NS_DATA_DIR" ]; then
  echo "NS_DATA_DIR does not exist: $NS_DATA_DIR"
  exit 2
fi

echo "Checking Splatfacto CLI ..."
ns-train splatfacto --help >/tmp/u3dgs_splatfacto_help.txt

echo "Starting Splatfacto training ..."
ns-train splatfacto \
  --output-dir "$RUN_ROOT/train" \
  --max-num-iterations "$MAX_NUM_ITERATIONS" \
  --vis "$VIS" \
  --viewer.quit-on-train-completion True \
  nerfstudio-data \
  --data "$NS_DATA_DIR" \
  --downscale-factor "$DOWNSCALE_FACTOR"

echo "Training complete."
echo "Next: render held-out views with ns-render once the exact camera path/test split is prepared."
