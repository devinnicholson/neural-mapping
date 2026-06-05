#!/bin/bash

#SBATCH --job-name=u3dgs-eval-latest
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SPACK_SETUP="${SPACK_SETUP:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV="${SPACK_ENV:-CS-2050-gpu}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venvs/nerfstudio}"

SCENE_NAME="${SCENE_NAME:-poster_available}"
BUDGET="${BUDGET:-25}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/outputs/runs/$SCENE_NAME/splatfacto/budget_$(printf '%03d' "$BUDGET")}"
CONFIG_PATH="${CONFIG_PATH:-}"
RENDER_OUTPUTS="${RENDER_OUTPUTS:-1}"
ALLOW_LOCAL_CHECKPOINT_PICKLE="${ALLOW_LOCAL_CHECKPOINT_PICKLE:-1}"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm "$RUN_ROOT/metrics" "$RUN_ROOT/renders"

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"
. "$VENV_DIR/bin/activate"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Project: $PROJECT_DIR"
echo "Scene: $SCENE_NAME"
echo "Budget: $BUDGET"
echo "Run root: $RUN_ROOT"
nvidia-smi

if [ "$ALLOW_LOCAL_CHECKPOINT_PICKLE" = "1" ]; then
  # PyTorch 2.6+ defaults torch.load(..., weights_only=True). Nerfstudio
  # checkpoints include non-tensor training metadata, so locally generated
  # checkpoints need the old loading behavior for ns-eval.
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  echo "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"
fi

if [ -z "$CONFIG_PATH" ]; then
  CONFIG_PATH="$(find "$RUN_ROOT/train" -path "*/config.yml" -type f | sort | tail -n 1)"
fi

if [ -z "$CONFIG_PATH" ] || [ ! -f "$CONFIG_PATH" ]; then
  echo "Could not find config.yml under $RUN_ROOT/train"
  echo "Set CONFIG_PATH explicitly if needed."
  exit 2
fi

echo "Evaluating config: $CONFIG_PATH"
OUTPUT_PATH="$RUN_ROOT/metrics/ns_eval.json"

if [ "$RENDER_OUTPUTS" = "1" ]; then
  ns-eval \
    --load-config "$CONFIG_PATH" \
    --output-path "$OUTPUT_PATH" \
    --render-output-path "$RUN_ROOT/renders/eval"
else
  ns-eval \
    --load-config "$CONFIG_PATH" \
    --output-path "$OUTPUT_PATH"
fi

echo "Evaluation complete: $OUTPUT_PATH"
cat "$OUTPUT_PATH"
