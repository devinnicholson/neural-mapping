#!/bin/bash

#SBATCH --job-name=u3dgs-metric-smoke
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SPACK_SETUP="${SPACK_SETUP:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV="${SPACK_ENV:-CS-2050-gpu}"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"

python scripts/generate_splits.py \
  --frames examples/frames.txt \
  --budgets 4 6 \
  --val-count 2 \
  --test-count 2 \
  --scene example \
  --seed 7 \
  --output data/splits/example_split.json

python scripts/compute_uncertainty_metrics.py \
  --input examples/metric_input.json \
  --bad-threshold 0.5 \
  --output outputs/reports/example_metrics.json

echo "Metric smoke job complete."

