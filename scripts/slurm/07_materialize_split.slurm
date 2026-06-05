#!/bin/bash

#SBATCH --job-name=u3dgs-mat-split
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SPACK_SETUP="${SPACK_SETUP:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV="${SPACK_ENV:-CS-2050-gpu}"

SCENE_NAME="${SCENE_NAME:-poster_available}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$PROJECT_DIR/data/nerfstudio/$SCENE_NAME}"
SPLIT_JSON="${SPLIT_JSON:-$PROJECT_DIR/data/splits/$SCENE_NAME.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/data/nerfstudio_splits/$SCENE_NAME}"
BUDGETS="${BUDGETS:-25 50}"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm data/nerfstudio_splits

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Source data: $SOURCE_DATA_DIR"
echo "Split JSON: $SPLIT_JSON"
echo "Output root: $OUTPUT_ROOT"
echo "Budgets: $BUDGETS"

python scripts/materialize_nerfstudio_split.py \
  --source-dir "$SOURCE_DATA_DIR" \
  --split-json "$SPLIT_JSON" \
  --output-root "$OUTPUT_ROOT" \
  --budgets $BUDGETS

echo "Materialized split datasets:"
find "$OUTPUT_ROOT" -maxdepth 2 -type f -name "transforms.json" -print | sort

