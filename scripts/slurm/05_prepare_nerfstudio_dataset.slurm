#!/bin/bash

#SBATCH --job-name=u3dgs-prepare-data
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SPACK_SETUP="${SPACK_SETUP:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV="${SPACK_ENV:-CS-2050-gpu}"

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$PROJECT_DIR/data/nerfstudio/poster}"
FILTERED_DATA_DIR="${FILTERED_DATA_DIR:-$PROJECT_DIR/data/nerfstudio/poster_available}"
MIN_FRAMES="${MIN_FRAMES:-8}"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm data/nerfstudio

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Source data: $SOURCE_DATA_DIR"
echo "Filtered data: $FILTERED_DATA_DIR"

python scripts/filter_nerfstudio_transforms.py \
  --input-dir "$SOURCE_DATA_DIR" \
  --output-dir "$FILTERED_DATA_DIR" \
  --min-frames "$MIN_FRAMES"

echo "Filtered transforms summary:"
python - <<PY
import json
from pathlib import Path

root = Path("$FILTERED_DATA_DIR")
payload = json.loads((root / "transforms.json").read_text())
missing = json.loads((root / "missing_frames.json").read_text())
print("frames:", len(payload["frames"]))
print("missing:", missing["missing_frame_count"])
print("first frame:", payload["frames"][0]["file_path"])
PY

echo "Prepared dataset complete: $FILTERED_DATA_DIR"

