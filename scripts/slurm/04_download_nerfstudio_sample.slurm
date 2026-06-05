#!/bin/bash

#SBATCH --job-name=u3dgs-download-sample
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
CAPTURE_NAME="${CAPTURE_NAME:-poster}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-auto}"
DATA_DIR="$PROJECT_DIR/data/nerfstudio/$CAPTURE_NAME"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm data/nerfstudio

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"
. "$VENV_DIR/bin/activate"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Project: $PROJECT_DIR"
echo "Capture: $CAPTURE_NAME"
echo "Expected data dir: $DATA_DIR"

download_with_nerfstudio() {
  if ! command -v ns-download-data >/dev/null 2>&1; then
    echo "ns-download-data not found. Run scripts/slurm/01_nerfstudio_setup_check.slurm first."
    return 2
  fi
  ns-download-data nerfstudio --capture-name="$CAPTURE_NAME"
}

download_with_huggingface() {
  echo "Downloading from Hugging Face mirror ..."
  python -m pip install --upgrade "huggingface_hub[cli,hf_xet]"
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nerfstudioteam/datasets",
    repo_type="dataset",
    local_dir="data/nerfstudio",
    allow_patterns=["${CAPTURE_NAME}/**"],
)
PY
}

if [ "$DOWNLOAD_SOURCE" = "nerfstudio" ]; then
  download_with_nerfstudio
elif [ "$DOWNLOAD_SOURCE" = "huggingface" ]; then
  download_with_huggingface
elif [ "$DOWNLOAD_SOURCE" = "auto" ]; then
  if ! download_with_nerfstudio; then
    echo "Nerfstudio/gdown download failed; falling back to Hugging Face."
    download_with_huggingface
  fi
else
  echo "Unknown DOWNLOAD_SOURCE=$DOWNLOAD_SOURCE. Use auto, nerfstudio, or huggingface."
  exit 2
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "Download command completed, but expected directory is missing: $DATA_DIR"
  echo "Listing data directory for debugging:"
  find data -maxdepth 3 -type d | sort
  exit 2
fi

echo "Downloaded dataset directory:"
find "$DATA_DIR" -maxdepth 2 -type f | head -n 40

echo "Nerfstudio sample download complete: $DATA_DIR"
