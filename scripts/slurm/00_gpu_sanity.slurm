#!/bin/bash

#SBATCH --job-name=u3dgs-gpu-sanity
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SPACK_SETUP="${SPACK_SETUP:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV="${SPACK_ENV:-CS-2050-gpu}"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm

echo "Project: $PROJECT_DIR"
echo "Host: $(hostname)"
echo "Date: $(date)"

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"

echo "GPU check ..."
nvidia-smi

echo "CUDA compiler check ..."
which nvcc || true
nvcc --version || true

echo "Python check ..."
which python
python --version

echo "Python CUDA package check ..."
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch cuda device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", repr(exc))
PY

echo "Project lightweight tests ..."
python -m unittest discover -s tests

echo "GPU sanity job complete."

