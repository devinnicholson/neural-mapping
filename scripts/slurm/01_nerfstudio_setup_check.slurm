#!/bin/bash

#SBATCH --job-name=u3dgs-ns-check
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
SPACK_SETUP="${SPACK_SETUP:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV="${SPACK_ENV:-CS-2050-gpu}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venvs/nerfstudio}"
INSTALL_NERFSTUDIO="${INSTALL_NERFSTUDIO:-0}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
FORCE_REINSTALL_TORCH="${FORCE_REINSTALL_TORCH:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SETUPTOOLS_SPEC="${SETUPTOOLS_SPEC:-setuptools<82}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy<2.0.0,>=1.26.0}"

cd "$PROJECT_DIR"
mkdir -p outputs/slurm "$PROJECT_DIR/.venvs"

. "$SPACK_SETUP"
spack env activate -p "$SPACK_ENV"

echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi
python --version

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv at $VENV_DIR"
  python -m venv "$VENV_DIR"
fi

. "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel "$SETUPTOOLS_SPEC"

if [ "$INSTALL_NERFSTUDIO" = "1" ]; then
  echo "Installing Nerfstudio stack into $VENV_DIR"
  python -m pip install ninja
  python -m pip install "$NUMPY_SPEC"
  if [ "$INSTALL_TORCH" = "1" ]; then
    echo "Installing PyTorch from $TORCH_INDEX_URL"
    if [ "$FORCE_REINSTALL_TORCH" = "1" ]; then
      python -m pip uninstall -y torch torchvision torchaudio || true
      python -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
    else
      python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
    fi
  fi
  python -m pip install nerfstudio --extra-index-url "$TORCH_INDEX_URL"
  python -m pip install gsplat --extra-index-url "$TORCH_INDEX_URL"
  python -m pip install "$SETUPTOOLS_SPEC" "$NUMPY_SPEC"
else
  echo "INSTALL_NERFSTUDIO=0, skipping pip installs."
  echo "Set INSTALL_NERFSTUDIO=1 when the cluster allows package installation."
fi

echo "Checking commands ..."
python - <<'PY'
import sys

try:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    else:
        print("torch cuda unavailable after install/check")
        sys.exit(3)
except Exception as exc:
    print("torch import failed:", repr(exc))
    sys.exit(3)

for module in ("nerfstudio", "gsplat"):
    try:
        imported = __import__(module)
        print(f"{module}: import ok")
    except Exception as exc:
        print(f"{module}: import failed: {exc!r}")

try:
    import numpy
    import setuptools
    print("numpy:", numpy.__version__)
    print("setuptools:", setuptools.__version__)
except Exception as exc:
    print("dependency version check failed:", repr(exc))
PY

if command -v ns-train >/dev/null 2>&1; then
  ns-train splatfacto --help >/tmp/u3dgs_ns_train_splatfacto_help.txt
  echo "ns-train splatfacto --help: ok"
else
  echo "ns-train not found in PATH."
  exit 2
fi

echo "Nerfstudio setup/check job complete."
