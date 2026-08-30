#!/usr/bin/env bash
# setup_node.sh — provision ONE node for the StreamInfer artifact.
#
#   bash experiment_utils/setup_node.sh
#
# Idempotent: creates the `streaminfer` conda env if missing, installs Python
# deps, applies the vLLM patch, builds disagmoe_c, and loads the gdrdrv kernel
# module. Run once on the head node AND on every worker node. Mirrors
# StreamInfer/readme.md but automated. Assumes system prerequisites from that
# readme are already present (CUDA, apt libnccl2/libnccl-dev, libzmq3-dev +
# cppzmq-dev, gdrcopy under /usr/local/gdrcopy, UCX).
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/throughput-itl/config.sh"
CONDA_ENV="${CONDA_ENV:-streaminfer}"
REPO_DIR="${REPO_DIR:-$ARTIFACT_ROOT/StreamInfer}"
log(){ echo "$(date '+%H:%M:%S') [setup $(hostname -s)] $*"; }

source "$MINICONDA/etc/profile.d/conda.sh"

# 1. conda env
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  log "creating conda env $CONDA_ENV (python 3.12.8) + torch + vllm ..."
  conda create -n "$CONDA_ENV" python=3.12.8 -y
  conda activate "$CONDA_ENV"
  pip install torch==2.6.0 torchvision torchaudio
  pip install vllm==0.8.2
else
  log "conda env $CONDA_ENV already exists"
  conda activate "$CONDA_ENV"
fi

# 2. Python deps
cd "$REPO_DIR"
log "pip install -r requirements.txt"
pip install -r requirements.txt
# StreamInfer is a plain directory in this repo (no longer a submodule) and its
# third_party/ headers ship with it, so there is nothing to fetch — just check.
MISSING=""
for d in third_party/cutlass/include third_party/cereal/include \
         third_party/NVTX/c/include third_party/pybind11/include; do
  [ -d "$d" ] || MISSING="$MISSING $d"
done
if [ -n "$MISSING" ]; then
  log "ERROR: missing third_party headers needed by setup.py:$MISSING"
  log "       re-clone the artifact repo (they are tracked in it)."
  exit 1
fi

# 3. vLLM patch (apply once; skip if already applied)
PATCH="$REPO_DIR/patches/vllm_0.8.2.patch"
SITE="$(python -c "import os, site; print(next(p for p in site.getsitepackages() if os.path.isdir(os.path.join(p,'vllm'))))")"
if git -C "$SITE" apply -R --check "$PATCH" >/dev/null 2>&1; then
  log "vLLM patch already applied"
elif git -C "$SITE" apply --check "$PATCH" >/dev/null 2>&1; then
  log "applying vLLM patch to $SITE"
  git -C "$SITE" apply "$PATCH"
else
  log "WARNING: vLLM patch neither applies nor is already applied — check vllm version"
fi

# 4. Build disagmoe_c (against apt NCCL)
log "building disagmoe_c (make pip) ..."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-/usr/include}"
export NCCL_LIBRARY_DIR="${NCCL_LIBRARY_DIR:-/usr/lib/x86_64-linux-gnu}"
export ZMQ_HOME="${ZMQ_HOME:-/usr}"
export GDRCOPY_HOME="${GDRCOPY_HOME:-/usr/local/gdrcopy}"
export C_INCLUDE_PATH="${C_INCLUDE_PATH:-/usr/include}"
export CPP_INCLUDE_PATH="${CPP_INCLUDE_PATH:-/usr/include}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$NCCL_LIBRARY_DIR:${LIBRARY_PATH:-}"
make pip

# 5. Kernel modules: nvidia_peermem (GPU-Direct RDMA for NCCL) + gdrdrv (GDRCopy).
#    nvidia_peermem is REQUIRED on every node — without it NCCL silently falls back
#    to a staged host path and every measured number is invalid.
if ! lsmod | grep -qw nvidia_peermem; then
  log "loading nvidia_peermem ..."
  sudo modprobe nvidia_peermem 2>/dev/null \
    || sudo modprobe nvidia-peermem 2>/dev/null \
    || log "WARNING: could not load nvidia_peermem — GPU-Direct RDMA will NOT be used"
fi
lsmod | grep -qw nvidia_peermem && log "nvidia_peermem loaded" \
  || log "WARNING: nvidia_peermem NOT loaded"

if ! lsmod | grep -qw gdrdrv; then
  if [ -d "$HOME/gdrcopy" ]; then
    log "loading gdrdrv kernel module ..."
    (cd "$HOME/gdrcopy" && sudo bash ./insmod.sh) || log "WARNING: could not load gdrdrv"
  else
    log "WARNING: gdrdrv not loaded and ~/gdrcopy not found"
  fi
fi

# 6. Verify
log "verifying imports ..."
python -c "import disagmoe_c; print('disagmoe_c OK')"
python -c "import vllm; print('vllm', vllm.__version__)"
log "NODE SETUP COMPLETE"
