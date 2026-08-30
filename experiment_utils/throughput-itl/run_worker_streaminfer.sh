#!/usr/bin/env bash
# run_worker_streaminfer.sh — ONE command on each WORKER node for the StreamInfer run.
#
#   bash experiment_utils/throughput-itl/run_worker_streaminfer.sh
#
# Joins the Ray head started by run_head_streaminfer.sh, then idles so this node's
# GPUs are available. Leave running for the whole run; Ctrl-C to stop.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
CONDA_ENV="${CONDA_ENV:-streaminfer}"
REPO_DIR="${REPO_DIR:-$ARTIFACT_ROOT/StreamInfer}"
RAY_PORT="${RAY_PORT:-6379}"
PYTHON_BIN="$MINICONDA/envs/$CONDA_ENV/bin/python"
RAY_BIN="$MINICONDA/envs/$CONDA_ENV/bin/ray"
source "$MINICONDA/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
# disagmoe_c.so links libtorch but its RPATH misses torch's lib dir; add it so the raw
# `import disagmoe_c` (server.py imports it before the torch-loading disagmoe pkg) resolves.
export LD_LIBRARY_PATH="$("$PYTHON_BIN" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MY_IP="$(ip -o -4 addr show "$HOST_IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
MY_IP="${MY_IP:-$(hostname -I | awk '{print $1}')}"
echo "[worker/streaminfer] host=$(hostname -s) ip=$MY_IP  ->  Ray head $HEAD_IP:$RAY_PORT  (env=$CONDA_ENV)"
"$PYTHON_BIN" -c "import disagmoe_c" 2>/dev/null \
  || { echo "[worker] ERROR: 'import disagmoe_c' failed in env '$CONDA_ENV' — build StreamInfer here first."; exit 1; }

"$RAY_BIN" stop >/dev/null 2>&1 || true; sleep 2
echo "[worker/streaminfer] joining Ray head (retries until head is up)..."
until "$RAY_BIN" start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$MY_IP" \
        --disable-usage-stats >/tmp/ray_worker_join.log 2>&1; do
  echo "[worker]   head not ready yet; retry in 5s ($(tail -n1 /tmp/ray_worker_join.log 2>/dev/null))"; sleep 5
done
echo "[worker/streaminfer] JOINED. This node contributes $(nvidia-smi -L | wc -l) GPU(s). Ctrl-C to stop."
cleanup(){ echo; echo "[worker] stopping Ray..."; "$RAY_BIN" stop >/dev/null 2>&1 || true; exit 0; }
trap cleanup INT TERM
while true; do sleep 3600; done
