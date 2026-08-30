#!/bin/bash
# PP4×TP2 worker — WITH recorder
# Cluster: CARC SLURM, 4 nodes × 2 A100-80GB-PCIe = 8 GPUs
# Usage: ./launch_worker_pptp_record.sh <node_rank> <dist_init_addr> <gating_profile> <log_file> <record_dir> [mem_frac] [cuda_graph_max_bs]
set -euo pipefail

NODE_RANK=$1
DIST_INIT_ADDR=$2
GATING_PROFILE=$3
LOG_FILE=$4
RECORD_DIR=$5
MEM_FRAC=${6:-0.85}
CUDA_GRAPH_MAX_BS=${7:-}

# --- Environment setup (CARC SLURM) ---
source /etc/profile.d/modules.sh
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda
eval "$(/home1/yizhuoli/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
cd /home1/yizhuoli/sglang-fake-prefill

export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
export PYTHONPATH=/home1/yizhuoli/sglang-fake-prefill/python:${PYTHONPATH:-}

mkdir -p "$(dirname "$LOG_FILE")" "$RECORD_DIR"

# CARC SLURM network: Native InfiniBand via ib0 / mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export SGLANG_LOCAL_IP_NIC=ib0
export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$RECORD_DIR"

python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 2 \
    --pp-size 4 \
    --nnodes 4 \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --enable-fake-prefill \
    --profile-driven-gate-path "$GATING_PROFILE" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static "$MEM_FRAC" \
    ${CUDA_GRAPH_MAX_BS:+--cuda-graph-max-bs $CUDA_GRAPH_MAX_BS} \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --disable-custom-all-reduce \
    --expert-distribution-recorder-mode stat \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
