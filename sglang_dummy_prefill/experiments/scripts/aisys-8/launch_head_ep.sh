#!/bin/bash
# DP-attention EP8 head (rank 0) — mooncake-nccl, no recorder
# Cluster: aisys-303, 2 nodes × 4 A6000 = 8 GPUs
# Usage: ./launch_head_ep.sh <dist_init_addr> <gating_profile> <log_file> [mem_frac] [cuda_graph_max_bs]
set -euo pipefail

DIST_INIT_ADDR=$1
GATING_PROFILE=$2
LOG_FILE=$3
MEM_FRAC=${4:-0.70}
CUDA_GRAPH_MAX_BS=${5:-}

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ~/coulson/conda_envs
cd ~/coulson/sglang-fake-prefill

mkdir -p "$(dirname "$LOG_FILE")"

export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export SGLANG_LOCAL_IP_NIC=ib0
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 8 \
    --moe-a2a-backend mooncake-nccl \
    --enable-dp-attention \
    --dp-size 8 \
    --nnodes 2 \
    --node-rank 0 \
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
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
