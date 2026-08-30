#!/bin/bash
# DP-attention EP16 worker — mooncake-nccl, WITH recorder
# Usage: ./launch_worker_ep_record.sh <node_rank> <gating_profile> <log_file> <record_dir> [mem_frac]
set -euo pipefail

NODE_RANK=$1
GATING_PROFILE=$2
LOG_FILE=$3
RECORD_DIR=$4
MEM_FRAC=${5:-0.80}
MAX_RUNNING_REQS=${6:-}
CUDA_GRAPH_MAX_BS=${7:-}

eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

mkdir -p "$(dirname "$LOG_FILE")" "$RECORD_DIR"

export NCCL_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$RECORD_DIR"

find ~/coulson/sglang-fake-prefill/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 16 \
    --moe-a2a-backend mooncake-nccl \
    --enable-dp-attention \
    --dp-size 16 \
    --nnodes 8 \
    --node-rank "$NODE_RANK" \
    --dist-init-addr 10.0.0.1:25000 \
    --enable-fake-prefill \
    --profile-driven-gate-path "$GATING_PROFILE" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static "$MEM_FRAC" \
    ${MAX_RUNNING_REQS:+--max-running-requests $MAX_RUNNING_REQS} \
    ${CUDA_GRAPH_MAX_BS:+--cuda-graph-max-bs $CUDA_GRAPH_MAX_BS} \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --expert-distribution-recorder-mode stat \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
