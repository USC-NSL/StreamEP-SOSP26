#!/bin/bash
# Qwen3-235B EP16 worker — AllGather+ReduceScatter, no recorder
# Usage: ./launch_worker_ep.sh <node_rank> <gating_profile> <log_file> [mem_frac] [max_running_reqs]
set -euo pipefail

NODE_RANK=$1
GATING_PROFILE=$2
LOG_FILE=$3
MEM_FRAC=${4:-0.80}
MAX_RUNNING_REQS=${5:-}

eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

mkdir -p "$(dirname "$LOG_FILE")"

export NCCL_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --load-format dummy \
    --tp-size 16 \
    --dp-size 16 \
    --ep-size 16 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --nnodes 8 \
    --node-rank "$NODE_RANK" \
    --dist-init-addr 10.0.0.1:25000 \
    --enable-fake-prefill \
    --profile-driven-gate-path "$GATING_PROFILE" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static "$MEM_FRAC" \
    --max-running-requests "$MAX_RUNNING_REQS" \
    --quantization fp8 \
    --disable-cuda-graph \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --watchdog-timeout 600 \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
