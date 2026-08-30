#!/bin/bash
# DP-attention sglang server — 4xA100-40GB single node (Delta gpua080)
# TP=4, DP=4, dummy weights, 9-layer gpt-oss (1/4 of original 36)
#
# Prerequisites:
#   1. conda activate sglang
#   2. python ~/sglang/scripts/prepare_model.py
#
# Usage: ./launch_server.sh <log_file>
set -euo pipefail

LOG_FILE=$1

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate sglang
cd ~/sglang

mkdir -p "$(dirname "$LOG_FILE")"

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m sglang.launch_server \
    --model-path ~/gpt-oss-120b-bf16-mini \
    --load-format dummy \
    --tp-size 4 \
    --enable-dp-attention \
    --dp-size 4 \
    --ep-size 4 \
    --moe-a2a-backend mooncake-nccl \
    --enable-dp-lm-head \
    --nnodes 1 \
    --enable-fake-prefill \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static 0.80 \
    --max-running-requests 256 \
    --cuda-graph-max-bs 16 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
