#!/bin/bash
# EP16 worker (rank 1-3) — Delta 4×A100-SXM4-40GB × 4 nodes
# Usage: ./launch_worker_ep.sh <node_rank> <dist_init_addr> <log_file> [mem_frac]
set -euo pipefail

NODE_RANK=$1
DIST_INIT_ADDR=$2
LOG_FILE=$3
MEM_FRAC=${4:-0.80}

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate sglang
cd ~/sglang

mkdir -p "$(dirname "$LOG_FILE")"

# Delta Slingshot network (hsn0, no InfiniBand)
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN
export SGLANG_LOCAL_IP_NIC=hsn0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sglang.launch_server \
    --model-path /projects/bgro/spark36/models/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 16 \
    --dp-size 16 \
    --ep-size 16 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend mooncake-nccl \
    --nnodes 4 \
    --node-rank "$NODE_RANK" \
    --dist-init-addr "$DIST_INIT_ADDR" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static "$MEM_FRAC" \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
