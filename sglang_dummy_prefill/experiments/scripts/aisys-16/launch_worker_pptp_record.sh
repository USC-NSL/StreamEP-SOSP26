#!/bin/bash
# PP4×TP4 worker — WITH recorder
# Cluster: aisys-303, 4 nodes × 4 A6000 = 16 GPUs
# Usage: ./launch_worker_pptp_record.sh <node_rank> <dist_init_addr> <gating_profile> <log_file> <record_dir> [mem_frac] [cuda_graph_max_bs]
set -euo pipefail

NODE_RANK=$1
DIST_INIT_ADDR=$2
GATING_PROFILE=$3
LOG_FILE=$4
RECORD_DIR=$5
MEM_FRAC=${6:-0.70}
CUDA_GRAPH_MAX_BS=${7:-}

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ~/coulson/conda_envs
cd ~/coulson/sglang-fake-prefill

mkdir -p "$(dirname "$LOG_FILE")" "$RECORD_DIR"

# aisys-303 network: InfiniBand via ib0 / mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export SGLANG_LOCAL_IP_NIC=ib0
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$RECORD_DIR"

python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 4 \
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
    --expert-distribution-recorder-mode stat \
    --dist-timeout 1800 \
    --log-level warning 2>&1 | tee "$LOG_FILE"
