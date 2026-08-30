#!/usr/bin/bash
# config.sh — Shared cluster / network / runtime / benchmark config for Delta eval
# Source this file; do not execute directly.
# Model-specific settings live in config_gptoss.sh.
#
# Usage: source experiments/delta/eval/config.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$REPO_DIR/experiments/delta"
GATING_DIR="$REPO_DIR/gating_profiles"
MINICONDA="$HOME/miniconda3"
CONDA_ENV="sglang"

# ── System identity ───────────────────────────────────────────────────────────
SYSTEM_NAME="sglang"

# ── Cluster — 4-node × 4-GPU A100-SXM4-40GB (Delta gpuA100x4) ───────────────
N_NODE=4
N_GPU_PER_NODE=4
WORLD_SIZE=16

# ── Common runtime ────────────────────────────────────────────────────────────
# ep16/pp4tp4: 0.65 (≥0.70 deadlocks with mooncake-nccl during CUDA graph capture)
# ep8: must be ≥0.85 (8-GPU sharding → ~30 GB weights/GPU for 120B model)
# Override per-profile in evallib/server.sh
MEM_FRAC=0.65
EP8_MEM_FRAC=0.85
SERVER_PORT=30000
DIST_INIT_PORT=25000
DIST_TIMEOUT=1800

# ── Network — HPE Slingshot (Delta) ───────────────────────────────────────────
HOST_IFNAME="hsn0"

# ── Server profiles ──────────────────────────────────────────────────────────
# Each profile defines parallelism strategy and optional server flags.
#   _build_server_cmd in evallib/server.sh dispatches on SERVER_PROFILE.
#
#   ep16          — tp=16, dp=16, ep=16, DP-attention
#   ep16_limited  — same as ep16 + --max-running-requests cap
#   pp4tp4        — pp=4, tp=4, no EP/DP-attention (pure pipeline+tensor parallel)
#   ep8           — tp=8, dp=8, ep=8, same flags as ep16 (2 nodes)

EP16_LIMITED_MAX_RUNNING_REQS=256

# ── EP8 — 2-node subset (head + first worker) ────────────────────────────────
EP8_N_NODE=2
EP8_WORLD_SIZE=8

# ── Benchmark — common ────────────────────────────────────────────────────────
BENCH_BACKEND="sglang"
BENCH_DATASET=${BENCH_DATASET:-"sharegpt"}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-10000}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-2000}
BENCH_TIMEOUT_EP16=600
BENCH_TIMEOUT_EP16_LIMITED=1200
BENCH_TIMEOUT_EP8=900
BENCH_TIMEOUT_PP4TP4=1500

# ── Benchmark — random dataset ───────────────────────────────────────────────
BENCH_RANDOM_INPUT_LEN=512
BENCH_RANDOM_OUTPUT_LEN=512
BENCH_RANDOM_RANGE_RATIO=0.5

# ── Benchmark — sharegpt dataset ─────────────────────────────────────────────
BENCH_SHAREGPT_CONTEXT_LEN=${BENCH_SHAREGPT_CONTEXT_LEN:-2048}
# BENCH_SHAREGPT_OUTPUT_LEN=       # unset → use natural completion length

# ── Benchmark — gsm8k dataset ────────────────────────────────────────────────
BENCH_GSM8K_CONTEXT_LEN=${BENCH_GSM8K_CONTEXT_LEN:-2048}
# BENCH_GSM8K_OUTPUT_LEN=          # unset → use natural answer length

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=300
