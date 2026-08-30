#!/usr/bin/bash
# config.sh — Shared cluster / network / runtime / benchmark config for Sphere eval
# Source this file; do not execute directly.
# Model-specific settings live in config_gptoss.sh / config_glm4air.sh.
#
# Usage: source experiments/sphere/eval/config.sh

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$REPO_DIR/experiments/sphere"
GATING_DIR="$REPO_DIR/gating_profiles"
MINICONDA="$HOME/miniconda3"
CONDA_ENV="sglang-fp"

# ── System identity ───────────────────────────────────────────────────────────
SYSTEM_NAME="sglang"

# ── Cluster — 8-node × 2-GPU L40S (Sphere) ───────────────────────────────────
N_NODE=8
N_GPU_PER_NODE=2
WORLD_SIZE=16

# ── Default node names (override via HEAD / WORKERS env vars) ─────────────────
DEFAULT_WORKERS="sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9"

# ── Common runtime ────────────────────────────────────────────────────────────
MEM_FRAC=${MEM_FRAC:-0.77}
SERVER_PORT=30000
DIST_INIT_PORT=25000
DIST_TIMEOUT=1800

# ── Network — InfiniBand (Sphere) ────────────────────────────────────────────
HOST_IFNAME="ens1f1np1"
NCCL_IB_HCA="mlx5_1"
NCCL_IB_GID_INDEX=3

# ── Server profiles ──────────────────────────────────────────────────────────
# Each profile defines parallelism strategy and optional server flags.
# _build_server_cmd in evallib/server.sh dispatches on SERVER_PROFILE.
#
#   ep16          — tp=16, dp=16, ep=16, DP-attention, mooncake-nccl a2a (8 nodes)
#   ep16_limited  — same as ep16 + --max-running-requests cap (8 nodes)
#   ep8           — tp=8, dp=8, ep=8, same flags as ep16 (4 nodes)
#   pp8tp2        — pp=8, tp=2, no EP/DP-attention (8 nodes)

EP16_LIMITED_MAX_RUNNING_REQS=256

# ── EP8 — 4-node subset (head + first 3 workers) ────────────────────────────
EP8_N_NODE=4
EP8_WORLD_SIZE=8

# ── Benchmark — common ────────────────────────────────────────────────────────
BENCH_BACKEND="sglang"
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-20000}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-2000}
BENCH_TIMEOUT=1500
BENCH_NPY_CONTEXT_LEN=${BENCH_NPY_CONTEXT_LEN:-2048}
BENCH_DISABLE_STREAM=${BENCH_DISABLE_STREAM:-0}

# ── Benchmark — .npy dataset paths (aligned with AsyncMoE) ──────────────────
DATASETS_DIR="${REPO_DIR}/datasets"
BENCH_DATASET_PATHS=(
    "${DATASETS_DIR}/sharegpt_lengths.npy:sharegpt"
    "${DATASETS_DIR}/gsm8k_lengths.npy:gsm8k"
)

# ── Server startup timeout ────────────────────────────────────────────────────
SERVER_READY_TIMEOUT=600
