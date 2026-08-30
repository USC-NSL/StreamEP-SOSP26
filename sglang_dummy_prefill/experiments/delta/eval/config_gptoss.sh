#!/usr/bin/bash
# config_gptoss.sh — Model-specific config for gpt-oss-120b-bf16
# Source this file; do not execute directly.
# Sources config.sh for shared cluster / network / benchmark settings.
#
# Usage: source experiments/delta/eval/config_gptoss.sh

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EVAL_DIR/config.sh"

# ── Model — gpt-oss-120b-bf16 (36 layers, 128 experts, dummy weights) ────────
MODEL_PATH="lmsys/gpt-oss-120b-bf16"
MODEL_NAME="lmsys/gpt-oss-120b-bf16"
LOAD_FORMAT="dummy"

# ── Gate profiles ─────────────────────────────────────────────────────────────
GATE_PROFILES=(
    "${GATING_DIR}/gating_gptoss120b_sharegpt_200.parquet:sharegpt_regular"
    "${GATING_DIR}/gptosss_balanced_output/balanced_gptoss120b_sharegpt_200.parquet:sharegpt_balanced"
    "${GATING_DIR}/gating_math_gsm8k_200.parquet:gsm8k_regular"
    "${GATING_DIR}/gptosss_balanced_output/balanced_math_gsm8k_200.parquet:gsm8k_balanced"
)

# ── Customized server args (appended to every sglang.launch_server command) ───
CUSTOMIZED_ARGS="--moe-a2a-backend mooncake-nccl"
