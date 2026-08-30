#!/usr/bin/bash
# config_glm4air.sh — Model-specific config for GLM-4.5-Air
# Source this file; do not execute directly.
# Sources config.sh for shared cluster / network / benchmark settings.
#
# Usage: source experiments/sphere/eval/config_glm4air.sh

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$EVAL_DIR/config.sh"

# ── Model — GLM-4.5-Air (46 layers, 128 experts, 106B total / 12B active) ────
MODEL_PATH="zai-org/GLM-4.5-Air"
MODEL_NAME="zai-org/GLM-4.5-Air"
LOAD_FORMAT="dummy"

# ── Gate profiles ─────────────────────────────────────────────────────────────
GATE_PROFILES=(
    "${GATING_DIR}/glm45air_gating_profiles/gating_glm45air_sharegpt_200.parquet:sharegpt_regular"
    "${GATING_DIR}/glm45air_gating_profiles/balanced_output/balanced_glm45air_sharegpt_200.parquet:sharegpt_balanced"
    "${GATING_DIR}/glm45air_gating_profiles/gating_glm45air_gsm8k_200.parquet:gsm8k_regular"
    "${GATING_DIR}/glm45air_gating_profiles/balanced_output/balanced_glm45air_gsm8k_200.parquet:gsm8k_balanced"
)

# ── Customized server args (appended to every sglang.launch_server command) ───
CUSTOMIZED_ARGS=""
