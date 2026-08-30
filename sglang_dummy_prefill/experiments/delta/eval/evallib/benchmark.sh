#!/usr/bin/bash
# evallib/benchmark.sh — SGLang benchmark runner
# Source this file; do not execute directly.
#
# Requires (from config.sh):
#   SERVER_PORT, BENCH_TIMEOUT_EP16, BENCH_TIMEOUT_EP16_LIMITED,
#   BENCH_TIMEOUT_EP8, BENCH_TIMEOUT_PP4TP4, MODEL_NAME, BENCH_BACKEND, BENCH_DATASET,
#   BENCH_NUM_PROMPTS, BENCH_REQUEST_RATE, MINICONDA, CONDA_ENV
#   For random:   BENCH_RANDOM_INPUT_LEN, BENCH_RANDOM_OUTPUT_LEN, BENCH_RANDOM_RANGE_RATIO
#   For sharegpt: BENCH_SHAREGPT_CONTEXT_LEN, (optional) BENCH_SHAREGPT_OUTPUT_LEN
#   For gsm8k:    BENCH_GSM8K_CONTEXT_LEN,    (optional) BENCH_GSM8K_OUTPUT_LEN
# Requires (from cluster.sh):
#   HEAD

log_bench() { echo "$(date '+%Y-%m-%d %H:%M:%S') [bench] $*"; }

_build_dataset_args() {
    local dataset_args=()
    case "$BENCH_DATASET" in
        random|random-ids)
            dataset_args+=(
                --random-input-len "$BENCH_RANDOM_INPUT_LEN"
                --random-output-len "$BENCH_RANDOM_OUTPUT_LEN"
                --random-range-ratio "$BENCH_RANDOM_RANGE_RATIO"
            )
            ;;
        sharegpt)
            dataset_args+=(--sharegpt-context-len "$BENCH_SHAREGPT_CONTEXT_LEN")
            [[ -n "${BENCH_SHAREGPT_OUTPUT_LEN:-}" ]] && \
                dataset_args+=(--sharegpt-output-len "$BENCH_SHAREGPT_OUTPUT_LEN")
            ;;
        gsm8k)
            dataset_args+=(--sharegpt-context-len "$BENCH_GSM8K_CONTEXT_LEN")
            [[ -n "${BENCH_GSM8K_OUTPUT_LEN:-}" ]] && \
                dataset_args+=(--gsm8k-output-len "$BENCH_GSM8K_OUTPUT_LEN")
            ;;
    esac
    echo "${dataset_args[@]}"
}

_log_dataset_info() {
    case "$BENCH_DATASET" in
        random|random-ids)
            log_bench "  input=[${BENCH_RANDOM_INPUT_LEN}*${BENCH_RANDOM_RANGE_RATIO}, ${BENCH_RANDOM_INPUT_LEN}]"
            log_bench "  output=[${BENCH_RANDOM_OUTPUT_LEN}*${BENCH_RANDOM_RANGE_RATIO}, ${BENCH_RANDOM_OUTPUT_LEN}]"
            ;;
        sharegpt)
            log_bench "  context_len=${BENCH_SHAREGPT_CONTEXT_LEN}"
            [[ -n "${BENCH_SHAREGPT_OUTPUT_LEN:-}" ]] && \
                log_bench "  fixed_output_len=${BENCH_SHAREGPT_OUTPUT_LEN}"
            ;;
        gsm8k)
            log_bench "  context_len=${BENCH_GSM8K_CONTEXT_LEN}"
            [[ -n "${BENCH_GSM8K_OUTPUT_LEN:-}" ]] && \
                log_bench "  fixed_output_len=${BENCH_GSM8K_OUTPUT_LEN}"
            ;;
    esac
}

run_benchmark() {
    local result_file="$1"
    local cmd_file="$2"
    shift 2
    local extra_args=("$@")

    local bench_timeout
    case "${SERVER_PROFILE:-}" in
        ep16)         bench_timeout="${BENCH_TIMEOUT_EP16:-600}" ;;
        ep16_limited) bench_timeout="${BENCH_TIMEOUT_EP16_LIMITED:-1200}" ;;
        ep8)          bench_timeout="${BENCH_TIMEOUT_EP8:-900}" ;;
        pp4tp4)       bench_timeout="${BENCH_TIMEOUT_PP4TP4:-1500}" ;;
        *)            bench_timeout="${BENCH_TIMEOUT_PP4TP4:-1500}" ;;
    esac

    local dataset_args
    read -ra dataset_args <<< "$(_build_dataset_args)"

    local cmd=(
        python -m sglang.bench_serving
        --backend "$BENCH_BACKEND"
        --host "$HEAD"
        --port "$SERVER_PORT"
        --model "$MODEL_NAME"
        --dataset-name "$BENCH_DATASET"
        --num-prompts "$BENCH_NUM_PROMPTS"
        --request-rate "$BENCH_REQUEST_RATE"
        "${dataset_args[@]}"
        --output-file "$result_file"
        "${extra_args[@]}"
    )

    rm -f "$result_file"
    log_bench "Running benchmark:"
    log_bench "  host=${HEAD}:${SERVER_PORT}"
    log_bench "  dataset=${BENCH_DATASET}, ${BENCH_NUM_PROMPTS} prompts, ${BENCH_REQUEST_RATE} rps, timeout=${bench_timeout}s"
    _log_dataset_info

    {
        printf '# Benchmark command\n'
        printf '# Generated: %s\n\n' "$(date)"
        printf '%s' "${cmd[0]}"
        for arg in "${cmd[@]:1}"; do printf ' \\\n    %s' "$arg"; done
        printf '\n'
    } > "$cmd_file"

    if timeout "$bench_timeout" "${MINICONDA}/envs/${CONDA_ENV}/bin/python" "${cmd[@]:1}" 2>&1 | tee "${result_file%.json}.log"; then
        if [ -f "$result_file" ]; then
            log_bench "Benchmark complete. Result: $result_file"
            return 0
        else
            log_bench "ERROR: Benchmark ran but no output file produced."
            printf '{"error":"no_output_file"}\n' > "$result_file"
            return 1
        fi
    else
        local exit_code=$?
        if [ "$exit_code" -eq 124 ]; then
            log_bench "ERROR: Benchmark timed out after ${bench_timeout}s."
            printf '{"error":"timeout"}\n' > "$result_file"
        else
            log_bench "ERROR: Benchmark failed with exit code $exit_code."
            printf '{"error":"exit_%d"}\n' "$exit_code" > "$result_file"
        fi
        return 1
    fi
}
