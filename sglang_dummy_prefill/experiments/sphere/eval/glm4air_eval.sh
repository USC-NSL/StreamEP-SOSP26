#!/usr/bin/bash
# glm4air_eval.sh — SGLang multi-baseline evaluation for GLM-4.5-Air, Sphere cluster
#
# Usage:
#   bash experiments/sphere/eval/glm4air_eval.sh <RESULTS_DIR> [OPTIONS]
#
#   RESULTS_DIR  required; a parent directory that holds one sub-dir per run.
#                Example: /scratch/myrun/results
#
#   Options:
#     --list          Print numbered experiment list and exit
#     --only FILTER   Run only experiments matching FILTER (comma-separated
#                     indices or name substrings). Examples:
#                       --only 1,5,9            # by index
#                       --only ep16-sharegpt    # name substring
#                       --only pp8tp2           # all pp8tp2 experiments
#                       --only sharegpt_regular # all sharegpt_regular across profiles
#
# Run directory naming: <RESULTS_DIR>/<system>_<server_profile>-<dataset_label>/
#   e.g.  sglang_ep16-sharegpt_regular/
#         sglang_pp8tp2-gsm8k_balanced/
#
# Prerequisites:
#   - 8 nodes × 2 L40S GPUs available via SSH
#   - conda env 'sglang-fp' installed on all nodes
#   - SSH access to compute nodes + tmux available
#   - Model config/tokenizer at $MODEL_PATH (weights not needed for dummy)
#   - Gate profile parquets in place (see EXPERIMENT MATRIX below)

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load model config (sources shared config.sh internally) and function libs ─
source "$EVAL_DIR/config_glm4air.sh"
source "$EVAL_DIR/evallib/cluster.sh"
source "$EVAL_DIR/evallib/server.sh"
source "$EVAL_DIR/evallib/benchmark.sh"

# ── Argument parsing ──────────────────────────────────────────────────────────
ONLY_FILTER=""
LIST_ONLY=0
RESULTS_DIR=""
DISABLE_STREAM=${BENCH_DISABLE_STREAM:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY_FILTER="${2:?ERROR: --only requires a comma-separated list}"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        --disable-stream) DISABLE_STREAM=1; shift ;;
        -*) echo "ERROR: Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$RESULTS_DIR" ]]; then RESULTS_DIR="$1"; shift
            else echo "ERROR: Unexpected argument: $1" >&2; exit 1; fi
            ;;
    esac
done

if [[ "$LIST_ONLY" -eq 0 ]] && [[ -z "$RESULTS_DIR" ]]; then
    echo "ERROR: RESULTS_DIR is required (e.g. /path/to/results)" >&2
    echo "Usage: $0 <RESULTS_DIR> [--list] [--only FILTER] [--disable-stream]" >&2
    exit 1
fi

export BENCH_DISABLE_STREAM="$DISABLE_STREAM"

# ── Server profiles to evaluate ──────────────────────────────────────────────
SERVER_PROFILES=( ep16 ep16_limited pp8tp2 ep8 )

# ── Full experiment matrix: SERVER_PROFILES × GATE_PROFILES ──────────────────
EXPERIMENTS=()
for _sp in "${SERVER_PROFILES[@]}"; do
    for _gp in "${GATE_PROFILES[@]}"; do
        EXPERIMENTS+=("${_sp}:${_gp}")
    done
done

MAX_RETRIES=2
MEM_FRAC_STEP=0.05

# ─────────────────────────────────────────────────────────────────────────────
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [main] $*"; }

resolve_dataset_path() {
    local label="$1"
    for entry in "${BENCH_DATASET_PATHS[@]}"; do
        IFS=: read -r path tag <<< "$entry"
        if [[ "$label" == *"$tag"* ]]; then
            echo "$path"
            return 0
        fi
    done
    echo ""
    return 1
}

# ── Experiment filter ─────────────────────────────────────────────────────────
should_run_experiment() {
    local idx="$1" label="$2"
    [[ -z "$ONLY_FILTER" ]] && return 0
    IFS=',' read -ra FILTERS <<< "$ONLY_FILTER"
    for f in "${FILTERS[@]}"; do
        f="${f#"${f%%[![:space:]]*}"}"
        f="${f%"${f##*[![:space:]]}"}"
        if [[ "$f" =~ ^[0-9]+$ ]]; then
            [[ "$f" -eq "$idx" ]] && return 0
        else
            [[ "$label" == *"$f"* ]] && return 0
        fi
    done
    return 1
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    echo "Available experiments:"
    _i=0
    for exp_entry in "${EXPERIMENTS[@]}"; do
        IFS=: read -r _sp _gp _ds <<< "$exp_entry"
        _i=$((_i + 1))
        printf "  %2d. %s_%s-%s\n" "$_i" "$SYSTEM_NAME" "$_sp" "$_ds"
    done
    exit 0
fi

mkdir -p "$RESULTS_DIR"

# ── Discover nodes ────────────────────────────────────────────────────────────
log "Discovering cluster nodes..."
discover_nodes || exit 1

log "Evaluation starting"
log "  System      : $SYSTEM_NAME"
log "  Model       : $MODEL_NAME"
log "  Results dir : $RESULTS_DIR"
log "  Head        : $HEAD ($HEAD_IP)"
log "  Workers     : ${WORKERS[*]}"
log "  Server profiles : ${SERVER_PROFILES[*]}"
log "  Gate profiles   : ${#GATE_PROFILES[@]}"
log "  Experiments     : ${#EXPERIMENTS[@]} (${#SERVER_PROFILES[@]} × ${#GATE_PROFILES[@]}), up to $MAX_RETRIES retries each"
log "  Initial MEM_FRAC: $MEM_FRAC"
log "  Benchmark stream: $([[ "$BENCH_DISABLE_STREAM" == "1" ]] && echo disabled || echo enabled)"
[[ -n "$ONLY_FILTER" ]] && log "  Filter          : --only $ONLY_FILTER"

EXP_NUM=0
TOTAL=${#EXPERIMENTS[@]}

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS=: read -r server_profile gate_profile dataset <<< "$exp_entry"
    EXP_NUM=$((EXP_NUM + 1))

    BENCH_DATASET_PATH=$(resolve_dataset_path "$dataset")
    if [[ -z "$BENCH_DATASET_PATH" ]]; then
        log "ERROR: No .npy dataset matches label '$dataset' in BENCH_DATASET_PATHS"
        continue
    fi
    export BENCH_DATASET_PATH

    run_name="${SYSTEM_NAME}_${server_profile}-${dataset}"

    if ! should_run_experiment "$EXP_NUM" "$run_name"; then
        log "[$EXP_NUM/$TOTAL] SKIP (--only filter): $run_name"
        continue
    fi

    run_dir="$RESULTS_DIR/$run_name"
    mkdir -p "$run_dir"

    log "================================================================"
    log "[$EXP_NUM/$TOTAL] $run_name  (server=$server_profile)"
    log "================================================================"

    if [ ! -f "$gate_profile" ]; then
        log "SKIP: profile not found: $gate_profile"
        printf '{"error":"profile_not_found","path":"%s"}\n' "$gate_profile" \
            > "$run_dir/result.json"
        continue
    fi

    SUCCESS=0
    server_log_dir="$run_dir/logs"
    server_cmd="$run_dir/server_cmd.sh"
    bench_result="$run_dir/bench_result.json"
    bench_cmd="$run_dir/bench_cmd.sh"

    for attempt in $(seq 1 "$MAX_RETRIES"); do
        log "Attempt $attempt/$MAX_RETRIES (MEM_FRAC=$MEM_FRAC)..."

        kill_server
        sleep 5

        if [ -d "$server_log_dir" ]; then
            mv "$server_log_dir" "${server_log_dir}_attempt$((attempt - 1))"
            log "Previous logs preserved as logs_attempt$((attempt - 1))/"
        fi
        launch_server "$server_profile" "$gate_profile" "$server_log_dir" "$server_cmd"

        if wait_for_server; then
            if run_benchmark "$bench_result" "$bench_cmd"; then
                cp "$bench_result" "$run_dir/result.json"
                SUCCESS=1
                break
            else
                log "Benchmark failed on attempt $attempt."
            fi
        else
            if is_oom "$server_log_dir"; then
                new_frac=$(awk "BEGIN {printf \"%.2f\", $MEM_FRAC - $MEM_FRAC_STEP}")
                log "OOM detected — reducing MEM_FRAC: $MEM_FRAC -> $new_frac"
                MEM_FRAC="$new_frac"
            else
                log "Server failed (non-OOM). See logs in: $server_log_dir/"
            fi
        fi

        kill_server
        sleep 10
    done

    if [ "$SUCCESS" -eq 0 ]; then
        log "FAILED: $run_name — all $MAX_RETRIES attempts unsuccessful."
    else
        log "SUCCESS: $run_name → $run_dir/result.json"
    fi
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill_server

log "================================================================"
log "All $TOTAL experiments done. Results in: $RESULTS_DIR"
log "================================================================"
