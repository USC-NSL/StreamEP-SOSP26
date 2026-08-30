#!/usr/bin/bash
# evallib/server.sh — SGLang server lifecycle helpers for Delta eval
# Source this file; do not execute directly.
#
# Requires (from config.sh):
#   REPO_DIR, MODEL_PATH, LOAD_FORMAT, MEM_FRAC, MINICONDA, CONDA_ENV
#   N_NODE, WORLD_SIZE, EP8_N_NODE, EP8_WORLD_SIZE, EP16_LIMITED_MAX_RUNNING_REQS
#   SERVER_PORT, SERVER_READY_TIMEOUT, DIST_TIMEOUT, HOST_IFNAME
# Requires (from cluster.sh):
#   HEAD, WORKERS, ALL_NODES, HEAD_IP, DIST_INIT_ADDR


log_server() { echo "$(date '+%Y-%m-%d %H:%M:%S') [server] $*"; }

_profile_nnodes() {
    case "$1" in
        ep8) echo "${EP8_N_NODE}" ;;
        *)   echo "${N_NODE}" ;;
    esac
}

# _build_server_cmd <server_profile> <node_rank> <gate_profile> <log_file>
_build_server_cmd() {
    local server_profile="$1"
    local node_rank="$2"
    local gate_profile="$3"
    local log_file="$4"

    local cmd="eval \"\$(${MINICONDA}/bin/conda shell.bash hook)\" && conda activate ${CONDA_ENV} && cd ${REPO_DIR} && mkdir -p \$(dirname ${log_file})"
    local apply_customized_args=0

    cmd+=" && export NCCL_SOCKET_IFNAME=${HOST_IFNAME}"
    cmd+=" && export GLOO_SOCKET_IFNAME=${HOST_IFNAME}"
    cmd+=" && export NCCL_DEBUG=WARN"
    cmd+=" && export SGLANG_LOCAL_IP_NIC=${HOST_IFNAME}"
    cmd+=" && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"


    local nnodes
    nnodes=$(_profile_nnodes "$server_profile")

    cmd+=" && python -m sglang.launch_server"
    cmd+=" --model-path ${MODEL_PATH}"
    cmd+=" --load-format ${LOAD_FORMAT}"
    cmd+=" --nnodes ${nnodes}"
    cmd+=" --node-rank ${node_rank}"
    cmd+=" --dist-init-addr ${DIST_INIT_ADDR}"
    cmd+=" --enable-fake-prefill"
    cmd+=" --disable-radix-cache"
    cmd+=" --chunked-prefill-size -1"
    cmd+=" --trust-remote-code"
    cmd+=" --moe-runner-backend triton"
    cmd+=" --dist-timeout ${DIST_TIMEOUT}"
    cmd+=" --log-level-http warning"
    cmd+=" --log-level warning"

    case "$server_profile" in
        ep16)
            apply_customized_args=1
            cmd+=" --mem-fraction-static ${MEM_FRAC}"
            cmd+=" --tp-size ${WORLD_SIZE}"
            cmd+=" --dp-size ${WORLD_SIZE}"
            cmd+=" --ep-size ${WORLD_SIZE}"
            cmd+=" --enable-dp-attention"
            cmd+=" --enable-dp-lm-head"
            cmd+=" --disable-custom-all-reduce"
            ;;
        ep16_limited)
            apply_customized_args=1
            cmd+=" --mem-fraction-static ${MEM_FRAC}"
            cmd+=" --tp-size ${WORLD_SIZE}"
            cmd+=" --dp-size ${WORLD_SIZE}"
            cmd+=" --ep-size ${WORLD_SIZE}"
            cmd+=" --enable-dp-attention"
            cmd+=" --enable-dp-lm-head"
            cmd+=" --max-running-requests ${EP16_LIMITED_MAX_RUNNING_REQS}"
            cmd+=" --disable-custom-all-reduce"
            ;;
        ep8)
            apply_customized_args=1
            cmd+=" --mem-fraction-static ${EP8_MEM_FRAC:-0.85}"
            cmd+=" --tp-size ${EP8_WORLD_SIZE}"
            cmd+=" --dp-size ${EP8_WORLD_SIZE}"
            cmd+=" --ep-size ${EP8_WORLD_SIZE}"
            cmd+=" --enable-dp-attention"
            cmd+=" --enable-dp-lm-head"
            cmd+=" --disable-custom-all-reduce"
            ;;
        pp4tp4)
            cmd+=" --mem-fraction-static ${MEM_FRAC}"
            cmd+=" --tp-size 4"
            cmd+=" --pp-size 4"
            cmd+=" --disable-custom-all-reduce"
            ;;
        *)
            log_server "FATAL: unknown server_profile='$server_profile'"
            return 1
            ;;
    esac

    if [ "$node_rank" -eq 0 ]; then
        cmd+=" --host 0.0.0.0"
    fi

    if [ -n "$gate_profile" ]; then
        cmd+=" --profile-driven-gate-path ${gate_profile}"
    fi

    if [ "$apply_customized_args" -eq 1 ] && [ -n "${CUSTOMIZED_ARGS:-}" ]; then
        cmd+=" ${CUSTOMIZED_ARGS}"
    fi

    cmd+=" 2>&1"
    echo "$cmd"
}

# launch_server <server_profile> <gate_profile> <log_dir> <cmd_file>
launch_server() {
    local server_profile="$1"
    local gate_profile="$2"
    local log_dir="$3"
    local cmd_file="$4"

    local nnodes
    nnodes=$(_profile_nnodes "$server_profile")
    local n_workers=$((nnodes - 1))

    log_server "Launching $server_profile | profile=$(basename "${gate_profile:-none}") | mem_frac=$MEM_FRAC | nnodes=$nnodes"
    mkdir -p "$log_dir"

    {
        printf '# Server launch commands\n'
        printf '# Generated: %s\n' "$(date)"
        printf '# server_profile: %s\n' "$server_profile"
        printf '# nnodes: %d\n' "$nnodes"
        printf '# mem_frac: %s\n' "$MEM_FRAC"
        printf '# gate_profile: %s\n' "${gate_profile:-none}"
        printf '# dist_init_addr: %s\n\n' "$DIST_INIT_ADDR"

        printf '# Head (rank 0) on %s:\n' "$HEAD"
        printf 'ssh %s '\''%s'\''\n\n' "$HEAD" \
            "$(_build_server_cmd "$server_profile" 0 "$gate_profile" "$log_dir/server_head.log")"

        for i in $(seq 0 $((n_workers - 1))); do
            local rank=$((i + 1))
            printf '# Worker rank %d on %s:\n' "$rank" "${WORKERS[$i]}"
            printf 'ssh %s '\''%s'\''\n\n' "${WORKERS[$i]}" \
                "$(_build_server_cmd "$server_profile" "$rank" "$gate_profile" "$log_dir/server_w${rank}.log")"
        done
    } > "$cmd_file"

    local head_cmd
    head_cmd=$(_build_server_cmd "$server_profile" 0 "$gate_profile" "$log_dir/server_head.log")
    tmux new-session -d -s sglang-head "ssh $HEAD '${head_cmd}' 2>&1 | tee $log_dir/server_head.log"

    sleep 5

    for i in $(seq 0 $((n_workers - 1))); do
        local w="${WORKERS[$i]}"
        local rank=$((i + 1))
        local sess="sglang-w$((i + 1))"
        local worker_cmd
        worker_cmd=$(_build_server_cmd "$server_profile" "$rank" "$gate_profile" "$log_dir/server_w${rank}.log")
        tmux new-session -d -s "$sess" "ssh $w '${worker_cmd}' 2>&1 | tee $log_dir/server_w${rank}.log"
    done

    log_server "Head + ${n_workers} workers launched (command saved to $(basename "$cmd_file"))"
}

# wait_for_server
#   Polls the health endpoint until server is ready or timeout.
#   Returns 0 on success, 1 on failure.
wait_for_server() {
    local elapsed=0

    log_server "Waiting for server health on ${HEAD}:${SERVER_PORT} (timeout ${SERVER_READY_TIMEOUT}s)..."
    while [ "$elapsed" -lt "$SERVER_READY_TIMEOUT" ]; do
        if curl -sf "http://${HEAD}:${SERVER_PORT}/health" > /dev/null 2>&1; then
            log_server "Server ready (${elapsed}s elapsed)."
            sleep 3
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    log_server "ERROR: Server did not become ready within ${SERVER_READY_TIMEOUT}s."
    return 1
}

# is_oom <log_dir>
#   Returns 0 if any server log contains CUDA/torch out-of-memory indication.
is_oom() {
    local log_dir="$1"
    grep -rqi \
        "out of memory\|OutOfMemoryError\|CUDA error: out of memory\|cudaMalloc failed" \
        "$log_dir"/server_*.log 2>/dev/null
}

# kill_server
#   Kills all sglang processes and tmux sessions.
kill_server() {
    log_server "Killing server..."
    for n in "${ALL_NODES[@]}"; do
        ssh "$n" 'pkill -9 -f "sglang\.launch_server" 2>/dev/null; pkill -9 -f "sglang\.srt" 2>/dev/null; pkill -9 -f "torch\._inductor\.compile_worker" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 5
    log_server "Server killed."
}
