#!/usr/bin/bash
# evallib/cluster.sh — SSH+tmux cluster management for Sphere
# Source this file; do not execute directly.
#
# Requires (from config.sh):  N_NODE, HOST_IFNAME, DIST_INIT_PORT, DEFAULT_WORKERS
#
# Exports:
#   HEAD, WORKERS, ALL_NODES, HEAD_IP, DIST_INIT_ADDR
#   discover_nodes()  — resolve nodes from env vars or defaults
#   kill_all()        — kill all sglang processes + tmux sessions

log_cluster() { echo "$(date '+%Y-%m-%d %H:%M:%S') [cluster] $*"; }

# discover_nodes
#   Resolves HEAD, WORKERS[], ALL_NODES[] from env vars or defaults.
#   Sphere does not use SLURM — nodes are specified directly.
#   Sets DIST_INIT_ADDR using the head node's IB IP.
discover_nodes() {
    if [ -z "${HEAD:-}" ]; then
        HEAD="$(hostname -s)"
        log_cluster "HEAD not set, using local hostname: $HEAD"
    fi

    if [ -z "${WORKERS:-}" ]; then
        WORKERS=($DEFAULT_WORKERS)
        log_cluster "WORKERS not set, using defaults: ${WORKERS[*]}"
    else
        WORKERS=($WORKERS)
    fi

    ALL_NODES=("$HEAD" "${WORKERS[@]}")

    if [ ${#ALL_NODES[@]} -lt "$N_NODE" ]; then
        log_cluster "FATAL: Need $N_NODE nodes, got ${#ALL_NODES[@]}: ${ALL_NODES[*]}"
        return 1
    fi

    HEAD_IP=$(ssh "$HEAD" "ip -4 addr show $HOST_IFNAME | grep -oP 'inet \K[0-9.]+'")
    DIST_INIT_ADDR="${HEAD_IP}:${DIST_INIT_PORT}"

    log_cluster "Head:    $HEAD ($HEAD_IP)"
    log_cluster "Workers: ${WORKERS[*]}"
    log_cluster "dist-init-addr: $DIST_INIT_ADDR"
}

# kill_all
#   Kills any existing sglang processes on all nodes and cleans up tmux sessions.
kill_all() {
    log_cluster "Killing all sglang processes..."
    for n in "${ALL_NODES[@]}"; do
        ssh "$n" 'pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang.srt" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3 sglang-w4 sglang-w5 sglang-w6 sglang-w7; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 5
    log_cluster "Cleanup complete."
}
