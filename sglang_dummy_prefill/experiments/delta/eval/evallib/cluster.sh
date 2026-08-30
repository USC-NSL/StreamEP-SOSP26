#!/usr/bin/bash
# evallib/cluster.sh — SSH+tmux cluster management for Delta EP16
# Source this file; do not execute directly.
#
# Requires (from config.sh):  N_NODE, HOST_IFNAME, DIST_INIT_PORT
# Requires (from environment): SLURM allocation active (squeue)
#
# Exports:
#   HEAD, WORKERS, ALL_NODES, HEAD_IP, DIST_INIT_ADDR
#   discover_nodes()  — resolve nodes from SLURM or env override
#   kill_all()        — kill all sglang processes + tmux sessions
#   resolve_head_ip() — get head node's hsn0 IP

log_cluster() { echo "$(date '+%Y-%m-%d %H:%M:%S') [cluster] $*"; }

# discover_nodes
#   Resolves HEAD, WORKERS[], ALL_NODES[] from SLURM or env vars.
#   Sets DIST_INIT_ADDR using the head node's hsn0 IP.
discover_nodes() {
    if [ -z "${HEAD:-}" ] || [ -z "${WORKERS:-}" ]; then
        local nodelist
        nodelist=$(squeue -u "$USER" -h -o "%N" | head -1)
        if [ -z "$nodelist" ]; then
            log_cluster "FATAL: No SLURM job found and HEAD/WORKERS not set."
            log_cluster "Usage: HEAD=gpua002 WORKERS='gpua007 gpua047 gpua076' ..."
            return 1
        fi
        ALL_NODES=($(scontrol show hostnames "$nodelist"))
        if [ ${#ALL_NODES[@]} -lt "$N_NODE" ]; then
            log_cluster "FATAL: Need $N_NODE nodes for EP16, got ${#ALL_NODES[@]}: ${ALL_NODES[*]}"
            return 1
        fi
        HEAD="${ALL_NODES[0]}"
        WORKERS=("${ALL_NODES[@]:1}")
    else
        WORKERS=($WORKERS)
        ALL_NODES=("$HEAD" "${WORKERS[@]}")
    fi

    # Resolve head node hsn0 IP for dist-init-addr
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
        ssh "$n" 'pkill -9 -f "sglang\.launch_server" 2>/dev/null; pkill -9 -f "sglang\.srt" 2>/dev/null; pkill -9 -f "torch\._inductor\.compile_worker" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 5
    log_cluster "Cleanup complete."
}
