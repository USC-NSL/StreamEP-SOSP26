#!/bin/bash
# EP16 orchestration — Delta 4×A100-SXM4-40GB × 4 nodes (16 GPUs)
# Launches head + 3 workers via SSH+tmux from the login node.
#
# Usage: ./run_ep16.sh
#   Node list is read from SLURM allocation (squeue).
#   Or override: HEAD=gpua002 WORKERS="gpua007 gpua047 gpua076" ./run_ep16.sh
#
# Requires: tmux, ssh access to compute nodes, conda env 'sglang'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Node discovery (from SLURM or env override) ──
if [ -z "${HEAD:-}" ] || [ -z "${WORKERS:-}" ]; then
    NODELIST=$(squeue -u "$USER" -h -o "%N" | head -1)
    if [ -z "$NODELIST" ]; then
        echo "ERROR: No SLURM job found and HEAD/WORKERS not set."
        echo "Usage: HEAD=gpua002 WORKERS='gpua007 gpua047 gpua076' $0"
        exit 1
    fi
    # Expand SLURM compact notation (e.g. gpua[002,007,047,076])
    ALL_NODES=($(scontrol show hostnames "$NODELIST"))
    if [ ${#ALL_NODES[@]} -lt 4 ]; then
        echo "ERROR: Need 4 nodes for EP16, got ${#ALL_NODES[@]}: ${ALL_NODES[*]}"
        exit 1
    fi
    HEAD="${ALL_NODES[0]}"
    WORKERS=("${ALL_NODES[1]}" "${ALL_NODES[2]}" "${ALL_NODES[3]}")
else
    WORKERS=($WORKERS)
fi

ALL=(${HEAD} ${WORKERS[@]})
WORKER_RANKS=(1 2 3)

# ── Resolve head node hsn0 IP for dist-init-addr ──
HEAD_IP=$(ssh "$HEAD" "ip -4 addr show hsn0 | grep -oP 'inet \K[0-9.]+'")
DIST_INIT_ADDR="${HEAD_IP}:25000"

# ── Experiment config ──
EXP_ID="ep16-delta"
EXP_DIR="experiments/${EXP_ID}"
MEM_FRAC="0.80"

echo "================================================================"
echo "EP16 Launch: Delta 4×4 A100-40GB"
echo "Head:    $HEAD ($HEAD_IP)"
echo "Workers: ${WORKERS[*]}"
echo "dist-init-addr: $DIST_INIT_ADDR"
echo "Experiment dir:  ~/${EXP_DIR}"
echo "Running from: $(hostname) (login node)"
echo "================================================================"

# ── Cleanup ──
kill_all() {
    echo "Killing any existing sglang processes..."
    for n in "${ALL[@]}"; do
        ssh "$n" 'pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang.srt" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 5
}

wait_for_server() {
    echo "Waiting for server health on ${HEAD}:30000..."
    for i in $(seq 1 180); do
        if curl -sf "http://${HEAD}:30000/health" > /dev/null 2>&1; then
            echo "Server UP after $((i * 10))s"
            return 0
        fi
        sleep 10
    done
    echo "TIMEOUT after 30 minutes"
    return 1
}

kill_all

# ── Launch head (rank 0) ──
echo "Starting head on $HEAD ..."
tmux new-session -d -s sglang-head \
    "ssh $HEAD 'bash ${SCRIPT_DIR}/launch_head_ep.sh \
     $DIST_INIT_ADDR ${EXP_DIR}/server_head.log $MEM_FRAC'"

sleep 5

# ── Launch workers (ranks 1-3) ──
for i in "${!WORKERS[@]}"; do
    w="${WORKERS[$i]}"
    rank="${WORKER_RANKS[$i]}"
    sess="sglang-w$((i+1))"
    echo "Starting worker rank $rank on $w ..."
    tmux new-session -d -s "$sess" \
        "ssh $w 'bash ${SCRIPT_DIR}/launch_worker_ep.sh \
         $rank $DIST_INIT_ADDR ${EXP_DIR}/server_w${rank}.log $MEM_FRAC'"
done

# ── Wait for server ──
if ! wait_for_server; then
    echo "FAILED to start server — check logs:"
    echo "  tmux attach -t sglang-head"
    echo "  tmux attach -t sglang-w1"
    echo "  cat ~/${EXP_DIR}/server_head.log"
    kill_all
    exit 1
fi

echo ""
echo "================================================================"
echo "EP16 server is RUNNING on ${HEAD}:30000"
echo ""
echo "Quick test:"
echo "  curl http://${HEAD}:30000/v1/models"
echo ""
echo "Benchmark:"
echo "  python -m sglang.bench_serving --backend sglang --host $HEAD --port 30000 \\"
echo "    --model lmsys/gpt-oss-120b-bf16 --dataset-name random \\"
echo "    --random-input-len 128 --random-output-len 512 --num-prompts 50 --request-rate 10"
echo ""
echo "Tmux sessions: sglang-head, sglang-w1, sglang-w2, sglang-w3"
echo "Logs: ~/${EXP_DIR}/server_*.log"
echo "================================================================"
