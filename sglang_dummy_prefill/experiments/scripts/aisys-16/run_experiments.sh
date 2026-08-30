#!/bin/bash
# Automated experiment runner for GPT-OSS 120B benchmarks on aisys-303 cluster
# Runs 8 experiments: 2 configs × 2 profiles × 2 loads
# Usage: bash run_experiments.sh <head_ib_ip> <worker1_host> <worker2_host> <worker3_host>
# Example: bash run_experiments.sh 10.0.0.11:25000 aisys-303-cluster02 aisys-303-cluster03 aisys-303-cluster07

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <dist_init_addr> <worker1_host> <worker2_host> <worker3_host>"
    echo "Example: $0 10.0.0.11:25000 aisys-303-cluster02 aisys-303-cluster03 aisys-303-cluster07"
    exit 1
fi

DIST_INIT_ADDR=$1
WORKER1=$2
WORKER2=$3
WORKER3=$4
WORKERS=("$WORKER1" "$WORKER2" "$WORKER3")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ~/coulson/conda_envs
cd ~/coulson/sglang-fake-prefill

PROFILES=("gating_gptoss120b_200.parquet" "gating_math_gsm8k_200.parquet")
PROFILE_NAMES=("gptoss" "gsm8k")

kill_all() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" 'pkill -f "sglang.launch_server" 2>/dev/null' || true
    done
    sleep 5
    tmux kill-session -t sglang-head 2>/dev/null || true
    tmux kill-session -t sglang-w1 2>/dev/null || true
    tmux kill-session -t sglang-w2 2>/dev/null || true
    tmux kill-session -t sglang-w3 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    echo "  Waiting for server..."
    for i in $(seq 1 90); do
        curl -s http://127.0.0.1:30000/health > /dev/null 2>&1 && echo "  Server UP after ${i}0s" && return 0
        sleep 10
    done
    echo "  TIMEOUT waiting for server"
    return 1
}

launch_ep16() {
    local profile=$1
    local mem_frac=$2

    tmux new-session -d -s sglang-head \
        "bash ${SCRIPT_DIR}/launch_head_ep.sh $DIST_INIT_ADDR ./gating_profiles/$profile logs/server_ep_head.log $mem_frac"

    sleep 3

    for i in "${!WORKERS[@]}"; do
        local w="${WORKERS[$i]}"
        local rank=$((i + 1))
        local sess="sglang-w$((i+1))"
        tmux new-session -d -s "$sess" \
            "ssh $w 'bash ${SCRIPT_DIR}/launch_worker_ep.sh $rank $DIST_INIT_ADDR ./gating_profiles/$profile logs/server_ep_w${rank}.log $mem_frac'"
    done
}

launch_pptp() {
    local profile=$1
    local mem_frac=$2

    tmux new-session -d -s sglang-head \
        "bash ${SCRIPT_DIR}/launch_head_pptp.sh $DIST_INIT_ADDR ./gating_profiles/$profile logs/server_pptp_head.log $mem_frac"

    sleep 3

    for i in "${!WORKERS[@]}"; do
        local w="${WORKERS[$i]}"
        local rank=$((i + 1))
        local sess="sglang-w$((i+1))"
        tmux new-session -d -s "$sess" \
            "ssh $w 'bash ${SCRIPT_DIR}/launch_worker_pptp.sh $rank $DIST_INIT_ADDR ./gating_profiles/$profile logs/server_pptp_w${rank}.log $mem_frac'"
    done
}

run_bench() {
    local num_prompts=$1
    local rate=$2
    local log_name=$3

    echo "  Running benchmark: $num_prompts prompts, rate=$rate -> $log_name"
    python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port 30000 \
        --model lmsys/gpt-oss-120b-bf16 \
        --dataset-name random \
        --random-input-len 128 --random-output-len 512 --random-range-ratio 0.5 \
        --num-prompts $num_prompts --request-rate $rate \
        --seed 1 --warmup-requests 1 \
        2>&1 | tee ./logs/${log_name}.log

    # Check if server is still alive
    if ! curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
        echo "  SERVER CRASHED (OOM) for $log_name"
        return 1
    fi
    return 0
}

run_config_experiments() {
    local config=$1  # "ep16" or "pptp"
    local profile=$2
    local profile_name=$3
    local mem_frac=$4

    echo ""
    echo "========================================="
    echo "Config: $config, Profile: $profile_name, mem_frac: $mem_frac"
    echo "========================================="

    kill_all

    if [ "$config" == "ep16" ]; then
        launch_ep16 "$profile" "$mem_frac"
    else
        launch_pptp "$profile" "$mem_frac"
    fi

    wait_for_server || return 1

    # Run 1000/r250
    local log1="${config}_${profile_name}_1k_r250"
    if [ ! -f "./logs/${log1}.log" ] || ! grep -q "Successful requests" "./logs/${log1}.log" 2>/dev/null; then
        run_bench 1000 250 "$log1"
        if [ $? -ne 0 ]; then
            echo "  FAILED at $mem_frac, will retry at lower frac"
            return 1
        fi
    else
        echo "  Skipping $log1 (already completed)"
    fi

    # Run 2000/r500
    local log2="${config}_${profile_name}_2k_r500"
    if [ ! -f "./logs/${log2}.log" ] || ! grep -q "Successful requests" "./logs/${log2}.log" 2>/dev/null; then
        run_bench 2000 500 "$log2"
        if [ $? -ne 0 ]; then
            echo "  FAILED at $mem_frac, will retry at lower frac"
            return 1
        fi
    else
        echo "  Skipping $log2 (already completed)"
    fi

    return 0
}

# Main experiment loop
mkdir -p logs

echo "Starting 8-experiment benchmark suite (aisys-303 cluster)"
echo "Head: $(hostname), Workers: ${WORKERS[*]}"
echo "dist-init-addr: $DIST_INIT_ADDR"
echo "Input: 64-128 tokens, Output: 256-512 tokens"
echo ""

for config in ep16 pptp; do
    for pi in 0 1; do
        profile="${PROFILES[$pi]}"
        pname="${PROFILE_NAMES[$pi]}"

        # Try 0.70 first (aisys A6000 has less VRAM headroom)
        run_config_experiments "$config" "$profile" "$pname" "0.70"
        result=$?

        if [ $result -ne 0 ]; then
            echo ""
            echo ">>> Retrying $config/$pname at 0.65..."
            run_config_experiments "$config" "$profile" "$pname" "0.65"
            result=$?
            if [ $result -ne 0 ]; then
                echo ">>> STILL FAILING at 0.65 for $config/$pname — skipping"
            fi
        fi
    done
done

echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================="
echo ""
echo "Results:"
for f in ./logs/ep16_*.log ./logs/pptp_*.log; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .log)
        success=$(grep "Successful requests" "$f" 2>/dev/null | awk '{print $NF}')
        output_tput=$(grep "Output token throughput" "$f" 2>/dev/null | awk '{print $NF}')
        median_itl=$(grep "Median ITL" "$f" 2>/dev/null | awk '{print $NF}')
        echo "  $name: success=$success, output_tput=$output_tput tok/s, median_itl=$median_itl ms"
    fi
done
