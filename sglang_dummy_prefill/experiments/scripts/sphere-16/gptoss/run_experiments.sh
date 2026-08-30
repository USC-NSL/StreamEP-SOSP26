#!/bin/bash
# Automated experiment runner for GPT-OSS 120B benchmarks
# Runs 8 experiments: 2 configs × 2 profiles × 2 loads
# Usage: bash run_experiments.sh

set -e

eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

PROFILES=("gating_gptoss120b_200.parquet" "gating_math_gsm8k_200.parquet")
PROFILE_NAMES=("gptoss" "gsm8k")

kill_all() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    ssh sgpu6 'pkill -f "sglang.launch_server" 2>/dev/null' || true
    ssh sgpu7 'pkill -f "sglang.launch_server" 2>/dev/null' || true
    ssh sgpu8 'pkill -f "sglang.launch_server" 2>/dev/null' || true
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

launch_ep8() {
    local profile=$1
    local mem_frac=$2
    # Update launch scripts with the mem fraction
    sed -i "s/--mem-fraction-static [0-9.]*/--mem-fraction-static $mem_frac/" launch_head_ep_nccl.sh
    sed -i "s/--mem-fraction-static [0-9.]*/--mem-fraction-static $mem_frac/" launch_worker_ep_nccl.sh
    rsync -az launch_head_ep_nccl.sh launch_worker_ep_nccl.sh sgpu6:/home/yizhuoliang/sglang-fake-prefill/
    rsync -az launch_head_ep_nccl.sh launch_worker_ep_nccl.sh sgpu7:/home/yizhuoliang/sglang-fake-prefill/
    rsync -az launch_head_ep_nccl.sh launch_worker_ep_nccl.sh sgpu8:/home/yizhuoliang/sglang-fake-prefill/
    
    tmux new-session -d -s sglang-head "bash launch_head_ep_nccl.sh ./gating_profiles/$profile server_ep_nccl_head.log"
    tmux new-session -d -s sglang-w1 "ssh sgpu6 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_ep_nccl.sh 1 ./gating_profiles/$profile server_ep_nccl_w1.log'"
    tmux new-session -d -s sglang-w2 "ssh sgpu7 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_ep_nccl.sh 2 ./gating_profiles/$profile server_ep_nccl_w2.log'"
    tmux new-session -d -s sglang-w3 "ssh sgpu8 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_ep_nccl.sh 3 ./gating_profiles/$profile server_ep_nccl_w3.log'"
}

launch_pp4() {
    local profile=$1
    local mem_frac=$2
    sed -i "s/--mem-fraction-static [0-9.]*/--mem-fraction-static $mem_frac/" launch_head_pp.sh
    sed -i "s/--mem-fraction-static [0-9.]*/--mem-fraction-static $mem_frac/" launch_worker_pp.sh
    rsync -az launch_head_pp.sh launch_worker_pp.sh sgpu6:/home/yizhuoliang/sglang-fake-prefill/
    rsync -az launch_head_pp.sh launch_worker_pp.sh sgpu7:/home/yizhuoliang/sglang-fake-prefill/
    rsync -az launch_head_pp.sh launch_worker_pp.sh sgpu8:/home/yizhuoliang/sglang-fake-prefill/

    tmux new-session -d -s sglang-head "bash launch_head_pp.sh ./gating_profiles/$profile server_pp4_head.log"
    tmux new-session -d -s sglang-w1 "ssh sgpu6 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 1 ./gating_profiles/$profile server_pp4_w1.log'"
    tmux new-session -d -s sglang-w2 "ssh sgpu7 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 2 ./gating_profiles/$profile server_pp4_w2.log'"
    tmux new-session -d -s sglang-w3 "ssh sgpu8 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 3 ./gating_profiles/$profile server_pp4_w3.log'"
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
    local config=$1  # "ep8" or "pp4"
    local profile=$2
    local profile_name=$3
    local mem_frac=$4
    
    echo ""
    echo "========================================="
    echo "Config: $config, Profile: $profile_name, mem_frac: $mem_frac"
    echo "========================================="
    
    kill_all
    
    if [ "$config" == "ep8" ]; then
        launch_ep8 "$profile" "$mem_frac"
    else
        launch_pp4 "$profile" "$mem_frac"
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
echo "Starting 8-experiment benchmark suite"
echo "Input: 64-128 tokens, Output: 256-512 tokens"
echo ""

for config in ep8 pp4; do
    for pi in 0 1; do
        profile="${PROFILES[$pi]}"
        pname="${PROFILE_NAMES[$pi]}"
        
        # Try 0.80 first
        run_config_experiments "$config" "$profile" "$pname" "0.80"
        result=$?
        
        if [ $result -ne 0 ]; then
            # Check which experiment failed and retry at 0.75
            echo ""
            echo ">>> Retrying $config/$pname at 0.75..."
            run_config_experiments "$config" "$profile" "$pname" "0.75"
            result=$?
            if [ $result -ne 0 ]; then
                echo ">>> STILL FAILING at 0.75 for $config/$pname — skipping"
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
for f in ./logs/ep8_*.log ./logs/pp4_*.log; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .log)
        success=$(grep "Successful requests" "$f" 2>/dev/null | awk '{print $NF}')
        output_tput=$(grep "Output token throughput" "$f" 2>/dev/null | awk '{print $NF}')
        median_itl=$(grep "Median ITL" "$f" 2>/dev/null | awk '{print $NF}')
        echo "  $name: success=$success, output_tput=$output_tput tok/s, median_itl=$median_itl ms"
    fi
done
