#!/bin/bash
# 16-GPU EP skew comparison: 4 profiles × 1 load level
# 8 nodes (sgpu0 head + sgpu2/3/4/6/7/8/9), 2 GPUs each
set -e

eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

WORKERS=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)
WORKER_RANKS=(1 2 3 4 5 6 7)

PROFILES=(
    "gating_gptoss120b_sharegpt_200.parquet"
    "gating_math_gsm8k_200.parquet"
    "gating_legal_court_opinions_200.parquet"
    "gating_chinese_zhihu_200.parquet"
)
PROFILE_NAMES=(gptoss gsm8k legal chinese)

EXP_ID="sgl-008"
MEM_FRAC="0.70"
NUM_PROMPTS=8000
REQUEST_RATE=2000
INPUT_LEN=128
OUTPUT_LEN=512
MAX_RUNNING_REQS=
CUDA_GRAPH_MAX_BS=

mkdir -p "experiments/${EXP_ID}" logs

kill_all() {
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang.srt" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" 'pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang.srt" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3 sglang-w4 sglang-w5 sglang-w6 sglang-w7; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 5
}

wait_for_server() {
    echo "  Waiting for server health..."
    for i in $(seq 1 180); do
        if curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1; then
            echo "  Server UP after ${i}0s"
            return 0
        fi
        sleep 10
    done
    echo "  TIMEOUT after 30 minutes"
    return 1
}

launch_ep16() {
    local profile=$1
    local exp_dir=$2

    mkdir -p "$exp_dir/recorder_raw"

    tmux new-session -d -s sglang-head \
        "bash experiments/scripts/sphere-16/launch_head_ep_record.sh \
         ./gating_profiles/$profile $exp_dir/server_head.log $exp_dir/recorder_raw '$MEM_FRAC' '$MAX_RUNNING_REQS' '$CUDA_GRAPH_MAX_BS'"

    sleep 3

    for i in "${!WORKERS[@]}"; do
        local w="${WORKERS[$i]}"
        local rank="${WORKER_RANKS[$i]}"
        local sess="sglang-w$((i+1))"
        tmux new-session -d -s "$sess" \
            "ssh $w 'bash /home/yizhuoliang/sglang-fake-prefill/experiments/scripts/sphere-16/launch_worker_ep_record.sh \
             $rank ./gating_profiles/$profile $exp_dir/server_w${rank}.log $exp_dir/recorder_raw $MEM_FRAC $MAX_RUNNING_REQS $CUDA_GRAPH_MAX_BS'"
    done
}

run_bench() {
    local log_file=$1
    echo "  Benchmark: ${NUM_PROMPTS} prompts, rate=${REQUEST_RATE}, input=${INPUT_LEN}, output=${OUTPUT_LEN}"
    python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port 30000 \
        --model lmsys/gpt-oss-120b-bf16 \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio 0.5 \
        --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
        --seed 1 --warmup-requests 1 \
        2>&1 | tee "$log_file"

    if ! curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1; then
        echo "  SERVER CRASHED"
        return 1
    fi
    return 0
}

echo "================================================================"
echo "EP16 Skew Comparison: ${#PROFILES[@]} profiles"
echo "Params: ${NUM_PROMPTS} prompts, rate=${REQUEST_RATE}, input=${INPUT_LEN}, output=${OUTPUT_LEN}"
echo "Cluster: 8 nodes × 2 GPUs = 16× L40S"
echo "================================================================"

for pi in "${!PROFILES[@]}"; do
    profile="${PROFILES[$pi]}"
    pname="${PROFILE_NAMES[$pi]}"
    exp_dir="experiments/${EXP_ID}/exp$((pi+1))_${pname}"

    echo ""
    echo "========================================="
    echo "[$((pi+1))/${#PROFILES[@]}] Profile: $pname ($profile)"
    echo "========================================="

    bench_log="experiments/${EXP_ID}/exp$((pi+1))_${pname}/bench.log"
    if [ -f "$bench_log" ] && grep -q "Successful requests" "$bench_log" 2>/dev/null; then
        echo "  Skipping (already completed: $bench_log)"
        continue
    fi

    kill_all
    launch_ep16 "$profile" "$exp_dir"
    if ! wait_for_server; then
        echo "  FAILED to start server for $pname — skipping"
        kill_all
        continue
    fi

    curl -X POST http://localhost:30000/start_expert_distribution_record 2>/dev/null || true

    if ! run_bench "$bench_log"; then
        echo "  Benchmark FAILED for $pname (possible OOM)"
        echo "  Retrying with mem_frac=0.65..."
        kill_all
        local ORIG_MEM_FRAC="$MEM_FRAC"
        MEM_FRAC="0.65"
        launch_ep16 "$profile" "$exp_dir"
        if wait_for_server; then
            curl -X POST http://localhost:30000/start_expert_distribution_record 2>/dev/null || true
            run_bench "$bench_log" || echo "  Still failed at 0.65"
        fi
        MEM_FRAC="$ORIG_MEM_FRAC"
    fi

    timeout 30 curl -X POST http://localhost:30000/stop_expert_distribution_record 2>/dev/null || true
    timeout 30 curl -X POST http://localhost:30000/dump_expert_distribution_record 2>/dev/null || true
    sleep 3

    kill_all
done

echo ""
echo "Collecting recorder .pt files from workers..."
for pi in "${!PROFILES[@]}"; do
    pname="${PROFILE_NAMES[$pi]}"
    exp_dir="experiments/${EXP_ID}/exp$((pi+1))_${pname}"
    for w in "${WORKERS[@]}"; do
        rsync -az "$w:/home/yizhuoliang/sglang-fake-prefill/$exp_dir/recorder_raw/" "$exp_dir/recorder_raw/" 2>/dev/null || true
    done
done
echo "Collection done."

echo ""
echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================"
echo ""
echo "Results summary:"
for pi in "${!PROFILES[@]}"; do
    pname="${PROFILE_NAMES[$pi]}"
    f="experiments/${EXP_ID}/exp$((pi+1))_${pname}/bench.log"
    if [ -f "$f" ]; then
        success=$(grep "Successful requests" "$f" 2>/dev/null | awk '{print $NF}')
        output_tput=$(grep "Output token throughput" "$f" 2>/dev/null | awk '{print $NF}')
        median_itl=$(grep "Median ITL" "$f" 2>/dev/null | awk '{print $NF}')
        echo "  $pname: success=$success, output_tput=${output_tput} tok/s, median_itl=${median_itl} ms"
    else
        echo "  $pname: NO RESULTS"
    fi
done
