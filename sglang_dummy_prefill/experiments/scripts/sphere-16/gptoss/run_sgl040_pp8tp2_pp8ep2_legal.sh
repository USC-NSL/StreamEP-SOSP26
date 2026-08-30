#!/bin/bash
set -euo pipefail

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

WORKERS=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)
WORKER_RANKS=(1 2 3 4 5 6 7)

EXP_ID="sgl-040"
MEM_FRAC="0.80"
NUM_PROMPTS=10000
REQUEST_RATE=2000
INPUT_LEN=512
OUTPUT_LEN=512
RANGE_RATIO=0.5
MAX_RUNNING_REQS=
CUDA_GRAPH_MAX_BS=

PROFILES=("gating_legal_court_opinions_200.parquet" "balanced_legal_court_opinions_200.parquet")
PROFILE_NAMES=(legal balanced_legal)
MODES=(pp8tp2 pp8ep2)

mkdir -p "experiments/${EXP_ID}"

kill_all() {
    pkill -9 -f "sglang" 2>/dev/null || true
    pkill -9 -f "ray" 2>/dev/null || true
    pkill -9 -f "torch._inductor" 2>/dev/null || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "ray" 2>/dev/null; pkill -9 -f "torch._inductor" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3 sglang-w4 sglang-w5 sglang-w6 sglang-w7; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 10
}

wait_for_server() {
    for i in $(seq 1 180); do
        if curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1; then
            echo "  Server UP after ${i}0s"
            return 0
        fi
        sleep 10
    done
    echo "  TIMEOUT waiting for server"
    return 1
}

launch_mode() {
    local mode=$1
    local profile=$2
    local exp_dir=$3
    local head_launcher worker_launcher

    mkdir -p "$exp_dir/recorder_raw"
    if [ "$mode" = "pp8tp2" ]; then
        head_launcher="launch_head_pp8tp2_record.sh"
        worker_launcher="launch_worker_pp8tp2_record.sh"
    else
        head_launcher="launch_head_pp8ep2_record.sh"
        worker_launcher="launch_worker_pp8ep2_record.sh"
    fi

    tmux new-session -d -s sglang-head \
        "bash experiments/scripts/sphere-16/${head_launcher} ./gating_profiles/${profile} ${exp_dir}/server_head.log ${exp_dir}/recorder_raw ${MEM_FRAC} ${MAX_RUNNING_REQS} ${CUDA_GRAPH_MAX_BS}"
    sleep 3
    for i in "${!WORKERS[@]}"; do
        local w="${WORKERS[$i]}"
        local rank="${WORKER_RANKS[$i]}"
        local sess="sglang-w$((i+1))"
        tmux new-session -d -s "$sess" \
            "ssh $w 'bash /home/yizhuoliang/sglang-fake-prefill/experiments/scripts/sphere-16/${worker_launcher} ${rank} ./gating_profiles/${profile} ${exp_dir}/server_w${rank}.log ${exp_dir}/recorder_raw ${MEM_FRAC} ${MAX_RUNNING_REQS} ${CUDA_GRAPH_MAX_BS}'"
    done
}

run_bench() {
    local log_file=$1
    python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port 30000 \
        --model lmsys/gpt-oss-120b-bf16 \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio "$RANGE_RATIO" \
        --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
        --seed 1 --warmup-requests 1 \
        2>&1 | tee "$log_file"
}

echo "==============================================================="
echo "${EXP_ID}: PP8xTP2 + PP8xEP2, legal + balanced_legal"
echo "10k reqs, 2000 rps, input/output uniform [256,512], mem_frac=${MEM_FRAC}"
echo "==============================================================="

for mode in "${MODES[@]}"; do
    for pi in "${!PROFILES[@]}"; do
        profile="${PROFILES[$pi]}"
        pname="${PROFILE_NAMES[$pi]}"
        exp_dir="experiments/${EXP_ID}/${mode}_${pname}"
        bench_log="${exp_dir}/bench.log"

        echo ""
        echo "==============================================================="
        echo "Mode=${mode} Profile=${pname}"
        echo "==============================================================="

        kill_all
        launch_mode "$mode" "$profile" "$exp_dir"
        if ! wait_for_server; then
            echo "  FAILED to launch ${mode} ${pname}"
            continue
        fi

        curl -X POST http://127.0.0.1:30000/start_expert_distribution_record 2>/dev/null || true
        if ! run_bench "$bench_log"; then
            echo "  BENCH FAILED for ${mode} ${pname}"
        fi
        timeout 30 curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record 2>/dev/null || true
        timeout 30 curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record 2>/dev/null || true
        sleep 5
        for w in "${WORKERS[@]}"; do
            rsync -az "$w:/home/yizhuoliang/sglang-fake-prefill/${exp_dir}/recorder_raw/" "${exp_dir}/recorder_raw/" 2>/dev/null || true
        done
        kill_all
    done
done

echo ""
echo "${EXP_ID} complete"
for mode in "${MODES[@]}"; do
    for pname in "${PROFILE_NAMES[@]}"; do
        f="experiments/${EXP_ID}/${mode}_${pname}/bench.log"
        if [ -f "$f" ]; then
            echo "--- ${mode}_${pname} ---"
            grep -E "Successful requests|Benchmark duration|Output token throughput|Median ITL" "$f" || true
        fi
    done
done
