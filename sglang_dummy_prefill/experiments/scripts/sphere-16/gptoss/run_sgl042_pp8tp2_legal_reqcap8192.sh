#!/bin/bash
set -euo pipefail

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

WORKERS=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)
WORKER_RANKS=(1 2 3 4 5 6 7)

EXP_ID="sgl-042"
PROFILE="gating_legal_court_opinions_200.parquet"
EXP_DIR="experiments/${EXP_ID}/pp8tp2_legal_reqcap8192"
MEM_FRAC="0.80"
NUM_PROMPTS=10000
REQUEST_RATE=2000
INPUT_LEN=512
OUTPUT_LEN=512
RANGE_RATIO=0.5

kill_all() {
    pkill -9 -f "python -m sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang.srt" 2>/dev/null || true
    pkill -9 -f "python -m sglang.bench_serving" 2>/dev/null || true
    pkill -9 -f "ray" 2>/dev/null || true
    pkill -9 -f "torch._inductor" 2>/dev/null || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" 'pkill -9 -f "python -m sglang.launch_server" 2>/dev/null || true; pkill -9 -f "sglang.srt" 2>/dev/null || true; pkill -9 -f "python -m sglang.bench_serving" 2>/dev/null || true; pkill -9 -f "ray" 2>/dev/null || true; pkill -9 -f "torch._inductor" 2>/dev/null || true' >/dev/null 2>&1 || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3 sglang-w4 sglang-w5 sglang-w6 sglang-w7; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 8
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

mkdir -p "$EXP_DIR/recorder_raw"

echo "==============================================================="
echo "${EXP_ID}: PP8xTP2 legal single-case (req cap raised to 8192 in code)"
echo "10k reqs, 2000 rps, input/output uniform [256,512], mem_frac=${MEM_FRAC}"
echo "==============================================================="

kill_all

tmux new-session -d -s sglang-head     "bash experiments/scripts/sphere-16/launch_head_pp8tp2_record.sh ./gating_profiles/${PROFILE} ${EXP_DIR}/server_head.log ${EXP_DIR}/recorder_raw ${MEM_FRAC}"

sleep 3
for i in "${!WORKERS[@]}"; do
    w="${WORKERS[$i]}"
    rank="${WORKER_RANKS[$i]}"
    sess="sglang-w$((i+1))"
    tmux new-session -d -s "$sess"         "ssh $w 'bash /home/yizhuoliang/sglang-fake-prefill/experiments/scripts/sphere-16/launch_worker_pp8tp2_record.sh ${rank} ./gating_profiles/${PROFILE} ${EXP_DIR}/server_w${rank}.log ${EXP_DIR}/recorder_raw ${MEM_FRAC}'"
done

if ! wait_for_server; then
    echo "FAILED to launch server"
    exit 1
fi

curl -X POST http://127.0.0.1:30000/start_expert_distribution_record >/dev/null 2>&1 || true

python -m sglang.bench_serving     --backend sglang --host 127.0.0.1 --port 30000     --model lmsys/gpt-oss-120b-bf16     --dataset-name random     --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio "$RANGE_RATIO"     --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE"     --seed 1 --warmup-requests 1     2>&1 | tee "$EXP_DIR/bench.log"

timeout 30 curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record >/dev/null 2>&1 || true
timeout 30 curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record >/dev/null 2>&1 || true
sleep 5
for w in "${WORKERS[@]}"; do
    rsync -az "$w:/home/yizhuoliang/sglang-fake-prefill/${EXP_DIR}/recorder_raw/" "$EXP_DIR/recorder_raw/" >/dev/null 2>&1 || true
done

kill_all

echo "${EXP_ID} single-case complete"
grep -E "Successful requests|Benchmark duration|Request throughput|Output token throughput|Concurrency|Median ITL" "$EXP_DIR/bench.log" || true
