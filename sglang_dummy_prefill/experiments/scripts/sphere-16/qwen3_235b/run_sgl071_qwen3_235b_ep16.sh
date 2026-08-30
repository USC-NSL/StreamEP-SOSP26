#!/bin/bash
# sgl-071: Qwen3-235B EP16 AllGather+ReduceScatter smoke test (50 requests)
set -euo pipefail

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

WORKERS=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)
WORKER_RANKS=(1 2 3 4 5 6 7)

EXP_ID="sgl-071"
MODEL="Qwen/Qwen3-235B-A22B"
PROFILE="qwen3_235b_profiles/gating_qwen3_235b_sharegpt_200.parquet"
MEM_FRAC="0.70"
MAX_RUNNING_REQS=512
NUM_PROMPTS=50
REQUEST_RATE=10
INPUT_LEN=128
OUTPUT_LEN=128
RANGE_RATIO=0.5
BENCH_TIMEOUT_S=600
EXP_DIR="experiments/${EXP_ID}"

kill_all() {
    pkill -9 -f "python -m sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang.srt" 2>/dev/null || true
    pkill -9 -f "python -m sglang.bench_serving" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true
    pkill -9 -f "torch._inductor" 2>/dev/null || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" 'pkill -9 -f "python -m sglang.launch_server" 2>/dev/null || true; pkill -9 -f "sglang.srt" 2>/dev/null || true; pkill -9 -f "python -m sglang.bench_serving" 2>/dev/null || true; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true; pkill -9 -f "torch._inductor" 2>/dev/null || true' >/dev/null 2>&1 || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3 sglang-w4 sglang-w5 sglang-w6 sglang-w7; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 10
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
    echo "  TIMEOUT waiting for server"
    return 1
}

# Clean old experiment data on head and workers
rm -rf "${EXP_DIR}"
for w in "${WORKERS[@]}"; do
    ssh "$w" "rm -rf /home/yizhuoliang/sglang-fake-prefill/${EXP_DIR}" &
done
wait
mkdir -p "${EXP_DIR}"
for w in "${WORKERS[@]}"; do
    ssh "$w" "mkdir -p /home/yizhuoliang/sglang-fake-prefill/${EXP_DIR}" &
done
wait

# Write experiment intention
cat > "${EXP_DIR}/intention.txt" <<EOF
sgl-071: Smoke test for Qwen3-235B-A22B EP16 on sphere-16 cluster.
Using AllGather+ReduceScatter EP (no mooncake-nccl), disable-cuda-graph, 
dummy weights, FP8 quantization, fake prefill with qwen3 sharegpt gating profile.
50 requests, random dataset, input=128, output=128.
Goal: Verify the model launches and serves correctly on this cluster.
EOF

echo "================================================================"
echo "${EXP_ID}: Qwen3-235B EP16 AllGather+ReduceScatter smoke test"
echo "================================================================"

kill_all

# Launch head node
tmux new-session -d -s sglang-head \
    "bash experiments/scripts/sphere-16/qwen3_235b/launch_head_ep.sh \
     ./gating_profiles/${PROFILE} ${EXP_DIR}/server_head.log ${MEM_FRAC} ${MAX_RUNNING_REQS}"

sleep 3

# Launch workers
for i in "${!WORKERS[@]}"; do
    w="${WORKERS[$i]}"
    rank="${WORKER_RANKS[$i]}"
    sess="sglang-w$((i+1))"
    tmux new-session -d -s "$sess" \
        "ssh $w 'bash /home/yizhuoliang/sglang-fake-prefill/experiments/scripts/sphere-16/qwen3_235b/launch_worker_ep.sh \
         ${rank} ./gating_profiles/${PROFILE} ${EXP_DIR}/server_w${rank}.log ${MEM_FRAC} ${MAX_RUNNING_REQS}'"
done

if ! wait_for_server; then
    echo "FAILED to launch server"
    echo "=== Head log tail ==="
    tail -100 "${EXP_DIR}/server_head.log" 2>/dev/null || true
    kill_all
    exit 1
fi

echo "Running benchmark (${NUM_PROMPTS} prompts)..."
if ! timeout "${BENCH_TIMEOUT_S}" python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio "$RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
    --seed 1 --warmup-requests 1 \
    2>&1 | tee "${EXP_DIR}/bench.log"; then
    echo "BENCH FAILED" | tee -a "${EXP_DIR}/bench.log"
fi

if ! curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1; then
    echo "SERVER UNHEALTHY after bench" | tee -a "${EXP_DIR}/bench.log"
fi

kill_all

echo ""
echo "=== RESULTS ==="
grep -E "Successful requests|Benchmark duration|Request throughput|Output token throughput|Concurrency|Median ITL|P99 ITL|Mean ITL|BENCH FAILED|SERVER UNHEALTHY" "${EXP_DIR}/bench.log" || true
