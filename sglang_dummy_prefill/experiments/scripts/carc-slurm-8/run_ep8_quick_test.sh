#!/bin/bash
# Quick 50-request test to validate EP8 on CARC SLURM
# MUST run from carcai login node (compute nodes lack tmux)
# Usage: bash run_ep8_quick_test.sh <head_host> <dist_init_addr> <worker1> <worker2> <worker3>
# Example: bash run_ep8_quick_test.sh b04-13 10.125.137.190:25000 b05-12 b05-14 b10-14
set -e

if [ $# -lt 5 ]; then
    echo "Usage: $0 <head_host> <dist_init_addr> <worker1> <worker2> <worker3>"
    echo "Example: $0 b04-13 10.125.137.190:25000 b05-12 b05-14 b10-14"
    exit 1
fi

HEAD=$1
DIST_INIT_ADDR=$2
WORKER1=$3
WORKER2=$4
WORKER3=$5
ALL_NODES=("$HEAD" "$WORKER1" "$WORKER2" "$WORKER3")
WORKERS=("$WORKER1" "$WORKER2" "$WORKER3")
WORKER_RANKS=(1 2 3)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="/home1/yizhuoli/sglang-fake-prefill"

PROFILE="gating_gptoss_sharegptv3_200.parquet"
EXP_ID="sgl-test-carc-ep8"
EXP_DIR="experiments/${EXP_ID}/exp_quick"
MEM_FRAC="0.85"
NUM_PROMPTS=50
REQUEST_RATE=100
INPUT_LEN=128
OUTPUT_LEN=512
CUDA_GRAPH_MAX_BS=

mkdir -p "${REPO_ROOT}/${EXP_DIR}/recorder_raw"

kill_all() {
    for n in "${ALL_NODES[@]}"; do
        ssh "$n" 'pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang.srt" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    sleep 5
}

wait_for_server() {
    echo "  Waiting for server health on ${HEAD}:30000..."
    for i in $(seq 1 180); do
        if curl -sf "http://${HEAD}:30000/health" > /dev/null 2>&1; then
            echo "  Server UP after ${i}0s"
            return 0
        fi
        sleep 10
    done
    echo "  TIMEOUT after 30 minutes"
    return 1
}

echo "================================================================"
echo "EP8 Quick Test: $NUM_PROMPTS requests on CARC SLURM"
echo "Profile: $PROFILE"
echo "Head: $HEAD, Workers: ${WORKERS[*]}"
echo "dist-init-addr: $DIST_INIT_ADDR"
echo "Running from: $(hostname) (login node)"
echo "================================================================"

kill_all

tmux new-session -d -s sglang-head \
    "ssh $HEAD 'bash ${SCRIPT_DIR}/launch_head_ep_record.sh \
     $DIST_INIT_ADDR ./gating_profiles/$PROFILE $EXP_DIR/server_head.log $EXP_DIR/recorder_raw $MEM_FRAC $CUDA_GRAPH_MAX_BS'"

sleep 3

for i in "${!WORKERS[@]}"; do
    w="${WORKERS[$i]}"
    rank="${WORKER_RANKS[$i]}"
    sess="sglang-w$((i+1))"
    tmux new-session -d -s "$sess" \
        "ssh $w 'bash ${SCRIPT_DIR}/launch_worker_ep_record.sh \
         $rank $DIST_INIT_ADDR ./gating_profiles/$PROFILE $EXP_DIR/server_w${rank}.log $EXP_DIR/recorder_raw $MEM_FRAC $CUDA_GRAPH_MAX_BS'"
done

if ! wait_for_server; then
    echo "FAILED to start server — check ${EXP_DIR}/server_head.log"
    echo "Debug: tmux attach -t sglang-head"
    kill_all
    exit 1
fi

echo ""
echo "Starting recording..."
curl -X POST "http://${HEAD}:30000/start_expert_distribution_record" 2>/dev/null || true

echo "Running benchmark: ${NUM_PROMPTS} prompts, rate=${REQUEST_RATE}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda 2>/dev/null || true
eval "$(/home1/yizhuoli/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
export PYTHONPATH=${REPO_ROOT}/python:${PYTHONPATH:-}
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp

cd "$REPO_ROOT"
python -m sglang.bench_serving \
    --backend sglang --host "$HEAD" --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio 0.5 \
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
    --seed 1 --warmup-requests 1 \
    2>&1 | tee "${EXP_DIR}/bench.log"

echo ""
echo "Stopping and dumping recording..."
timeout 30 curl -X POST "http://${HEAD}:30000/stop_expert_distribution_record" 2>/dev/null || true
timeout 30 curl -X POST "http://${HEAD}:30000/dump_expert_distribution_record" 2>/dev/null || true
sleep 3

kill_all

echo ""
if grep -q "Successful requests" "${EXP_DIR}/bench.log" 2>/dev/null; then
    success=$(grep "Successful requests" "${EXP_DIR}/bench.log" | awk '{print $NF}')
    output_tput=$(grep "Output token throughput" "${EXP_DIR}/bench.log" | awk '{print $NF}')
    median_itl=$(grep "Median ITL" "${EXP_DIR}/bench.log" | awk '{print $NF}')
    echo "================================================================"
    echo "EP8 QUICK TEST PASSED"
    echo "  Successful: $success, Output: ${output_tput} tok/s, Median ITL: ${median_itl} ms"
    echo "================================================================"
else
    echo "================================================================"
    echo "EP8 QUICK TEST FAILED — check ${EXP_DIR}/server_head.log"
    echo "================================================================"
    exit 1
fi
