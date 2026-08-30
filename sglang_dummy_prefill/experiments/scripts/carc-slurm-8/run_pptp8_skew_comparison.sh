#!/bin/bash
# 8-GPU PP4×TP2 skew comparison: 4 profiles × 1 load level
# MUST run from carcai login node (compute nodes lack tmux)
# Usage: bash run_pptp8_skew_comparison.sh <head_host> <dist_init_addr> <worker1> <worker2> <worker3>
# Example: bash run_pptp8_skew_comparison.sh b04-13 10.125.137.190:25000 b05-12 b05-14 b10-14
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

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda 2>/dev/null || true
eval "$(/home1/yizhuoli/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
export PYTHONPATH=${REPO_ROOT}/python:${PYTHONPATH:-}
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
cd "$REPO_ROOT"

PROFILES=(
    "gating_gptoss_sharegptv3_200.parquet"
    "gating_math_gsm8k_200.parquet"
)
PROFILE_NAMES=(gptoss gsm8k)

EXP_ID="${EXP_ID:-sgl-carc-pptp8}"
MEM_FRAC="0.85"
NUM_PROMPTS=8000
REQUEST_RATE=2000
INPUT_LEN=128
OUTPUT_LEN=512
CUDA_GRAPH_MAX_BS=256

mkdir -p "experiments/${EXP_ID}"

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

launch_pptp8() {
    local profile=$1
    local exp_dir=$2

    mkdir -p "$exp_dir/recorder_raw"

    tmux new-session -d -s sglang-head \
        "ssh $HEAD 'bash ${SCRIPT_DIR}/launch_head_pptp_record.sh \
         $DIST_INIT_ADDR ./gating_profiles/$profile $exp_dir/server_head.log $exp_dir/recorder_raw $MEM_FRAC $CUDA_GRAPH_MAX_BS'"

    sleep 3

    for i in "${!WORKERS[@]}"; do
        local w="${WORKERS[$i]}"
        local rank="${WORKER_RANKS[$i]}"
        local sess="sglang-w$((i+1))"
        tmux new-session -d -s "$sess" \
            "ssh $w 'bash ${SCRIPT_DIR}/launch_worker_pptp_record.sh \
             $rank $DIST_INIT_ADDR ./gating_profiles/$profile $exp_dir/server_w${rank}.log $exp_dir/recorder_raw $MEM_FRAC $CUDA_GRAPH_MAX_BS'"
    done
}

run_bench() {
    local log_file=$1
    echo "  Benchmark: ${NUM_PROMPTS} prompts, rate=${REQUEST_RATE}, input=${INPUT_LEN}, output=${OUTPUT_LEN}"
    python -m sglang.bench_serving \
        --backend sglang --host "$HEAD" --port 30000 \
        --model lmsys/gpt-oss-120b-bf16 \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio 0.5 \
        --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
        --seed 1 --warmup-requests 1 \
        2>&1 | tee "$log_file"

    if ! curl -sf "http://${HEAD}:30000/health" > /dev/null 2>&1; then
        echo "  SERVER CRASHED"
        return 1
    fi
    return 0
}

echo "================================================================"
echo "PP4×TP2 Skew Comparison: ${#PROFILES[@]} profiles"
echo "Params: ${NUM_PROMPTS} prompts, rate=${REQUEST_RATE}, input=${INPUT_LEN}, output=${OUTPUT_LEN}"
echo "Cluster: CARC SLURM, 4 nodes × 2 GPUs = 8× A100-80GB-PCIe"
echo "Head: $HEAD, Workers: ${WORKERS[*]}"
echo "dist-init-addr: $DIST_INIT_ADDR"
echo "Running from: $(hostname) (login node)"
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
    launch_pptp8 "$profile" "$exp_dir"
    if ! wait_for_server; then
        echo "  FAILED to start server for $pname — skipping"
        kill_all
        continue
    fi

    curl -X POST "http://${HEAD}:30000/start_expert_distribution_record" 2>/dev/null || true

    if ! run_bench "$bench_log"; then
        echo "  Benchmark FAILED for $pname (possible OOM)"
        echo "  Retrying with mem_frac=0.80..."
        kill_all
        ORIG_MEM_FRAC="$MEM_FRAC"
        MEM_FRAC="0.80"
        launch_pptp8 "$profile" "$exp_dir"
        if wait_for_server; then
            curl -X POST "http://${HEAD}:30000/start_expert_distribution_record" 2>/dev/null || true
            run_bench "$bench_log" || echo "  Still failed at 0.80"
        fi
        MEM_FRAC="$ORIG_MEM_FRAC"
    fi

    curl -X POST "http://${HEAD}:30000/stop_expert_distribution_record" 2>/dev/null || true
    curl -X POST "http://${HEAD}:30000/dump_expert_distribution_record" 2>/dev/null || true
    sleep 3

    kill_all
done

echo ""
echo "================================================================"
echo "ALL PP4×TP2 EXPERIMENTS COMPLETE"
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
