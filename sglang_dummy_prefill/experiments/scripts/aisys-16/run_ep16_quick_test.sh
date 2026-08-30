#!/bin/bash
# Quick 50-request test to validate MoE time recorder with cudagraph
# Usage: bash run_ep16_quick_test.sh <dist_init_addr> <worker1_host> <worker2_host> <worker3_host>
# Example: bash run_ep16_quick_test.sh 10.0.0.11:25000 aisys-303-cluster02 aisys-303-cluster03 aisys-303-cluster07
set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <dist_init_addr> <worker1_host> <worker2_host> <worker3_host>"
    exit 1
fi

DIST_INIT_ADDR=$1
WORKER1=$2
WORKER2=$3
WORKER3=$4
WORKERS=("$WORKER1" "$WORKER2" "$WORKER3")
WORKER_RANKS=(1 2 3)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ~/coulson/conda_envs
cd ~/coulson/sglang-fake-prefill

PROFILE="gating_gptoss120b_sharegpt_200.parquet"
EXP_ID="sgl-test-recorder"
EXP_DIR="experiments/${EXP_ID}/exp_quick"
MEM_FRAC="0.70"
NUM_PROMPTS=50
REQUEST_RATE=100
INPUT_LEN=128
OUTPUT_LEN=512
CUDA_GRAPH_MAX_BS=

mkdir -p "$EXP_DIR" logs

kill_all() {
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang.srt" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" 'pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang.srt" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null' 2>/dev/null || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
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

echo "================================================================"
echo "EP16 Quick Recorder Test: 50 requests"
echo "Profile: $PROFILE"
echo "Head: $(hostname), Workers: ${WORKERS[*]}"
echo "================================================================"

kill_all

mkdir -p "$EXP_DIR/recorder_raw"

tmux new-session -d -s sglang-head \
    "bash ${SCRIPT_DIR}/launch_head_ep_record.sh \
     $DIST_INIT_ADDR ./gating_profiles/$PROFILE $EXP_DIR/server_head.log $EXP_DIR/recorder_raw '$MEM_FRAC' '$CUDA_GRAPH_MAX_BS'"

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
    echo "FAILED to start server"
    kill_all
    exit 1
fi

echo ""
echo "Starting recording..."
curl -X POST http://localhost:30000/start_expert_distribution_record 2>/dev/null || true

echo "Running benchmark: ${NUM_PROMPTS} prompts, rate=${REQUEST_RATE}"
python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio 0.5 \
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
    --seed 1 --warmup-requests 1 \
    2>&1 | tee "$EXP_DIR/bench.log"

echo ""
echo "Stopping and dumping recording..."
timeout 30 curl -X POST http://localhost:30000/stop_expert_distribution_record 2>/dev/null || true
timeout 30 curl -X POST http://localhost:30000/dump_expert_distribution_record 2>/dev/null || true
sleep 3

kill_all

echo ""
echo "Collecting recorder .pt files from workers..."
for w in "${WORKERS[@]}"; do
    rsync -az "$w:~/coulson/sglang-fake-prefill/$EXP_DIR/recorder_raw/" "$EXP_DIR/recorder_raw/" 2>/dev/null || true
done
echo "Collection done."

echo ""
echo "Checking for .pt files..."
PT_FILES=$(find "$EXP_DIR/recorder_raw" -name "*.pt" 2>/dev/null)
if [ -z "$PT_FILES" ]; then
    echo "ERROR: No .pt files found in $EXP_DIR/recorder_raw/"
    echo "Recorder may not have dumped data."
    exit 1
fi
echo "Found .pt files:"
echo "$PT_FILES"

echo ""
echo "Validating MoE times (checking for non-zero values)..."
python3 -c "
import torch, sys, os, glob
pt_files = sorted(glob.glob('$EXP_DIR/recorder_raw/moe_kernel_balance_*.pt'))
if not pt_files:
    print('ERROR: No moe_kernel_balance_*.pt files found')
    sys.exit(1)
data = torch.load(pt_files[-1], map_location='cpu', weights_only=False)
moe_times = data['moe_times'].float().numpy()
ltok = data['local_token_counts'].int().numpy()
print(f'moe_times shape: {moe_times.shape} (steps, layers, ranks)')
print(f'local_token_counts shape: {ltok.shape}')
print(f'moe_times — min: {moe_times.min():.4f}, max: {moe_times.max():.4f}, mean: {moe_times.mean():.4f} ms')
print(f'non-zero entries: {(moe_times > 0).sum()} / {moe_times.size} ({100*(moe_times > 0).mean():.1f}%)')
print(f'local_token_counts — min: {ltok.min()}, max: {ltok.max()}, mean: {ltok.mean():.1f}')
if moe_times.max() > 0:
    print('SUCCESS: MoE times are non-zero — cudagraph timing fix works!')
else:
    print('FAILURE: All MoE times are zero — cudagraph timing fix did not work')
    sys.exit(1)
"
VALIDATE_RC=$?

if [ $VALIDATE_RC -eq 0 ]; then
    echo ""
    echo "Generating plots..."
    PLOT_DIR="experiments/plots/${EXP_ID}"
    mkdir -p "$PLOT_DIR"
    python experiments/plot_moe_kernel_balance.py \
        "$EXP_DIR/recorder_raw" \
        --output-dir "$PLOT_DIR" \
        --warmup 5 --peak-pct 0 2>&1 || echo "WARNING: plot_moe_kernel_balance.py failed"

    python experiments/plot_moe_recorder_compare.py \
        --experiments "quick_test:$EXP_DIR" \
        --output-dir "$PLOT_DIR" \
        --peak-pct 0 2>&1 || echo "WARNING: plot_moe_recorder_compare.py failed"

    echo ""
    echo "================================================================"
    echo "QUICK TEST PASSED — MoE time recording works with cudagraph!"
    echo "Plots saved to: $PLOT_DIR/"
    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "QUICK TEST FAILED — MoE times are still zero"
    echo "Will need to evaluate --disable-cuda-graph approach"
    echo "================================================================"
    exit 1
fi
