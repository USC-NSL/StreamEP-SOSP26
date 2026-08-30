#!/bin/bash
set -euo pipefail

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

WORKERS=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)
WORKER_RANKS=(1 2 3 4 5 6 7)
PEER_HOST="sgpu2"
PEER_IP="10.0.0.2"
TRACES=("aws_hpc_metal" "azure_hpc_200g")
MEM_FRACS=("0.77" "0.55")
EXP_ID="sgl-045"
PROFILE="balanced_legal_court_opinions_200.parquet"
PROFILE_NAME="balanced_legal"
NUM_PROMPTS=10000
REQUEST_RATE=2000
INPUT_LEN=512
OUTPUT_LEN=512
RANGE_RATIO=0.5
MAX_RUNNING_REQS=
CUDA_GRAPH_MAX_BS=
BENCH_TIMEOUT_S=1800
LINK_CAP_GBPS=200
INTERFERE_WARMUP_S=5

mkdir -p "experiments/${EXP_ID}"

kill_all() {
    pkill -9 -f "python -m sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang.srt" 2>/dev/null || true
    pkill -9 -f "python -m sglang.bench_serving" 2>/dev/null || true
    pkill -9 -f "ray" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true
    pkill -9 -f "torch._inductor" 2>/dev/null || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" 'pkill -9 -f "python -m sglang.launch_server" 2>/dev/null || true; pkill -9 -f "sglang.srt" 2>/dev/null || true; pkill -9 -f "python -m sglang.bench_serving" 2>/dev/null || true; pkill -9 -f "ray" 2>/dev/null || true; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true; pkill -9 -f "torch._inductor" 2>/dev/null || true' >/dev/null 2>&1 || true
    done
    for s in sglang-head sglang-w1 sglang-w2 sglang-w3 sglang-w4 sglang-w5 sglang-w6 sglang-w7; do
        tmux kill-session -t "$s" 2>/dev/null || true
    done
    bash interference_gen/run_interfere.sh --peer-host "$PEER_HOST" --stop >/dev/null 2>&1 || true
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

launch_ep16() {
    local mem_frac=$1
    local exp_dir=$2
    mkdir -p "$exp_dir/recorder_raw"
    tmux new-session -d -s sglang-head         "bash experiments/scripts/sphere-16/launch_head_ep_record.sh ./gating_profiles/${PROFILE} ${exp_dir}/server_head.log ${exp_dir}/recorder_raw ${mem_frac} ${MAX_RUNNING_REQS} ${CUDA_GRAPH_MAX_BS}"
    sleep 3
    for i in "${!WORKERS[@]}"; do
        local w="${WORKERS[$i]}"
        local rank="${WORKER_RANKS[$i]}"
        local sess="sglang-w$((i+1))"
        tmux new-session -d -s "$sess"             "ssh $w 'bash /home/yizhuoliang/sglang-fake-prefill/experiments/scripts/sphere-16/launch_worker_ep_record.sh ${rank} ./gating_profiles/${PROFILE} ${exp_dir}/server_w${rank}.log ${exp_dir}/recorder_raw ${mem_frac} ${MAX_RUNNING_REQS} ${CUDA_GRAPH_MAX_BS}'"
    done
}

start_interference() {
    local trace=$1
    local exp_dir=$2
    echo "  Starting interference trace=${trace} on sgpu0<->${PEER_HOST}"
    tmux new-session -d -s interfere-run         "cd interference_gen && ./run_interfere.sh --peer-host ${PEER_HOST} --peer-ip ${PEER_IP} --trace ${trace} --link-capacity-gbps ${LINK_CAP_GBPS} --profile --profile-output /home/yizhuoliang/sglang-fake-prefill/${exp_dir}/bw_profile_${trace}.csv"
    sleep ${INTERFERE_WARMUP_S}
}

stop_interference() {
    bash interference_gen/run_interfere.sh --peer-host "$PEER_HOST" --stop >/dev/null 2>&1 || true
    tmux kill-session -t interfere-run 2>/dev/null || true
}

run_bench() {
    local log_file=$1
    timeout "${BENCH_TIMEOUT_S}" python -m sglang.bench_serving         --backend sglang --host 127.0.0.1 --port 30000         --model lmsys/gpt-oss-120b-bf16         --dataset-name random         --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" --random-range-ratio "$RANGE_RATIO"         --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE"         --seed 1 --warmup-requests 1         2>&1 | tee "$log_file"
}

echo "==============================================================="
echo "${EXP_ID}: EP16 balanced legal with two-node interference"
echo "10k reqs, 2000 rps, input/output uniform [256,512], fake-prefill, recorder on"
echo "mem_fracs=${MEM_FRACS[*]} traces=${TRACES[*]} peer=${PEER_HOST}"
echo "==============================================================="

for trace in "${TRACES[@]}"; do
    for mem_frac in "${MEM_FRACS[@]}"; do
        tag="memfrac_${mem_frac/./}"
        exp_dir="experiments/${EXP_ID}/${tag}_${PROFILE_NAME}_trace_${trace}"
        bench_log="${exp_dir}/bench.log"

        echo ""
        echo "==============================================================="
        echo "Profile=${PROFILE_NAME} mem_frac=${mem_frac} trace=${trace}"
        echo "==============================================================="

        kill_all
        launch_ep16 "$mem_frac" "$exp_dir"
        if ! wait_for_server; then
            echo "  FAILED to launch mem_frac=${mem_frac} trace=${trace}"
            continue
        fi

        start_interference "$trace" "$exp_dir"
        curl -X POST http://127.0.0.1:30000/start_expert_distribution_record >/dev/null 2>&1 || true
        if ! run_bench "$bench_log"; then
            echo "  BENCH FAILED for mem_frac=${mem_frac} trace=${trace}" | tee -a "$bench_log"
        fi
        if ! curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1; then
            echo "  SERVER UNHEALTHY after bench for mem_frac=${mem_frac} trace=${trace}" | tee -a "$bench_log"
        fi
        timeout 30 curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record >/dev/null 2>&1 || true
        timeout 30 curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record >/dev/null 2>&1 || true
        stop_interference
        sleep 5
        for w in "${WORKERS[@]}"; do
            rsync -az "$w:/home/yizhuoliang/sglang-fake-prefill/${exp_dir}/recorder_raw/" "$exp_dir/recorder_raw/" >/dev/null 2>&1 || true
        done
        kill_all
    done
done

echo ""
echo "${EXP_ID} complete"
for trace in "${TRACES[@]}"; do
    for mem_frac in "${MEM_FRACS[@]}"; do
        tag="memfrac_${mem_frac/./}"
        f="experiments/${EXP_ID}/${tag}_${PROFILE_NAME}_trace_${trace}/bench.log"
        if [ -f "$f" ]; then
            echo "--- ${tag}_${PROFILE_NAME}_trace_${trace} ---"
            grep -E "Successful requests|Benchmark duration|Request throughput|Output token throughput|Concurrency|Median ITL|BENCH FAILED|SERVER UNHEALTHY" "$f" || true
        fi
    done
done
