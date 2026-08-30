#!/usr/bin/env bash
# run_head_streaminfer.sh — ONE command on the HEAD for the StreamInfer interference sweep.
#
#   bash experiment_utils/interference-resist/run_head_streaminfer.sh
#
# Workers are launched automatically over passwordless SSH — nothing to run on them.
# Starts Ray head, waits for all GPUs, launches ONE StreamInfer server (reused for all
# conditions), then for each interference condition: start trace-driven RDMA interference,
# run the fixed-rate benchmark, stop interference, record. Finally parses.
set -uo pipefail
UTIL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$UTIL_DIR/config.sh"
source "$UTIL_DIR/lib_interference.sh"

# ── StreamInfer-specific settings (mirror throughput-itl) ─────────────────────
CONDA_ENV="${CONDA_ENV:-streaminfer}"
REPO_DIR="${REPO_DIR:-$ARTIFACT_ROOT/StreamInfer}"
RESULTS_DIR="${RESULTS_DIR:-$RESULTS_BASE/streaminfer}"
PYTHON_BIN="$MINICONDA/envs/$CONDA_ENV/bin/python"
RAY_BIN="$MINICONDA/envs/$CONDA_ENV/bin/ray"
RAY_PORT="${RAY_PORT:-6379}"; SERVER_PORT="${SERVER_PORT:-6699}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"   # boot fast-fail: healthy boots take ~2-4min (8 Ray-actor engines init serially); hangs go to retry (BENCH_CURL_TIMEOUT still guards serving)
MODEL="${MODEL:-gptoss_120b}"
DP_SIZE="${DP_SIZE:-$WORLD_SIZE}"; EP_SIZE="${EP_SIZE:-$WORLD_SIZE}"
PLACEMENT="${PLACEMENT:-colocate}"; TRANSPORT="${TRANSPORT:-zmq}"
MEM_FRAC="${MEM_FRAC:-0.90}"
MAX_BATCH_SIZE_ATTN="${MAX_BATCH_SIZE_ATTN:-256}"; MAX_BATCH_SIZE_EXP="${MAX_BATCH_SIZE_EXP:-1024}"
MAX_PENDING_SENDS="${MAX_PENDING_SENDS:-16}"; BLOCK_SIZE="${BLOCK_SIZE:-16}"
UNIFIED_SCHEDULER_TYPE="${UNIFIED_SCHEDULER_TYPE:-defrag}"
DEFRAG_WEIGHT_DECAY="${DEFRAG_WEIGHT_DECAY:-0.8}"
DEFRAG_LOOKAHEAD_STEPS="${DEFRAG_LOOKAHEAD_STEPS:-4}"; DEFRAG_LOOKBACK_STEPS="${DEFRAG_LOOKBACK_STEPS:-4}"
DATASET_PATH="${DATASET_PATH:-$REPO_DIR/datasets/sharegpt_lengths.npy}"
BENCH_CURL_TIMEOUT="${BENCH_CURL_TIMEOUT:-3600}"

source "$MINICONDA/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
# disagmoe_c.so links libtorch but its RPATH misses torch's lib dir; add it so the raw
# `import disagmoe_c` (server.py imports it before the torch-loading disagmoe pkg) resolves.
export LD_LIBRARY_PATH="$("$PYTHON_BIN" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
log(){ echo "$(date '+%H:%M:%S') [head/streaminfer] $*"; }

[ -f "$DATASET_PATH" ] || { log "ERROR: dataset not found: $DATASET_PATH"; exit 1; }
[ -f "$GATE_PROFILE" ] || { log "ERROR: gate profile not found: $GATE_PROFILE"; exit 1; }
"$PYTHON_BIN" -c "import disagmoe_c" 2>/dev/null \
  || { log "ERROR: 'import disagmoe_c' failed in env '$CONDA_ENV'."; exit 1; }
# start clean: wipe THIS system's results dir (not the shared base — the other
# system's results must survive for the combined plot).
rm -rf "$RESULTS_DIR"; mkdir -p "$RESULTS_DIR"
AW_END=$(( BENCH_TIME - 10 )); [ "$AW_END" -lt 5 ] && AW_END="$BENCH_TIME"
if [ "$BENCH_TIME" -ge 120 ]; then AW_START=30; else AW_START=$(( BENCH_TIME / 5 )); fi

kill_server(){ pkill -f "[b]enchmark/server.py" 2>/dev/null || true; sleep 4; pkill -9 -f "[b]enchmark/server.py" 2>/dev/null || true; }
# workers are SSH-launched from the head (after the Ray head is up, so they never
# join a stale head) and torn down automatically at exit — nothing to run on them.
launch_workers(){
  local w
  for w in $WORKER_HOSTS; do
    # CONDA_ENV/MINICONDA are passed explicitly: ssh does not carry the head's
    # environment, so without this the worker would silently fall back to its
    # own defaults and join Ray from a different env than the head's engines.
    ssh -n -o BatchMode=yes "$w" "CONDA_ENV=$CONDA_ENV MINICONDA=$MINICONDA setsid nohup bash $UTIL_DIR/run_worker_streaminfer.sh > /tmp/si_worker.log 2>&1 < /dev/null & sleep 1; exit 0" 2>/dev/null \
      && log "  worker launched on $w" || log "  WARNING: could not launch worker on $w"
  done
}
stop_workers(){
  local w
  for w in $WORKER_HOSTS; do
    ssh -n -o BatchMode=yes "$w" "pkill -f '[r]un_worker_streaminfer' 2>/dev/null; sleep 2; pkill -9 -f '[r]un_worker_streaminfer' 2>/dev/null; '$MINICONDA/envs/$CONDA_ENV/bin/ray' stop --force >/dev/null 2>&1; true" 2>/dev/null || true
  done
}
cleanup(){ iface_stop 2>/dev/null || true; kill_server; stop_workers; "$RAY_BIN" stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

start_ray_head(){
  log "Starting Ray head at $HEAD_IP:$RAY_PORT ..."
  kill_server; "$RAY_BIN" stop --force >/dev/null 2>&1 || true; sleep 3
  "$RAY_BIN" start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
     --dashboard-port=8265 --min-worker-port=30000 --max-worker-port=39999 \
     --disable-usage-stats >/dev/null || { log "ERROR: ray head failed"; exit 1; }
}
ray_gpu_count(){ "$PYTHON_BIN" - <<'PY' 2>/dev/null
import ray
try:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    print(int(ray.cluster_resources().get("GPU", 0)))
except Exception:
    print(0)
PY
}
wait_for_gpus(){
  log "Waiting for $WORLD_SIZE GPUs to join Ray..."
  local waited=0 n
  while :; do
    n="$(ray_gpu_count)"; n="${n:-0}"
    [ "$n" -ge "$WORLD_SIZE" ] && { log "Ray has $n/$WORLD_SIZE GPUs. Ready."; return 0; }
    [ $(( waited % 30 )) -eq 0 ] && log "  ...ray GPUs=$n/$WORLD_SIZE (${waited}s)"
    sleep 5; waited=$(( waited + 5 )); [ "$waited" -ge 600 ] && { log "ERROR: only $n/$WORLD_SIZE GPUs after 600s"; return 1; }
  done
}
launch_server(){
  local server_log="$1"
  local -a cmd=(
    "$PYTHON_BIN" benchmark/server.py -N "$N_NODE" -g "$N_GPU_PER_NODE" -u "$MEM_FRAC"
    --model "$MODEL" --num-layers "$NUM_LAYERS" --attn-qkv-quant none --moe-linear-quant none
    --router-mode "${ROUTER_MODE:-random}"
    --max-batch-size-attn "$MAX_BATCH_SIZE_ATTN" --max-attn-graph-bsz "$MAX_BATCH_SIZE_ATTN"
    --max-pending-sends "$MAX_PENDING_SENDS" --max-batch-size-expert "$MAX_BATCH_SIZE_EXP"
    --block-size "$BLOCK_SIZE" --placement "$PLACEMENT" --dp-size "$DP_SIZE" --ep-size "$EP_SIZE"
    --transport "$TRANSPORT" --host-ifname "$HOST_IFNAME" --nccl-ib-hca "$NCCL_IB_HCA"
    --nccl-ib-gid-index "$NCCL_IB_GID_INDEX" --unified-scheduler-type "$UNIFIED_SCHEDULER_TYPE"
    --defrag-weight-decay "$DEFRAG_WEIGHT_DECAY" --defrag-lookahead-steps "$DEFRAG_LOOKAHEAD_STEPS"
    --defrag-lookback-steps "$DEFRAG_LOOKBACK_STEPS" --less-than-sm90 --cuda-graph-attn --cuda-graph-expert
    --analyze-throughput --analyze-throughput-window "$AW_START,$AW_END"
    --gate-profile-file "$GATE_PROFILE"
  )
  cd "$REPO_DIR"; NCCL_RUNTIME_CONNECT=0 "${cmd[@]}" > "$server_log" 2>&1 & SERVER_PID=$!
}
wait_for_server(){
  local server_log="$1" waited=0
  log "  waiting for server ready (timeout ${SERVER_READY_TIMEOUT}s)..."
  while [ "$waited" -lt "$SERVER_READY_TIMEOUT" ]; do
    grep -qE "Running on http://0\.0\.0\.0|Running on all addresses" "$server_log" 2>/dev/null && { log "  server ready (${waited}s)."; sleep 3; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { log "  ERROR: server exited early. See $server_log"; return 1; }
    sleep 10; waited=$(( waited + 10 ))
  done
  log "  ERROR: server not ready within ${SERVER_READY_TIMEOUT}s."; return 1
}
run_benchmark(){
  local result_file="$1" payload
  payload=$(printf '{"rate":%d,"time":%d,"distribution":"dataset","dataset_path":"%s","dataset_max_context_len":%d,"min_input_len":%d,"max_input_len":%d,"min_output_len":%d,"max_output_len":%d}' \
    "$RATE" "$BENCH_TIME" "$DATASET_PATH" "$BENCH_MAX_CONTEXT_LEN" "$BENCH_MIN_IN" "$BENCH_MAX_IN" "$BENCH_MIN_OUT" "$BENCH_MAX_OUT")
  log "  POST /run_once  rate=${RATE} time=${BENCH_TIME}s"
  local code
  code=$(curl -s -o "$result_file" -w "%{http_code}" -X POST "http://localhost:${SERVER_PORT}/run_once" \
    -H "Content-Type: application/json" -d "$payload" --max-time "$BENCH_CURL_TIMEOUT")
  [ "$code" = "200" ] && { log "  benchmark OK (HTTP 200)."; return 0; } || { log "  benchmark FAILED (HTTP $code)."; return 1; }
}

log "StreamInfer interference sweep | model=$MODEL layers=$NUM_LAYERS world=$WORLD_SIZE gate_profile=$(basename "$GATE_PROFILE") rate=$RATE time=${BENCH_TIME}s | workers=[$WORKER_HOSTS] conditions=[$CONDITIONS]"
start_ray_head
launch_workers
wait_for_gpus || exit 1
server_log="$RESULTS_DIR/server.log"
# a hung boot leaves Ray-actor engines holding GPU memory + the ZMQ port on every node
# (they survive the frontend's death) — kill them before relaunching.
kill_leftover_engines(){
  kill_server
  local w
  for w in $WORKER_HOSTS; do
    ssh -o BatchMode=yes "$w" 'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; true' 2>/dev/null || true
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 15
}
BOOT_RETRIES="${BOOT_RETRIES:-5}"
booted=0
for battempt in $(seq 1 "$BOOT_RETRIES"); do
  log "Launching StreamInfer server (once, reused for all conditions; boot attempt ${battempt}/${BOOT_RETRIES})..."
  launch_server "$server_log"
  wait_for_server "$server_log" && { booted=1; break; }
  log "  boot attempt ${battempt} failed; killing leftover engines and retrying..."
  kill_leftover_engines
done
[ "$booted" -eq 1 ] || { tail -30 "$server_log"; exit 1; }
for COND in $CONDITIONS; do
  run_dir="$RESULTS_DIR/$COND"; mkdir -p "$run_dir"
  log "======================  condition=${COND}  ======================"
  iface_start "$COND" "$run_dir/interference.log" || log "  (interference start reported an issue; continuing)"
  run_benchmark "$run_dir/result.json" || true
  grep -iE "token_throughput|req_throughput|itl_latency_(mean|median)" "$run_dir/result.json" 2>/dev/null | sed 's/^/      /' || true
  iface_report || true
  iface_stop
  sleep 5
done
log "Sweep done. Parsing ->"
"$PYTHON_BIN" "$UTIL_DIR/parse_results.py" "$RESULTS_BASE" | tee "$RESULTS_DIR/summary.txt"
log "Results in $RESULTS_DIR"
