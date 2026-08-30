#!/usr/bin/env bash
# run_head_sglang.sh — THE single command for the sglang baseline (run on the HEAD).
#
#   bash experiment_utils/throughput-itl/run_head_sglang.sh
#
# For each request rate it launches a FRESH distributed sglang server (rank 0 here
# + rank N on each worker over SSH), benchmarks, and RETRIES on the transient
# crashes sglang shows under sustained load (OOM / NCCL abort / CUDA illegal
# access). Workers are launched automatically over passwordless SSH — set
# WORKER_HOSTS. gpt-oss, 1/4 layers, fake prefill, uncapped, EP (DP-attention
# + mooncake-nccl a2a) — matches the paper's ep16 profile scaled to WORLD_SIZE.
set -uo pipefail
UTIL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$UTIL_DIR/config.sh"
CONDA_ENV="${CONDA_ENV:-sglang}"
REPO_DIR="${REPO_DIR:-$ARTIFACT_ROOT/sglang_dummy_prefill}"
RESULTS_DIR="${RESULTS_DIR:-$RESULTS_BASE/sglang}"
WORKER_HOSTS="${WORKER_HOSTS:-sgpu7 sgpu8 sgpu9}"     # worker hostnames (become ranks 1..N); config.sh already sets this
PYTHON_BIN="$MINICONDA/envs/$CONDA_ENV/bin/python"
SERVER_PORT="${SERVER_PORT:-30000}"; DIST_INIT_PORT="${DIST_INIT_PORT:-25000}"
DIST_TIMEOUT="${DIST_TIMEOUT:-1800}"; SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-180}"   # boot fast-fail: healthy 8-rank boots are ready in ~1-2.5min; hangs go to retry (BENCH_TIMEOUT still guards serving)
MODEL_PATH="${MODEL_PATH:-lmsys/gpt-oss-120b-bf16}"; MODEL_NAME="${MODEL_NAME:-$MODEL_PATH}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
MEM_FRAC="${MEM_FRAC:-0.76}"                          # VRAM-bounds concurrency (uncapped); fits rates <=100
MEM_FRAC_HIGH="${MEM_FRAC_HIGH:-0.70}"                # rates > threshold saturate deeper -> lower mem for activation headroom
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-300}"
HIGH_RATE_THRESHOLD="${HIGH_RATE_THRESHOLD:-100}"
MEM_FRAC_EFF="$MEM_FRAC"                               # effective mem for the current rate (set per-rate in the loop)
DATASET_PATH="${DATASET_PATH:-$REPO_DIR/datasets/sharegpt_lengths.npy}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-1200}"
MAX_RETRIES="${MAX_RETRIES:-5}"
GLOO_IFNAME="${GLOO_IFNAME:-ens1f0np0}"               # dedicated NIC for the DP-attention GLOO barrier — keeps the per-step CPU/TCP sync off the NCCL datapath interface (else it drops under load)
NCCL_ENV="NCCL_SOCKET_IFNAME=$HOST_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA GLOO_SOCKET_IFNAME=$GLOO_IFNAME NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX NCCL_DEBUG=WARN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
source "$MINICONDA/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
log(){ echo "$(date '+%H:%M:%S') [head/sglang] $*"; }

[ -f "$DATASET_PATH" ] || { log "ERROR: dataset not found: $DATASET_PATH"; exit 1; }
[ -f "$GATE_PROFILE" ] || { log "ERROR: gate profile not found: $GATE_PROFILE"; exit 1; }
"$PYTHON_BIN" -c "import sglang" 2>/dev/null || { log "ERROR: 'import sglang' failed in env '$CONDA_ENV'."; exit 1; }
# start clean: wipe THIS system's results dir (not the shared base — the other
# system's results must survive for the combined plot).
rm -rf "$RESULTS_DIR"; mkdir -p "$RESULTS_DIR"

# single-line launch command for a given node rank (--host only on rank 0)
_launch_cmd(){
  local rank="$1" hostarg=""
  [ "$rank" -eq 0 ] && hostarg="--host 0.0.0.0 --port $SERVER_PORT"
  echo "python -m sglang.launch_server --model-path $MODEL_PATH --load-format $LOAD_FORMAT --nnodes $N_NODE --node-rank $rank --dist-init-addr $HEAD_IP:$DIST_INIT_PORT $hostarg --enable-fake-prefill --disable-radix-cache --chunked-prefill-size -1 --mem-fraction-static $MEM_FRAC_EFF --trust-remote-code --moe-runner-backend triton --dist-timeout $DIST_TIMEOUT --log-level-http warning --log-level warning --num-hidden-layers-override $NUM_LAYERS --tp-size $WORLD_SIZE --dp-size $WORLD_SIZE --ep-size $WORLD_SIZE --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend mooncake-nccl --cuda-graph-max-bs $CUDA_GRAPH_MAX_BS --profile-driven-gate-path $GATE_PROFILE ${SGLANG_EXTRA_ARGS:-}"
}

# sglang renames its subprocesses via setproctitle to "sglang::scheduler_TPx" /
# "sglang::detoken" etc., which the launch_server/srt patterns miss — so they survive
# pkill, keep the GPUs busy and hold ports 25000-25005/30000, and the next relaunch dies
# with "metrics_ipc at 25004 is not available". Also kill "sglang::*", fuser -k the ports,
# then wait for the ports AND GPU memory to actually clear. ([s]/[t] brackets keep pkill
# from matching this script.)
SGLANG_PORTS="${SGLANG_PORTS:-25000 25001 25002 25003 25004 25005 30000}"
_KILL_CMD='pkill -9 -f "[s]glang.launch_server"; pkill -9 -f "[s]glang::"; pkill -9 -f "[s]glang.srt"; pkill -9 -f "[t]orch._inductor.compile_worker"; for p in '"$SGLANG_PORTS"'; do fuser -k ${p}/tcp >/dev/null 2>&1; done; true'
kill_all_sglang(){
  bash -c "$_KILL_CMD" 2>/dev/null || true
  for w in $WORKER_HOSTS; do ssh -o BatchMode=yes "$w" "$_KILL_CMD" 2>/dev/null || true; done
  # wait for the head's sglang PROCESSES to actually die — a SIGKILLed server can
  # linger ~1 min in CUDA/NCCL teardown (D-state), still holding dist_init_port 25000
  # in a non-LISTEN state the ss check below can't see ("dist_init_port ... not available").
  local waited=0
  while pgrep -f "[s]glang.launch_server|[s]glang::" >/dev/null 2>&1; do
    sleep 2; waited=$(( waited + 2 )); [ "$waited" -ge 90 ] && { log "  WARN: sglang procs still alive after ${waited}s"; break; }
  done
  waited=0
  while ss -tln 2>/dev/null | grep -qE ":2500[0-5] |:30000 "; do
    sleep 2; waited=$(( waited + 2 )); [ "$waited" -ge 40 ] && { log "  WARN: sglang ports still busy after ${waited}s"; break; }
  done
  waited=0
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)" -gt "${GPU_FREE_MIB:-2000}" ] 2>/dev/null; do
    sleep 3; waited=$(( waited + 3 )); [ "$waited" -ge 60 ] && { log "  WARN: GPU mem not released after ${waited}s"; break; }
  done
  sleep 4
}

launch_servers(){          # $1 = log dir for this attempt
  local logdir="$1" rank=1
  cd "$REPO_DIR"
  env $NCCL_ENV $(_launch_cmd 0) > "$logdir/server_head.log" 2>&1 &
  SERVER_PID=$!
  for w in $WORKER_HOSTS; do
    ssh -n -o BatchMode=yes "$w" "cd $REPO_DIR && source $MINICONDA/etc/profile.d/conda.sh && conda activate $CONDA_ENV && nohup env $NCCL_ENV $(_launch_cmd $rank) > /tmp/sglang_worker_rank${rank}.log 2>&1 </dev/null &" >/dev/null 2>&1 &
    rank=$((rank + 1))
  done
  disown -a 2>/dev/null || true   # keep the shell from printing "Killed" when kill_all_sglang reaps these bg procs between rates
  sleep 3   # let the backgrounded SSH worker dispatch(es) fire and return
}

wait_for_server(){
  local waited=0
  while [ "$waited" -lt "$SERVER_READY_TIMEOUT" ]; do
    curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1 && { sleep 3; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || return 1
    sleep 10; waited=$(( waited + 10 ))
  done
  return 1
}

run_benchmark(){
  local rate="$1" result_file="$2"; local nprompts=$(( rate * BENCH_TIME ))
  timeout "$BENCH_TIMEOUT" "$PYTHON_BIN" -m sglang.bench_serving \
    --backend sglang --host localhost --port "$SERVER_PORT" --model "$MODEL_NAME" \
    --dataset-name npy --dataset-path "$DATASET_PATH" --sharegpt-context-len "$BENCH_MAX_CONTEXT_LEN" \
    --num-prompts "$nprompts" --request-rate "$rate" --output-file "$result_file" \
    > "${result_file%.json}.log" 2>&1
}

log "sglang baseline (fresh server + retries per rate) | layers=$NUM_LAYERS world=$WORLD_SIZE mem=$MEM_FRAC uncapped gate_profile=$(basename "$GATE_PROFILE") | workers=[$WORKER_HOSTS] rates=[$RATES] time=${BENCH_TIME}s"
for RATE in $RATES; do
  run_dir="$RESULTS_DIR/sharegpt-${RATE}rps"; mkdir -p "$run_dir"; rm -f "$run_dir/result.json"
  MEM_FRAC_EFF="$MEM_FRAC"; [ "$RATE" -gt "$HIGH_RATE_THRESHOLD" ] && MEM_FRAC_EFF="$MEM_FRAC_HIGH"
  ok=0
  for attempt in $(seq 1 "$MAX_RETRIES"); do
    log "======= rate=${RATE} rps | mem=${MEM_FRAC_EFF} | attempt ${attempt}/${MAX_RETRIES} ======="
    kill_all_sglang
    launch_servers "$run_dir"
    if wait_for_server; then
      log "  server ready; benchmarking ${RATE} rps ..."
      if run_benchmark "$RATE" "$run_dir/result.json" && grep -q output_throughput "$run_dir/result.json" 2>/dev/null; then
        # report the SERVER-side averages (Detokenizer Manager 10s windows) — the
        # client-side result.json numbers deviate under overload; parse_results.py
        # uses the same server-log averages for the final table.
        log "  OK: $(awk '/from Detokenizer Manager/ && match($0,/Throughput: [0-9.]+/){t=substr($0,RSTART+12,RLENGTH-12); if(match($0,/ITL mean=[0-9.]+/)){st+=t; si+=substr($0,RSTART+9,RLENGTH-9); n++}} END{if(n) printf "server-log avg (%d windows): tput=%.0f tok/s, itl_mean=%.1f ms", n, st/n, si/n; else print "no Detokenizer lines parsed (see server_head.log)"}' "$run_dir/server_head.log")"
        nprompts=$(( RATE * BENCH_TIME ))
        completed=$(grep -oE '"completed": *[0-9]+' "$run_dir/result.json" | grep -oE '[0-9]+' || echo "?")
        log "  requests completed: ${completed}/${nprompts} ($(awk -v c="$completed" -v n="$nprompts" 'BEGIN{if(n>0&&c!="?")printf "%.1f",100*c/n; else printf "?"}')% success)"
        ok=1; break
      fi
      log "  benchmark crashed/failed on attempt ${attempt}."
    else
      log "  server not ready on attempt ${attempt} (see $run_dir/server_head.log)."
    fi
  done
  [ "$ok" -eq 0 ] && log "  rate ${RATE} FAILED after ${MAX_RETRIES} attempts."
  kill_all_sglang; sleep 3
done
log "Sweep done. Parsing ->"
"$PYTHON_BIN" "$UTIL_DIR/parse_results.py" "$RESULTS_BASE" | tee "$RESULTS_DIR/summary.txt"
log "Results in $RESULTS_DIR"
