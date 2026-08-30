#!/usr/bin/env bash
# lib_interference.sh — start/stop trace-driven RDMA network interference around a
# serving benchmark. Sourced by run_head_{streaminfer,sglang}.sh (config.sh first).
#
# Implements the paper's four interference modes as ONE-directional point-to-point
# flows of the `aws_hpc_metal` trace, driven over UCX **rc (RDMA)** — TCP interference
# can't contend with NCCL's RDMA on a lossless RoCE link, so all bulk traffic must be
# RDMA. Each flow = IFACE_STREAMS parallel ucx_sender→ucx_receiver pairs replaying a
# rate schedule. The schedule is generated once on the head (from the trace, with the
# per-mode intensity multiplier) and deployed to the flow endpoints; the C binaries are
# launched directly with the validated rc env (no rdma_cm — see README "UCX / RDMA").

_iface_local="$(hostname -s)"
# env prefix for every ucx_sender/ucx_receiver (rc data + tcp_sockcm wireup).
# UCX_IB_TRAFFIC_CLASS is pinned to NCCL's default TC (NCCL_IB_TC=0) so both
# workloads land in the SAME switch queue / DCQCN loop — UCX's 'auto' TC (106)
# could otherwise be queued separately by a DSCP-trusting switch, silently
# preventing any real contention.
IFACE_UCX_ENV="LD_LIBRARY_PATH=${UCX_LIB_DIR} UCX_TLS=rc,tcp UCX_NET_DEVICES=${UCX_NET_DEV_LIST} UCX_IB_GID_INDEX=${UCX_IB_GID_INDEX} UCX_SOCKADDR_TLS_PRIORITY=${UCX_SOCKADDR_TLS_PRIORITY} UCX_IB_TRAFFIC_CLASS=${UCX_IB_TRAFFIC_CLASS:-0}"

# parse IFACE_NODES -> IFACE_H[] hosts, IFACE_IP[] datapath IPs
_iface_parse_nodes(){
  local e; IFACE_H=(); IFACE_IP=()
  IFS=',' read -ra e <<< "$IFACE_NODES"
  local x; for x in "${e[@]}"; do IFACE_H+=("${x%%:*}"); IFACE_IP+=("${x##*:}"); done
}

# iface_resolve <cond> -> sets IFACE_FLOWS (array of "srcH srcIP dstH dstIP") + IFACE_MULT.
# Nodes are paired (n0,n1) (n2,n3) ... — the all-links modes touch every pair, so
# "half the nodes drive the other half" at any (even) cluster size.
# Returns 1 for the "none" baseline, 2 for an unknown condition.
iface_resolve(){
  _iface_parse_nodes
  IFACE_FLOWS=(); IFACE_MULT=1
  local n=${#IFACE_H[@]} i
  local A="${IFACE_H[0]} ${IFACE_IP[0]}" B="${IFACE_H[1]} ${IFACE_IP[1]}"
  case "$1" in
    none)            return 1 ;;
    single-link)     IFACE_FLOWS=("$A $B") ;;
    single-link-2x)  IFACE_FLOWS=("$A $B"); IFACE_MULT=2 ;;
    all-links)       for (( i=0; i+1<n; i+=2 )); do
                       IFACE_FLOWS+=("${IFACE_H[$i]} ${IFACE_IP[$i]} ${IFACE_H[$((i+1))]} ${IFACE_IP[$((i+1))]}")
                     done ;;
    bidir-all-links) for (( i=0; i+1<n; i+=2 )); do
                       IFACE_FLOWS+=("${IFACE_H[$i]} ${IFACE_IP[$i]} ${IFACE_H[$((i+1))]} ${IFACE_IP[$((i+1))]}")
                       IFACE_FLOWS+=("${IFACE_H[$((i+1))]} ${IFACE_IP[$((i+1))]} ${IFACE_H[$i]} ${IFACE_IP[$i]}")
                     done ;;
    *) echo "iface: unknown condition '$1'" >&2; return 2 ;;
  esac
  # optional global intensity override (e.g. IFACE_EXTRA_MULT=20 to stress-test whether
  # congestion actually lands on the inference's traffic path). Rates are capped at the
  # link capacity in the schedule generator, so this just pins the trace near saturation.
  IFACE_MULT=$(( IFACE_MULT * ${IFACE_EXTRA_MULT:-1} ))
  return 0
}

_iface_on(){ local h="$1"; shift; if [ "$h" = "$_iface_local" ]; then bash -c "$*"; else ssh -n -o BatchMode=yes "$h" "$*"; fi; }

# generate the rate schedule on the head: trace -> per-window interference rates, x mult,
# divided across IFACE_STREAMS. (interfere.py uses only stdlib, so system python3 works.)
_iface_gen_schedule(){   # $1=out.bin
  python3 - "$INTERFERENCE_DIR" "$IFACE_TRACE" "$LINK_CAP_GBPS" "$IFACE_MULT" "$IFACE_STREAMS" "$1" <<'PY'
import sys
root, trace, cap, mult, streams, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, root)
from interfere import resolve_trace_path, load_trace_rates, write_schedule_v2
cap_bps = cap * 1e9 / 8.0
rates, wsec = load_trace_rates(resolve_trace_path(trace), cap_bps, window_ms=1)
if mult != 1.0:
    rates = [min(r * mult, cap_bps) for r in rates]
rates = [r / streams for r in rates]     # split across parallel streams per flow
write_schedule_v2(rates, wsec, 65536, out)
PY
}

# deploy the schedule + binaries to a node's /tmp (local cp for the head)
_iface_deploy(){   # $1=host $2=schedule
  local h="$1" s="$2"
  if [ "$h" = "$_iface_local" ]; then
    cp -f "$INTERFERENCE_DIR/ucx_sender" "$INTERFERENCE_DIR/ucx_receiver" "$s" /tmp/ 2>/dev/null
  else
    scp -q -o BatchMode=yes "$INTERFERENCE_DIR/ucx_sender" "$INTERFERENCE_DIR/ucx_receiver" "$s" "$h":/tmp/ 2>/dev/null
  fi
}

# launch one one-directional flow: IFACE_STREAMS receivers on dst, senders on src.
# Sender stdout goes to /tmp/iface_snd_<port>.log — ucx_sender prints achieved-vs-
# target Gbps every 5s, which is the ground truth for whether the interference
# actually held its offered rate while contending with NCCL (iface_report sums it).
# --max-outstanding/--burst-bytes deepen the send pipeline so achieved ~= offered
# under congestion (the defaults collapse to ~1MB in flight once RTT inflates).
_iface_launch_flow(){   # $1=srcH $2=srcIP $3=dstH $4=dstIP $5=port_base $6=duration $7=schedule_basename
  local sH="$1" dstH="$3" dstIP="$4" pb="$5" dur="$6" sched="$7" S=$(( IFACE_STREAMS - 1 ))
  local sargs="--max-outstanding ${IFACE_MAX_OUTSTANDING:-64} --burst-bytes ${IFACE_BURST_BYTES:-8388608}"
  _iface_on "$dstH" "cd /tmp && for i in \$(seq 0 $S); do env $IFACE_UCX_ENV setsid ./ucx_receiver \$(( $pb + i )) --max-msg 65536 --duration $(( dur + 90 )) >/tmp/iface_rcv_\$(( $pb + i )).log 2>&1 </dev/null & done; disown -a 2>/dev/null; true"
  sleep 1
  _iface_on "$sH" "cd /tmp && rm -f /tmp/iface_snd_1*.log; for i in \$(seq 0 $S); do env $IFACE_UCX_ENV setsid ./ucx_sender $dstIP \$(( $pb + i )) --schedule $sched --duration $dur $sargs >/tmp/iface_snd_\$(( $pb + i )).log 2>&1 </dev/null & done; disown -a 2>/dev/null; true"
}

# Head-node datapath IB TX counter in bytes (device mlx5_1, or rocep225s0f1 on sgpu9).
_iface_head_ib_tx(){
  local d c
  for d in mlx5_1 rocep225s0f1; do
    c="/sys/class/infiniband/$d/ports/1/counters/port_xmit_data"
    [ -e "$c" ] && { echo $(( $(cat "$c") * 4 )); return; }
  done
  echo 0
}

# Background sampler on the head: IB TX/RX bytes + RoCE congestion hw counters
# (ECN marks / CNPs = DCQCN engaging = proof the two workloads really contend)
# every 2s into <interference log>.counters.csv, for the whole condition.
_iface_sampler_start(){   # $1=csv path
  local csv="$1" dev=mlx5_1
  [ -e "/sys/class/infiniband/$dev/ports/1/counters/port_xmit_data" ] || dev=rocep225s0f1
  (
    local p="/sys/class/infiniband/$dev/ports/1"
    echo "ts,tx_bytes,rx_bytes,np_cnp_sent,rp_cnp_handled,np_ecn_marked" > "$csv"
    while :; do
      echo "$(date +%s),$(( $(cat $p/counters/port_xmit_data) * 4 )),$(( $(cat $p/counters/port_rcv_data) * 4 )),$(cat $p/hw_counters/np_cnp_sent 2>/dev/null || echo 0),$(cat $p/hw_counters/rp_cnp_handled 2>/dev/null || echo 0),$(cat $p/hw_counters/np_ecn_marked_roce_packets 2>/dev/null || echo 0)" >> "$csv"
      sleep 2
    done
  ) &
  IFACE_SAMPLER_PID=$!
}
_iface_sampler_stop(){ [ -n "${IFACE_SAMPLER_PID:-}" ] && kill "$IFACE_SAMPLER_PID" 2>/dev/null; IFACE_SAMPLER_PID=""; }

# Start interference for a condition (no-op for "none"). $1=condition, $2=log path.
# Blocks until the head's IB counter shows real RDMA traffic + a warmup. Sets IFACE_ACTIVE.
iface_start(){
  local cond="$1" log="$2"; : > "$log"
  IFACE_LOG="$log"; IFACE_ACTIVE=0
  iface_resolve "$cond" || { echo "$(date '+%H:%M:%S') [iface] condition=none (baseline, no interference)"; return 0; }
  # generous margin: the benchmark's drain tail can outlive BENCH_TIME by minutes; the
  # senders must outlive IT (they are TERM'd at iface_stop anyway, so this costs nothing)
  local dur=$(( IFACE_WARMUP + BENCH_TIME + 600 ))
  echo "$(date '+%H:%M:%S') [iface] '$cond': ${#IFACE_FLOWS[@]} flow(s) x${IFACE_MULT} trace=$IFACE_TRACE cap=${LINK_CAP_GBPS}Gbps streams=$IFACE_STREAMS (rc/RDMA)" | tee -a "$log"
  local sched="/tmp/iface_sched.bin"
  _iface_gen_schedule "$sched" >>"$log" 2>&1 || { echo "$(date '+%H:%M:%S') [iface] ERROR: schedule gen failed (see $log)"; return 1; }
  IFACE_HOSTS=""                                  # unique hosts touched (for cleanup)
  local f pb=18515
  for f in "${IFACE_FLOWS[@]}"; do
    set -- $f; local sH="$1" sIP="$2" dH="$3" dIP="$4"
    echo "$(date '+%H:%M:%S') [iface]   flow $sH -> $dH  ports $pb-$(( pb + IFACE_STREAMS - 1 ))" | tee -a "$log"
    _iface_deploy "$sH" "$sched"; _iface_deploy "$dH" "$sched"
    _iface_launch_flow "$sH" "$sIP" "$dH" "$dIP" "$pb" "$dur" "$(basename "$sched")"
    case " $IFACE_HOSTS " in *" $sH "*) :;; *) IFACE_HOSTS="$IFACE_HOSTS $sH";; esac
    case " $IFACE_HOSTS " in *" $dH "*) :;; *) IFACE_HOSTS="$IFACE_HOSTS $dH";; esac
    pb=$(( pb + IFACE_STREAMS ))
  done
  IFACE_ACTIVE=1
  _iface_sampler_start "${log%.log}.counters.csv"
  # confirm real RDMA traffic on the head (poll IB TX; flows settle a startup race ~15s)
  local waited=0 gbps=0 t0 t1
  while [ "$waited" -lt 75 ]; do
    t0=$(_iface_head_ib_tx); sleep 3; t1=$(_iface_head_ib_tx)
    gbps=$(( (t1 - t0) / 3 / 125000000 ))
    [ "$gbps" -ge 1 ] && break
    waited=$(( waited + 3 ))
  done
  if [ "$gbps" -ge 1 ]; then
    echo "$(date '+%H:%M:%S') [iface] interference live: head TX ~${gbps} Gbps RDMA (settled ~${waited}s); warmup ${IFACE_WARMUP}s ..." | tee -a "$log"
  else
    echo "$(date '+%H:%M:%S') [iface] WARNING: no RDMA traffic after ${waited}s — mark '$cond' suspect (see $log)" | tee -a "$log"
  fi
  sleep "$IFACE_WARMUP"
  return 0
}

# Stop all interference flows (no-op if inactive). SIGTERM first so senders
# drain and print their "=== Sender Summary ===" (harvested into the log),
# then SIGKILL stragglers.
iface_stop(){
  [ "${IFACE_ACTIVE:-0}" -eq 1 ] || return 0
  echo "$(date '+%H:%M:%S') [iface] stopping interference ..."
  _iface_sampler_stop
  local h
  for h in ${IFACE_HOSTS:-$_iface_local}; do
    _iface_on "$h" 'pkill -TERM -f ucx_sender 2>/dev/null; true'
  done
  sleep 2
  if [ -n "${IFACE_LOG:-}" ]; then
    for h in ${IFACE_HOSTS:-$_iface_local}; do
      _iface_on "$h" 'grep -h "Avg BW" /tmp/iface_snd_*.log 2>/dev/null | awk -v h="$(hostname -s)" "{s+=\$4} END {if (NR>0) printf \"[iface] %s sender whole-run avg: %.1f Gbps across %d streams\n\", h, s, NR}"' >> "$IFACE_LOG" 2>/dev/null || true
    done
  fi
  for h in ${IFACE_HOSTS:-$_iface_local}; do
    _iface_on "$h" 'pkill -9 -f ucx_sender 2>/dev/null; pkill -9 -f ucx_receiver 2>/dev/null; true'
  done
  IFACE_ACTIVE=0
  sleep 3
}

# Report the interference actually achieved DURING contention: per flow, sum the
# senders' latest 5s achieved/target Gbps from their logs; plus the head's total
# IB TX (NCCL share = head total − head's own senders' achieved).
iface_report(){
  [ "${IFACE_ACTIVE:-0}" -eq 1 ] || return 0
  local f pb=18515 S=$(( IFACE_STREAMS - 1 ))
  for f in "${IFACE_FLOWS[@]}"; do
    set -- $f; local sH="$1" dH="$3"
    local line
    line=$(_iface_on "$sH" "for i in \$(seq 0 $S); do tail -n 4 /tmp/iface_snd_\$(( $pb + i )).log 2>/dev/null | grep '\[sender\] t=' | tail -1; done" 2>/dev/null | \
      awk '{ ach += $3; t = $6; sub(/\(target=/, "", t); tgt += t + 0 } END { if (NR > 0) printf "achieved %.1f / target %.1f Gbps (%d streams reporting)", ach, tgt, NR; else printf "no sender stats" }')
    echo "$(date '+%H:%M:%S') [iface]   flow $sH -> $dH: $line" | tee -a "${IFACE_LOG:-/dev/null}"
    pb=$(( pb + IFACE_STREAMS ))
  done
  local t0 t1; t0=$(_iface_head_ib_tx); sleep 3; t1=$(_iface_head_ib_tx)
  echo "$(date '+%H:%M:%S') [iface] head TOTAL IB TX ~$(( (t1 - t0) / 3 / 125000000 )) Gbps (interference + NCCL)" | tee -a "${IFACE_LOG:-/dev/null}"
}
